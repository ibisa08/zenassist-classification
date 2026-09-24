import { NextResponse } from 'next/server';
import { ALLOWED_TAGS } from '@/constants/tags.js';

// Cette route delegue la classification a l'API Python du projet
// zenassist-classification (POST /tags). Elle ne fait que valider le corps,
// relayer le texte et controler la reponse avant de la rendre au client.

// Borne haute d'attente de l'API de classification. La prediction elle-meme
// est mesuree sous 10 ms : ce delai ne sert qu'a ne pas rester suspendu sur
// un service bloque (processus gele, port en ecoute sans reponse).
const DELAI_MAX_MS = 5000;

export async function POST(request) {
  // Adresse de l'API, sans slash final pour construire `${base}/tags`.
  const base = (process.env.ML_API_URL ?? '').trim().replace(/\/+$/, '');
  if (!base) {
    console.error('ML_API_URL absente ou vide dans l\'environnement');
    return NextResponse.json({ error: 'config_error' }, { status: 500 });
  }

  let content;
  try {
    ({ content } = await request.json());
  } catch {
    return NextResponse.json({ error: 'invalid_body' }, { status: 400 });
  }

  if (typeof content !== 'string' || !content.trim()) {
    return NextResponse.json({ error: 'invalid_body' }, { status: 400 });
  }

  // Appel de l'API. `no-store` : une reclamation identique doit toujours
  // repasser par le modele en service, jamais par un cache de Next.
  let response;
  try {
    response = await fetch(`${base}/tags`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ user_claim: content }),
      signal: AbortSignal.timeout(DELAI_MAX_MS),
      cache: 'no-store',
    });
  } catch (error) {
    // Le texte de la reclamation n'est JAMAIS journalise : seul le type
    // d'erreur l'est.
    const name = error?.name;
    if (name === 'TimeoutError' || name === 'AbortError') {
      console.error('API de classification: delai depasse', name, DELAI_MAX_MS);
      return NextResponse.json({ error: 'timeout' }, { status: 504 });
    }
    console.error('API de classification injoignable:', name ?? 'erreur inconnue');
    return NextResponse.json({ error: 'network_error' }, { status: 502 });
  }

  // 422 : l'API a refuse le corps (validation Pydantic). C'est une erreur du
  // demandeur, pas du service : on la rend comme telle.
  if (response.status === 422) {
    console.error('API de classification: corps refuse', response.status);
    return NextResponse.json({ error: 'invalid_body' }, { status: 400 });
  }

  if (!response.ok) {
    console.error('API de classification: erreur amont', response.status);
    return NextResponse.json({ error: 'upstream_error' }, { status: 502 });
  }

  // Parsing STRICT : un corps non JSON, un tag absent ou hors referentiel
  // n'est pas une classification, meme si un libelle serait devinable.
  let parsed;
  try {
    parsed = await response.json();
  } catch {
    console.error('API de classification: corps non JSON', response.status);
    return NextResponse.json({ error: 'parse_error' }, { status: 502 });
  }

  if (
    parsed === null ||
    typeof parsed !== 'object' ||
    Array.isArray(parsed) ||
    typeof parsed.tag !== 'string'
  ) {
    console.error('API de classification: cle "tag" absente ou non textuelle', response.status);
    return NextResponse.json({ error: 'parse_error' }, { status: 502 });
  }

  const tag = parsed.tag;
  if (!ALLOWED_TAGS.includes(tag)) {
    console.error('API de classification: tag hors referentiel', response.status);
    return NextResponse.json({ error: 'parse_error' }, { status: 502 });
  }

  // La version du modele est recopiee telle quelle si elle est textuelle,
  // sinon null : le client n'en depend pas, elle sert a la tracabilite.
  const version_modele =
    typeof parsed.version_modele === 'string' ? parsed.version_modele : null;

  return NextResponse.json({ tag, version_modele });
}
