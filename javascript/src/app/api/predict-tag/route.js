import { NextResponse } from 'next/server';
import { Mistral } from '@mistralai/mistralai';
import { ALLOWED_TAGS } from '@/constants/tags.js';

// Cette route porte le prompt `v1_zeroshot` mesure a l'etape 1 du projet
// zenassist-classification (`src/llm_prompts.py`, `_INSTRUCTION_V1_EN`).
// Le texte est recopie A L'OCTET PRES : c'est la variante dont le F1-macro est
// documente, la reecrire meme legerement rendrait cette mesure caduque.
const INSTRUCTION_V1_ZEROSHOT = `You are a classification system for consumer finance complaints submitted to the US Consumer Financial Protection Bureau.

Read the complaint provided by the user and assign it to exactly one of the product categories listed below.

Redaction notice: the complaints contain personal data replaced by runs of the letter X (for example XXXX, XX/XX/XXXX, $XXXX). This masking is normal and expected. Do not treat it as missing information, do not refuse to classify because of it, and do not comment on it.

You must pick exactly one category from the list, copied verbatim. Do not invent a category and do not return more than one.

Reply with a single minimal JSON object and nothing else:
{"label": "<one category, copied exactly from the list>"}

No explanation, no confidence score, no markdown code fences, no text before or after the JSON object.`;

// Reproduit `prefixe_fige("v1_zeroshot")` : instruction, puis les libelles dans
// l'ordre de `CLASS_ORDER`. La liste est derivee d'ALLOWED_TAGS plutot que
// recopiee, pour que le dropdown et le prompt ne puissent pas diverger.
const SYSTEM_PROMPT = `${INSTRUCTION_V1_ZEROSHOT}\n\nCategories:\n${ALLOWED_TAGS.map(
  (tag) => `- ${tag}`
).join('\n')}`;

const MODEL = 'mistral-small-latest';

export async function POST(request) {
  const apiKey = process.env.MISTRAL_API_KEY;
  if (!apiKey) {
    console.error('MISTRAL_API_KEY absente de l\'environnement');
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

  // `strategy: "none"` est deja le defaut du SDK, mais on l'ecrit : un retry
  // silencieux multiplierait les appels factures sans que l'UI le voie.
  const client = new Mistral({
    apiKey,
    retryConfig: { strategy: 'none' },
  });

  let raw;
  try {
    const response = await client.chat.complete({
      model: MODEL,
      temperature: 0,
      maxTokens: 20,
      responseFormat: { type: 'json_object' },
      messages: [
        { role: 'system', content: SYSTEM_PROMPT },
        { role: 'user', content },
      ],
    });
    raw = response?.choices?.[0]?.message?.content ?? null;
  } catch (error) {
    // `statusCode` n'existe que sur les erreurs HTTP du SDK (MistralError) ;
    // une panne reseau arrive ici sans ce champ.
    const status = error?.statusCode;
    if (status === 429) {
      console.error('Mistral: quota atteint');
      return NextResponse.json({ error: 'rate_limited' }, { status: 429 });
    }
    if (typeof status === 'number' && status >= 500) {
      console.error('Mistral: erreur serveur', status);
      return NextResponse.json({ error: 'upstream_error' }, { status: 502 });
    }
    if (typeof status === 'number') {
      console.error('Mistral: requete refusee', status);
      return NextResponse.json({ error: 'upstream_error' }, { status: 502 });
    }
    console.error('Mistral injoignable:', error);
    return NextResponse.json({ error: 'network_error' }, { status: 502 });
  }

  // Parsing STRICT, aligne sur `parse_reponse()` : on ne rattrape rien. Un
  // format non respecte n'est pas une classification, meme si un libelle est
  // devinable dans la reponse.
  if (typeof raw !== 'string' || !raw.trim()) {
    console.error('Mistral: reponse vide');
    return NextResponse.json({ error: 'parse_error' }, { status: 502 });
  }

  let parsed;
  try {
    parsed = JSON.parse(raw.trim());
  } catch {
    console.error('Mistral: JSON invalide:', raw);
    return NextResponse.json({ error: 'parse_error' }, { status: 502 });
  }

  if (
    parsed === null ||
    typeof parsed !== 'object' ||
    Array.isArray(parsed) ||
    typeof parsed.label !== 'string'
  ) {
    console.error('Mistral: cle "label" absente ou non textuelle:', raw);
    return NextResponse.json({ error: 'parse_error' }, { status: 502 });
  }

  const tag = parsed.label.trim();
  if (!ALLOWED_TAGS.includes(tag)) {
    console.error('Mistral: libelle hors referentiel:', tag);
    return NextResponse.json({ error: 'parse_error' }, { status: 502 });
  }

  return NextResponse.json({ tag });
}
