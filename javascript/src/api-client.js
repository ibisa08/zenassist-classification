// Removed Claim type import as it's not needed in JavaScript

const BASE_URL = '/api/claims';

export async function fetchClaims(tag) {
  const url = tag
    ? `${BASE_URL}?tag=${encodeURIComponent(tag)}`
    : BASE_URL;

  const response = await fetch(url);

  if (!response.ok) {
    throw new Error('Failed to fetch claims', { cause: response });
  }

  return response.json();
}

export async function updateClaimTag(claimId, tag) {
  const response = await fetch(`${BASE_URL}/${claimId}/tag`, {
    method: 'PUT',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ tag }),
  });

  if (!response.ok) {
    throw new Error('Failed to update tag');
  }
}

// Messages destines a l'utilisateur, indexes sur les codes rendus par
// /api/predict-tag. Ils disent quoi faire, pas ce qui a casse : le detail
// technique est deja dans les logs serveur.
const PREDICTION_ERRORS = {
  config_error: "L'auto-etiquetage n'est pas configure sur ce serveur.",
  rate_limited: 'Quota Mistral atteint, reessayez dans un moment.',
  upstream_error: 'Le service Mistral est indisponible, reessayez plus tard.',
  network_error: 'Impossible de joindre Mistral, verifiez la connexion.',
  parse_error: "L'IA n'a pas rendu de categorie exploitable, choisissez un tag manuellement.",
  invalid_body: 'Cette reclamation ne contient pas de texte a analyser.',
};

/**
 * Demande une categorie a Mistral pour le texte d'une reclamation.
 *
 * N'ENREGISTRE RIEN : la valeur rendue est une suggestion, la sauvegarde reste
 * le geste de l'utilisateur via updateClaimTag.
 *
 * @param {string} content texte de la reclamation
 * @returns {Promise<string>} un tag appartenant a ALLOWED_TAGS
 * @throws {Error} message pret a etre affiche dans l'UI
 */
export async function predictClaimTag(content) {
  let response;
  try {
    response = await fetch('/api/predict-tag', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ content }),
    });
  } catch (error) {
    throw new Error(PREDICTION_ERRORS.network_error, { cause: error });
  }

  let payload = null;
  try {
    payload = await response.json();
  } catch {
    // corps illisible : on retombe sur le code HTTP ci-dessous
  }

  if (!response.ok) {
    const message =
      PREDICTION_ERRORS[payload?.error] ?? "L'auto-etiquetage a echoue.";
    throw new Error(message);
  }

  if (typeof payload?.tag !== 'string' || !payload.tag) {
    throw new Error(PREDICTION_ERRORS.parse_error);
  }

  return payload.tag;
}
