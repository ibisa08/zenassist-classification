// Les 9 categories du referentiel zenassist-classification.
// L'ORDRE EST CELUI DE `config.CLASS_ORDER` cote LLM, donc celui de la liste
// envoyee a Mistral par /api/predict-tag : les deux doivent rester alignes,
// sinon un tag predit pourrait ne pas exister dans ce dropdown.
export const ALLOWED_TAGS = [
  'Credit reporting',
  'Debt collection',
  'Mortgage',
  'Credit card or prepaid card',
  'Bank account or service',
  'Student loan',
  'Money transfer or virtual currency',
  'Payday, title or personal loan',
  'Vehicle loan or lease'
];
