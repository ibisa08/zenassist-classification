// Les 9 categories du referentiel zenassist-classification.
// Cette liste doit rester IDENTIQUE aux classes du modele servi par l'API de
// classification (models/<version>/LinearSVC.metadata.json, cle
// `vocabulaire.classes`, dans le depot zenassist-classification) : les deux
// doivent contenir les memes 9 chaines, sinon un tag predit pourrait ne pas
// exister dans ce dropdown.
// L'ORDRE D'AFFICHAGE est independant de l'ordre des classes du modele : ici
// du plus frequent au plus rare (choix d'ergonomie du menu deroulant), alors
// que le metadata les liste par ordre alphabetique. L'API renvoie une chaine,
// jamais un indice : seule l'appartenance a la liste compte.
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
