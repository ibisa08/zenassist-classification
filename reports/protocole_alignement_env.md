# Protocole — alignement de l'environnement de référence du modèle LinearSVC

## Constat établi
- requirements.txt (commit de l'étape 1, 2026-08-11) désigne le .venv du
  projet comme environnement de référence : scikit-learn 1.9.0,
  pandas 3.0.5, matplotlib 3.11.1.
- Le paquet scikit-learn 1.9.0 est présent dans le .venv depuis le
  2026-08-11 17:41 (date de création de son dossier dist-info, inchangée).
- L'export du modèle du 2026-08-19 (models/LinearSVC.metadata.json)
  enregistre scikit-learn 1.6.1, numpy 2.1.3, pandas 2.2.3 : versions de
  l'interpréteur Anaconda de base, et non du .venv.
- Le même jour, le notebook d'exploration et la figure
  label_distribution.png ont été ré-exécutés sous ce même interpréteur
  (affichage pandas 2, métadonnée PNG Matplotlib 3.10.0). Ces sorties ne
  sont pas commitées.
- Les fichiers data/processed/ (train, test, échantillon de 2 000) n'ont
  pas été réécrits depuis le 2026-08-11 : leurs sha256 sont égaux à ceux
  de split_metadata.json.
- Chargé sous scikit-learn 1.9.0, le pickle du 2026-08-19 produit sur
  l'échantillon de 2 000 des prédictions identiques à celles obtenues
  sous 1.6.1.
Ce protocole établit une référence sous l'environnement déclaré.
Critères fixés avant toute mesure.

## Critères
- C0 — chaîne de données : src.data_prep rejoué sous le .venv, sur une
  copie, produit train.csv, test.csv et test_sample_2000.csv dont les
  sha256 sont égaux à ceux de split_metadata.json. Sinon : arrêt, aucun
  entraînement.
- C1 — identité : les prédictions du modèle réentraîné sous le .venv sont
  identiques à celles du pickle du 2026-08-19 évalué sous l'interpréteur
  Anaconda de base, sur test.csv (70 863 lignes) et sur
  test_sample_2000.csv. Conséquence : métriques publiées inchangées.
- C2 — équivalence tolérée (si C1 échoue) : taux d'accord >= 99,9 % sur
  test.csv ET |delta F1-macro| <= 0,002 sur test.csv non pondéré ET sur
  l'échantillon de 2 000 pondéré Horvitz-Thompson. Conséquence : nouvelle
  référence acceptée, métriques corrigées par erratum.
- C3 — rejet : tout autre cas. Arrêt et investigation.

## Justification des seuils
Choix raisonné, non dérivé analytiquement. 0,002 représente environ un
dixième de la demi-largeur de l'IC 95 % du F1-macro sur l'échantillon de
2 000 (environ 0,02), et un ordre de grandeur inférieur à la borne basse
de l'écart ML - LLM mesuré (+0,007). 99,9 % correspond à au plus 70
prédictions divergentes sur 70 863.

## Hors critères
L'empreinte de contenu est rapportée mais non décisionnelle : elle
vérifie le déterminisme à version de scikit-learn égale, pas entre
versions. Les critères C1 à C3 sont réutilisés comme contrôle dans
l'intégration continue, relativement à la nouvelle référence.
