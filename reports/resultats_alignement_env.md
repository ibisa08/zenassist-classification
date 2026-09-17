# Résultats — alignement de l'environnement de référence du modèle LinearSVC

Protocole pré-enregistré : `reports/protocole_alignement_env.md`
(commit `84dd2d4f1886a520d3ad0709a1cde0bf1fb35e25`, antérieur à toute mesure).

Mesures réalisées le 2026-09-17 sur une copie du projet
(`/tmp/zenassist-verif`), sous l'environnement déclaré dans
`requirements.txt` : `.venv/bin/python`, Python 3.13.5, scikit-learn 1.9.0,
numpy 2.5.2, scipy 1.18.0, pandas 3.0.5.

L'ancien pickle a été évalué sous l'interpréteur d'origine
(`/opt/anaconda3/bin/python3`, scikit-learn 1.6.1, numpy 2.1.3, scipy 1.15.3,
pandas 2.2.3).

## C0 — chaîne de données rejouée

`src.data_prep` rejoué sous le `.venv` sur la copie, à partir de
`data/raw/dataset.csv`. Empreintes comparées à celles inscrites dans
`data/processed/split_metadata.json`.

| Fichier | sha256 attendu | sha256 obtenu | EGAL |
| --- | --- | --- | --- |
| `data/processed/train.csv` | `eeaf757019362752104c97be5b55286a90f2ef48608070c2112803b1e15ba458` | `eeaf757019362752104c97be5b55286a90f2ef48608070c2112803b1e15ba458` | True |
| `data/processed/test.csv` | `f4d9c0c4beae54bacaa8140b63190287eace117914ca1c6ac2820a62ff0d43c7` | `f4d9c0c4beae54bacaa8140b63190287eace117914ca1c6ac2820a62ff0d43c7` | True |
| `data/processed/test_sample_2000.csv` | `8cfc0205ec70549e64e455bb3cd14e1ba8eb11ee6767703b07182a49b2856691` | `8cfc0205ec70549e64e455bb3cd14e1ba8eb11ee6767703b07182a49b2856691` | True |

**C0=True.**

Ressources du rejeu, relevées par `/usr/bin/time -l` :

| Grandeur | Valeur |
| --- | --- |
| durée réelle | 15.22 s |
| maximum resident set size | 1434714112 octets |

## Réentraînement sous le `.venv`

`tools/export_modele.py --modele LinearSVC`, exécuté depuis la copie.

| Grandeur | Valeur |
| --- | --- |
| durée réelle | 81.23 s |
| maximum resident set size | 2397683712 octets |
| taille du pickle | 124.0 Mo (123970605 octets) |
| taille du vocabulaire | 1 235 192 |

Empreintes des deux exports :

| Export | Date | scikit-learn | empreinte de contenu | sha256 du pickle |
| --- | --- | --- | --- | --- |
| ancien | 2026-08-19 | 1.6.1 | `73a3831a69dc34e14d98d1e62e42a264a52682c28ffb00d58832afe23bf3fc74` | `75d19298f0de445ee045f37134b435d7ec6f596a41457a1d3333578e01ea3689` |
| nouveau | 2026-09-17 | 1.9.0 | `73a3831a69dc34e14d98d1e62e42a264a52682c28ffb00d58832afe23bf3fc74` | `52504691ce1e914e2b939f81ceb24ffbee22609f47f060d646130cb4660bfb9e` |

**EMPREINTE_EGALE=True.** Les `sha256_pickle` diffèrent, ce que
`tools/export_modele.py` documente comme attendu : seule l'empreinte de
contenu porte le déterminisme. Cette ligne est rapportée hors critères,
conformément au protocole.

## C1 — comparaison des prédictions

Seuils codés en dur dans le script de comparaison : accord >= 0.999,
|delta F1| <= 0.002, tolérance de reproduction 1e-09.

| Jeu | n | divergences | taux d'accord | IDENTIQUE |
| --- | --- | --- | --- | --- |
| `test.csv` | 70863 | 0 | 1.000000 | True |
| `test_sample_2000.csv` | 2000 | 0 | 1.000000 | True |

| Jeu | pondération | F1-macro ancien | F1-macro nouveau | delta |
| --- | --- | --- | --- | --- |
| `test.csv` | non pondéré | 0.8192847674418321 | 0.8192847674418321 | 0.000000 |
| `test_sample_2000.csv` | Horvitz-Thompson | 0.8187435068316256 | 0.8187435068316256 | 0.000000 |

Contrôle de reproduction des valeurs publiées dans
`reports/etape3_evaluation.json` :

| Contrôle | Référence | Écart | Résultat |
| --- | --- | --- | --- |
| F1 ancien sur `test.csv` | 0.8192847674418321 | 0.000000000000e+00 | REPRODUIT_TEST=True |
| F1 ancien pondéré sur les 2 000 | 0.8187435068 | 3.162559103487e-11 | REPRODUIT_2000=True |

La référence pondérée est stockée avec dix décimales dans le rapport de
l'étape 3, ce qui explique l'écart résiduel de 3,16e-11.

**C1=True.** **C2=True** (satisfait également, non décisionnel puisque C1
est vérifié).

## Latence unitaire du nouveau modèle

Méthode reprise de `tools/etape3_evaluation.py`, fonction
`predit_avec_latence` : préchauffage sur les 50 premiers textes, non mesuré,
puis une réclamation à la fois chronométrée par `src.metrics.Timer`,
agrégation par `src.metrics.latency_stats`. Jeu utilisé :
`test_sample_2000.csv`, comme le bloc `latences_2000` de l'étape 3.

| Grandeur | Valeur (s) |
| --- | --- |
| latence_n | 2000 |
| latence_moyenne_s | 0.00031206506059970704 |
| latence_p50_s | 0.00027281249640509486 |
| latence_p95_s | 0.0005587833526078612 |
| latence_max_s | 0.0020088329911231995 |

Soit p50 = 0,273 ms et p95 = 0,559 ms.

`tools/export_modele.py` ne contient aucune mesure de latence : la méthode
a été reprise de l'outil d'évaluation de l'étape 3.

## Tests de non-régression

Exécutés depuis la copie sous le `.venv`, après réentraînement.

| Fichier | Résultat | Code de sortie |
| --- | --- | --- |
| `tests/test_export_modele.py` | RESULTAT : 28/28 verifications passees | 0 |
| `tests/test_metrics_gel.py` | RESULTAT : 44/44 verifications passees | 0 |
| `tests/test_pipeline.py` | RESULTAT : 50/50 verifications passees | 0 |

## Verdict

Verdict selon le protocole : C1 vérifié. Nouvelle référence : export du
2026-09-17 sous scikit-learn 1.9.0. Métriques publiées inchangées.
Réserve : mesures réalisées sous macOS ; la reproductibilité sous Linux sera
contrôlée par l'intégration continue selon les mêmes critères.

## Contrôle sous Linux (intégration continue)

La réserve ci-dessus est levée. Le workflow `export-modele.yml` a rejoué la
chaîne complète sur `ubuntu-latest` lors de la release `modele-v1.0.0` :
téléchargement du corpus publié, `src.data_prep`, tests, export, puis les
mêmes contrôles C0 et C1/C2/C3. Chiffres repris des assets
`controle_donnees.json` et `controle_modele.json` de cette release.

### Environnement du runner

| Grandeur | macOS (référence) | Linux (intégration continue) |
| --- | --- | --- |
| plateforme | Darwin | Linux |
| python | 3.13.5 | 3.13.15 |
| scikit-learn | 1.9.0 | 1.9.0 |
| numpy | 2.5.2 | 2.5.2 |
| scipy | 1.18.0 | 1.18.0 |
| pandas | 3.0.5 | 3.0.5 |

### C0 — chaîne de données

Les quatre empreintes sont égales à celles de la référence, y compris celle du
corpus brut téléchargé depuis la release `data-v1`.

**C0 = true.**

### C1 — prédictions

| Jeu | n | divergences | taux d'accord | F1 référence | F1 obtenu | delta |
| --- | --- | --- | --- | --- | --- | --- |
| `test` | 70863 | 0 | 1.0 | 0.8192847674418321 | 0.8192847674418321 | 0.0 |
| `sample2000` | 2000 | 0 | 1.0 | 0.8187435068316256 | 0.8187435068316256 | 0.0 |

Alignement par `complaint_id`, `ORDRE_ALIGNE` vrai sur les deux jeux.

**VERDICT = C1.**

### Empreintes du modèle Linux

| Grandeur | Valeur |
| --- | --- |
| sha256 du pickle Linux | `b63addf5a508f1a78ef8805b9a2468e017f3191c1a62c543b3fd16effe81336c` |
| sha256 du pickle de référence | `52504691ce1e914e2b939f81ceb24ffbee22609f47f060d646130cb4660bfb9e` |
| empreinte de contenu Linux | `0ea0da08fa39aaec17ec08fc7a51c0d6270cb3f2b5970ba12913eb6b7e4c6aa3` |
| empreinte de contenu de référence | `73a3831a69dc34e14d98d1e62e42a264a52682c28ffb00d58832afe23bf3fc74` |
| `EMPREINTE_EGALE` | false |

L'empreinte de contenu diffère entre macOS et Linux alors que les prédictions
sont identiques : elle n'est pas portable d'une plateforme à l'autre et reste
non décisionnelle, comme le prévoyait le protocole. Seule la comparaison des
prédictions sert de critère.
