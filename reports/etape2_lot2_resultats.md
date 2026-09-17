# Étape 2, extension — lot 2 : résultats bruts

Mesure du 2026-09-10 sur les **1 835 lignes** de `data/processed/train_selection_1835.csv`, évaluées par les quatre variantes.

Aucune interprétation, aucun choix de variante. Les chiffres, et le gate tel qu'il a été pré-enregistré.

## 1. Pré-enregistrement rappelé

Écrit dans [`reports/protocole_lot2.md`](protocole_lot2.md) **avant** toute campagne.

| hypothèse | référence | variante |
|---|---|---|
| **H1** | `v1_zeroshot` | `v7_fewshot_court` |
| **H2** | `v1_zeroshot` | `v8_fewshot_filtre` |
| **H3** | `v7_fewshot_court` | `v8_fewshot_filtre` |

Gate : McNemar exact bilatéral, correction de Holm-Bonferroni, FWER 5 %, famille de **m = 3**. Seuils par rang : 0,0166667 / 0,0250000 / 0,0500000. Procédure arrêtée au premier échec.

Critère de rejet en faveur de la variante : `p_brut <= seuil_Holm` **et** `c > b`, avec `b` = référence juste / variante fausse (lignes **cassées**) et `c` = référence fausse / variante juste (lignes **corrigées**), convention de `src/llm_eval.py:1010-1011`.

`v6_fewshot` est mesurée pour référence descriptive et **n'appartient pas à la famille** : l'y ajouter aurait durci les seuils des trois hypothèses testées.

## 2. Les quatre variantes

| | `v1_zeroshot` | `v6_fewshot` | `v7_fewshot_court` | `v8_fewshot_filtre` |
|---|---|---|---|---|
| préfixe (mots) | 180 | 2 519 | 789 | 670 |
| lignes évaluées | 1 835 | 1 835 | 1 835 | 1 835 |
| **F1-macro** | **0.7729** | **0.7463** | **0.7522** | **0.7567** |
| IC 95 % (bootstrap) | [0.7425 ; 0.8018] | [0.7166 ; 0.7741] | [0.7211 ; 0.7807] | [0.7236 ; 0.7861] |
| exactitude | 0.8223 | 0.8093 | 0.8065 | 0.8174 |
| F1 pondéré | 0.8252 | 0.8112 | 0.8061 | 0.8168 |
| parse_error (n) | 12 | 0 | 2 | 2 |
| parse_error (taux) | 0.65% | 0.00% | 0.11% | 0.11% |
| `label_inconnu` | 0 | 0 | 0 | 0 |
| `json_invalide` | 12 | 0 | 2 | 2 |
| `vide` | 0 | 0 | 0 | 0 |
| erreurs HTTP | 0 | 0 | 0 | 0 |
| tokens_in moyens | 499.5 | 3 508.5 | 1 395.5 | 1 242.5 |
| tokens_caches moyens | 195.8 | 2 846.9 | 948.2 | 791.5 |
| coût total | 0.0991 $ | 0.2707 $ | 0.1593 $ | 0.1560 $ |
| latence p50 (s) | 0.368 | 0.395 | 0.409 | 0.406 |
| latence p95 (s) | 0.600 | 0.655 | 0.739 | 0.656 |

IC 95 % : bootstrap par percentiles, **10 000 tirages**, graine **42**, rééchantillonnage stratifié sur la classe vraie (`src/metrics.py:299`).

## 3. F1 par classe

Descriptif. Le jeu est alloué **proportionnellement au train et sans plancher** : la classe la plus rare n'y compte que 29 lignes, et son F1 n'est pas exploitable.

| classe | n | `v1_zeroshot` | `v6_fewshot` | `v7_fewshot_court` | `v8_fewshot_filtre` |
|---|---:|---|---|---|---|
| Credit reporting | 558 | 0.8302 | 0.8190 | 0.8164 | 0.8333 |
| Debt collection | 435 | 0.8062 | 0.8078 | 0.7481 | 0.7721 |
| Mortgage | 274 | 0.9319 | 0.9174 | 0.9293 | 0.9321 |
| Credit card or prepaid card | 214 | 0.7942 | 0.7586 | 0.8019 | 0.8106 |
| Bank account or service | 144 | 0.8076 | 0.7788 | 0.8113 | 0.7915 |
| Student loan | 113 | 0.8448 | 0.8475 | 0.8498 | 0.8472 |
| Money transfer or virtual currency | 36 | 0.7333 | 0.7458 | 0.7119 | 0.7143 |
| Payday, title or personal loan | 32 | 0.6102 | 0.4912 | 0.5517 | 0.5306 |
| Vehicle loan or lease | 29 | 0.5974 | 0.5510 | 0.5495 | 0.5783 |

## 4. Les trois comparaisons appariées

| | H1 — `v7` vs `v1` | H2 — `v8` vs `v1` | H3 — `v8` vs `v7` |
|---|---|---|---|
| accord — justes des deux | 1 431 | 1 454 | 1 455 |
| accord — fautes des deux | 277 | 280 | 310 |
| **b** (variante casse) | **78** | **55** | **25** |
| **c** (variante corrige) | **49** | **46** | **45** |
| paires discordantes | 127 | 101 | 70 |
| bilan net (c − b) | -29 | -9 | +20 |
| p brut (exact bilatéral) | 0.01266 | 0.42616 | 0.02246 |
| rang Holm | 1 | 3 | 2 |
| seuil du rang | 0.0166667 | 0.0500000 | 0.0250000 |
| p ≤ seuil ? | oui | non | oui |
| c > b ? | non | non | oui |
| **décision** | **rejetee EN DEFAVEUR de la variante** | **non rejetee (arret de la procedure)** | **rejetee EN FAVEUR de la variante** |

Hypothèses rejetées par la procédure de Holm : **H1, H3**.

## 5. Provenance des chiffres

| élément | source |
|---|---|
| journaux d'appels | `data/llm/lot2_v{1,6,7,8}.jsonl` |
| jeu évalué | `data/processed/train_selection_1835.csv` |
| F1-macro, exactitude, F1 par classe | `src/metrics.py:245-260` |
| IC bootstrap | `src/metrics.py:299` |
| b, c | `src/llm_eval.py:1010-1011` |
| p exact bilatéral | `src/llm_eval.py:1018-1019` |
| seuils de Holm | `src/llm_eval.py:1215` |
| vérification « mêmes lignes » | `src/llm_eval.py:476` |
| conformité de format | `src/llm_eval.py:217` |
| ce script | `tools/lot2_analyse.py` |
