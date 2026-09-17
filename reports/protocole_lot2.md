# Protocole de mesure — étape 2, extension, lot 2

**Écrit le 2026-09-10, AVANT tout appel API du lot 2.** Aucune métrique des
campagnes `lot2_*` n'a été observée à la date de rédaction : les quatre JSONL
`data/llm/lot2_v{1,6,7,8}.jsonl` n'existaient pas.

Ce document existe pour une seule raison : rendre le critère de décision
**non révisable après la mesure**. Un gate choisi en regardant les chiffres
n'est plus un gate.

---

## 1. Objet

Mesurer quatre variantes de prompt sur les **mêmes 1 835 lignes** du jeu de
sélection `data/processed/train_selection_1835.csv`, et décider si `v7` ou `v8`
se distingue de `v1`, ou l'une de l'autre.

| style | nature | préfixe |
|---|---|---|
| `v1_zeroshot` | référence, zéro-shot | 180 mots, sha256 `16007502179e43f4` |
| `v6_fewshot` | few-shot 1ʳᵉ génération, sans plafond de longueur | 2 519 mots, sha256 `116478e1351d4163` |
| `v7_fewshot_court` | few-shot, plafond 120 mots, tirage aléatoire | 789 mots, sha256 `ff97381ad271806d` |
| `v8_fewshot_filtre` | few-shot, plafond 120 mots + filtre zéro-shot 3 passes | 670 mots, sha256 `3861c2829202b2bb` |

`v6` est mesurée pour référence descriptive. **Elle n'entre pas dans la famille
de tests** : elle a déjà été mesurée en phase 3 sur 200 lignes, et l'ajouter
comme quatrième comparaison durcirait les seuils de Holm des trois hypothèses
qui portent la question.

---

## 2. Famille de trois comparaisons appariées — PRÉ-ENREGISTRÉE

| hypothèse | comparaison | référence | variante |
|---|---|---|---|
| **H1** | `v7` vs `v1` | `v1_zeroshot` | `v7_fewshot_court` |
| **H2** | `v8` vs `v1` | `v1_zeroshot` | `v8_fewshot_filtre` |
| **H3** | `v8` vs `v7` | `v7_fewshot_court` | `v8_fewshot_filtre` |

La famille compte **m = 3** hypothèses. Elle est close : aucune comparaison ne
sera ajoutée après la mesure, et aucune ne sera retirée si son résultat déplaît.

---

## 3. Gate

**Test** : McNemar exact bilatéral (binomial), tel qu'implémenté par
`src/llm_eval.py:970` — version exacte et non l'approximation du χ², le nombre
de paires discordantes attendu étant faible.

Sur chaque paire, avec `A` = référence et `B` = variante :

- `b` = A juste / B faux — lignes **cassées** par la variante ;
- `c` = A faux / B juste — lignes **corrigées** par la variante.

Convention identique à `src/llm_eval.py:1010-1011` et à celle des lots 0 et 1.

**Correction de multiplicité** : Holm-Bonferroni, FWER contrôlé à 5 %,
`src/llm_eval.py:1174`. Les trois p bruts sont triés par ordre croissant et
comparés aux seuils :

| rang | seuil |
|---|---|
| 1 | 0,05 / 3 = 0,0166667 |
| 2 | 0,05 / 2 = 0,0250000 |
| 3 | 0,05 / 1 = 0,0500000 |

La procédure **s'arrête au premier échec** : toutes les hypothèses de rang
supérieur sont non rejetées, quel que soit leur p brut.

**Critère de rejet, en faveur de la variante :**

```
p_brut <= seuil_Holm_du_rang   ET   c > b
```

### Note sur le sens de `b` et `c` — corrigée avant toute mesure

La consigne du lot 2 énonçait le critère comme `b > c`. Sous la convention
`b` / `c` **confirmée au lot 1** et implémentée en
`src/llm_eval.py:1010-1011` — `b` = référence juste / variante fausse — c'est
`c > b` qui signifie « la variante fait mieux que la référence » :

- `b` = lignes que la variante **casse** ;
- `c` = lignes que la variante **corrige**.

Les deux formulations sont contradictoires ; garder `b > c` aurait fait
rejeter en faveur d'une variante qui dégrade. La convention confirmée est
conservée et le critère est écrit `c > b`, ce qui préserve l'intention
(« rejet en faveur de la variante »).

Repère : au lot 0, `v6` affichait `b = 10, c = 9` contre `v1`, soit un bilan
net de **−1** pour `v6` — dix lignes cassées pour neuf corrigées.

Un p sous le seuil avec `b > c` est un rejet **en défaveur** de la variante :
il est rapporté comme tel et ne vaut pas retenue.

---

## 4. Ce qu'un non-rejet signifie, et ce qu'il ne signifie pas

Un non-rejet **borne** l'effet, il ne l'annule pas. Le jeu de 1 835 lignes est
dimensionné pour 80 % de chances de détecter un gain d'exactitude de 0,02 sous
ce gate exact, avec un taux de discordance supposé de 6,75 %. Un gain réel plus
petit a une chance proportionnellement plus faible d'être détecté, et un non-rejet
ne permet pas de conclure à l'équivalence.

---

## 5. Conditions de mesure, identiques pour les quatre campagnes

| paramètre | valeur |
|---|---|
| jeu | `data/processed/train_selection_1835.csv`, 1 835 lignes |
| modèle | `mistral-small-latest` (alias vérifié → `mistral-small-2603`) |
| température | 0,0 |
| parsing | STRICT (`llm_prompts.parse_reponse`), aucun rattrapage |
| cadence | 0,33 req/s |
| réessais | 5 avec backoff exponentiel sur 408/409/429/5xx |
| budget par campagne | 1,00 $ |
| sortie | `data/llm/lot2_v{1,6,7,8}.jsonl`, un fichier par campagne |

Ordre d'exécution : `v1`, `v6`, `v7`, `v8`.

**Point d'arrêt après chaque campagne.** Trois conditions, toutes exigées avant
de lancer la suivante :

1. 1 835 lignes journalisées ;
2. 0 erreur HTTP non rejouée ;
3. taux de `parse_error` < 5 %.

Si l'une échoue, la séquence s'arrête et rien n'est enchaîné.

**Avant le McNemar**, vérification que les quatre campagnes portent sur le même
jeu de 1 835 `complaint_id`. Un test apparié sur des lignes différentes n'a pas
de sens ; en cas de divergence, la mesure s'arrête sans produire de comparaison.

---

## 6. Métriques descriptives, hors gate

Rapportées pour les quatre variantes, sans valeur décisionnelle : F1-macro et
son IC 95 % bootstrap (10 000 tirages, graine 42), exactitude, F1 par classe,
taux de conformité de format (`parse_error`, `label_inconnu`), `tokens_in` et
`tokens_caches` moyens, coût total, latences p50 et p95.

**Le F1-macro n'est pas le gate.** La phase 3 a montré pourquoi : `v5` y
affichait le meilleur F1-macro du tableau tout en n'ayant corrigé que 6 lignes
pour 4 cassées. Le test apparié, qui ne regarde que les lignes où les deux
variantes diffèrent, est le seul critère de décision de ce protocole.

Le jeu étant alloué **proportionnellement au train et sans plancher**, la classe
la plus rare n'y compte que 29 lignes : le **F1 par classe des classes rares n'y
est pas exploitable**, et il est rapporté à titre descriptif uniquement.
