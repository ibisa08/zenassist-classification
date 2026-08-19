# Résultats — étape 3, phase A (choix du vectoriseur)

Règles appliquées : [protocole_selection_phaseA.md](protocole_selection_phaseA.md).
Ce fichier ne contient que des agrégats, aucun texte de réclamation.

> **`[CV train, NON comparable au LLM]`** — F1-macro de validation croisée à
> 5 plis sur le train (283 449 lignes), **non pondérés**. Données différentes,
> pondération différente, rôle différent du 0,7885 de l'étape 2. Aucun chiffre
> de ce fichier ne se compare à un chiffre de l'étape 2.

Classifieur tenu fixe : `LinearSVC(class_weight='balanced')`.
Plis : `StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)`.

## 1. Grille et écarts appariés contre la référence

| configuration | F1 CV | vocab | écart apparié | SD app. | `t` | `t_corrigé` | verdict §3 |
|---|---|---|---|---|---|---|---|
| `ngram=(1,3)` | **0,81816** | 1 760 221 | +0,00586 | 0,00164 | +8,01 | +5,34 | détecté et pertinent |
| `min_df=2` | 0,81448 | 1 235 192 | +0,00218 | 0,00056 | +8,68 | +5,79 | détecté, sous le seuil |
| référence (1,2) `min_df=5` | 0,81230 | 550 116 | — | — | — | — | — |
| `max_features=200000` | 0,80797 | 200 000 | −0,00433 | 0,00051 | −19,14 | −12,76 | détecté, sous le seuil |
| `min_df=20` | 0,80534 | 194 220 | −0,00696 | 0,00060 | −25,97 | −17,31 | détecté et pertinent |
| `sublinear_tf=False` | 0,80503 | 550 116 | −0,00727 | 0,00128 | −12,74 | −8,49 | détecté et pertinent |
| **témoin** `ngram=(1,1)` | 0,77858 | 27 480 | −0,03372 | 0,00390 | −19,32 | −12,88 | détecté et pertinent |

**Deux comparateurs distincts, à ne pas confondre.** La colonne verdict compare
**à la référence** (§3) ; l'ensemble candidat se définit **au meilleur** (§4.2).
`min_df=2` est « sous le seuil » face à la référence et pourtant candidat : ce
n'est pas une contradiction, ce sont deux questions différentes.

**Le témoin positif joue son rôle** : −0,034, soit sept fois le seuil de
pertinence. Le dispositif détecte ce qu'il doit détecter.

**Aucun verdict ne bascule sous la correction de variance.** Le `|t_corrigé|` le
plus faible vaut 5,34, près du double du seuil de 2,78. La non-indépendance des
plis, réelle, ne change ici aucune lecture.

## 2. Ensemble candidat

`F* = 0,81816`. Sont candidates les configurations à `F* − F ≤ 0,005` :

| configuration | écart au meilleur | statut |
|---|---|---|
| `ngram=(1,3)` | 0 | candidate |
| `min_df=2` | 0,00368 | candidate |
| référence (1,2) `min_df=5` | **0,00586** | **hors, de 0,00086** |
| `max_features=200000` | 0,01019 | hors |
| `min_df=20` | 0,01282 | hors |
| `sublinear_tf=False` | 0,01313 | hors |
| `ngram=(1,1)` | 0,03958 | hors |

**Réserve à reporter dans la synthèse.** La référence sort de l'ensemble
candidat pour 0,00086 — moins d'un millième de F1. La coupure est franche et a
été appliquée telle qu'écrite, sans ajustement après coup, mais une décision qui
tient à cette marge ne doit pas être présentée comme robuste.

## 3. Latence de prédiction unitaire

Pipeline complet `transform` + `predict`, **une réclamation à la fois**,
500 mesures individuelles via `metrics.Timer`, après 50 prédictions de
préchauffage non mesurées. Modèles entraînés sur le train complet.

| configuration | statut | vocab | p50 | p95 | temps de `fit` |
|---|---|---|---|---|---|
| `ngram=(1,3)` | candidate | 1 760 221 | 0,32 ms | 0,83 ms | 122 s |
| **`min_df=2`** | **candidate** | 1 235 192 | **0,23 ms** | **0,47 ms** | 64 s |
| référence (1,2) `min_df=5` | hors ensemble | 550 116 | 0,23 ms | 0,45 ms | 62 s |

Le p95 porte la décision, conformément à la doctrine de `metrics.py`. Le p50 va
dans le même sens.

**Observation à conserver — la latence ne dépend presque pas de la taille du
vocabulaire.** La référence et `min_df=2` ont des latences indiscernables
(0,45 contre 0,47 ms au p95) alors que leurs vocabulaires vont du simple au
double et quart. Le coût d'une prédiction unitaire est dominé par l'**analyseur**
— l'extraction des n-grammes du texte — et non par la taille du dictionnaire,
qui n'ajoute qu'une recherche par table de hachage. C'est pourquoi seul
`ngram_range` déplace la latence : passer aux trigrammes la multiplie par 1,8 au
p95.

**Critique de la règle d'arrêt, à conserver.** L'ordre lexicographique de §4
place la latence avant le vocabulaire, en supposant implicitement que la latence
est le critère le plus discriminant après le F1. La mesure montre l'inverse :
entre deux configurations de même `ngram_range`, la latence est **quasi inerte**
et c'est le critère de vocabulaire (§4.4), placé en second, qui départage
réellement. La règle n'a donc mordu ici que parce que l'ensemble candidat
contenait deux `ngram_range` différents — un fait de la grille, pas une
propriété de la règle. Sur un ensemble candidat de même `ngram_range`, la règle
telle qu'écrite aurait glissé jusqu'à §4.4 sans que §4.3 tranche quoi que ce
soit. L'ordre reste défendable — la latence est bien le coût de service, et le
vocabulaire un coût de mémoire — mais il est présenté ici avec ce que la mesure
lui retire : son mordant supposé.

Le temps de `fit` n'entre pas dans la règle d'arrêt : c'est un coût
d'entraînement ponctuel, pas un coût de service. Il est rapporté pour mémoire.

## 4. Configuration retenue

```python
TfidfVectorizer(lowercase=True, sublinear_tf=True, ngram_range=(1, 2), min_df=2)
```

Chemin de décision, par les étapes de §4 du protocole :

1. `F* = 0,81816` (`ngram=(1,3)`).
2. Ensemble candidat : `ngram=(1,3)` et `min_df=2`.
3. **Latence d'abord** : 0,47 ms contre 0,83 ms au p95, soit 77 % d'écart
   relatif — bien au-delà des 10 % qui vaudraient égalité. `min_df=2` l'emporte
   ici, et la règle s'arrête.
4. Le critère de vocabulaire n'est pas atteint. Il aurait désigné la même
   configuration (1,24 M contre 1,76 M).

Le protocole écarte donc la meilleure moyenne CV, comme il le prévoit
explicitement : 0,0037 de F1 en validation croisée sur le train ne justifie pas
une inférence 1,8 fois plus lente et un vocabulaire de 42 % plus gros.

## 5. Ordre de grandeur, à confirmer en phase 2

À titre indicatif seulement, la latence du LLM mesurée à l'étape 2 était de
470 ms au p50 et 740 ms au p95 par appel. Le rapport est de l'ordre de **2 000**.
Deux réserves : la latence du LLM inclut l'aller-retour réseau, et les deux
mesures ne portent pas sur les mêmes textes. Le chiffre comparable sera produit
en phase 2, sur les mêmes 2 000 lignes.

## 6. Décisions antérieures reconduites, non rejouées ici

- **`max_df` : aucun filtre.** `max_df=0.9` n'exclut qu'un seul token (`to`) sur
  550 116, pour un F1 identique à quatre décimales. `xxxx` a un `df` de 84,32 %
  et n'est atteint par aucun `max_df` défendable.
- **`xxxx` conservé.** Le retrait modifie le F1 de moins de 0,0011 en valeur
  absolue (écart minimal détectable de cette configuration), et l'estimation
  ponctuelle est −0,00059 — donc légèrement en défaveur du retrait. Réserve de
  transférabilité maintenue : le modèle apprend en partie la chaîne
  d'anonymisation du CFPB, pas seulement le domaine.
- **`stop_words` : aucun.** `stop_words='english'` **augmente** le vocabulaire de
  30 863 entrées par pontage de bigrammes (342 374 supprimés, 373 547 apparus),
  effet vérifié et non un artefact.
