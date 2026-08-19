# Résultats — étape 3, phase B (choix du classifieur)

Règles appliquées : [protocole_selection_phaseA.md](protocole_selection_phaseA.md),
§7 (stabilité du vectoriseur), §8 (sélection), §9 (sensibilité à `alpha`).
Agrégats uniquement, aucun texte de réclamation.

> **`[CV train, NON comparable au LLM]`** — F1-macro de validation croisée à
> 5 plis sur le train, **non pondérés**. Aucun chiffre de ce fichier ne se
> compare à un chiffre de l'étape 2.

Vectorisation **ajustée par pli** sur la partie entraînement seulement, puis
réutilisée par les trois classifieurs du même pli : aucune fuite, cinq
vectorisations au lieu de quinze.

## 1. Les trois classifieurs

| classifieur | F1 CV | écart apparié au meilleur | SD app. | `t` | `t_corrigé` | verdict §3 |
|---|---|---|---|---|---|---|
| **`LinearSVC`** | **0,81448** | — | — | — | — | retenu |
| `LogisticRegression` | 0,80275 | −0,01174 | 0,00146 | −18,02 | −12,01 | détecté et pertinent |
| `MultinomialNB` | 0,71800 | −0,09649 | 0,00309 | −69,72 | −46,48 | détecté et pertinent |

**Ensemble candidat : `LinearSVC` seul.** `LogisticRegression` est à 0,01174 du
meilleur, plus du double du seuil de pertinence. La règle §4 s'arrête à
l'étape 2 et la latence n'a pas à départager. Contrairement à la phase A, la
coupure n'est ici pas serrée.

`LogisticRegression` et `MultinomialNB` restent entraînés et exportés : le §8 le
prévoit, la règle §4 désigne le modèle de la comparaison principale, pas la
liste des modèles produits.

### Contrôle de cohérence entre phases

La moyenne de `LinearSVC` en phase B est **identique à 0,0 près, pli par pli**,
à celle de `min_df=2` en phase A — alors que les deux passent par des chemins de
code indépendants :

| | phase A | phase B |
|---|---|---|
| découpage | `Pipeline` + `cross_val_score` | boucle explicite sur les plis |
| vectorisation | interne au `Pipeline` | ajustée à la main sur la partie entraînement |
| métrique | scorer `'f1_macro'` de scikit-learn | `src.metrics.f1_macro`, `labels=CLASS_ORDER` |
| scores par pli | 0,818137 / 0,819569 / 0,813066 / 0,810891 / 0,810754 | identiques |

Les deux coïncident parce que **les 9 classes sont présentes dans chaque pli de
validation** — enregistré par le script, pas supposé. Le contrôle vérifie du même
coup qu'aucune fuite ne s'est glissée dans le découpage réécrit.

## 2. Sonde §7 — l'hypothèse de stabilité ne tient pas

| point | F1 CV | vocab | écart au retenu | SD app. | `t` | `t_corrigé` |
|---|---|---|---|---|---|---|
| retenu (1,2) `min_df=2` | 0,71800 | 1 235 192 | — | — | — | — |
| **(1,2) `min_df=20`** | **0,72836** | 194 220 | **+0,01036** | 0,00278 | +8,34 | +5,56 |
| (1,1) `min_df=5` | 0,72001 | 27 480 | +0,00201 | 0,00090 | +5,02 | +3,35 |
| (1,2) `min_df=2`, `sublinear_tf=False` | 0,71988 | 1 235 192 | +0,00189 | 0,00044 | +9,52 | +6,34 |

**CAS 2 de la règle §7.** L'écart de +0,01036 dépasse le seuil de 0,005 :
**l'optimum du vectoriseur n'est pas stable d'une famille de classifieur à
l'autre.** C'est un résultat, pas un incident, et la règle qui s'applique avait
été écrite avant la mesure.

La **prédiction directionnelle du §7 est vérifiée** : le déplacement est bien
pour NB, et vers un vocabulaire plus petit. Le détail corrige toutefois une
lecture trop simple. Ce n'est pas « moins de variables » qui aide :
`(1,1) min_df=5` descend à 27 480 variables et ne gagne que 0,002, alors que
`min_df=20` en garde 194 220 et gagne cinq fois plus. **Ce qui pénalise NB, ce
sont les bigrammes rares, pas les bigrammes.** Le mécanisme de redondance
prédisait la direction ; il ne prédisait pas que le remède serait d'élaguer
plutôt que de supprimer.

Conformément à la règle : les deux chiffres de NB sont publiés, son vectoriseur
propre ne sert qu'au rôle d'ancrage, et **le vectoriseur de la comparaison
principale reste celui de la phase A**.

## 3. Sensibilité à `alpha` — caractérisation, `alpha=1` conservé

| `alpha` | F1 CV | écart à `alpha=1` | SD app. | `t` | `t_corrigé` |
|---|---|---|---|---|---|
| 0,001 | 0,72709 | +0,00909 | 0,00260 | +7,81 | +5,21 |
| 0,01 | 0,75526 | +0,03726 | 0,00174 | +48,00 | +32,00 |
| **0,1** | **0,75964** | **+0,04164** | 0,00156 | +59,70 | +39,80 |
| **1** *(retenu)* | 0,71800 | — | — | — | — |
| 10 | 0,68537 | −0,03263 | 0,00105 | −69,25 | −46,17 |

**Δ = +0,04164**, contre un seuil de contamination fixé d'avance à 0,022. Le
verdict ne tient à aucune marge et ne repose sur aucune statistique de test :
c'est l'amplitude seule qui tranche, ce qui le met hors d'atteinte de
l'anti-conservatisme signalé au §3.

**La réponse est non monotone**, avec un optimum intérieur vers 0,1 — `alpha`
descendu à 0,001 redescend à 0,727. Cohérent avec le mécanisme : trop peu de
lissage et `log(alpha)` s'effondre sur chaque variable jamais vue par une classe,
or avec 1,24 M de variables presque toutes les classes en ont un nombre
considérable, si bien que le score se réduit à un décompte de variables
inconnues. Trop de lissage et il noie la vraisemblance. **`alpha=1` est du
mauvais côté du pic.**

**Borne sur la portée.** Même à `alpha=0,1`, NB plafonne à 0,7596 contre 0,81448
pour `LinearSVC`. La contingence ne change **pas** quel modèle est retenu ; elle
change ce que le chiffre de NB signifie comme ancrage de famille. Conséquence
enregistrée dans l'erratum de `protocole_analyse_croisee.md` : **ancrage
intra-famille ML déclaré contaminé**, critère `R` lu avec le seul ancrage
LLM/LLM.

`alpha=1` est conservé, conformément à la condition ferme du §9.

## 4. Latence de prédiction unitaire

Mêmes conditions qu'en phase A : pipeline complet `transform` + `predict`, une
réclamation à la fois, 500 mesures via `metrics.Timer`, après 50 prédictions de
préchauffage. Modèles ajustés sur le train complet.

| modèle | vocab | p50 | p95 | `fit` |
|---|---|---|---|---|
| `LinearSVC` | 1 235 192 | 0,21 ms | 0,52 ms | 72 s |
| `LogisticRegression` | 1 235 192 | 0,22 ms | 0,53 ms | 167 s |
| `MultinomialNB` (vect. retenu) | 1 235 192 | **9,93 ms** | 11,48 ms | 32 s |
| `MultinomialNB` (vect. propre) | 194 220 | 1,56 ms | 1,98 ms | 30 s |

### `MultinomialNB` est 20 fois plus lent — et ce n'est pas l'algorithme

Le rapport entre les deux vectoriseurs de NB (9,93 / 1,56 = **6,37**) reproduit
exactement le rapport des vocabulaires (1 235 192 / 194 220 = **6,36**). Le coût
est donc **linéaire en la taille du vocabulaire et indépendant de la longueur du
document** — ce qui n'a aucun sens pour une somme sur les termes présents.

Mesure du décomposé, sur un corpus réduit (533 715 variables) :

| | |
|---|---|
| `transform` seul | 0,16 ms |
| `predict` seul, `LinearSVC` | **0,03 ms** |
| `predict` seul, `MultinomialNB` | **3,72 ms** |
| produit creux `x @ coef_.T` (`LinearSVC`) | 0,00 ms |
| produit creux `x @ feature_log_prob_.T` (NB) | **3,65 ms** |

Le produit matriciel **est** tout le coût. Sa cause est une **disposition
mémoire**, pas une différence de calcul :

| tableau | forme | C-contigu | `.T` C-contigu |
|---|---|---|---|
| `LinearSVC.coef_` | (9, V) | **non** | **oui** |
| `MultinomialNB.feature_log_prob_` | (9, V) | **oui** | **non** |

Le produit creux×dense exige l'opérande dense en disposition C. `coef_` sort de
liblinear en ordre Fortran, donc sa transposée est déjà contiguë et le produit
est gratuit. `feature_log_prob_` est en ordre C, donc sa transposée ne l'est pas
et **scipy recopie les 89 Mo du tableau à chaque prédiction**.

**Ce n'est donc pas une propriété de la famille Naive Bayes.** L'arithmétique de
NB est de taille identique à celle de `LinearSVC`, dont le même produit se mesure
à 0,00 ms. Rapporter « NB est 20 fois plus lent » sans cette cause attribuerait à
l'algorithme ce qui appartient à un détail d'implémentation, levable par un
`np.ascontiguousarray` au chargement.

**Rien n'a été corrigé.** Le chiffre publié est celui de `MultinomialNB` tel que
scikit-learn 1.6.1 le sert par défaut, ce qui est la grandeur pertinente pour un
client qui déploierait ce code. La cause est documentée à côté pour que personne
n'en tire la mauvaise conclusion.

> **Chiffre daté : mesuré le 2026-08-19 avec scikit-learn 1.6.1, scipy 1.15.3,
> numpy 2.1.3, Python 3.13.5.** Il ne décrit pas l'algorithme, il décrit une
> implémentation à une version donnée. Une version future qui rendrait
> `feature_log_prob_.T` contiguë ferait tomber ces 9,93 ms à l'ordre de grandeur
> de `LinearSVC` **sans que rien ne change dans le modèle**. À revérifier à
> chaque montée de version, au même titre que les tarifs de l'étape 2, et à ne
> jamais citer sans sa version.

### Ordre de grandeur contre le LLM

0,21 ms au p50 pour `LinearSVC` contre 470 ms mesurés à l'étape 2, soit un
rapport de l'ordre de **2 200**. Deux réserves inchangées : la latence du LLM
inclut l'aller-retour réseau, et les deux mesures ne portent pas sur les mêmes
textes. Le chiffre comparable sera produit en phase 2, sur les mêmes 2 000
lignes.

## 5. Modèle retenu pour la comparaison principale

```python
Pipeline([
    ("tfidf", TfidfVectorizer(lowercase=True, sublinear_tf=True,
                              ngram_range=(1, 2), min_df=2)),
    ("clf",   LinearSVC(class_weight="balanced", random_state=RANDOM_SEED)),
])
```

Export, empreintes et métadonnées : `tools/export_modele.py`, dont le
déterminisme est vérifié par `tests/test_export_modele.py`. Journal du
vocabulaire : `reports/vocabulaire_top_tokens.md`. Scores lisibles par machine :
`reports/etape3_scores.json`.
