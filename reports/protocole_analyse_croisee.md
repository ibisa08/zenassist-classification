# Protocole de l'analyse croisée LLM / ML

**Écrit le 2026-08-19, AVANT tout entraînement de modèle ML.** Les erreurs du ML
sont inconnues à cette date. Les définitions, références et critères ci-dessous
sont arrêtés maintenant et ne seront pas révisés après les avoir vues.

## Pourquoi ce document existe

L'analyse croisée est la **seule mesure empirique** du plafond de performance
annoncé au §C.4 du diagnostic. L'argument est le suivant : une réclamation
manquée par deux approches aux mécanismes indépendants relève probablement d'une
ambiguïté d'étiquetage plutôt que d'une faiblesse de modèle.

Cet argument est **invérifiable si le protocole est écrit après coup**. Il existe
au moins quatre définitions défendables de « même erreur », deux façons de
calculer un recouvrement attendu, et aucun seuil naturel. Choisir après avoir vu
les chiffres revient à choisir la conclusion.

---

## 1. Définition de « erreur commune »

**Définition primaire — CO-ÉCHEC.** Un `complaint_id` est une *erreur commune*
si les deux approches le classent mal, **quelle que soit la classe prédite par
chacune**.

C'est la définition qui correspond à la question posée. Ce qui est en jeu est la
difficulté de l'item, pas la convergence des modèles. Deux approches qui échouent
en se trompant différemment échouent quand même — et l'item est sans doute plus
difficile, pas moins, puisqu'il attire des réponses fausses distinctes.

**Définition secondaire — CO-ÉCHEC ORIENTÉ.** Sous-ensemble du précédent où les
deux approches prédisent **la même classe fausse**.

Elle mesure autre chose : une attraction partagée vers une classe précise, ce qui
suggère que le texte ressemble réellement à cette classe. C'est un indice
d'**ambiguïté directionnelle**, plus fort que le simple co-échec.

Les deux sont rapportées. La primaire porte le verdict ; la secondaire le
qualifie. Référence intra-famille déjà mesurée : sur les 298 erreurs communes aux
deux LLM, **83,6 % portent la même classe prédite**.

### Exclusion pré-spécifiée : les PARSE_ERROR

Les items où le LLM a produit `PARSE_ERROR` sont **exclus** de l'analyse croisée,
et leur nombre est rapporté.

Motif : ce sont des échecs de **format**, pas de classification. Le ML ne peut
pas en produire — il rend toujours une classe du référentiel. Les compter
attribuerait à une ambiguïté d'étiquetage ce qui est un défaut de sérialisation,
et gonflerait mécaniquement le co-échec sans rapport avec la difficulté du texte.

Sur la campagne mistral-small, cette exclusion porte sur **19 items** (0,95 %).

---

## 2. Référence de comparaison sous indépendance

Si les deux approches se trompaient indépendamment, le nombre d'erreurs communes
attendu serait :

```
E[|A ∩ B|] = N × p_A × p_B
```

où `p_A` et `p_B` sont les taux d'erreur marginaux **mesurés sur les mêmes
2 000 lignes**.

Deux indicateurs sont rapportés :

| indicateur | formule | ce qu'il dit |
|---|---|---|
| **lift** | `|A ∩ B| / (N × p_A × p_B)` | combien de fois le recouvrement dépasse le hasard |
| Jaccard | `|A ∩ B| / |A ∪ B|` | part de l'union partagée — sensible aux taux marginaux |

Le **lift** porte le verdict : il neutralise les taux marginaux, alors que le
Jaccard monte mécaniquement quand les deux approches se trompent beaucoup.

---

## 3. À quoi comparer le recouvrement LLM / ML

Deux ancrages, l'un bas, l'autre haut.

**Ancrage bas — indépendance : lift = 1,0.** Aucun recouvrement au-delà du
hasard : les erreurs des deux approches n'ont rien de commun.

**Ancrage haut — recouvrement intra-famille, déjà mesuré :**

| | valeur |
|---|---|
| erreurs mistral-small | 385 (19,25 %) |
| erreurs ministral-3b | 485 (24,25 %) |
| attendu sous indépendance | 93,4 |
| **observé** | **298** |
| **lift intra-famille** | **3,19** |
| Jaccard | 0,521 |

Ces deux modèles partagent la famille, le tokenizer, le prompt et le paradigme
d'entraînement. **3,19 est donc une borne haute de « mécanisme partagé »**, et non
une mesure pure de difficulté des données : une part de ce recouvrement vient de
ce que les deux modèles se ressemblent.

**Ancrage haut n° 2 — recouvrement intra-famille ML**, à mesurer.

Symétrique du précédent, et disponible à coût nul puisque plusieurs classifieurs
seront de toute façon entraînés. **Deux modèles de familles différentes** seront
retenus — une régression logistique et un Naive Bayes multinomial, qui ne
partagent ni fonction de perte, ni hypothèses génératives, ni traitement des
corrélations entre variables — et leur recouvrement mutuel sera calculé
exactement comme les autres.

Ce troisième ancrage change ce qu'on peut conclure :

| configuration observée | lecture |
|---|---|
| les deux lifts intra-famille sont proches **et** le lift croisé LLM/ML est nettement en dessous des deux | le recouvrement s'explique surtout par la **similarité d'architecture** ; l'argument du plafond faiblit |
| le lift croisé est **du même ordre** que les deux intra-famille | le recouvrement ne dépend pas de l'architecture ; l'argument du plafond se renforce nettement |
| les deux lifts intra-famille divergent fortement | l'ancrage haut n'est pas une grandeur stable ; le critère `R` devient peu lisible et le rapport doit le dire |

Sans ce troisième point, `R ≥ 0,75` compare le croisé à un seul ancrage dont on
ne sait pas s'il est représentatif. Avec lui, on dispose de deux mesures du
« mécanisme partagé » et l'on peut vérifier qu'elles se ressemblent avant de s'y
appuyer.

Le recouvrement LLM / ML doit se lire **entre ces ancrages**. Un TF-IDF
suivi d'un classifieur linéaire ne partage avec un LLM ni l'architecture, ni la
représentation du texte, ni les données d'entraînement : ce qu'ils manquent
ensemble ne peut guère s'expliquer par un mécanisme commun.

---

## 4. Critères de lecture — dans les deux sens

Soit `R = lift(LLM, ML) / 3,19`, la position du recouvrement croisé par rapport
à la référence intra-famille.

| condition | lecture |
|---|---|
| **R ≥ 0,75** (lift ≥ 2,39) | **Plafond dans les données.** Deux approches sans mécanisme commun se recouvrent presque autant que deux modèles de la même famille. L'explication par l'architecture devient peu plausible. |
| **0,40 ≤ R < 0,75** (1,28 ≤ lift < 2,39) | **Plafond partiel.** Une part des erreurs est structurelle, une part est propre à chaque approche. |
| **R < 0,40** (lift < 1,28) | **Limites propres aux modèles.** Le recouvrement s'écarte peu du hasard : chaque approche échoue sur ses propres cas, et rien n'établit un plafond dans les données. |

### Critère complémentaire — concentration sur les paires ambiguës

Indépendamment du lift, on compare :

- la part des **erreurs communes** tombant sur les paires du §C.4 ;
- à la part des **erreurs propres à chaque approche** tombant sur ces mêmes paires.

| condition | lecture |
|---|---|
| communes **plus** concentrées sur les paires ambiguës que les erreurs propres | appuie l'interprétation par l'ambiguïté d'étiquetage |
| concentration **égale ou moindre** | le co-échec ne s'explique pas par l'ambiguïté annoncée ; chercher ailleurs |

Écart minimal retenu comme significatif : **10 points de pourcentage**.

### Ce qui vaudrait infirmation, explicitement

Le plafond serait **infirmé** si l'une de ces situations se présentait :

1. `R < 0,40` — le recouvrement ne dépasse pas nettement le hasard ;
2. les erreurs communes ne sont **pas** plus concentrées sur les paires
   ambiguës que les erreurs propres ;
3. le ML corrige une **part substantielle** des erreurs du LLM sur les classes
   que le §C.4 désignait comme les plus difficiles — ce qui montrerait que ces
   classes ne sont pas intrinsèquement dures, mais mal traitées par le LLM.

Pour le point 3, la base de calcul est fixée sans ambiguïté :

```
taux de rattrapage = (erreurs du LLM sur les 3 classes, que le ML classe juste)
                     ---------------------------------------------------------
                            (erreurs du LLM sur les 3 classes)
```

Le dénominateur est donc **les erreurs du LLM dont la classe VRAIE appartient aux
trois classes de plus faible F1** — `Vehicle loan or lease` (0,6854),
`Payday, title or personal loan` (0,6966) et `Debt collection` (0,7708) — et
**non** le total des erreurs du LLM.

Motif : la question posée est « ces classes sont-elles intrinsèquement dures, ou
seulement mal traitées par le LLM ? ». Elle porte sur le taux de rattrapage **au
sein** de ces classes. Rapporter au total des erreurs mêlerait la réponse au
poids de ces classes dans l'échantillon, qui est un fait d'échantillonnage sans
rapport avec leur difficulté.

**Seuil : 30 %.** Au-delà, le ML rattrape une part substantielle de ce que le LLM
manque sur les classes réputées les plus difficiles, et le plafond serait une
propriété du LLM plutôt que des données.

---

## 5. Limites du protocole, énoncées d'avance

- **L'ancrage haut n'est pas une référence pure.** 3,19 mélange difficulté des
  données et similarité des deux LLM. Il majore donc ce qu'un « mécanisme
  partagé » explique, et rend le critère `R ≥ 0,75` d'autant plus exigeant.
- **Le co-échec ne prouve pas l'ambiguïté.** Deux approches peuvent échouer
  ensemble sur des textes simplement courts, mal rédigés ou hors sujet — sans que
  l'étiquette soit contestable. Le critère complémentaire de concentration
  atténue ce risque sans l'éliminer.
- **Deux modèles ML de familles différentes sont comparés au LLM**, et leur
  recouvrement mutuel fournit un second ancrage intra-famille. Cela lève la
  limite d'une mesure sur une seule paire, sans l'éliminer : quatre paires
  d'approches restent un petit nombre d'observations, pas une loi.
- **L'échantillon est stratifié à plancher.** Les taux d'erreur marginaux et le
  recouvrement sont calculés **sans pondération**, sur l'échantillon tel qu'il a
  été tiré : il s'agit de dénombrer des cas à lire, pas d'estimer une grandeur de
  population. Les métriques de performance, elles, restent pondérées.

## 6. Complément qualitatif — documenté, non décisionnel

Vingt erreurs communes seront lues à la main et classées selon la grille de la
phase 2 : **(A)** ambiguïté réelle — le texte décrit un préjudice pointant vers
une autre classe, ou relève d'une paire du §C.4 ; **(B)** défaut de modèle.

Ce complément **n'entre dans aucun critère**. Il illustre, il ne tranche pas :
une lecture à la main de 20 cas, faite par la même personne qui connaît
l'hypothèse, n'a pas la valeur d'une mesure.

---

# ERRATUM du 2026-08-19 — le troisième ancrage est indisponible

**Ce qui change dans les critères : rien.** Aucun seuil n'est déplacé, aucune
définition n'est revue. `R ≥ 0,75`, `0,40 ≤ R < 0,75`, `R < 0,40`, l'écart de
10 points sur la concentration et le seuil de 30 % du rattrapage restent
exactement tels qu'écrits avant tout entraînement. Cet erratum enregistre un
**fait mesuré** qui rend l'une des mesures prévues inutilisable, et rien d'autre.

## Le fait

Le §3 prévoyait un second ancrage intra-famille, tiré du recouvrement entre
`LogisticRegression` et `MultinomialNB`, « deux modèles de familles différentes
qui ne partagent ni fonction de perte, ni hypothèses génératives, ni traitement
des corrélations entre variables ».

Deux mesures de la phase B de l'étape 3 retirent à `MultinomialNB` la qualité
qui le rendait utilisable comme second point :

1. **Inadéquation du vectoriseur** (sonde §7 du protocole de sélection, CAS 2).
   L'optimum du vectoriseur n'est pas stable entre familles : NB gagne
   **+0,01036** de F1 en validation croisée avec `min_df=20` plutôt qu'avec le
   vectoriseur retenu.
2. **Contingence au lissage** (§9 du même protocole). Le rééquilibrage de NB
   transite à 85 % par le lissage de Laplace, et non par une repondération des
   classes. Le défaut `alpha=1` lui coûte **0,04164** de F1 par rapport à
   `alpha=0,1`.

La règle de lecture du §9 avait été fixée **avant** cette mesure : un écart
atteignant 0,022 — la demi-largeur de l'IC de la mesure finale — déclare
l'ancrage contaminé. L'écart mesuré vaut 0,0416, soit près du double. Le verdict
ne tient donc à aucune marge, et ne repose sur aucune statistique de test : c'est
l'amplitude seule qui tranche.

## Pourquoi cela disqualifie l'ancrage, et pas seulement le chiffre de NB

L'ancrage intra-famille sert à mesurer ce qu'un **mécanisme partagé** explique du
recouvrement, pour le comparer au croisé LLM/ML. Il suppose que la différence
entre les deux modèles ML soit une différence **d'architecture**. Ici, les
erreurs de NB portent en plus deux inadéquations de représentation dont l'ampleur
cumulée est du même ordre que la différence de famille que l'ancrage devait
isoler. Le recouvrement mesuré mélangerait les deux sans qu'on puisse les
séparer.

## Ce qui en découle, et qui était déjà prévu

- **Le critère `R` est lu avec le seul ancrage LLM/LLM, `lift = 3,19`.**
- **La limite énoncée au §5, premier point, cesse d'être compensée.** Elle disait
  que 3,19 mélange difficulté des données et similarité des deux LLM, et qu'il
  majore donc ce qu'un mécanisme partagé explique. Le second ancrage devait
  permettre de vérifier que les deux mesures se ressemblent avant de s'appuyer
  sur l'une. Cette vérification n'aura pas lieu, et la limite doit être répétée
  partout où `R` est cité.
- **La troisième ligne du tableau du §3 devient inapplicable** — « les deux lifts
  intra-famille divergent fortement » suppose deux lifts.
- Le recouvrement `LogisticRegression` / `MultinomialNB` **sera tout de même
  calculé et rapporté**, explicitement étiqueté comme contaminé et **non
  décisionnel**, au même titre que le complément qualitatif du §6. Il documente,
  il ne tranche pas.

## Ce qui reste ouvert

Rétablir un second ancrage supposerait un modèle d'une troisième famille qui ne
souffre pas du même défaut. Aucun n'est engagé : ce serait un choix de modèle
pris **après** avoir vu les résultats, et il doit être décidé explicitement, pas
glissé dans une correction. En l'état, le rapport final conclura avec un seul
ancrage haut et le dira.

## Décision du 2026-08-19 — on conclut avec un seul ancrage, et on le dit

Le second ancrage **n'est pas rétabli**. Aucun modèle d'une troisième famille
n'est ajouté. Trois raisons.

1. **La conduite en cas d'échec était écrite d'avance, et elle s'applique.** Le
   protocole prévoyait ce cas : `R` lu avec le seul ancrage LLM/LLM, et la limite
   du §5 répétée partout où `R` est cité. Un protocole qui prévoit son propre
   échec puis qu'on contourne quand l'échec survient ne protège de rien.
2. **Ajouter `ComplementNB` maintenant serait choisir un modèle après avoir vu
   que le premier ne convenait pas.** Le motif serait pourtant bon — il est conçu
   contre exactement la mauvaise spécification mesurée. Mais même avec un bon
   motif, plus personne ne pourrait distinguer « prévu » de « retenu parce qu'il
   donnait le résultat attendu ». C'est le même défaut que celui contre lequel ce
   protocole a été écrit avant tout entraînement.
3. **L'ancrage n'était pas au protocole d'origine.** Il y a été ajouté en croyant
   lever une limite à coût nul. Le retirer ramène donc à **l'état validé**, pas à
   un état dégradé : ce qui est perdu est un supplément espéré, pas un acquis.
