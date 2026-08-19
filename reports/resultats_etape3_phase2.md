# Résultats — étape 3, phase 2 (évaluation et analyse croisée)

Produit par `tools/etape3_evaluation.py`. Données brutes lisibles par machine :
`reports/etape3_evaluation.json`. Agrégats seulement, aucun texte de réclamation.

Modèles **chargés depuis `models/`**, pas réentraînés : le chiffre publié vient
de l'artefact livré, dont l'empreinte est enregistrée dans les métadonnées.
`metrics.py` et `data_prep.py` inchangés ; toutes les métriques passent par
`metrics.evaluate()` avec `sample_weight` fourni explicitement.

## 1. Les 2 000 lignes pondérées Horvitz-Thompson

**Seule comparaison valide avec le LLM.** Mêmes 2 000 lignes, somme des poids
70 863.

| modèle | accuracy | **F1-macro** | IC 95 % | F1-pondéré | précision macro | rappel macro | latence p50 | p95 |
|---|---|---|---|---|---|---|---|---|
| **`LinearSVC`** | 0,8668 | **0,8187** | [0,798 ; 0,839] | 0,8665 | 0,8219 | 0,8165 | 0,23 ms | 0,50 ms |
| `LogisticRegression` | 0,8546 | 0,8136 | [0,792 ; 0,835] | 0,8551 | 0,7972 | 0,8340 | 0,23 ms | 0,48 ms |
| `MultinomialNB` | 0,7867 | 0,7233 | [0,704 ; 0,744] | 0,7940 | 0,6876 | 0,8077 | 10,41 ms | 12,50 ms |
| `mistral-small` (LLM) | 0,8149 | 0,7885 | [0,766 ; 0,810] | 0,8171 | 0,8084 | 0,7821 | 470 ms | 740 ms |

### F1 par classe

| classe | `LinearSVC` | `LogisticRegression` | `MultinomialNB` | `mistral-small` |
|---|---|---|---|---|
| Credit reporting | 0,8881 | 0,8658 | 0,8239 | 0,8175 |
| Debt collection | 0,8533 | 0,8337 | 0,7597 | 0,7708 |
| Mortgage | 0,9338 | 0,9216 | 0,8909 | 0,9342 |
| Credit card or prepaid card | 0,8197 | 0,8332 | 0,7530 | 0,7959 |
| Bank account or service | 0,8517 | 0,8569 | 0,8055 | 0,8112 |
| Student loan | 0,8797 | 0,8831 | 0,7994 | 0,8532 |
| Money transfer or virtual currency | 0,8254 | 0,8003 | 0,7075 | 0,7315 |
| Payday, title or personal loan | 0,6980 | 0,6774 | 0,5742 | 0,6966 |
| **Vehicle loan or lease** | 0,6190 | 0,6500 | 0,3955 | **0,6854** |

Matrice de confusion : `reports/figures/etape3_confusion_linearsvc.png`,
9 classes, pondérée. **Aucune colonne `PARSE_ERROR`** — le ML rend toujours une
classe du référentiel.

La latence du LLM inclut l'aller-retour réseau. Rapport p50 `LinearSVC` / LLM :
**≈ 2 030**.

## 2. Test complet — tableau SÉPARÉ, non pondéré

> **Ce chiffre ne se compare pas au LLM.** Population, pondération et effectif
> différents. Il montre ce que le volume d'entraînement apporte au ML.

| | `LinearSVC` sur le test complet |
|---|---|
| n | 70 863 |
| F1-macro | **0,8193** |
| accuracy | 0,8747 |
| F1-pondéré | 0,8745 |

## 3. McNemar apparié — `LinearSVC` vs `mistral-small`

Mêmes 2 000 lignes. Les `PARSE_ERROR` du LLM comptent comme des erreurs, comme à
l'étape 2.

| | |
|---|---|
| paires | 2 000 |
| `LinearSVC` juste / LLM faux — **b** | **196** |
| `LinearSVC` faux / LLM juste — **c** | **100** |
| paires discordantes | 296 |
| accord | 1 704 |
| **p bilatérale exacte** | **2,556 × 10⁻⁸** |

Écart de F1-macro pondéré, `LinearSVC` − `mistral-small` :

| | |
|---|---|
| écart | **+0,0303** |
| IC 95 % apparié | **[+0,0071 ; +0,0525]** |
| plan | stratifié intra-classe, **même tirage pour les deux modèles**, 1 000 rééchantillonnages |

L'IC est apparié : deux IC calculés séparément traiteraient les modèles comme
indépendants alors qu'ils portent sur les mêmes réclamations.

## 4. Analyse croisée

Protocole du 2026-08-19 et son erratum. Dénombrement **non pondéré** (§5,
dernier point). `PARSE_ERROR` exclus (§1) : **19 lignes**, soit 1 981 retenues.
Contrôle : 385 erreurs du LLM sur 2 000 moins 19 `PARSE_ERROR` = 366, ce que
les trois tableaux retrouvent.

### Mesures

| | `LinearSVC` / LLM | `LogisticRegression` / LLM | `MultinomialNB` / LLM |
|---|---|---|---|
| lignes retenues | 1 981 | 1 981 | 1 981 |
| erreurs ML | 282 (14,24 %) | 294 (14,84 %) | 411 (20,75 %) |
| erreurs LLM | 366 (18,48 %) | 366 (18,48 %) | 366 (18,48 %) |
| attendu sous indépendance | 52,1 | 54,3 | 75,9 |
| **co-échec** (déf. primaire) | **182** | **209** | **221** |
| co-échec **orienté** (même classe fausse) | 144 — 79,1 % | 170 — 81,3 % | 159 — 71,9 % |
| **lift** | **3,493** | **3,848** | **2,910** |
| Jaccard | 0,391 | 0,463 | 0,397 |
| **R = lift / 3,19** | **1,095** | **1,206** | **0,912** |

Référence intra-famille LLM/LLM, mesurée à l'étape 2 et figée : lift 3,19,
part orientée 83,6 %.

### Critère complémentaire — concentration sur les paires du §C.4

Seuil pré-fixé : **10 points de pourcentage**.

| | `LinearSVC` / LLM | `LogisticRegression` / LLM | `MultinomialNB` / LLM |
|---|---|---|---|
| erreurs communes | 37,4 % de 182 | 41,1 % de 209 | 40,3 % de 221 |
| propres au ML | 46,0 % de 100 | 50,6 % de 85 | 36,3 % de 190 |
| propres au LLM | 58,7 % de 184 | 56,7 % de 157 | 56,6 % de 145 |
| **écart communes − propres** | **−15,0 pts** | **−12,5 pts** | **−6,2 pts** |

Le signe est **négatif dans les trois cas** : les erreurs communes sont **moins**
concentrées sur les paires annoncées que les erreurs propres.

### Point 3 — taux de rattrapage sur les 3 classes de plus faible F1

Dénominateur : erreurs du LLM dont la classe **vraie** appartient à
`Vehicle loan or lease`, `Payday, title or personal loan`, `Debt collection`.
Seuil pré-fixé : **30 %**.

| | numérateur | dénominateur | taux |
|---|---|---|---|
| `LinearSVC` | 80 | 157 | **51,0 %** |
| `LogisticRegression` | 67 | 157 | 42,7 % |
| `MultinomialNB` | 55 | 157 | 35,0 % |

## 5. Ancrage intra-famille ML — CONTAMINÉ, NON DÉCISIONNEL

Rapporté pour documentation seulement, conformément à l'erratum. `MultinomialNB`
porte deux inadéquations de représentation mesurées — vectoriseur +0,010,
lissage +0,042 — de sorte que son recouvrement avec `LogisticRegression` mélange
différence d'architecture et défaut de représentation sans qu'on puisse les
séparer. **Le critère `R` est lu sans cet ancrage.**

| `LogisticRegression` / `MultinomialNB` | |
|---|---|
| lignes retenues | **2 000** (aucun `PARSE_ERROR` : ni l'un ni l'autre n'en produit) |
| erreurs `LogisticRegression` | 300 (15,00 %) |
| erreurs `MultinomialNB` | 417 (20,85 %) |
| attendu sous indépendance | 62,5 |
| co-échec | 243 |
| co-échec orienté | 203 — 83,5 % |
| lift | 3,885 |
| Jaccard | 0,513 |
| concentration, écart communes − propres | **+15,9 pts** |
| rattrapage 3 classes | 75 / 169 = 44,4 % |

**Base non identique.** Cette paire porte sur 2 000 lignes, les trois autres sur
1 981. Les taux marginaux ne se comparent donc pas terme à terme aux tableaux
du §4.

Le signe de la concentration est **positif** ici, à l'inverse des trois paires
LLM/ML.

## Confrontation aux critères pré-enregistrés

Les valeurs mesurées, en regard des seuils fixés avant tout entraînement. **La
lecture n'est pas faite ici.**

| critère du protocole | seuil pré-fixé | valeur mesurée (`LinearSVC` / LLM) |
|---|---|---|
| §4 — plafond dans les données | `R ≥ 0,75` | `R` = **1,095** |
| §4 — plafond partiel | `0,40 ≤ R < 0,75` | — |
| §4 — limites propres aux modèles | `R < 0,40` | — |
| §4 — infirmation, point 1 | `R < 0,40` | `R` = 1,095 |
| §4 — infirmation, point 2 | communes **pas plus** concentrées que propres | écart **−15,0** points |
| §4 — infirmation, point 3 | rattrapage **≥ 30 %** | **51,0 %** |

`R = 1,095` place le recouvrement croisé **au-dessus** de l'ancrage intra-famille
LLM/LLM. La grille du §4 s'arrête à `R ≥ 0,75` et ne distingue rien au-delà de 1 :
le cas n'était pas anticipé, et c'est un fait de la mesure, pas une lecture.

---

# Lecture des critères — le plafond du §C.4 est INFIRMÉ

**Verdict arrêté le 2026-08-19 par le responsable du projet, après application
des critères pré-enregistrés.** Deux clauses d'infirmation sur trois se
déclenchent, et le protocole dit qu'une seule suffit.

| clause | seuil pré-fixé | mesuré | déclenchée |
|---|---|---|---|
| §4 point 1 — `R < 0,40` | 0,40 | `R` = 1,095 | non |
| §4 point 2 — communes **pas plus** concentrées que propres | écart ≥ 10 pts | **−15,0 pts** | **oui** |
| §4 point 3 — rattrapage `≥ 30 %` sur les 3 classes | 30 % | **51,0 %** | **oui** |

Le critère principal `R ≥ 0,75` ne l'emporte pas : les clauses d'infirmation
priment, c'est ce qui en fait des clauses d'infirmation.

## Comment `R = 1,095` et les deux infirmations coexistent

Elles ne mesurent pas la même chose, et rien ne les oppose.

**`R` mesure s'il existe un socle d'items partagé.** 182 co-échecs contre 52,1
attendus sous indépendance : oui, il existe. Deux approches qui ne partagent ni
architecture, ni représentation du texte, ni données d'entraînement échouent sur
les mêmes réclamations bien plus souvent que le hasard ne l'expliquerait, et
aussi souvent que deux modèles de la même famille. **Cette partie de l'intuition
du §C.4 tient.**

**Les deux clauses mesurent où ce socle se trouve et s'il est irréductible.**

- La concentration dit **où** : les erreurs communes sont *moins* portées par les
  paires annoncées (37,4 %) que les erreurs propres à chaque approche (46,0 % et
  58,7 %). Le recouvrement existe, mais il ne tombe pas là où le §C.4 l'avait
  situé. Les paires annoncées décrivent surtout ce que **chaque approche rate de
  son côté**.
- Le rattrapage dit **s'il est irréductible** : sur les erreurs du LLM dont la
  classe vraie appartient aux trois classes désignées comme les plus difficiles,
  le ML en corrige **51,0 %**. Ces classes ne sont donc pas intrinsèquement
  dures ; elles étaient mal traitées par le LLM.

**Ce que les trois disent ensemble.** Il existe un socle de difficulté partagé,
mais **le §C.4 l'a mal localisé et surestimé son étendue**. L'hypothèse infirmée
n'est pas « certains items sont durs » — énoncé trivial et intestable. Elle était
« le plafond se situe sur ces paires-là et sur ces trois classes-là », et c'est
cela qui échoue. Un `R` élevé confirme l'existence d'un plancher ; il ne confirme
rien sur son contenu, et c'est précisément le contenu que les deux clauses
réfutent.

---

# Caractérisation du socle partagé

Le protocole prévoyait 20 lectures manuelles non décisionnelles. Les données
permettaient en outre des mesures quantitatives **qu'il n'avait pas prévues** ;
elles sont ajoutées ici et signalées comme telles.

## A. Spectre de difficulté sur 5 approches — mesure non prévue

Les mêmes lignes passées par `LinearSVC`, `LogisticRegression`,
`MultinomialNB`, `mistral-small` et `ministral-3b`. Base : 1 976 lignes
(`PARSE_ERROR` de l'un ou l'autre LLM exclus, 24 lignes).

| échecs sur 5 | lignes | part | part de population (HT) | médiane de mots |
|---|---|---|---|---|
| 0 | 1 244 | 63,0 % | 63,3 % | 149 |
| 1 | 288 | 14,6 % | 14,9 % | 147 |
| 2 | 132 | 6,7 % | 6,3 % | 152 |
| 3 | 111 | 5,6 % | 5,7 % | 113 |
| 4 | 70 | 3,5 % | 3,5 % | 108 |
| **5** | **131** | **6,6 %** | **6,3 %** | 127 |

**131 réclamations sont manquées par les cinq approches.** Attendu sous
indépendance : **0,38**. Le rapport est de **345**.

La distribution n'est pas décroissante : **plus d'items échouent aux cinq (131)
qu'à exactement quatre (70) ou exactement trois (111)**. Un noyau dur existe, et
il est séparé du reste plutôt que situé au bout d'un continuum.

## B. Les 182 co-échecs `LinearSVC` / `mistral-small`

Base du protocole : 1 981 lignes.

### Poids de population — mesure non prévue

L'échantillon est stratifié à plancher ; un décompte de lignes ne dit pas ce que
le groupe pèse dans la population.

| groupe | lignes | part des lignes | part de la population (HT) |
|---|---|---|---|
| co-échecs | 182 | 9,19 % | **8,49 %** |
| propres au ML | 100 | 5,05 % | 4,61 % |
| propres au LLM | 184 | 9,29 % | 9,22 % |
| justes des deux | 1 515 | 76,48 % | 77,68 % |

Le socle partagé pèse **légèrement moins** en population qu'en lignes. Il n'est
pas amplifié par la pondération.

### Longueur du texte — mesure non prévue

| groupe | q1 | médiane | q3 |
|---|---|---|---|
| **co-échecs** | **57** | **116** | 242 |
| propres au ML | 88 | 152 | 244 |
| propres au LLM | 68 | 126 | 229 |
| justes des deux | 82 | 149 | 276 |

**Les co-échecs sont le groupe le plus court**, au premier quartile comme à la
médiane : 57 mots contre 82 pour les items correctement classés, 116 contre 149.
Les erreurs propres au ML sont, elles, les plus *longues*.

### Composition par classe vraie

| classe vraie | n | part du groupe | part de l'échantillon | sur-représentation | taux de co-échec **dans** la classe |
|---|---|---|---|---|---|
| Credit reporting | 35 | 19,2 % | 26,0 % | 0,74 | 6,8 % |
| Debt collection | 48 | 26,4 % | 20,8 % | 1,27 | 11,7 % |
| Mortgage | 2 | 1,1 % | 14,2 % | **0,08** | **0,7 %** |
| Credit card or prepaid card | 29 | 15,9 % | 11,5 % | 1,38 | 12,7 % |
| Bank account or service | 14 | 7,7 % | 8,6 % | 0,89 | 8,2 % |
| Student loan | 10 | 5,5 % | 7,2 % | 0,76 | 7,0 % |
| Money transfer or virtual currency | 15 | 8,2 % | 4,0 % | 2,04 | 18,8 % |
| **Payday, title or personal loan** | 17 | 9,3 % | 3,9 % | **2,40** | **22,1 %** |
| Vehicle loan or lease | 12 | 6,6 % | 3,7 % | 1,77 | 16,2 % |

`Mortgage` est presque absente du socle : 2 co-échecs sur 282 lignes de la
classe, soit 0,7 %. `Payday` y est trente fois plus exposée.

### Directions partagées — 144 co-échecs orientés sur 182 (79,1 %)

| n | direction (vraie → prédite par les deux) | paire §C.4 |
|---|---|---|
| 18 | Debt collection → Credit reporting | oui |
| **13** | **Credit card or prepaid card → Credit reporting** | **non** |
| 12 | Credit reporting → Debt collection | oui |
| 9 | Debt collection → Student loan | non |
| 8 | Payday, title or personal loan → Debt collection | oui |
| 8 | Money transfer or virtual currency → Bank account or service | oui |
| 7 | Credit reporting → Mortgage | non |
| 7 | Debt collection → Credit card or prepaid card | non |
| 7 | Debt collection → Mortgage | non |
| 5 | Bank account or service → Mortgage | non |

`Credit card or prepaid card ↔ Credit reporting` occupe le **rang 2** et n'est
pas au §C.4. C'est **la même paire** qui avait fait échouer le critère C de
l'étape 2, où elle apparaissait au rang 2 chez les deux LLM. Elle réapparaît ici
dans une configuration entièrement différente — ML contre LLM, sur les seuls
co-échecs.

## C. Vingt lectures manuelles — NON DÉCISIONNELLES

Tirage à graine `RANDOM_SEED`. Textes exportés dans `data/llm/` (répertoire
exclu du dépôt), jamais dans `reports/`.

**Répartition selon la grille de la phase 2 : 20 items en (A) ambiguïté réelle,
0 en (B) défaut de modèle.**

Un mécanisme domine, et il n'était pas anticipé : **l'étiquette du CFPB suit le
PRODUIT, le texte de la réclamation décrit la CONDUITE.** Un consommateur dont
la carte de crédit est passée en recouvrement écrit une histoire de recouvrement ;
l'étiquette reste `Credit card`. Les deux approches suivent la conduite, pas le
produit. Cas 02 (`Payday` étiquetée, harcèlement téléphonique décrit), 08
(`Debt collection` étiquetée, « Target Credit Card » écrit), 14, 17, 20 relèvent
tous de ce schéma.

Second mécanisme : **l'anonymisation efface l'identité du produit.** Cas 07,
« my loan is with synchrony bank who finances XXXX » — le nom du bien financé,
qui déterminerait `Vehicle loan`, est masqué. Cas 18, « WESTLAKE FINANCIAL » sans
mention d'un véhicule.

Trois cas (04, 16, 18) sont mieux décrits par **« le texte ne détermine pas
l'étiquette »** que par « deux lectures défendables » : 39, 28 et 25 mots, aucun
produit nommé. La grille A/B de la phase 2 n'offrait pas cette catégorie ; je la
signale sans la substituer.

> **Portée de cette section : nulle pour la décision.** Vingt cas lus à la main,
> par la personne qui connaît l'hypothèse, ne constituent pas une mesure. Elle
> illustre ce que les chiffres ci-dessus établissent, elle ne l'établit pas.
>
> **Et le 20/0 est plus faible encore qu'il n'en a l'air.** Le critère A —
> « le texte décrit un préjudice pointant vers une autre classe » — est **quasi
> tautologique** sur un ensemble retenu précisément parce que deux approches y
> ont prédit une autre classe : il mesure la sélection, pas les items. S'y
> ajoutent un intervalle de Clopper-Pearson de [83,2 % ; 100 %] sur 20/20, un
> lecteur unique non aveugle sans arbitrage, et une grille à choix forcé sans
> case « ni l'un ni l'autre ». Le développement complet est au §7 de
> `etape3_synthese.md`, avec la part de cette section qui repose sur des mesures
> — tout le reste — et la raison pour laquelle les trois cas hors grille sont
> plus informatifs que le score.

---

# Limite du protocole — la grille du §4 ne distingue rien au-dessus de `R = 1`

`R` a été construit comme `lift(croisé) / lift(intra-famille)`, avec trois bandes
dont la plus haute est `R ≥ 0,75`. **Cette construction contenait une supposition
non formulée : que l'ancrage intra-famille est un majorant du recouvrement.**

Autrement dit, qu'aucune paire d'approches ne pourrait se recouvrir *davantage*
que deux modèles partageant famille, tokenizer, prompt et paradigme
d'entraînement. `R` a été écrit comme une **position sous un plafond**, et un
rapport à un plafond n'a pas de sens au-dessus de 1. C'est pourquoi rien n'y est
distingué : `R > 1` n'était pas traité comme improbable, mais comme impossible.

Deux raisons rendent la supposition fausse.

**1. Le recouvrement a deux sources, pas une.** Le modèle mental de la grille est
monotone — plus de mécanisme partagé, plus de recouvrement. Mais le recouvrement
tient aussi à la difficulté des items, qui est une propriété des données et non
des modèles. Le §5 du protocole disait déjà que 3,19 mélange les deux ; la grille
traite ensuite ce mélange comme un plafond pur sur la composante « mécanisme ».

**2. Le `lift` a un plafond arithmétique propre à chaque paire.** Puisque
`|A ∩ B| ≤ min(|A|, |B|)`, on a `lift ≤ 1 / max(pA, pB)`. Les quatre paires
mesurées n'ont donc pas le même maximum atteignable :

| paire | pA | pB | plafond du lift | lift observé | **% du plafond** |
|---|---|---|---|---|---|
| `LinearSVC` / mistral-small | 0,1422 | 0,1842 | 5,429 | 3,493 | **64,3 %** |
| `LogisticRegression` / mistral-small | 0,1478 | 0,1842 | 5,429 | 3,848 | 70,9 % |
| `MultinomialNB` / mistral-small | 0,2075 | 0,1842 | 4,820 | 2,910 | 60,4 % |
| **mistral-small / ministral-3b** *(ancrage)* | 0,1842 | 0,2394 | **4,178** | 3,190 | **76,4 %** |

Diviser par une constante de 3,19 compare des grandeurs dont les maxima diffèrent
de 30 %. **Rapporté à son propre plafond, l'ancrage intra-famille (76,4 %) est
au-dessus du croisé LLM/ML (64,3 %) — l'ordre inverse de celui que `R = 1,095`
affiche.** Le classement produit par `R` n'est pas robuste au choix de
normalisation.

> **Ce constat ne change pas le verdict.** `R` n'est pas le critère qui a tranché :
> l'infirmation vient des clauses 2 et 3, qui ne passent pas par le `lift`. Ce
> paragraphe documente une faiblesse du critère qui **n'a pas décidé**, et il
> serait malhonnête de s'en servir pour rouvrir celui qui a décidé.

**Ce qu'il aurait fallu écrire.** Une grille sur le `lift` normalisé par son
plafond arithmétique, ou une comparaison directe des parts de plafond entre
paires, plutôt qu'un rapport à une constante mesurée sur une paire aux taux
marginaux différents.

---

# Confrontation à la simulation h1 §7

La simulation du 2026-08-11 prévoyait, pour la stratégie retenue B3 avec erreurs
corrélées, un `RMSE` de l'écart apparié de **0,0157** et un écart minimal
détectable de **3,1 points** de F1-macro.

| grandeur | prévu (h1 §7) | mesuré | rapport |
|---|---|---|---|
| écart-type de l'écart apparié | 0,0157 | **0,01157** | **0,737** — réalisé 26,3 % plus petit |
| écart minimal détectable | 0,0308 (3,1 pts) | 0,0227 (2,27 pts) | 0,737 |
| écart observé | — | **0,0303 (3,03 pts)** | **0,983 × le MDE prévu** |
| facteur d'annulation de l'appariement | 0,89 | **0,754** | l'appariement aide plus que prévu |

Trois constats.

**L'écart observé tombe à 1,7 % sous le seuil de détection prévu.** 3,03 points
contre 3,1 annoncés. Si la précision réalisée avait été celle de la simulation,
l'IC à 95 % aurait contenu zéro et la comparaison n'aurait pas tranché sur le
F1-macro. La conclusion tient à ce que la précision réelle s'est trouvée
meilleure que prévue.

**Elle l'est de 26 %.** L'écart-type apparié réalisé vaut 0,01157 contre 0,0157
simulé. Rapporté à cette précision, l'écart observé fait 1,33 fois le MDE et l'IC
exclut zéro : `[+0,0071 ; +0,0525]`.

**La raison est mesurable, et c'est le même chiffre que l'analyse croisée.** La
simulation modélisait la corrélation des erreurs par une variable latente de
difficulté et en tirait un facteur d'annulation de 0,89, soit 11 % de réduction.
Le facteur réalisé est de **0,754**, soit 24,6 % : les erreurs des deux approches
sont **plus corrélées** que la simulation ne le supposait. C'est exactement ce que
le `lift` de 3,49 dit par ailleurs. La simulation a sous-estimé le recouvrement,
donc sous-estimé le bénéfice de l'appariement, donc surestimé le seuil de
détection.

**L'avertissement du §7 est confirmé sur le fond.** Il annonçait qu'« il est très
possible que la comparaison ne tranche pas sur le F1-macro global » et qu'il
faudrait s'appuyer sur le F1 par classe et sur les critères non statistiques.
L'écart global tranche de justesse ; McNemar, qui ne passe pas par le F1, tranche
largement (b = 196, c = 100, p = 2,56 × 10⁻⁸). Les deux ne mesurent pas la même
chose, et c'est le second qui porte la conclusion la plus solide.
