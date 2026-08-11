# Limites et précautions d'interprétation

**Projet ZenAssist — étape 1**
Source des mesures : `tools/limites_analyse.py` sur `data/raw/dataset.csv`
Date : 2026-08-11

Ce document recense les limites de la solution et les précautions à prendre pour
interpréter ses résultats. Chaque section distingue explicitement trois statuts :

> 📊 **MESURÉ** — chiffre calculé sur les données, vérifiable
> 🔶 **SUPPOSÉ** — hypothèse de travail, non validée
> 🔜 **À FAIRE** — action planifiée, avec son étape

Deux des quatre limites ne sont **pas corrigeables** (§1 et §2) : elles sont des
propriétés du jeu de données, pas des défauts du pipeline. Elles sont donc
quantifiées plutôt que supposées.

---

## 1. Représentativité du corpus

### 1.a Canal de dépôt — 26 % du flux est hors périmètre par construction

📊 **MESURÉ**

| Canal | Total | % du flux | Avec narratif | % avec narratif |
|---|---:|---:|---:|---:|
| **Web** | 945 329 | **73,72** | 383 564 | **40,57** |
| Referral | 173 640 | 13,54 | 0 | **0,00** |
| Phone | 76 509 | 5,97 | 0 | **0,00** |
| Postal mail | 67 522 | 5,27 | 0 | **0,00** |
| Fax | 18 972 | 1,48 | 0 | **0,00** |
| Email | 383 | 0,03 | 0 | **0,00** |

Cinq canaux sur six ne produisent **jamais** de narratif — pas « rarement » :
zéro ligne sur 337 026. **26,28 % du flux réel est structurellement hors
périmètre**, et aucune amélioration du modèle ne changera cela : il n'y a pas de
texte à classer.

Parmi les 73,72 % déposés sur le Web, seuls 40,57 % portent un narratif — la
raison est établie en 1.c.

### 1.b Composition par catégorie — écart réel mais modéré

📊 **MESURÉ**

Distribution des 18 libellés bruts, lignes avec narratif contre lignes sans :

| Tag (libellé brut) | % avec | % sans | **écart (pp)** | taux de narratif |
|---|---:|---:|---:|---:|
| Credit reporting, credit repair services… | 24,08 | 14,86 | **+9,22** | 40,88 % |
| Debt collection | 22,61 | 17,60 | **+5,01** | 35,41 % |
| Credit card or prepaid card | 5,57 | 2,92 | +2,65 | 44,86 % |
| Student loan | 5,69 | 3,32 | +2,37 | 42,20 % |
| Money transfer, virtual currency… | 1,43 | 0,52 | +0,91 | 53,78 % |
| Vehicle loan or lease | 1,50 | 0,63 | +0,87 | 50,50 % |
| Payday loan, title loan, or personal loan | 1,15 | 0,47 | +0,68 | 51,06 % |
| Checking or savings account | 3,36 | 3,09 | +0,27 | 31,69 % |
| Prepaid card | 0,38 | 0,26 | +0,12 | 37,97 % |
| Payday loan | 0,46 | 0,42 | +0,04 | 31,51 % |
| Consumer Loan | 2,47 | 2,46 | +0,01 | 29,98 % |
| Virtual currency | 0,00 | 0,00 | 0,00 | 88,89 % |
| Other financial service | 0,08 | 0,09 | −0,01 | 27,57 % |
| Money transfers | 0,39 | 0,43 | −0,04 | 27,96 % |
| Credit card | 4,91 | 7,83 | −2,92 | 21,12 % |
| Credit reporting | 8,24 | 12,11 | −3,87 | 22,49 % |
| Bank account or service | 3,88 | 7,94 | −4,06 | 17,27 % |
| **Mortgage** | 13,81 | 25,05 | **−11,24** | **19,05 %** |

**Test du khi-deux d'indépendance** (Tag × présence de narratif) :
χ² = 61 627, ddl = 17, **p < 10⁻³⁰⁰**, **V de Cramér = 0,219**.

Lecture honnête de ces deux chiffres, qui disent des choses différentes :

- Le **p < 10⁻³⁰⁰** ne signifie pas « biais énorme ». Sur 1,28 million
  d'observations, le khi-deux détecte le moindre écart. Il faut le lire comme
  « l'indépendance est exclue », rien de plus.
- Le **V de Cramér de 0,219** est la mesure d'intensité : association réelle mais
  **modérée**. Ce n'est ni négligeable ni disqualifiant.
- L'**écart maximal est de 11,24 points** sur `Mortgage`, et la **distance de
  variation totale** (somme des écarts positifs) vaut **22,15 points** — c'est-à-dire
  qu'il faudrait déplacer 22 % de la masse pour passer d'une distribution à
  l'autre.

Le biais a une direction interprétable : les catégories **sous-représentées** sont
les produits bancaires anciens et lourds (`Mortgage` −11,2 pp, `Bank account`
−4,1 pp, `Credit card` −2,9 pp), les **sur-représentées** sont les litiges de
signalement de crédit (+9,2 pp) et de recouvrement (+5,0 pp). Cohérent avec
1.c et 1.d : ces dernières sont des catégories récentes, majoritairement
déposées en ligne, à une période où le narratif était collecté.

### 1.c Le consentement explique tout

📊 **MESURÉ** — c'est le résultat le plus net de cette section.

| `Consumer consent provided?` | Sans texte | Avec texte | Total | % avec texte |
|---|---:|---:|---:|---:|
| **Consent provided** | **321** | **383 564** | 383 885 | **99,92 %** |
| (non renseigné) | 591 701 | 0 | 591 701 | 0,00 % |
| Consent not provided | 285 087 | 0 | 285 087 | 0,00 % |
| Other | 20 482 | 0 | 20 482 | 0,00 % |
| Consent withdrawn | 1 200 | 0 | 1 200 | 0,00 % |

**La présence d'un narratif est équivalente au consentement à sa publication.**
Zéro ligne a un texte sans consentement ; 321 lignes ont un consentement sans
texte (0,08 % — probablement un consentement donné puis un formulaire laissé
vide).

Ce n'est donc pas un filtre technique mais un **filtre juridique** : le CFPB ne
publie le récit du consommateur qu'avec son accord explicite. 285 087
consommateurs l'ont explicitement refusé.

🔶 **SUPPOSÉ** — On ne sait pas si les consommateurs qui refusent la publication
rédigent des réclamations différentes de ceux qui l'acceptent. C'est le seul
biais de cette section qui pourrait dégrader les performances plutôt que
seulement réduire le périmètre, et **il n'est pas mesurable ici** : les textes
non consentis n'existent pas dans le fichier. Un mécanisme plausible — les
réclamations les plus sensibles ou les plus identifiantes seraient moins souvent
consenties — reste une conjecture.

### 1.d Le narratif n'existe que depuis 2015

📊 **MESURÉ**

| Année | Total | Avec narratif | % |
|---|---:|---:|---:|
| 2011 | 2 536 | 0 | **0,00** |
| 2012 | 72 373 | 0 | **0,00** |
| 2013 | 108 218 | 0 | **0,00** |
| 2014 | 153 047 | 0 | **0,00** |
| 2015 | 168 487 | 54 758 | 32,50 |
| 2016 | 191 473 | 77 823 | 40,64 |
| 2017 | 242 975 | 115 172 | **47,40** |
| 2018 | 257 379 | 118 479 | 46,03 |
| 2019 (partielle) | 85 867 | 17 332 | 20,18 |

Confirmé : **aucun narratif avant 2015**, ce qui écarte d'office 336 174 lignes
(26,2 % du fichier). Le taux monte ensuite régulièrement jusqu'à ~47 % en 2017.

⚠️ **2019 n'est pas une rupture** : l'extraction s'arrête au 10/05/2019 et le
narratif n'est publié qu'après traitement de la réclamation. Les 20,18 % de 2019
reflètent des dossiers encore en cours au moment de l'extraction, pas une baisse
du consentement. **Ne pas présenter ce chiffre comme une tendance.**

### 1.e Géographie et entreprises — écarts faibles

📊 **MESURÉ**

**États** — les rangs sont quasi identiques ; le plus gros écart est de
1,62 point.

| État | % avec | % sans | écart (pp) | rang avec | rang sans |
|---|---:|---:|---:|---:|---:|
| CA | 13,58 | 14,09 | −0,52 | 1 | 1 |
| FL | 9,74 | 10,13 | −0,39 | 2 | 2 |
| TX | 9,56 | 8,07 | +1,49 | 3 | 3 |
| GA | 5,90 | 5,06 | +0,85 | 4 | 5 |
| NY | 5,69 | 7,31 | **−1,62** | 5 | 4 |
| NJ | 3,33 | 3,98 | −0,66 | 8 | 6 |

**Entreprises** — les trois bureaux de crédit dominent les deux
sous-populations ; les banques universelles sont sous-représentées parmi les
lignes avec narratif.

| Entreprise | % avec | % sans | écart (pp) | rang avec | rang sans |
|---|---:|---:|---:|---:|---:|
| EQUIFAX, INC. | 9,96 | 8,62 | +1,34 | 1 | 1 |
| Experian Information Solutions Inc. | 8,18 | 8,06 | +0,12 | 2 | 2 |
| TRANSUNION INTERMEDIATE HOLDINGS | 7,92 | 7,37 | +0,55 | 3 | 4 |
| WELLS FARGO & COMPANY | 3,63 | 6,34 | **−2,72** | 4 | 5 |
| BANK OF AMERICA, N.A. | 3,41 | 7,68 | **−4,27** | 6 | 3 |
| CITIBANK, N.A. | 3,48 | 3,97 | −0,49 | 5 | 7 |

Même direction qu'en 1.b : les banques universelles (Bank of America −4,3 pp,
Wells Fargo −2,7 pp) sont moins présentes, en cohérence avec la
sous-représentation de `Mortgage` et `Bank account or service`.

---

### 📦 PORTÉE DE LA SOLUTION

> **Ce que le modèle couvre**
> Les réclamations déposées **en ligne**, dont le consommateur a **consenti** à la
> publication du récit, **depuis 2015**. Soit **383 564 réclamations sur 1 282 355**,
> c'est-à-dire **29,9 % du flux historique** et environ **46 % du flux courant**
> (taux 2017-2018).
>
> **Ce que le modèle ne couvre pas**
> - **26,3 % du flux** déposé par téléphone, courrier, fax, e-mail ou transmission
>   inter-agences : **aucun texte n'existe**, la classification automatique est
>   impossible par nature.
> - **22,2 % du flux** déposé en ligne mais sans consentement à publication.
> - L'intégralité des dépôts antérieurs à 2015.

**Le biais mesuré réduit-il le périmètre ou dégrade-t-il les performances ?**
La distinction est essentielle et les deux effets coexistent, à des degrés très
différents :

| | Nature | Ampleur | Effet |
|---|---|---|---|
| **Réduction de périmètre** | Certaine, mesurée | 70 % du flux | Le modèle ne traitera jamais qu'une partie du flux. **Ce n'est pas une dégradation de performance** : sur son périmètre, il travaille sur des données de même nature que celles de l'entraînement. |
| **Dégradation potentielle** | Conjecturale | Inconnue | Ne se produirait que si les réclamations non consenties étaient rédigées différemment. Non mesurable ici. |

**Position à défendre** : le biais principal est un **biais de périmètre, pas un
biais de performance**. Le corpus d'entraînement et le flux réellement traité en
production sont la *même* population — les réclamations web consenties — puisque
seules celles-ci ont un texte à classer. La composition par catégorie diffère de
celle du flux global (V de Cramér 0,219), mais elle ne diffère pas de celle du
flux *classifiable*, qui est le seul qui compte.

**Ce qu'il faut dire au client**, en une phrase : *« La solution automatise la
catégorisation de la part du flux qui arrive sous forme de texte exploitable, soit
environ 46 % des réclamations reçues aujourd'hui. Le reste continue de relever du
traitement manuel, non par limite du modèle mais parce qu'il n'y a rien à
lire. »*

🔜 **À FAIRE** — Chiffrer le gain sur le seul périmètre couvert dans la
recommandation finale (étape 4), et ne jamais présenter un taux d'automatisation
rapporté au flux total.

---

## 2. Plafond de performance

### 2.a Borne inférieure dure

📊 **MESURÉ**

**Principe du calcul.** Un classifieur déterministe rend la même prédiction pour
deux textes identiques. Si un même texte porte plusieurs étiquettes dans les
données, le classifieur en satisfait au plus une : toutes les lignes portant une
autre étiquette sont **nécessairement** mal classées, quelle que soit l'approche.

Pour chaque groupe de textes identiques, le minimum d'erreurs vaut
`n_groupe − max(effectif de l'étiquette majoritaire)`. Sommé sur tous les
groupes, divisé par la taille du corpus.

Population de référence : **373 651 lignes** — corpus après filtrage du texte,
exclusions, fusion des libellés et seuil de longueur, mais **avant** vote
majoritaire et dédoublonnage (c'est la population qui reflète le flux réel).

| Définition de « même texte » | Groupes en conflit | Lignes | **Erreurs nécessaires** | Taux |
|---|---:|---:|---:|---:|
| **a)** texte exact (espaces normalisés) | 166 | 1 752 | **208** | **0,0557 %** |
| **b1)** insensible à la casse | 169 | 1 761 | 211 | 0,0565 % |
| **b2)** casse + ponctuation retirée | 177 | 1 835 | 234 | 0,0626 % |
| **b3)** 200 premiers caractères normalisés | 431 | 3 338 | **655** | **0,1753 %** |

> **Borne inférieure dure du taux d'erreur : 0,056 %** (définition stricte),
> **0,175 %** en tolérant les quasi-doublons de préambule.
> Autrement dit : **une accuracy de 99,94 % est mathématiquement inatteignable**,
> et 99,82 % en lecture élargie.

**Cette borne est vraie mais faible, et il faut le dire.** Elle prouve
rigoureusement qu'un plafond existe ; elle ne dit **rien** sur son niveau réel,
qui est certainement bien plus bas. Présenter 0,056 % comme « le plafond » serait
une faute de raisonnement.

### 2.b Ce que les quasi-doublons ajoutent

📊 **MESURÉ**

Le passage de l'exact au préfixe de 200 caractères **triple** le nombre de
groupes en conflit (166 → 431) et **plus que triple** les erreurs nécessaires
(208 → 655). L'évidence s'élargit bien au-delà des doublons exacts.

*Note de méthode* : la consigne prévoyait un échantillon de 50 000 lignes. Le
hachage étant peu coûteux, l'analyse a été menée sur **l'intégralité des 373 651
lignes** — les chiffres ci-dessus sont exhaustifs, pas extrapolés.

Assouplir encore la définition continuerait probablement à faire monter le
chiffre, ce qui est précisément le point : **la borne mesurée croît avec la
finesse de la mesure, donc elle sous-estime le plafond réel par construction.**

### 2.c Les courriers types FCRA — l'évidence la plus parlante

📊 **MESURÉ**

**14 962 textes (4,00 % du corpus)** citent la FCRA, 15 USC ou les sections
609/611. Leur répartition :

| Classe | n | % |
|---|---:|---:|
| Credit reporting | 9 929 | 66,36 |
| Debt collection | 4 016 | **26,84** |
| Credit card or prepaid card | 398 | 2,66 |
| Mortgage | 325 | 2,17 |
| Student loan | 110 | 0,74 |
| Vehicle loan or lease | 76 | 0,51 |
| Bank account or service | 75 | 0,50 |
| Payday, title or personal loan | 30 | 0,20 |
| Money transfer or virtual currency | 3 | 0,02 |

En regroupant par **modèle de lettre** (préfixe de 200 caractères identique, vu
au moins 5 fois) : **145 modèles couvrant 1 657 lignes**.

| Nombre de catégories par modèle | Modèles | % |
|---|---:|---:|
| 1 seule | 112 | **77,2** |
| 2 | 30 | 20,7 |
| 3 | 2 | 1,4 |
| 4 | 1 | 0,7 |

> **22,8 % des modèles de lettre (33 sur 145, soit 436 lignes) sont classés dans
> plusieurs catégories différentes.**

Trois exemples réels :

```
n=82 | Credit reporting: 78 | Debt collection: 3 | Credit card or prepaid card: 1
  "according to the fair credit reporting act section 609 a 1 a you are required
   by federal law to verify through the physical verification of the origin..."

n=39 | Debt collection: 38 | Credit card or prepaid card: 1
  "i have disputed this item with the credit reporting agency and they reported
   you confirmed the account as valid i honestly do not believe to ever have..."

n=25 | Debt collection: 17 | Credit reporting: 8
  "i noticed that you are reporting inaccurate data such as reporting an account
   as a collection account multiple months in a row when an account can onl..."
```

Le troisième est le plus instructif : **17 contre 8**, pas 24 contre 1. Ce n'est
pas une erreur isolée d'un agent distrait — c'est un **désaccord de fond** sur un
texte réellement ambigu. La paire `Credit reporting` ↔ `Debt collection`, déjà
identifiée comme la plus problématique (54 % du corpus, 35,8 % de recouvrement
lexical), se retrouve ici comme la ligne de fracture principale.

### 2.d Ce qui N'EST PAS mesurable dans ce projet

🔶 **NON MESURABLE — et volontairement non estimé**

Les bornes ci-dessus ne portent que sur les textes **répétés**. Elles ne disent
rien du **taux d'erreur d'annotation sur les textes uniques**, qui représentent
la quasi-totalité du corpus (342 411 groupes distincts sur 373 651 lignes en
définition élargie).

Mesurer ce taux exigerait une **double annotation indépendante** d'un échantillon
de textes uniques par deux annotateurs, puis un accord inter-annotateurs (kappa
de Cohen). Ce dispositif n'existe pas et ne peut pas être reconstruit a
posteriori : le fichier CFPB ne contient qu'une étiquette par réclamation, sans
trace de l'annotateur ni des éventuelles révisions.

**Aucune extrapolation n'est faite ici.** Il serait tentant d'écrire « si 22,8 %
des modèles de lettre sont ambigus, alors ~20 % du corpus l'est » — ce serait
invalide : les textes répétés sont précisément les courriers types génériques,
donc les plus ambigus par nature. Ils ne sont pas un échantillon représentatif du
corpus. Tout chiffre obtenu ainsi serait une invention habillée en mesure.

### 📦 FORMULATION DÉFENDABLE DU PLAFOND

> **Ce qui est prouvé** : un plafond de performance existe. La preuve est
> arithmétique et non contestable : **au moins 208 lignes (0,056 %) sont
> nécessairement mal classées**, 655 (0,175 %) en tolérant les quasi-doublons,
> parce que des textes identiques portent des étiquettes différentes.
>
> **Ce qui est fortement indiqué** : le plafond réel est très supérieur à cette
> borne. Trois faisceaux convergent — 22,8 % des modèles de lettre classés dans
> plusieurs catégories ; 35,8 % des réclamations `Debt collection` parlant
> explicitement du dossier de crédit ; une borne qui triple dès qu'on assouplit
> légèrement la définition de « même texte ».
>
> **Ce qui n'est pas mesurable** : la valeur de ce plafond. Elle nécessiterait une
> double annotation qui n'existe pas.
>
> **Formulation à retenir** : *« Une part de l'erreur résiduelle
> vient de l'ambiguïté de l'étiquetage humain, pas du modèle. Nous en avons la
> preuve arithmétique mais pas la mesure : les données ne permettent pas de la
> chiffrer. »*

🔜 **À FAIRE — étape 3 : analyse d'erreurs croisée LLM / ML**

C'est **la seule mesure honnête du plafond disponible dans ce projet**, et elle
ne coûte rien de plus puisque les deux séries de prédictions existeront déjà.

Protocole, sur les 2 000 lignes de `test_sample_2000.csv` :

1. Construire la table `(texte, vraie étiquette, prédiction LLM, prédiction ML)`.
2. Calculer le **taux d'erreurs communes** : parmi les lignes où au moins une
   approche se trompe, quelle fraction les voit se tromper **toutes les deux** ?
3. Parmi ces erreurs communes, quelle fraction produit la **même prédiction
   erronée** ? Deux modèles d'architectures totalement différentes convergeant
   vers la même mauvaise étiquette est le signal le plus fort possible que
   l'étiquette de référence est contestable.
4. Croiser avec les paires identifiées en C.4 du diagnostic. Si les erreurs
   communes se concentrent sur `Credit reporting` ↔ `Debt collection` et
   `Credit card` ↔ `Bank account`, la limite est **dans les données**.
5. Échantillonner 20 erreurs communes et les lire. Une inspection manuelle sur
   20 cas est un résultat exploitable à part entière.

**Interprétation prévue à l'avance**, pour éviter la rationalisation a
posteriori :

| Observation | Conclusion |
|---|---|
| Erreurs largement **communes**, concentrées sur les paires prévues | La limite est dans les données. Le plafond est réel et les deux approches l'atteignent. |
| Erreurs largement **disjointes** | La limite est dans les modèles. Il reste de la marge, et un ensemble des deux approches serait pertinent. |
| Cas **intermédiaire** | Décomposer par paire de classes plutôt que conclure globalement. |

---

## 3. Un seul tirage d'échantillon

### Le problème

📊 **MESURÉ (dans la simulation) / 🔶 NON VÉRIFIÉ (sur le tirage retenu)**

`test_sample_2000.csv` est un tirage **unique**, seed 42. La simulation de
`reports/h1_sampling_simulation.md` portait sur la **distribution** des tirages
possibles : elle établit que l'écart-type du F1-macro estimé vaut 0,0123 et que
l'IC à 95 % fait environ ±0,024.

Elle ne dit **rien** sur le tirage effectivement retenu. Celui-ci peut être
favorable ou défavorable — et rien ne garantit qu'il le soit de la même façon
pour le LLM et pour le ML, ce qui est le point sensible puisque la mission
consiste à les départager.

L'IC bootstrap ne répond pas à cette question : il mesure la variabilité
**interne** à l'échantillon, pas la variabilité **entre** échantillons.

### La parade — outil livré

✅ **FAIT** — `data_prep.construit_echantillon_alternatif(seed)` produit un second
échantillon B3 avec **exactement la même allocation par classe** (donc les mêmes
poids de Horvitz-Thompson) et un tirage différent. Vérifié : allocation et poids
identiques au premier échantillon, **recouvrement de 2,3 %** entre les deux — les
tirages sont pratiquement indépendants.

La fonction n'est **pas** appelée par `main()` : c'est un outil de vérification
pour l'étape 2, pas une étape du pipeline. Elle refuse `seed=42`, qui produirait
un échantillon identique au premier.

🔜 **À FAIRE — étape 2, une fois le prompt figé**

```python
from src.data_prep import construit_echantillon_alternatif, load_eval_sample
from src.metrics import evaluate, compare_results

# 1. échantillon de référence (déjà évalué)
df1, w1 = load_eval_sample()

# 2. second tirage — ~0,11 $ et ~35 min d'appels
df2, chemin = construit_echantillon_alternatif(seed=1337)
pred2 = classifie_avec_le_llm(df2["text"])          # MÊME prompt, figé

res1 = evaluate(df1["label"], pred1, sample_weight=w1, source=df1)
res2 = evaluate(df2["label"], pred2, sample_weight=df2["sampling_weight"],
                source=df2)
```

**Coût : ~0,11 $ et une trentaine de minutes.** À faire une seule fois, après
avoir figé le prompt — le refaire à chaque itération reviendrait à optimiser sur
les deux échantillons et détruirait la valeur du second.

**Critère de lecture, fixé à l'avance :**

| |Écart entre les deux F1-macro | Interprétation |
|---|---|---|
| ✅ | **< 0,024** (largeur de l'IC prévu) | Stabilité confirmée empiriquement. Preuve **plus forte** que l'IC bootstrap, car elle intègre la variabilité entre échantillons. À mettre en avant. |
| ⚠️ | 0,024 à 0,05 | Cohérent avec la simulation mais à mentionner explicitement. Publier les deux chiffres. |
| 🔴 | **> 0,05** | Anomalie. Chercher la cause (classe rare mal tirée, instabilité du LLM) **avant** la restitution, pas pendant. |

Le troisième cas est précisément la raison d'être de cette vérification :
**mieux vaut découvrir une divergence soi-même que se la voir signaler par le
client.**

---

## 4. Hypothèses de coût

### Statut actuel

🔶 **SUPPOSÉ — aucune valeur ci-dessous n'est mesurée sur le système réel**

Les constantes de coût sont désormais regroupées dans `src/config.py` sous un
bloc explicitement marqué `HYPOTHESES PROVISOIRES — A REMPLACER PAR DES MESURES
REELLES A L'ETAPE 2`, chacune portant son origine et son statut.

| Constante | Valeur | Origine | Fiabilité |
|---|---:|---|---|
| `MODELS_PRICING` | 10 modèles | pages tarifaires publiques, **consultées le 2026-08-11** | **volatile** — voir 4.1 |
| `AVG_COMPLAINT_TOKENS` | **257** | mots × 1,3 sur le corpus nettoyé | **la plus douteuse** |
| `PROMPT_INSTRUCTION_TOKENS` | 200 | comptage à la main d'un prompt non encore écrit | grossière |
| `PROMPT_LABELS_TOKENS` | 60 | idem | grossière |
| `AVG_OUTPUT_TOKENS` | 8 | idem | grossière |
| `CACHED_PREFIX_TOKENS` | 260 | instruction + 9 étiquettes | dépend du prompt final |

### 4.1 Tarifs — vérifiés le 2026-08-11

📊 **RELEVÉ** — sources : `mistral.ai/pricing`, `openai.com/pricing`,
`api-docs.deepseek.com`, `platform.claude.com/docs/en/about-claude/pricing`.

Dollars par million de tokens. Coût pour **1 000 prédictions par jour pendant un an**,
hypothèse de 517 tokens d'entrée et 8 de sortie par appel.

| Modèle | Fournisseur | $/M entrée | $/M sortie | $/an sans cache | $/an avec cache |
|---|---|---:|---:|---:|---:|
| ministral-3b | Mistral | 0,10 | 0,10 | **19** | 11 |
| deepseek-v4-flash | DeepSeek | 0,14 | 0,28 | 27 | 14 |
| **mistral-small-4** ← défaut | Mistral | 0,15 | 0,60 | **30** | **17** |
| gpt-5.6-luna | OpenAI | 0,20 | 1,20 | 41 | 24 |
| mistral-large-3 | Mistral | 0,50 | 1,50 | 99 | 56 |
| gemini-3.5-flash | Google | 0,75 | 4,50 | 155 | 155 ¹ |
| claude-haiku-4.5 | Anthropic | 1,00 | 5,00 | 203 | 118 |
| claude-sonnet-5 ² | Anthropic | 2,00 | 10,00 | 407 | 236 |
| claude-opus-5 | Anthropic | 5,00 | 25,00 | 1 016 | 589 |
| gpt-5.6-sol | OpenAI | 5,00 | 30,00 | **1 031** | 604 |

¹ ⚠️ **Le 0 % d'économie affiché pour `gemini-3.5-flash` est une absence de mesure de
notre côté, pas une absence de remise chez Google.** Nous n'avons pas relevé la remise
de cache de ce fournisseur au 2026-08-11 ; nous avons choisi de **ne pas la modéliser**
plutôt que de la supposer identique aux autres. Le coût affiché pour ce modèle est donc
**majoré**, et il n'est pas comparable en l'état aux neuf autres lignes de la colonne
« avec cache ». `estimate_cost()` retourne `cache_modelise = False` et un avertissement
explicite. **À relever à l'étape 2 si ce modèle entre dans la comparaison.**

² **Tarif d'introduction.** Voir 4.2.

> **Le coût n'est pas un facteur discriminant.** De 19 $ à 1 031 $ par an, l'écart
> couvre deux ordres de grandeur — mais même le plus cher reste négligeable devant le
> coût du traitement manuel qu'il remplace. La conclusion du diagnostic tient : **le
> facteur limitant est la latence**, pas le budget.
>
> Corollaire pour la recommandation : on peut choisir le modèle sur sa **qualité** et
> ses **contraintes de conformité**, pas sur son prix.

### 4.2 ⚠️ `claude-sonnet-5` : tarif d'introduction expirant le 31/08/2026

Le tarif de **2,00 / 10,00 $** est promotionnel. Le tarif standard de
**3,00 / 15,00 $** (**+50 %**) s'applique au **01/09/2026**. Une projection annuelle
bâtie sur le tarif d'introduction serait fausse dans trois semaines : 407 $/an
deviennent **610 $/an**.

C'est encodé dans `config.MODELS_PRICING` (`tarif_provisoire_jusquau`,
`tarif_apres`), et `estimate_cost()` **bascule automatiquement sur le tarif standard
et lève un `UserWarning`** dès lors que la date de référence dépasse l'échéance.
Vérifié :

```
au 2026-08-31 : tarif 2.0/10.0   coût 0,1114 $   warning = non
au 2026-09-01 : tarif 3.0/15.0   coût 0,1671 $   warning = OUI
```

### 4.3 ⚠️ Volatilité des tarifs — le bandeau reste

Deux mouvements sur les six dernières semaines suffisent à justifier le maintien du
bandeau `HYPOTHESES PROVISOIRES` :

- **OpenAI** a baissé `gpt-5.6-luna` de **80 %** le 30/07/2026.
- **DeepSeek** a annoncé une **hausse le 06/08/2026**, sans en préciser ni la date ni
  le montant, et applique une **tarification heures pleines doublée**.

Conséquence opérationnelle : **DeepSeek est utilisable pour comparer, pas pour une
projection à 12 mois.** Le mentionner si ce modèle figure dans la recommandation.

### 4.4 Le cache de préfixe — un levier plus fort que le choix du modèle

Le prompt de l'étape 2 aura un **préfixe constant de ~260 tokens** (instruction +
liste des 9 étiquettes), soit **la moitié des 517 tokens d'entrée par appel**. Ce
préfixe est éligible au cache chez tous les fournisseurs relevés :

| Fournisseur | Remise sur l'entrée cachée | Mode |
|---|---:|---|
| Anthropic | 90 % | déclaratif |
| Mistral | 90 % | déclaratif |
| OpenAI (familles 5.4 – 5.6) | 90 % | déclaratif |
| DeepSeek | **98 %** | **automatique** |
| Google | *non relevé de notre côté* | — |

⚠️ La ligne Google signale une **lacune de notre relevé**, pas une absence de cache chez
ce fournisseur. Tant qu'elle n'est pas comblée, `gemini-3.5-flash` apparaît plus cher
qu'il ne l'est probablement.

**Effet mesuré sur le modèle par défaut** : 0,0824 $ → **0,0473 $** pour 1 000
prédictions, soit **−42,6 %**.

Autrement dit, **activer le cache sur `mistral-small-4` (17 $/an) fait mieux que passer
à `ministral-3b` sans cache (19 $/an)** — alors que ce dernier est un modèle nettement
plus petit. C'est modélisé par le paramètre `cached_prefix_tokens` d'`estimate_cost()`.

*Simplification assumée* : le surcoût d'écriture du cache à la première requête est
négligé. Amorti sur des milliers d'appels, il pèse moins de 0,1 %.

### 4.5 Trois points opérationnels à ne pas manquer

**Tier gratuit Mistral — à écarter malgré son attrait.** Le tier gratuit
(~1 milliard de tokens/mois, ~1 requête/seconde) couvrirait **largement** toute
l'évaluation de l'étape 2. Mais les données soumises **peuvent servir à l'entraînement
sauf opt-out explicite**. Sur des réclamations clients réelles — contenant nom, adresse,
situation financière — c'est **inacceptable**. À signaler dans la recommandation : le
coût nul du tier gratuit se paie en exposition de données.

**API Batch — incompatible avec la mesure de latence.** Le mode Batch offre **−50 %**
chez OpenAI et Anthropic. Mais il traite les requêtes de façon asynchrone : la latence
observée n'a **rien à voir** avec celle d'un appel interactif. Si l'étape 2 l'utilise
pour réduire le coût, la **latence doit être mesurée séparément sur des appels
normaux** — sans quoi le p95 rapporté serait dénué de sens. Chez Anthropic, **cache et
batch se cumulent**.

**RGPD — un critère non statistique à faire figurer dans la recommandation.**
ZenAssist traite des données de clients européens. **Mistral est un fournisseur
européen**, ce qui simplifie la conformité (localisation des traitements, absence de
transfert hors UE à encadrer). Ce critère ne se mesure pas en F1-macro mais il peut
peser autant que lui dans la décision du client, et il doit apparaître explicitement
dans le tableau de recommandation finale — pas en note de bas de page.

### 4.6 Note de méthode — indépendance de la recommandation

Ce projet a été développé avec l'assistance de **Claude (Anthropic)**. Le catalogue de
modèles comparé inclut des modèles Anthropic.

**Si un modèle Anthropic figure dans la recommandation finale, celle-ci doit s'appuyer
exclusivement sur les mesures produites à l'étape 2 — F1-macro, latence p95, coût — et
jamais sur une préférence de fournisseur.** Les tarifs relevés ci-dessus placent
d'ailleurs les modèles Anthropic parmi les plus chers du catalogue (118 à 589 $/an
contre 17 $ pour le défaut Mistral), et le critère RGPD de 4.5 joue en faveur d'un
fournisseur européen. La démonstration devra être faite sur les chiffres.

**Pourquoi 257 est la valeur la plus susceptible d'être fausse** : le facteur 1,3
est une règle empirique pour un texte anglais courant et un tokenizer BPE
générique. Il n'a pas été vérifié sur le tokenizer de Mistral. Surtout, ce corpus
est atypique — **85 % des textes contiennent du masquage `XXXX`**, qui se
tokenise mal (une séquence répétée de X ne se découpe pas comme du langage
naturel). Le facteur réel pourrait être sensiblement supérieur à 1,3.

### 4.7 Garde-fous livrés

✅ **FAIT** — `metrics.estimate_cost(n, model=..., cached_prefix_tokens=...)` retourne
son statut avec le chiffre :

| Clé | Signification |
|---|---|
| `hypotheses` | `True` tant que les valeurs par défaut de `config.py` sont utilisées ; `False` dès que l'appelant fournit à la fois `avg_tokens` et `prompt_overhead_tokens` mesurés |
| `tarif_verifie_le` | date de consultation des grilles tarifaires |
| `model`, `fournisseur`, `input_per_1m`, `output_per_1m` | tarif effectivement appliqué |
| `cache_modelise` | `False` si la remise du fournisseur n'est pas vérifiée |
| `avertissements` | liste explicite (tarif promotionnel, cache non modélisé…) |

✅ **FAIT** — `metrics.compare_models_cost(n)` produit le tableau de la section 4.1,
avec et sans cache, trié du moins cher au plus cher. C'est le support de la restitution
sur le coût d'exploitation : **un tableau, pas un chiffre unique**, parce que la
conclusion à retenir est la fourchette et son ordre de grandeur, pas une valeur.

**Aucun coût ne peut donc être présenté sans que son statut soit explicite.**

🔜 **À FAIRE — étape 2, dès la première campagne d'appels**

Chaque réponse de l'API Mistral porte un champ `usage` avec les comptes de tokens
réels :

```python
reponse.usage.prompt_tokens        # tokens d'entrée effectivement facturés
reponse.usage.completion_tokens    # tokens de sortie
```

1. Agréger ces comptes sur les 2 000 appels de l'échantillon d'évaluation.
2. Comparer `prompt_tokens` moyen à l'hypothèse de **517** tokens d'entrée par
   appel (257 + 260 de surcoût de prompt). Consigner l'écart.
3. Relever au passage les tokens **effectivement servis depuis le cache**
   (`cache_read_input_tokens` ou équivalent) pour valider l'hypothèse de 260.
4. **Revérifier les dix tarifs** et mettre à jour `PRICING_CHECKED_ON`. Ils ont
   bougé deux fois en six semaines (4.3), et le tarif `claude-sonnet-5` change
   mécaniquement le 01/09/2026 (4.2).
5. Remplacer les constantes de tokens de `config.py` par les mesures et retirer le
   bandeau `HYPOTHESES PROVISOIRES`.
6. Recalculer toutes les projections, y compris celles du diagnostic (§ D.3), du
   README et de la section 9.3 du notebook.

**Ce qui ne changera pas** : la conclusion. Le coût n'est pas le facteur
discriminant — l'écart entre le modèle le moins cher et le plus cher du catalogue
couvre déjà deux ordres de grandeur (19 $ à 1 031 $ par an) sans qu'aucun ne soit
significatif devant le coût du traitement manuel remplacé. **Ce qui changera** : les
chiffres présentés au client, qui doivent être justes.

---

## Récapitulatif

| # | Limite | Statut | Corrigeable ? | Action |
|---|---|---|---|---|
| **1** | Corpus limité au canal web consenti — **29,9 % du flux historique, ~46 % du flux courant** | 📊 mesuré | **Non** — propriété du dataset | Chiffrer le gain sur le seul périmètre couvert |
| **2** | Plafond de performance — **borne dure 0,056 %**, valeur réelle inconnue | 📊 borne mesurée, 🔶 valeur non mesurable | **Non** — nécessiterait une double annotation | Analyse d'erreurs croisée LLM/ML à l'étape 3 |
| **3** | Un seul tirage d'échantillon (seed 42) | 🔶 non vérifié | **Oui** | Second tirage à l'étape 2 — outil livré, ~0,11 $ |
| **4** | Hypothèses de coût non mesurées ; tarifs volatils (**19 à 1 031 $/an** selon le modèle) | 🔶 supposé | **Oui** | Champ `usage` de l'API à l'étape 2 ; revérifier les tarifs — garde-fous livrés |

**Les deux limites non corrigeables sont les plus importantes**, et c'est
volontairement qu'elles ouvrent ce document : une limite quantifiée et assumée
est un signe de rigueur ; une limite passée sous silence est une faute que le
client découvrira à notre place, en production.

---

*Fin — phase 3bis. La phase 4 (notebook d'exploration) suit.*
