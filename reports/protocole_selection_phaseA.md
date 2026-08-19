# Protocole de sélection — étape 3, phase A (vectoriseur TF-IDF)

**Écrit le 2026-08-19.** Ce document transcrit des règles arrêtées **avant** le
lancement de la grille. Il n'introduit aucune règle nouvelle.

## Statut de préenregistrement — ce qui est vérifiable et ce qui ne l'est pas

Distinction à faire, parce qu'elle change ce que le document prouve.

| règle | fixée avant les résultats ? | preuve indépendante |
|---|---|---|
| seuil de pertinence 0,005 | oui | `SEUIL_PERTINENCE = 0.005` dans le script de la grille, écrit et lancé avant tout résultat |
| trois verdicts de lecture | oui | logique `lecture` du même script |
| seuil `\|t\| > 2,78` | oui | même script |
| conservation des scores par pli | oui | même script |
| règle d'arrêt avec latence | oui, en conversation | **aucune** — ce document est la seule trace |
| correction de variance (§3) | oui, en conversation | **aucune** — ce document est la seule trace |

Au moment de la rédaction, six des sept moyennes de la grille étaient visibles ;
**aucun écart apparié ni aucun `t` ne l'était**, ces quantités étant calculées en
fin de script. Les règles portant sur les écarts appariés sont donc écrites en
aveugle sur ce qu'elles vont trancher. Les règles portant sur les moyennes ne le
sont pas, et je ne les présente pas comme telles : leur préenregistrement repose
sur le script, daté et antérieur.

---

## 1. Ce que la phase A mesure — et ce qu'elle ne mesure pas

Les scores de la phase A sont des **F1-macro de validation croisée à 5 plis sur
le train** (283 449 lignes), **non pondérés**, servant **uniquement** au choix
des hyperparamètres du vectoriseur.

> **Ces scores ne sont comparables ni au 0,7885 du LLM, ni à aucun chiffre de
> l'étape 2.** Données différentes (train contre échantillon d'évaluation),
> pondération différente (aucune contre Horvitz-Thompson), rôle différent
> (sélection contre évaluation).

Cet avertissement est repris **sur chaque ligne** de sortie du script, et doit
l'être sur chaque tableau qui en dérive.

Le classifieur de la phase A est `LinearSVC(class_weight='balanced')`, tenu
**fixe**. La phase A choisit un vectoriseur, pas un modèle. Le choix du
classifieur est la phase B et se fait sur le vectoriseur retenu ici.

---

## 2. Seuil de pertinence : 0,005

**Règle.** Un écart de F1-macro CV inférieur à **0,005** en valeur absolue est
traité comme sans portée pour la sélection, quelle que soit sa régularité.

**Justification — ce sur quoi elle repose.** Le seuil est **arrimé à la
résolution de la mesure finale**, pas à un plancher de bruit de la validation
croisée.

La grandeur effectivement rapportée à la fin de l'étape 3 est un F1-macro
pondéré Horvitz-Thompson sur 2 000 lignes, dont l'intervalle de confiance
bootstrap à 95 % a une demi-largeur d'environ **±0,022** (mesuré à l'étape 2 :
mistral-small 0,7885 [0,7664 ; 0,8104]). Un écart de 0,005 est **plus de quatre
fois plus petit** que cette demi-largeur : deux configurations séparées de moins
de 0,005 produiront des résultats finaux que l'évaluation ne saura pas
distinguer. Arbitrer entre elles sur le F1 reviendrait à trancher au-dessous de
la résolution de l'instrument.

**Justification écartée, et pourquoi.** J'ai d'abord justifié un seuil de cet
ordre par le bruit de la validation croisée, en m'appuyant sur un écart-type
**inter-plis** de 0,0030. C'était une erreur de nature : les configurations
partagent exactement les mêmes plis, une part du bruit leur est donc commune et
s'annule dans la différence. La grandeur pertinente est l'écart-type des
**différences appariées**, mesuré entre 0,0002 et 0,0009 selon la configuration
— soit 4 à 18 fois plus petit. La résolution réelle de la validation croisée est
donc bien meilleure que 0,005 ; le seuil ne peut pas s'en réclamer. Il ne vaut
que par l'argument de la mesure finale ci-dessus. La valeur numérique est
inchangée, sa justification est remplacée.

**Conséquence assumée.** Le seuil est délibérément **plus grossier** que ce que
la validation croisée sait mesurer. C'est voulu : optimiser au-delà reviendrait
à sur-ajuster le train sur des différences que le rapport final ne pourra pas
montrer.

---

## 3. Statut du `t` : indicateur descriptif, jamais test de significativité

**Règle.** Le `t` apparié est rapporté comme **indicateur de régularité d'un
écart entre plis**. Aucune valeur *p* n'est produite, et aucun écart n'est
qualifié de « significatif ».

**Motif.** En validation croisée à k plis, les plis **ne sont pas
indépendants** : pour k = 5, deux ensembles d'entraînement partagent 3/4 de
leurs données. Les scores par pli sont positivement corrélés, leurs différences
aussi. L'écart-type naïf des k différences **sous-estime** la variance réelle,
le `t` est **gonflé**, et les valeurs *p* qui en dériveraient seraient
**anti-conservatrices**. C'est le résultat de Dietterich (1998), précisé par
Bengio et Grandvalet (2004) : **il n'existe pas d'estimateur sans biais de la
variance de la validation croisée à k plis.**

**Correction appliquée, et sa limite.** Faute d'estimateur exact, on applique le
facteur de Nadeau et Bengio (2003) — le *corrected resampled t-test* — qui
multiplie la variance par `(1/k + n_test/n_train) / (1/k)`. Pour k = 5,
`n_test/n_train = 0,25`, d'où un facteur **2,25** sur la variance, soit **1,5**
sur l'écart-type :

```
t_corrigé = t / 1,5        seuil de régularité : |t_corrigé| > 2,78  (ddl = 4)
```

Cette correction est dérivée pour le sous-échantillonnage aléatoire répété ; son
application au k-plis est une **approximation d'usage, pas un correctif exact**.
Les deux valeurs, `t` et `t_corrigé`, sont rapportées, et le verdict de
régularité s'appuie sur la seconde.

**Ce que le `t` ne dit pas.** Un test apparié répond à « cet écart est-il
constant d'un pli à l'autre ? », **pas** à « cet écart est-il grand ? ». Le
dénominateur est la variabilité de la différence, pas la différence. Deux
conséquences de lecture, à appliquer systématiquement :

- Un écart **non détecté** ne signifie **jamais** « effet plus petit ». Il peut
  signifier « effet de même taille, mais moins régulier ».
- Un écart **détecté** peut être sans portée. Cas mesuré : `max_df=0.5` donne
  `t = −6,45` pour un écart de 0,00057, soit six centièmes de point de F1.

**Illustration mesurée, et ce qu'elle établit.** Deux configurations d'effet
moyen identique à 0,00002 près, de détectabilité opposée :

| configuration | écart moyen | SD apparié | `t` | `t_corrigé` | lecture |
|---|---|---|---|---|---|
| `stop_words=['xxxx']` | −0,00059 | 0,00089 | −1,50 | −1,00 | non détecté |
| `max_df=0.5` | −0,00057 | 0,00020 | −6,45 | −4,30 | détecté, **sous** le seuil de sélection |
| témoin `ngram=(1,1)` | −0,03372 | 0,00390 | −19,32 | −12,88 | détecté **et** pertinent |

Ce qui les sépare est la **stabilité de l'intervention**, pas sa taille.
`max_df=0.5` retire 22 tokens fixes : le vocabulaire résultant est le même quel
que soit le pli, l'effet est un décalage quasi constant. `stop_words=['xxxx']`
remanie environ 9 000 bigrammes par pontage, et lesquels survivent au `min_df`
dépend des données de chaque pli — d'où un effet de même amplitude mais bien
plus variable.

**Règle générale qui en découle :** les configurations modifiant les **règles de
construction du vocabulaire** (`min_df`, `stop_words`, `ngram_range`) auront
structurellement un SD apparié plus élevé que celles appliquant un **filtre
fixe**. Leur non-détection doit être lue comme telle, et non comme la preuve
d'un effet moindre.

**Le témoin positif est obligatoire.** Toute campagne de tests appariés inclut
une configuration dont l'effet est connu pour être grand (ici `ngram=(1,1)`).
Sans lui, une série de non-détections ne se distingue pas d'un dispositif
incapable de détecter quoi que ce soit.

---

## 4. Règle d'arrêt

Appliquée **dans cet ordre**, sans retour en arrière :

1. **F* = meilleure moyenne CV** de la grille.
2. **Ensemble candidat** = toute configuration telle que `F* − F ≤ 0,005`.
   Cet ensemble est publié en entier, pas seulement la retenue.
3. **Latence d'abord.** Parmi les candidates, retenir celle dont la **latence de
   prédiction unitaire** est la plus faible, mesurée avec `metrics.Timer` sur un
   échantillon fixe, pipeline complet (`transform` + `predict`), après un
   préchauffage. La latence n'est mesurée **que sur l'ensemble candidat**.
4. **Égalité de latence** (moins de 10 % d'écart relatif) : départager par la
   **taille du vocabulaire**, la plus petite l'emportant — moindre empreinte
   mémoire, export pickle plus léger, vocabulaire plus lisible.
5. **Égalité résiduelle** : retenir la configuration ayant le **moins de
   paramètres non-défauts**, et à défaut la configuration de référence.

**Motif de l'ordre.** Le critère 2 est le seul critère de performance, et il est
délibérément peu discriminant (§2). Au-delà, ce qui départage relève du coût
d'exploitation : le ML est comparé à un LLM dont la latence médiane mesurée est
de 0,47 s par appel, et l'écart de latence est un résultat de l'étude, pas un
détail d'implémentation. Un gain de F1 de 0,003 payé par un vocabulaire trois
fois plus gros et une inférence plus lente n'est pas un gain.

**Ce que la règle interdit explicitement :** choisir la meilleure moyenne CV
quand elle appartient à un ensemble candidat de plusieurs configurations.
Prendre le maximum revient à sélectionner sur du bruit non résolu par la mesure
finale.

---

## 5. Hypothèse structurelle du seuil — non établie

Le seuil de 0,005 est justifié par la résolution de la **mesure finale**
(±0,022 sur 2 000 lignes pondérées). Il est **appliqué** à des écarts de
**validation croisée sur le train**. Le raisonnement suppose donc :

> **qu'un écart mesuré en validation croisée non pondérée sur le train se
> retrouve, de même signe et d'ordre de grandeur comparable, sur les 2 000
> lignes pondérées Horvitz-Thompson.**

**Cette hypothèse n'est pas établie.** Trois écarts entre les deux dispositifs,
par ordre de gravité croissante :

1. **Population.** Train (283 449) contre échantillon tiré du test (70 863). Le
   découpage est aléatoire et stratifié : c'est le maillon le plus solide, et il
   n'est pas la source principale de doute.
2. **Pondération.** La CV est non pondérée sur la distribution naturelle du
   train ; l'évaluation est pondérée Horvitz-Thompson sur un échantillon
   stratifié à plancher. En espérance les deux visent le F1-macro de population,
   mais les supports par classe diffèrent fortement, et une configuration qui
   déplace les erreurs vers les classes rares sera créditée différemment dans
   les deux dispositifs.
3. **Résolution, et non-monotonie.** C'est le point réel. La CV appariée résout
   des écarts de 0,0002 ; l'évaluation finale résout ±0,022, soit cent fois
   plus grossièrement. Rien ne garantit que l'**ordre** des configurations soit
   préservé d'un dispositif à l'autre. La sélection s'opère sur une grandeur
   **corrélée à** la grandeur rapportée, pas sur elle.

**Direction du risque.** Le seuil rend la sélection insensible à des différences
que l'évaluation finale ne saurait de toute façon pas trancher : de ce côté-là,
il est cohérent. Le risque inverse subsiste et n'est couvert par rien : deux
configurations séparées de moins de 0,005 en CV train pourraient différer
davantage sur les 2 000 lignes pondérées, notamment via les classes rares.
**Le protocole ne détecte pas ce cas.**

Une **seconde hypothèse structurelle**, distincte de celle-ci, porte sur la
stabilité de l'optimum du vectoriseur d'un classifieur à l'autre : voir §7.

**Engagement de conduite.** La configuration retenue en phase A **n'est pas
révisée après avoir vu les résultats sur les 2 000 lignes**. Re-sélectionner à
ce stade transformerait l'échantillon d'évaluation en jeu de développement et
invaliderait la comparaison avec le LLM, dont les 2 000 lignes n'ont servi
qu'une fois. Si les résultats finaux suggèrent qu'une autre configuration aurait
mieux fait, **le fait est rapporté comme une limite, et rien n'est refait.**

---

## 6. Ce que la phase A doit produire

- Le tableau **complet** de la grille — toutes les configurations, pas
  seulement les candidates — avec moyenne CV, scores **par pli**, taille du
  vocabulaire et durée.
- Pour chaque configuration : écart apparié contre la référence, SD apparié,
  `t`, `t_corrigé`, et le verdict parmi les trois de §3.
- L'ensemble candidat au sens de §4.2, et les latences mesurées dessus.
- L'avertissement de non-comparabilité de §1 sur chaque tableau.

**Les scores par pli sont conservés systématiquement.** Une première campagne ne
stockant que moyenne et écart-type a rendu l'écart-type apparié irrécupérable et
imposé un recalcul complet. La règle est désormais explicite.

---

## 7. Seconde hypothèse structurelle — stabilité de l'optimum du vectoriseur entre classifieurs

**Énoncé.** Le plan en deux phases — vectoriseur choisi avec `LinearSVC` tenu
fixe, puis trois classifieurs comparés sur ce vectoriseur — suppose que :

> l'argmax sur les configurations de vectoriseur est le même quel que soit le
> classifieur.

C'est une factorisation gloutonne d'une grille conjointe. Elle n'est exacte que
si l'objectif est séparable entre les deux blocs. Rien ne le garantit.

**Ce que la phase A dit là-dessus : rien.** Les SD appariés faibles et l'ordre
stable d'un pli à l'autre mesurent la stabilité **entre plis, à classifieur
fixé**. Aucune mesure de la phase A ne porte sur la variation **entre
classifieurs**. L'hypothèse n'est pas étayée par le travail déjà fait ; elle est
simplement non testée.

**Une partie de la décision est pourtant à l'abri.** La règle d'arrêt §4 se
décompose :

- **§4.2, l'ensemble candidat**, dépend du F1 et donc du classifieur —
  **exposé** ;
- **§4.3–4.5, latence puis vocabulaire**, ne dépendent pas du classifieur. La
  latence de prédiction unitaire est dominée par l'analyseur (mesuré : 0,45
  contre 0,47 ms au p95 pour des vocabulaires de 550 116 et 1 235 192), et la
  taille du vocabulaire est une propriété du vectoriseur seul — **non exposés**.

Le risque porte donc uniquement sur la composition de l'ensemble candidat.

### Où l'hypothèse est fragile, et par quel mécanisme

Le risque n'est pas réparti également entre les trois classifieurs.

`LogisticRegression` et `LinearSVC` sont tous deux linéaires, discriminatifs,
régularisés en L2, et ne diffèrent que par la fonction de perte (log contre
charnière). Face à des variables corrélées, ils se comportent de la même
manière : le poids se répartit entre elles. Un optimum de vectoriseur trouvé
avec l'un a peu de raisons de se déplacer avec l'autre.

`MultinomialNB` diffère sur trois points, et chacun interagit avec un réglage du
vectoriseur.

1. **Redondance des variables.** NB suppose l'indépendance conditionnelle. Un
   bigramme est presque déterminé par ses unigrammes : NB compte donc la même
   évidence plusieurs fois. Plus le vocabulaire est grand et redondant, plus il
   est pénalisé — alors qu'un modèle discriminatif y est essentiellement
   insensible. **Le vectoriseur retenu se situe précisément à l'extrémité la
   plus défavorable de cet axe** : `min_df=2`, 1 235 192 variables, en majorité
   des bigrammes rares.
2. **Nature des valeurs.** Le modèle multinomial suppose des comptes.
   `sublinear_tf=True` et la normalisation L2 produisent des valeurs
   fractionnaires : la vraisemblance est mal spécifiée. Le TF-IDF reste un
   correctif connu et efficace pour NB (Rennie et al., 2003), mais le degré de
   mauvaise spécification dépend du réglage — et `sublinear_tf` est justement un
   axe où `LinearSVC` a tranché nettement (+0,0073 pour `True`).
3. **Rééquilibrage des classes.** Chez les modèles linéaires,
   `class_weight='balanced'` déforme une marge. Chez NB, les poids agissent sur
   les comptes et modifient les probabilités conditionnelles estimées
   elles-mêmes. Mécanisme différent, interaction différente avec le vocabulaire.

**Prédiction directionnelle qui en découle :** si l'optimum se déplace, il se
déplace **pour NB**, et **vers un vocabulaire plus petit** — `min_df` plus
élevé, voire unigrammes seuls.

### Ce que cela risque de coûter

Trois portées distinctes, à ne pas confondre.

| ce qui est en jeu | exposition |
|---|---|
| **comparaison principale ML contre LLM** | **non exposée** — le meilleur ML sera selon toute vraisemblance un modèle discriminatif, sélectionné avec un vectoriseur choisi lui-même sur un modèle discriminatif ; la chaîne est cohérente |
| **troisième ancrage de l'analyse croisée** | **exposé, et c'est le point sérieux** |
| **énoncé « discriminatif contre génératif »** | **exposé** |

Sur le troisième ancrage : le protocole d'analyse croisée fait du recouvrement
`LogisticRegression` / `MultinomialNB` une mesure du « mécanisme partagé » entre
deux familles. Si NB tourne avec un vectoriseur qui lui convient mal, ses
erreurs reflètent en partie une **inadéquation de représentation** et non une
différence d'architecture — c'est-à-dire exactement la grandeur que l'ancrage
prétend isoler. Un ancrage contaminé fausse la lecture du critère `R`.

Sur le troisième point : sans vérification, toute phrase opposant les deux
familles confondrait famille de classifieur et adéquation du vectoriseur.

**Amplitude possible, et pourquoi je ne la borne pas.** L'étendue mesurée en
phase A entre configurations raisonnables (témoin unigramme exclu) est de
**0,0131**. Ce chiffre donne une échelle, **mais ne majore pas** ce que NB
risque : le mécanisme n° 1 prédit précisément une sensibilité *plus grande* chez
NB. Je n'ai pas de borne supérieure à proposer et je n'en invente pas.

### Ce que je propose — une sonde ciblée, pas une reprise de grille

Pour `MultinomialNB` seul, quatre points sur les **mêmes 5 plis**, chacun choisi
pour un mécanisme énoncé ci-dessus :

| point | mécanisme testé |
|---|---|
| vectoriseur retenu — (1,2) `min_df=2` | référence |
| (1,2) `min_df=20` | redondance — moins de bigrammes rares |
| (1,1) `min_df=5` | redondance — bigrammes supprimés |
| (1,2) `min_df=2`, `sublinear_tf=False` | mauvaise spécification du modèle de comptes |

Coût : quatre ajustements de NB, dont trois vectorisations supplémentaires. NB
s'ajuste par comptage, sans optimisation itérative ; le coût est celui de la
vectorisation. **Aucune sonde pour `LogisticRegression`** : l'argument de
mécanisme ci-dessus ne désigne chez lui aucune fragilité, et le budget va là où
il désigne quelque chose.

### Règle de décision, fixée d'avance

1. Si le meilleur point de la sonde **est** le vectoriseur retenu, ou s'en écarte
   de **moins de 0,005** : l'hypothèse tient sur l'axe testé. NB reste sur le
   vectoriseur retenu ; le résultat de la sonde est rapporté.
2. Si un autre point le dépasse de **0,005 ou plus** : l'hypothèse ne tient pas
   entre familles, et c'est un **résultat à rapporter comme tel**, pas un
   incident à corriger discrètement. NB est alors rapporté avec **son propre
   vectoriseur**, explicitement étiqueté, et **les deux chiffres sont donnés** —
   NB sur le vectoriseur retenu et NB sur le sien. Jamais le meilleur seul.
3. Dans le cas 2, le vectoriseur propre à NB sert **au seul rôle d'ancrage
   intra-famille ML** de l'analyse croisée, et ce fait est rappelé à l'endroit
   de l'ancrage.
4. **Dans tous les cas, le vectoriseur de la comparaison principale contre le
   LLM reste celui de la phase A.** Le modifier après avoir vu la phase B
   rouvrirait une sélection close, ce que l'engagement de §5 interdit dans son
   esprit.

**Limites de la sonde, énoncées d'avance.** Elle teste un axe et demi —
redondance du vocabulaire, et la nature des valeurs par un seul point — sur les
six réglages de la grille. Elle ne dit rien de `max_features`, des `min_df`
intermédiaires, ni des trigrammes pour NB. Un résultat de cas 1 signifie
**« pas de déplacement détecté sur les axes testés »**, et non « l'hypothèse est
vérifiée ».

---

## 8. Règle de sélection de la phase B (classifieurs)

Trois classifieurs sur le vectoriseur retenu en phase A. Tous rééquilibrés :
`class_weight='balanced'` pour `LogisticRegression` et `LinearSVC`,
`sample_weight` issu de `compute_sample_weight('balanced', y)` pour
`MultinomialNB` — c'est l'équivalent exploitable chez NB, `alpha` (lissage de
Laplace) et `fit_prior` (a priori de classe seulement) répondant à d'autres
questions.

**Vectorisation ajustée par pli**, sur la partie entraînement seulement, et
réutilisée par les trois classifieurs du même pli : aucune fuite, et cinq
vectorisations au lieu de quinze.

Même appareil qu'en phase A : scores **par pli** conservés, écarts appariés, SD
apparié, `t` et `t_corrigé`, seuil de pertinence 0,005, et **même règle d'arrêt
§4** — ensemble candidat au meilleur, puis latence, puis simplicité.

Deux différences de portée, à ne pas confondre :

- `LogisticRegression` et `MultinomialNB` sont **conservés quoi qu'il arrive** ;
  la sélection ne les élimine pas. Le protocole d'analyse croisée en fait les
  deux familles du troisième ancrage. La règle §4 désigne le modèle de la
  **comparaison principale contre le LLM**, pas la liste des modèles entraînés.
- Le seuil de pertinence garde exactement le statut et la réserve de §5 : il est
  arrimé à la résolution de la mesure finale, et le transfert du train vers les
  2 000 lignes pondérées n'est pas établi.

**Sorties exigées :** latence de prédiction unitaire par classifieur
(`metrics.Timer`, mêmes conditions qu'en phase A), export pickle du modèle
retenu, journal du vocabulaire avec les tokens de plus fort poids par classe.

---

## 9. Sensibilité de MultinomialNB à `alpha` — caractérisation, pas réglage

**Écrit avant la mesure.**

### Condition ferme

> **La valeur retenue pour `MultinomialNB` reste `alpha=1` (le défaut de
> scikit-learn), quel que soit le résultat de cette mesure.**

Motif. `LogisticRegression` et `LinearSVC` tournent avec leurs valeurs par
défaut, `C=1` en particulier, sans qu'aucune recherche n'ait été faite sur elles.
Optimiser `alpha` sur la validation croisée du train pour le seul NB rendrait la
comparaison inéquitable **dans l'autre sens** : NB arriverait réglé face à deux
modèles qui ne le sont pas. La mesure documente une contingence ; elle ne choisit
pas un point.

### Pourquoi la faire quand même — trois raisons

1. **Le chiffre de NB dépend de `alpha` d'une façon dont ceux des deux modèles
   linéaires ne dépendent pas.** Les rapporter côte à côte sans le dire les
   présenterait comme comparables au même titre. Le mécanisme est établi : avec
   des poids constants par classe, `w_c` se simplifie dans la vraisemblance des
   variables que la classe utilise, et **ne se simplifie pas** sur le plancher
   lissé des variables qu'elle n'utilise jamais, où le décalage vaut exactement
   `−log(w_c)` — prédit à 1,9255, mesuré à 1,926. Le rééquilibrage de NB transite
   donc à 85 % par le lissage (0,3276 de gain de F1 sur 0,3854 au total, le reste
   venant de l'a priori). `alpha` n'est pas un paramètre de nuisance chez NB :
   c'est le canal du rééquilibrage.
2. **Le §7 signalait une inadéquation de représentation ; il y en a maintenant
   deux.** La mesure dira si l'ancrage intra-famille ML reste utilisable ou doit
   être déclaré contaminé.
3. **`alpha=1` sur une masse documentaire médiane de 12,36 est un défaut par
   valeur par défaut, pas un choix.** Le lissage effectif varie d'un facteur 18,6
   entre classes (`alpha/w_c`, de 0,146 à 2,715). Le dire sans le chiffrer serait
   une affirmation de plus.

### Ce qui est mesuré

`MultinomialNB` sur le vectoriseur retenu, **mêmes 5 plis**, `sample_weight`
équilibré inchangé, pour `alpha ∈ {0,001 ; 0,01 ; 0,1 ; 1 ; 10}`. La
vectorisation est faite une fois par pli et réutilisée par les cinq ajustements :
le coût est celui de cinq comptages.

### Règles de lecture, fixées d'avance

Soit `Δ = F1(meilleur alpha) − F1(alpha=1)`, en moyenne sur les 5 plis.

| condition | lecture |
|---|---|
| `Δ < 0,005` | Contingence **faible** — sous le seuil de pertinence de §2. L'ancrage intra-famille ML reste utilisable ; la réserve est rapportée, sans plus. |
| `0,005 ≤ Δ < 0,022` | Contingence **notable**. L'ancrage reste utilisable, mais tout énoncé opposant les deux familles doit porter la mention que NB tourne à un `alpha` par défaut qui lui coûte `Δ`. |
| `Δ ≥ 0,022` | L'écart atteint la demi-largeur de l'IC de la mesure finale. NB n'est plus interprétable comme « la famille générative » : ce serait confondre la famille et un défaut de lissage. **L'ancrage intra-famille ML est déclaré CONTAMINÉ**, le critère `R` est lu sans lui, avec le seul ancrage LLM/LLM et la limite déjà énoncée au §3 du protocole d'analyse croisée. |

### Ce que la mesure ne dira pas

Elle teste `alpha` **à vectoriseur fixé**. Elle ne dit rien de l'interaction entre
`alpha` et le vectoriseur, ni de `fit_prior`. Surtout, le mécanisme identifié
implique que `alpha` et le rééquilibrage **ne sont pas séparables** : faire varier
`alpha` fait varier la force du rééquilibrage. C'est précisément ce qui interdit
de se servir du résultat pour choisir `alpha` — on ne choisirait pas un lissage,
on choisirait une repondération par un canal détourné.
