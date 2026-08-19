# Étape 3 — Approche ML classique · rapport de synthèse

**Question posée au projet :** une chaîne TF-IDF + classifieur linéaire tient-elle
la comparaison face à un LLM pour router 9 catégories de réclamations, et que
disent les deux approches ensemble sur la difficulté de la tâche ?

**Réponse en une ligne :** `LinearSVC` fait **+3,03 points de F1-macro** sur le
LLM (McNemar `p = 2,6 × 10⁻⁸`), pour une latence **2 000 fois plus faible** et un
coût d'API nul — et l'analyse croisée **infirme** le plafond de performance
annoncé au §C.4 du diagnostic.

Détail des mesures : `resultats_phaseA.md`, `resultats_phaseB.md`,
`resultats_etape3_phase2.md`. Règles : `protocole_selection_phaseA.md`,
`protocole_analyse_croisee.md`. Chiffres lisibles par machine :
`etape3_scores.json`, `etape3_evaluation.json`.

---

## 1. Les quatre approches sur les 2 000 réclamations pondérées

**Seule comparaison valide.** Mêmes 2 000 lignes que l'étape 2, poids de
Horvitz-Thompson, somme 70 863, `metrics.evaluate()` inchangé.

| approche | **F1-macro** | IC 95 % | accuracy | latence p50 | coût / 1000 préd. |
|---|---|---|---|---|---|
| **`LinearSVC`** | **0,8187** | [0,798 ; 0,839] | 0,8668 | **0,23 ms** | **0 $** |
| `LogisticRegression` | 0,8136 | [0,792 ; 0,835] | 0,8546 | 0,23 ms | 0 $ |
| `mistral-small` (LLM) | 0,7885 | [0,766 ; 0,810] | 0,8149 | 470 ms | 0,0516 $ |
| `MultinomialNB` | 0,7233 | [0,704 ; 0,744] | 0,7867 | 10,41 ms | 0 $ |

Le ML n'a aucun coût par prédiction : sa dépense est l'entraînement, faite une
fois (72 s), et le stockage du modèle (118 Mo).

### Là où le LLM reste devant

| classe | `LinearSVC` | `mistral-small` | écart |
|---|---|---|---|
| **Vehicle loan or lease** | 0,6190 | **0,6854** | **−0,066** |
| Payday, title or personal loan | 0,6980 | 0,6966 | +0,001 |
| Mortgage | 0,9338 | 0,9342 | −0,000 |
| Money transfer or virtual currency | **0,8254** | 0,7315 | +0,094 |
| Debt collection | **0,8533** | 0,7708 | +0,083 |
| Credit reporting | **0,8881** | 0,8175 | +0,071 |

**`Vehicle loan or lease` est la seule classe où le LLM devance nettement le
ML** — et c'est la classe la plus rare du corpus (0,34 % du train). C'est le
profil que la simulation h1 avait supposé : le ML exploite le volume, le LLM
tient mieux les classes rares. Le profil est vérifié, mais sur **une seule
classe** : le ML est devant sur **7 classes sur 9**, dont `Payday` (3,9 %) et
`Money transfer` (4,0 %), qui sont aussi rares. `Mortgage` est un ex æquo —
0,9338 contre 0,9342, soit 0,0004 en faveur du LLM, très en deçà de ce que
2 000 lignes résolvent.

Matrice de confusion : `figures/etape3_confusion_linearsvc.png`, 9 classes,
pondérée, **sans colonne `PARSE_ERROR`** — le ML rend toujours une classe du
référentiel. C'est une différence de nature, pas de degré : les 19
non-conformités du LLM n'ont pas d'équivalent côté ML.

## 2. McNemar apparié et l'écart de F1

| | |
|---|---|
| paires | 2 000 |
| `LinearSVC` juste / LLM faux — **b** | **196** |
| `LinearSVC` faux / LLM juste — **c** | **100** |
| paires discordantes | 296 |
| **p bilatérale exacte** | **2,56 × 10⁻⁸** |
| écart de F1-macro pondéré | **+0,0303** |
| IC 95 % **apparié** | **[+0,0071 ; +0,0525]** |

L'IC est apparié — même tirage bootstrap pour les deux modèles, plan stratifié
intra-classe. Deux IC calculés séparément traiteraient les modèles comme
indépendants alors qu'ils portent sur les mêmes réclamations.

## 3. Test complet — tableau SÉPARÉ

> **Ce chiffre ne se compare à aucun chiffre de l'étape 2.**

`LinearSVC` sur les 70 863 lignes du test, non pondéré : **F1-macro 0,8193**,
accuracy 0,8747. Il montre ce que le volume apporte, rien d'autre.

## 4. Le vectoriseur — phase A

`TfidfVectorizer(lowercase=True, sublinear_tf=True, ngram_range=(1,2), min_df=2)`,
1 235 192 variables.

La règle d'arrêt a été appliquée **là où elle coûtait quelque chose, deux fois** :
elle a écarté la meilleure moyenne CV (`ngram=(1,3)`, 0,81816) au profit d'une
configuration 1,8 fois plus rapide, et elle a exclu la configuration de référence
de l'ensemble candidat pour **0,00086** — moins d'un millième. Aucune des deux
coupures n'a été ajustée après coup.

Trois décisions documentées et non rejouées : **aucun `max_df`** (le filet
annoncé n'exclut qu'un token sur 550 116), **`xxxx` conservé** (le retrait change
le F1 de moins de 0,0011, estimation ponctuelle −0,00059), **aucun `stop_words`**
(`'english'` *augmente* le vocabulaire de 30 863 entrées par pontage de
bigrammes).

**Critique de la règle, à conserver.** L'ordre lexicographique place la latence
avant le vocabulaire, en supposant qu'elle est le critère le plus discriminant
après le F1. La mesure dit l'inverse : la latence de prédiction unitaire est
dominée par l'analyseur et non par la taille du dictionnaire — 0,45 contre
0,47 ms au p95 pour des vocabulaires allant du simple au double et quart. Entre
deux configurations de même `ngram_range`, le critère de latence est **quasi
inerte**. La règle n'a mordu ici que parce que l'ensemble candidat contenait deux
`ngram_range` différents : un fait de la grille, pas une propriété de la règle.

## 5. Le classifieur — phase B, et deux hypothèses tombées

| classifieur | F1 CV train | écart apparié au meilleur | `t_corrigé` |
|---|---|---|---|
| **`LinearSVC`** | **0,81448** | — | — |
| `LogisticRegression` | 0,80275 | −0,01174 | −12,01 |
| `MultinomialNB` | 0,71800 | −0,09649 | −46,48 |

Ensemble candidat réduit à `LinearSVC` : la latence n'a pas eu à départager.

**Contrôle de cohérence.** La moyenne de `LinearSVC` est identique **à 0,0 près,
pli par pli**, à celle de `min_df=2` en phase A, obtenue par un chemin de code
indépendant (`Pipeline` + scorer sklearn contre boucle explicite + `mt.f1_macro`).
Les 9 classes sont présentes dans chaque pli de validation, enregistré et non
supposé.

### Hypothèse 1 tombée — l'optimum du vectoriseur n'est pas stable entre familles

La sonde du §7, écrite avant mesure, donne **CAS 2** : `MultinomialNB` gagne
**+0,01036** avec `min_df=20` plutôt qu'avec le vectoriseur retenu. Le plan en
deux phases supposait la séparabilité ; elle est fausse.

La prédiction directionnelle du §7 est vérifiée — le déplacement est pour NB et
vers un vocabulaire plus petit — mais le détail corrige une lecture trop simple.
`(1,1) min_df=5` descend à 27 480 variables et ne gagne que 0,002, quand
`min_df=20` en garde 194 220 et gagne cinq fois plus. **Ce qui pénalise NB, ce
sont les bigrammes rares, pas les bigrammes.**

### Hypothèse 2 tombée — le rééquilibrage de NB est un artefact de lissage

`sample_weight='balanced'` n'agit pas comme une repondération de classes. Avec
des poids constants par classe, `w_c` se simplifie dans la vraisemblance des
variables que la classe **utilise**, et ne se simplifie pas sur le plancher lissé
des variables qu'elle **n'utilise jamais**, où le décalage vaut exactement
`−log(w_c)` — prédit 1,9255, mesuré 1,926. Décomposition du gain de F1 :
**15 % pour l'a priori, 85 % pour le décalage du plancher**. `alpha` n'est pas un
paramètre de nuisance chez NB : c'est le canal du rééquilibrage.

Caractérisation à `alpha` variable, **`alpha=1` conservé** conformément à la
condition écrite d'avance : le défaut coûte **0,0416** de F1 par rapport à
`alpha=0,1`, soit près du double de la demi-largeur de l'IC final. Conséquence
appliquée : **ancrage intra-famille ML déclaré contaminé.**

### Journal du vocabulaire

`vocabulaire_top_tokens.md`, 25 termes par classe. Sur `Credit reporting`, NB
place dans ses dix premiers `equifax`, `equifax and`, `to equifax`,
`with equifax`, `equifax is`, `xxxx equifax`, `equifax has` — **sept termes pour
une seule évidence**, comptée sept fois. `LinearSVC` sur la même classe donne dix
termes distincts. Le mécanisme du §7 y est visible sans mesure.

**Réserve de transférabilité :** `xxxx` et ses composés reviennent dans plusieurs
classements. Le modèle apprend en partie la chaîne d'anonymisation du CFPB, pas
seulement le domaine.

## 6. Analyse croisée — le plafond du §C.4 est INFIRMÉ

| clause | seuil pré-fixé | mesuré | déclenchée |
|---|---|---|---|
| §4 point 1 — `R < 0,40` | 0,40 | `R` = 1,095 | non |
| §4 point 2 — communes pas plus concentrées | 10 pts | **−15,0 pts** | **oui** |
| §4 point 3 — rattrapage ≥ 30 % | 30 % | **51,0 %** | **oui** |

Une seule clause suffit ; deux se déclenchent. Le critère principal ne l'emporte
pas — c'est ce qui fait des clauses d'infirmation des clauses d'infirmation.

**Comment `R = 1,095` et les deux infirmations coexistent.** Elles ne mesurent pas
la même chose. `R` dit qu'un **socle d'items partagé existe** : 182 co-échecs
contre 52,1 attendus, autant de recouvrement qu'entre deux modèles de la même
famille. Les deux clauses disent **où** ce socle se trouve et **s'il est
irréductible** : il ne tombe pas sur les paires annoncées (les erreurs communes y
sont *moins* concentrées que les erreurs propres), et plus de la moitié des
erreurs du LLM sur les trois classes réputées les plus dures sont corrigées par
le ML. Ensemble : **un socle de difficulté partagé existe, mais le §C.4 l'a mal
localisé et a surestimé son étendue.** Ce qui est infirmé n'est pas « certains
items sont durs » — énoncé trivial — mais « le plafond se situe sur ces paires-là
et ces trois classes-là ».

L'ancrage intra-famille ML (`LogisticRegression` / `MultinomialNB`, lift 3,885)
est rapporté **contaminé et non décisionnel**. Le second ancrage n'a pas été
rétabli : ajouter un modèle après avoir vu que le premier ne convenait pas
rendrait indistinguables « prévu » et « retenu parce qu'il donnait le résultat
attendu ».

## 7. Ce qu'est le socle partagé

Mesures **non prévues au protocole**, ajoutées parce que les données les
permettaient.

**Un noyau dur séparé, pas un continuum.** Sur les cinq approches évaluées,
**131 réclamations sont manquées par toutes** — attendu sous indépendance 0,38,
soit un rapport de 345. Plus d'items échouent aux cinq (131) qu'à exactement
quatre (70) ou trois (111).

**Les co-échecs sont les textes les plus courts** : médiane 116 mots contre 149
pour les items correctement classés, premier quartile 57 contre 82. Les erreurs
propres au ML sont, elles, les plus longues.

**Ils pèsent 8,49 % de la population** pour 9,19 % des lignes : la pondération ne
les amplifie pas.

**`Mortgage` en est presque absente** (0,7 % de la classe), `Payday` trente fois
plus exposée (22,1 %).

### Vingt lectures manuelles — et ce que le 20/0 ne peut pas établir

Vingt co-échecs tirés à la graine `RANDOM_SEED`, classés selon la grille binaire
de la phase 2 : **20 en (A) ambiguïté réelle, 0 en (B) défaut de modèle.**

**Ce score ne vaut pas ce qu'il paraît valoir, et il faut le dire avant de le
lire.** Un résultat parfait sur une grille binaire, appliquée par la personne qui
connaît l'hypothèse, se lit spontanément comme un chiffre fort. Il ne l'est pas,
pour quatre raisons dont la deuxième suffirait.

1. **Ce n'est pas un taux.** Vingt tirés sur 182. Même en tenant la
   classification pour acquise, l'intervalle de Clopper-Pearson à 95 % sur 20/20
   va de **83,2 % à 100 %** — dix-sept points de large.
2. **Le critère A est quasi tautologique sur cet ensemble.** La grille définit A
   comme « le texte décrit un préjudice pointant vers une autre classe ». Or les
   182 items ont été retenus **parce que** deux approches indépendantes ont
   prédit une autre classe. Demander si le texte pointe ailleurs, sur un
   ensemble sélectionné pour le fait qu'il pointe ailleurs, ne peut guère rendre
   autre chose. **Le 20/0 mesure la sélection, pas les items.**
3. **Un seul lecteur, non aveugle, sans arbitrage.** Aucun second annotateur,
   aucun accord inter-juges, et la connaissance de l'hypothèse au moment de
   classer. Un annotateur seul qui croit à une catégorie produit un score qui
   mesure sa croyance autant que les textes.
4. **La grille est à choix forcé.** Elle n'a pas de case « ni l'un ni l'autre ».
   Un score déséquilibré est ce qu'elle produit mécaniquement quand la structure
   réelle comporte un troisième mode — et c'est le cas, voir plus bas.

**Ce qui, dans cette section, repose sur des mesures et non sur la lecture.**
Tout le reste : les 131 échecs sur cinq approches et leur rapport de 345, les
quartiles de longueur, le poids de population, la composition par classe et les
taux intra-classe, les 144 co-échecs orientés et le rang 2 de
`Credit card ↔ Credit reporting`. **Aucun de ces chiffres ne dépend de ce que
quiconque a lu.** Ils tiennent si les vingt lectures sont retirées.

**Ce que seule la lecture soutient.** Le mécanisme ci-dessous. C'est une
hypothèse produite par la lecture, non une mesure — testable, mais non testée
ici. Elle le serait par un indicateur lexical : fréquence du vocabulaire de
recouvrement (`validation letter`, `charge off`, `collection agency`) dans les
items dont l'étiquette vraie est une classe de produit, comparée entre co-échecs
et items bien classés. Ce test n'a pas été fait.

**Le mécanisme, énoncé comme hypothèse.** L'étiquette du CFPB suit le
**PRODUIT**, le texte décrit la **CONDUITE**. Un consommateur dont la carte passe
en recouvrement écrit une histoire de recouvrement ; l'étiquette reste
`Credit card`, et les deux approches suivent la conduite. Second mécanisme :
l'anonymisation efface l'identité du produit (« synchrony bank who finances
XXXX »).

### Les trois cas hors grille — plus informatifs que le score

Trois des vingt (04, 16, 18 — 39, 28 et 25 mots, aucun produit nommé) ne sont
décrits ni par A ni par B. Ils ne relèvent pas de « deux lectures défendables »
mais de **« le texte ne détermine aucune réponse »**.

**Pourquoi la grille binaire ne les couvrait pas.** Ses deux branches
présupposent toutes deux que le texte contient de quoi trancher : A dit qu'il
supporte deux réponses, B dit qu'il en supporte une et que le modèle l'a manquée.
Aucune ne couvre « il n'en supporte aucune ». Classés en A, ces cas gonflent
l'ambiguïté ; classés en B, ils imputeraient au modèle une information qu'il n'a
jamais eue.

**Et pourquoi l'angle mort s'ouvre ici précisément.** La grille a été construite
en phase 2 sur des erreurs du LLM tirées de l'ensemble de l'échantillon, où les
textes très courts et sans produit nommé sont une minorité. Elle est appliquée
ici aux co-échecs, qui sont **le groupe le plus court** — premier quartile à 57
mots contre 82 pour les items bien classés. **Une grille conçue sur une
population où son angle mort était rare a été appliquée à une population
sélectionnée pour cet angle mort.** Trois cas sur vingt en est la trace ; le
chiffre exact importe moins que le fait que la catégorie manque.

`Credit card or prepaid card ↔ Credit reporting` occupe le **rang 2** des
directions partagées et n'est pas au §C.4. C'est la **même paire** qui avait fait
échouer le critère C de l'étape 2. Mesure, non lecture.

## 8. Latence et coût

| | `LinearSVC` | `mistral-small` | rapport |
|---|---|---|---|
| p50 | 0,23 ms | 470 ms | **≈ 2 030** |
| p95 | 0,50 ms | 740 ms | ≈ 1 480 |
| coût / 1000 prédictions | 0 $ | 0,0516 $ | — |

La latence du LLM inclut l'aller-retour réseau. Le ML n'a pas de coût par
prédiction ; son coût est un entraînement de 72 s et 118 Mo de stockage.

**`MultinomialNB` mesuré à 10,41 ms, et ce n'est pas l'algorithme.** Le rapport
entre ses deux vectoriseurs (6,37) reproduit celui des vocabulaires (6,36) : le
coût est linéaire en la taille du dictionnaire et indépendant de la longueur du
document, ce qui n'a aucun sens pour une somme sur les termes présents. Cause
localisée : `feature_log_prob_` est en ordre C, donc sa transposée ne l'est pas,
et scipy recopie 89 Mo à **chaque prédiction** ; `coef_` sort de liblinear en
ordre Fortran et échappe à la recopie. **Chiffre daté : scikit-learn 1.6.1, le
2026-08-19.** Une version corrigeant la disposition mémoire le changerait sans
que l'algorithme change.

## 9. Confrontation aux prévisions

| prévision | source | résultat |
|---|---|---|
| MDE de l'écart apparié : 3,1 points | h1 §7, B3 corrélé | écart observé **3,03 points** — 1,7 % **sous** le seuil prévu |
| RMSE de l'écart : 0,0157 | h1 §7 | réalisé **0,01157**, soit **26,3 % plus petit** |
| facteur d'annulation de l'appariement : 0,89 | h1 §7 | réalisé **0,754** — l'appariement aide plus que prévu |
| « la comparaison pourrait ne pas trancher sur le F1 global » | h1 §7 | **confirmé** : elle tranche de justesse |
| plafond du §C.4 | diagnostic | **infirmé**, 2 clauses sur 3 |
| stabilité du vectoriseur entre familles | protocole §7 | **infirmée**, CAS 2 |
| optimum de `alpha` proche du défaut | protocole §9 | **infirmé**, Δ = 0,0416 |

**Le point le plus instructif.** L'écart observé tombe à 1,7 % sous le seuil de
détection prévu. Si la précision réalisée avait été celle de la simulation, l'IC
aurait contenu zéro. La conclusion tient à ce que la précision s'est trouvée
meilleure — de 26 % — et **la raison est le même chiffre que l'analyse croisée** :
les erreurs des deux approches sont plus corrélées que la simulation ne le
supposait (`lift` 3,49), donc l'appariement annule davantage, donc le seuil de
détection était surestimé.

## 10. Limites, énoncées

**La grille du §4 ne distingue rien au-dessus de `R = 1`.** Sa construction
contenait une supposition non formulée : que l'ancrage intra-famille est un
**majorant** du recouvrement. `R` a été écrit comme une position sous un plafond,
et `R > 1` n'était pas traité comme improbable mais comme impossible. Deux
raisons le rendent faux : le recouvrement a deux sources (mécanisme partagé *et*
difficulté des items), et le `lift` a un plafond arithmétique propre à chaque
paire — `lift ≤ 1/max(pA, pB)`. Rapporté à son propre plafond, l'ancrage
intra-famille est à **76,4 %** et le croisé LLM/ML à **64,3 %** : l'ordre inverse
de celui que `R = 1,095` affiche. Ce constat porte sur le critère qui **n'a pas**
décidé, et ne rouvre pas celui qui a décidé.

**La sélection s'est faite sur une grandeur corrélée à celle qui est rapportée.**
Les écarts de validation croisée sur le train se résolvent à 0,0002 ; la mesure
finale se résout à ±0,022. Rien ne garantit que l'ordre des configurations soit
préservé. La configuration retenue **n'a pas été révisée** après avoir vu les
2 000 lignes.

**Un seul ancrage haut.** La vérification que deux mesures du « mécanisme
partagé » se ressemblent n'aura pas lieu ; `lift = 3,19` mélange difficulté des
données et similarité des deux LLM, et cette limite doit accompagner chaque
citation de `R`.

**Transférabilité.** Le modèle apprend en partie la chaîne d'anonymisation du
CFPB. Sur des réclamations anonymisées autrement, les variables contenant `xxxx`
ne se retrouveraient pas.

## 11. Leçons de méthode

**Un écart-type inter-plis ne juge pas une différence entre configurations qui
partagent les plis.** Une part du bruit leur est commune et s'annule dans la
différence ; l'écart-type apparié est 4 à 18 fois plus petit. Une première
campagne ne stockant que moyenne et écart-type a rendu la quantité irrécupérable
et imposé un recalcul. Les scores par pli sont désormais conservés
systématiquement.

**Un test apparié répond à « cet écart est-il constant ? », pas « est-il
grand ? ».** Deux configurations d'effet moyen identique à 0,00002 près ont eu
des détectabilités opposées, parce que l'une modifiait un filtre fixe et l'autre
les règles de construction du vocabulaire. Une non-détection ne signifie jamais
« effet plus petit ».

**Les plis d'une validation croisée ne sont pas indépendants.** Pour `k = 5`, deux
ensembles d'entraînement partagent 3/4 de leurs données. Aucun estimateur sans
biais de la variance n'existe. Tous les `t` sont rapportés avec la correction de
Nadeau-Bengio et **aucune décision ne repose sur eux** : les seuils portent sur
des tailles d'effet, qui ne demandent pas d'estimation de variance.

**Un rapport qui reproduit un autre rapport est un signal.** La latence de NB
entre deux vectoriseurs (6,37) reproduisait celui des vocabulaires (6,36) — un
coût indépendant de la longueur du document n'a aucun sens pour une somme sur les
termes présents. Suivre ce signal a mené à la disposition mémoire ; publier
« NB est 20 fois plus lent » sans lui aurait attribué à l'algorithme ce qui
appartient à un `np.ascontiguousarray` manquant.

**Un échec bruyant vaut mieux qu'une jointure silencieuse.** Le journal de
campagne stocke `complaint_id` en texte, le CSV en entier. L'indexation `.loc` a
levé ; un `merge` aurait joint à vide et produit un tableau calculé sur zéro
ligne appariée.

**Les règles écrites d'avance ont tranché contre l'attente, trois fois.** Sonde
§7 en CAS 2, sensibilité `alpha` au-delà du seuil de contamination, plafond du
§C.4 infirmé. Aucun seuil n'a été déplacé après mesure. Le second ancrage a été
retiré plutôt que remplacé, parce qu'ajouter un modèle après avoir vu que le
premier ne convenait pas aurait rendu « prévu » et « retenu parce qu'il donnait
le résultat attendu » indistinguables.

## 12. Ce qui est transmis

**Le modèle.** `models/LinearSVC.pkl` (118 Mo) et sa métadonnée versionnée.
Export scripté, déterministe, vérifié : `tools/export_modele.py --verifie`
réentraîne sur le train complet et compare l'empreinte de contenu.
`tests/test_export_modele.py`, 28 vérifications.

**Deux empreintes, et la raison écrite à côté.** Deux exports du même modèle ont
des `sha256_pickle` différents ; la cause est `_stop_words_id`, une adresse
mémoire servant de clé de cache. `empreinte_contenu` porte le déterminisme,
`sha256_pickle` l'intégrité d'un transfert. Les comparer entre deux constructions
n'a pas de sens, et c'est documenté dans le fichier lui-même.

**Une contrainte pour l'intégration continue.** `reports/contrainte_ci_export.md` :
`data/` étant exclu du dépôt, un runner n'a pas `train.csv` après un checkout.
Deux options chiffrées, recommandation B avec repli sur A, et un garde-fou
obligatoire dans les deux branches — comparer le `sha256` de `train.csv` porté
par la métadonnée avant d'entraîner. La boucle ne se ferme qu'une fois le premier
export committé.

**Pour la recommandation finale.** Le ML est devant sur 7 classes sur 9
(`Mortgage` étant un ex æquo à 0,0004 près), avec une latence 2 000 fois plus
faible et sans coût par prédiction. Le LLM garde
`Vehicle loan or lease`, la classe la plus rare, et ne demande aucun
réentraînement pour absorber une nouvelle catégorie. La nature des
non-conformités diffère : le ML ne peut pas produire de hors-référentiel, le LLM
en a produit 19 dont 3 catégories hallucinées côté `ministral-3b`. Le socle de
182 réclamations que les deux manquent relève majoritairement d'un décalage entre
l'étiquette (le produit) et le texte (la conduite) — ce qui désigne une action sur
la convention d'étiquetage, pas sur le modèle.

---

## Annexe — reproductibilité

| élément | chemin |
|---|---|
| protocole de sélection, écrit avant mesure | `reports/protocole_selection_phaseA.md` |
| protocole d'analyse croisée + erratum | `reports/protocole_analyse_croisee.md` |
| résultats phases A, B, 2 | `reports/resultats_phaseA.md`, `resultats_phaseB.md`, `resultats_etape3_phase2.md` |
| journal du vocabulaire | `reports/vocabulaire_top_tokens.md` |
| scores et évaluation, lisibles par machine | `reports/etape3_scores.json`, `etape3_evaluation.json` |
| évaluation rejouable | `python tools/etape3_evaluation.py` |
| export du modèle | `python tools/export_modele.py [--verifie]` |
| contrainte d'intégration continue | `reports/contrainte_ci_export.md` |

`src/metrics.py` et `src/data_prep.py` **n'ont pas été modifiés**. Toutes les
métriques de l'étape 3 passent par `metrics.evaluate()` avec `sample_weight`
fourni explicitement, et `CLASS_ORDER` est inchangé.
