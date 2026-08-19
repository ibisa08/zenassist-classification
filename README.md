# ZenAssist — classification automatique de réclamations clients

Comparaison de deux approches sur la même tâche : un **LLM** (Mistral) et un
**modèle de Machine Learning classique** (TF-IDF + classifieur linéaire), afin de
recommander l'une des deux au client.

Corpus : réclamations de consommateurs déposées auprès du CFPB (Consumer
Financial Protection Bureau), en anglais. **354 312 réclamations, 9 classes.**

> **État du projet : étape 1 terminée** — analyse exploratoire, nettoyage,
> split et métriques. Les étapes 2 (LLM) et 3 (ML) ne sont pas commencées.

---

## Démarrage

### Environnement

**L'interpréteur de référence est le `.venv` du projet, en Python 3.13.5.** Les
versions de `requirements.txt` sont celles de cet environnement — pas d'une
installation système (anaconda ou autre). Un `requirements.txt` épinglé sur le mauvais
interpréteur donne un projet qui ne s'installe pas ailleurs.

```bash
python3.13 -m venv .venv
source .venv/bin/activate            # Windows : .venv\Scripts\activate
pip install -r requirements.txt
python -c "import pandas; print(pandas.__version__)"   # doit afficher 3.0.5
```

> Sous VS Code, sélectionner explicitement l'interpréteur du `.venv`
> (**Python: Select Interpreter** → `./.venv/bin/python`). Sinon l'éditeur signale à
> tort les paquets de `requirements.txt` comme absents et exécute les notebooks avec le
> mauvais noyau.

**Compatibilité vérifiée** : le pipeline produit des fichiers **byte-identiques** sous
`pandas 3.0.5 / numpy 2.5.2 / scikit-learn 1.9.0` (le `.venv`) et sous
`pandas 2.2.3 / numpy 2.1.3 / scikit-learn 1.6.1` — mêmes empreintes `sha256`, et les
49 vérifications passent sur les deux. Les versions épinglées sont celles de
l'environnement de livraison.

### Pipeline

`data/` est intégralement exclu du dépôt : sur un clone, le répertoire n'existe pas et
doit être créé avant de déposer le fichier source.

```bash
mkdir -p data/raw
# y placer le fichier source : data/raw/dataset.csv (639 Mo, non versionné)
python -m src.data_prep          # ~15 s : nettoie, découpe, écrit data/processed/
```

Le pipeline est **idempotent** : le relancer reproduit des fichiers strictement
identiques (vérifiable par les empreintes `sha256` de `split_metadata.json`).
Seul le champ `generated_at` change d'une exécution à l'autre.

```bash
jupyter lab notebooks/01_exploration.ipynb    # notebook narratif d'exploration
```

---

## Arborescence

```
zenassist-classification/
├── data/
│   ├── raw/                          # non versionné
│   │   └── dataset.csv               # 639 Mo, 1 282 355 lignes, 15 colonnes
│   └── processed/                    # non versionné, régénérable
│       ├── train.csv                 # 283 449 lignes
│       ├── test.csv                  #  70 863 lignes
│       ├── test_sample_2000.csv      #   2 000 lignes — évaluation LLM ⚠️ voir plus bas
│       ├── test_sample_2000.README.txt
│       ├── split_metadata.json       # seed, tailles, distributions, poids, sha256
│       └── cleaning_report.json      # lignes supprimées à chaque étape
│
├── src/
│   ├── config.py                     # chemins, constantes, référentiel d'étiquettes
│   ├── data_prep.py                  # chargement, nettoyage, split
│   └── metrics.py                    # métriques partagées par les étapes 2 et 3
│
├── tests/
│   └── test_pipeline.py              # garde-fous + cohérence du pipeline
│
├── notebooks/
│   └── 01_exploration.ipynb          # exploration narrative (livrable client)
│
├── reports/
│   ├── diagnostic.md                 # diagnostic exploratoire chiffré
│   ├── h1_sampling_simulation.md     # arbitrage du plan d'échantillonnage
│   ├── limites.md                    # limites et précautions d'interprétation
│   └── figures/                      # figures des rapports et du notebook
│
├── tools/                            # génère le notebook et les rapports chiffrés
│   ├── build_notebook.py             # ← source de vérité du notebook
│   ├── limites_analyse.py            # ← source de vérité de limites.md
│   ├── h1_simulation.py              # ← arbitrage du plan d'échantillonnage
│   ├── h1_apparie.py · h1_decomposition.py · h1_figures.py
│   └── README.md
│
├── scratch/                          # résultats intermédiaires, non versionné
├── requirements.txt                  # versions épinglées
├── .env.example                      # modèle de configuration (clé API)
└── .gitignore
```

⚠️ **`notebooks/01_exploration.ipynb` est un artefact généré.** Il ne doit pas être
édité à la main : toute correction se fait dans `tools/build_notebook.py`, puis on
régénère avec `python tools/build_notebook.py`. Éditer le `.ipynb` directement fait
diverger les deux, et la correction sera perdue à la génération suivante. Voir
[`tools/README.md`](tools/README.md).

### Vérifications

```bash
python tests/test_pipeline.py     # 49 vérifications, ~30 s
```

À lancer après `python -m src.data_prep`, dont il relit les fichiers. Le script
retourne un code non nul si une vérification échoue, donc il s'intègre tel quel à une
chaîne d'intégration continue.

Il couvre deux choses. D'abord la **cohérence du pipeline** : aucun texte partagé entre
train et test, plancher d'échantillonnage respecté, somme des poids égale à la
population de référence. Ensuite, et c'est sa raison d'être, il vérifie que **les
garde-fous échouent bien quand ils doivent échouer** — `sample_weight` omis lève un
`TypeError`, une source pondérée sans poids lève une `ValueError`, `load_eval_sample()`
refuse de servir un échantillon de développement, un modèle de tarification inconnu lève
une `KeyError`. Un garde-fou qui n'est jamais mis en défaut n'est pas un garde-fou,
c'est une intention.

---

## Le pipeline de préparation

`src/data_prep.py` applique un ordre imposé, chaque étape étant justifiée dans
[`reports/diagnostic.md`](reports/diagnostic.md).

| # | Étape | Lignes supprimées | Restantes |
|---|---|---:|---:|
| 0 | Fichier brut | — | 1 282 355 |
| 1 | Tri par `complaint_id` *(idempotence)* | 0 | 1 282 355 |
| 2 | Texte ou étiquette manquants | 898 791 | 383 564 |
| 3 | Exclusion `Consumer Loan` + `Other financial service` | 9 766 | 373 798 |
| 4 | Fusion des libellés (16 → 9 classes) | 0 | 373 798 |
| 5 | Normalisation des espaces | 0 | 373 798 |
| 6 | `MIN_TEXT_LENGTH = 20` | 147 | 373 651 |
| 7 | Contradictions d'étiquetage — vote majoritaire | 179 | 373 472 |
| 8 | Doublons exacts sur le texte | 19 160 | **354 312** |
| 9 | Garde-fou `MIN_SAMPLES_PER_CLASS = 1000` | 0 | **354 312** |
| 10 | Split stratifié 80 / 20 | — | 283 449 / 70 863 |

Deux contraintes d'ordre sont structurantes :

- **Le vote majoritaire précède le dédoublonnage.** 166 textes portent au moins
  deux étiquettes différentes — ce sont des courriers types de credit repair
  déposés en masse, que les agents du CFPB ont classés différemment. 83 sont
  résolus par majorité stricte (1 573 lignes réétiquetées), 83 sont supprimés
  pour égalité parfaite (179 lignes).
- **Le dédoublonnage précède le split.** Sans lui, les 11 354 textes dupliqués
  se retrouveraient simultanément dans le train et le test, et le score de test
  serait artificiellement gonflé.

Le texte n'est **ni mis en minuscules ni dépouillé de sa ponctuation** : le texte
brut est l'entrée du LLM. Le prétraitement lourd appartient au pipeline TF-IDF
de l'étape 3.

### Distribution finale

| Classe | corpus | % | train | test | éch. éval. |
|---|---:|---:|---:|---:|---:|
| Credit reporting | 107 734 | 30,4 | 86 187 | 21 547 | 521 |
| Debt collection | 83 988 | 23,7 | 67 190 | 16 798 | 417 |
| Mortgage | 52 921 | 14,9 | 42 337 | 10 584 | 282 |
| Credit card or prepaid card | 41 392 | 11,7 | 33 113 | 8 279 | 231 |
| Bank account or service | 27 701 | 7,8 | 22 161 | 5 540 | 171 |
| Student loan | 21 766 | 6,1 | 17 413 | 4 353 | 145 |
| Money transfer or virtual currency | 6 966 | 2,0 | 5 573 | 1 393 | 81 |
| Payday, title or personal loan | 6 146 | 1,7 | 4 917 | 1 229 | 77 |
| Vehicle loan or lease | 5 698 | 1,6 | 4 558 | 1 140 | 75 |

**Ratio majoritaire / minoritaire = 18,9** (contre 5 774 avant fusion des
libellés). C'est ce déséquilibre qui impose le F1-macro comme métrique
principale.

---

## ⚠️ L'échantillon d'évaluation LLM — à lire avant de l'utiliser

`data/processed/test_sample_2000.csv` est **stratifié à plancher** (50 items
minimum par classe, solde réparti proportionnellement). Il n'est donc **pas**
représentatif de la distribution du test : les classes rares y sont
volontairement sur-échantillonnées.

**Toute métrique calculée sans la colonne `sampling_weight` est biaisée :**

| Métrique | Biais si non pondérée |
|---|---:|
| F1-macro | **+0,022** |
| Précision de `Vehicle loan or lease` | **+0,29** |
| Précision de `Credit reporting` | **−0,082** |

Ces valeurs sont mesurées, pas estimées : 2 000 réplicats Monte-Carlo,
[`reports/h1_sampling_simulation.md`](reports/h1_sampling_simulation.md).

### Le seul chargement supporté

```python
from src.data_prep import load_eval_sample
from src.metrics import evaluate

df, weights = load_eval_sample()           # retourne un TUPLE, jamais un DataFrame seul
resultats = evaluate(df["label"], y_pred, sample_weight=weights, source=df)
```

Cinq garde-fous rendent l'oubli difficile :

1. **`sample_weight` n'a pas de valeur par défaut** dans `evaluate()` et
   `plot_confusion_matrix()`. C'est un argument nommé requis : l'omettre lève un
   `TypeError` immédiat et inconditionnel. Passer `None` reste possible, mais
   devient une *déclaration d'intention* — « cette évaluation n'est pas
   pondérée » — et non un défaut subi.
2. `evaluate(..., source=df)` **lève une `ValueError`** — pas un avertissement —
   si `source` porte une colonne de poids et que `sample_weight` vaut `None`.
   C'est le filet qui rattrape un `None` passé à tort.
3. `load_eval_sample()` retourne un **tuple** — un `df = load_eval_sample()`
   suivi d'un `df["label"]` échoue immédiatement.
4. `load_eval_sample()` **lève une `RuntimeError` si `SAMPLE_FRACTION < 1.0`**,
   sauf `autoriser_dev=True`. C'est le seul endroit du projet où une erreur coûte
   de l'argent et des heures d'attente : lancer 2 000 appels d'API sur un
   échantillon de développement est irrattrapable une fois les appels partis.
5. Le CSV porte une ligne d'avertissement en tête, ce qui fait produire à un
   `pd.read_csv()` nu un DataFrame à une seule colonne dont l'en-tête *est*
   l'avertissement : tout accès à `label` lève un `KeyError`. Attention, le
   fichier se « lit » sans exception — il est seulement inexploitable.

`compare_results()` refuse par ailleurs d'aligner un résultat pondéré et un
résultat non pondéré, ou deux échantillons de tailles différentes.

### Les deux tableaux de résultats

La comparaison finale produit **deux tableaux distincts**, et les fusionner
serait exactement l'erreur que le garde-fou 2 empêche :

| Tableau | Population | Pondération | Ce qu'il montre |
|---|---|---|---|
| **1** | les 2 000 lignes de `test_sample_2000.csv` | oui | LLM vs ML — **seule comparaison valide** |
| **2** | les 70 863 lignes de `test.csv` | non | ML seul — **l'apport du volume d'entraînement** |

```python
# tableau 1 — comparaison
df, weights = load_eval_sample()
res_llm = evaluate(df["label"], pred_llm, sample_weight=weights, source=df)
res_ml  = evaluate(df["label"], pred_ml,  sample_weight=weights, source=df)
compare_results({"LLM": res_llm, "ML": res_ml})

# tableau 2 — apport du volume, sans IC (voir ci-dessous)
res_ml_complet = evaluate(test["label"], pred_ml_complet,
                          sample_weight=None, with_ci=False)
```

### Pourquoi ce plan plutôt qu'un échantillon proportionnel

À coût LLM identique (2 000 appels), la stratification à plancher avec
repondération de Horvitz-Thompson réduit le **RMSE du F1-macro de 20 %** et
celui du F1 de la classe la plus faible de **23 %**. Surtout, elle fait passer
l'**écart minimal détectable entre les deux approches de 4,0 à 3,1 points de
F1-macro** — ce qui est décisif, la mission consistant précisément à départager
LLM et ML. Le classement est stable sur quatre scénarios de matrice de confusion.

---

## Choix des métriques

**F1-macro** est la métrique principale. Avec un ratio de déséquilibre de 18,9,
l'accuracy est trompeuse : prédire systématiquement `Credit reporting` donne
déjà 30 % d'accuracy sans rien comprendre. Le F1-macro pondère les 9 classes à
égalité, ce qui correspond au besoin métier — une réclamation mal routée coûte
la même chose au client quelle que soit sa catégorie.

**Tout résultat affiché porte son intervalle de confiance**, obtenu par bootstrap
**intra-strates** (`metrics.bootstrap_ci`). Le rééchantillonnage se fait avec
remise à l'intérieur de chaque classe, en conservant l'effectif de chacune : un
bootstrap uniforme ferait varier le nombre d'items des classes rares d'un tirage
à l'autre, ce qui ajouterait une variance absente du plan de sondage réel.

Le plan est **exposé en paramètre** plutôt que codé en dur, parce que c'est un
choix statistique et non un détail d'implémentation. `evaluate(strata=...)`
accepte `"auto"` (défaut — `y_true` si des poids sont fournis, `None` sinon),
`None`, ou une séquence de strates. Le mode retenu est journalisé dans la clé
`bootstrap_strata` du dictionnaire de résultats.

**`with_ci=False` sur le test complet.** Sur 70 863 lignes l'intervalle fait
~0,005 de large : il n'apprend rien et coûte plusieurs minutes. L'IC a du sens
sur les 2 000 lignes de l'échantillon d'évaluation, pas sur la population.

**Latence : le p95, pas la moyenne.** La moyenne est écrasée par la masse des
appels rapides et masque la queue de distribution — précisément ce que
l'utilisateur perçoit comme « le service rame ». Le chronomètre
(`metrics.Timer`) mesure **chaque appel individuellement**, jamais un temps total
divisé par `n`. Si l'étape 2 parallélise les appels, la latence enregistrée reste
celle de l'appel unitaire : c'est la latence perçue par l'utilisateur final, à ne
pas confondre avec le débit du batch.

---

## Ce que disent déjà les données

Trois constats du diagnostic qui conditionnent la lecture des résultats à venir.

- **Un plafond de performance est attendu, quelle que soit l'approche.**
  L'étiquette CFPB désigne le *produit financier*, alors que le texte décrit
  souvent le *préjudice ressenti* — presque toujours « ça abîme mon crédit ».
  35,8 % des réclamations `Debt collection` et 26,7 % des `Vehicle loan or lease`
  parlent explicitement du dossier de crédit. La paire
  `Credit reporting` ↔ `Debt collection` pèse 54 % du corpus et sa frontière est
  réellement floue.
- **Le coût n'est pas un facteur discriminant.** Sur les dix modèles comparés
  (tarifs relevés le 11/08/2026), la projection annuelle pour 1 000 réclamations par
  jour va de **19 $** (`ministral-3b`) à **1 031 $** (`gpt-5.6-sol`) — deux ordres de
  grandeur, tous négligeables devant le traitement manuel remplacé. Le **cache de
  préfixe** (~260 tokens constants) réduit encore la facture de **~43 %**, soit un
  levier plus fort que le choix du modèle. Le facteur limitant est la **latence** :
  ~20 h pour les 70 863 lignes du test complet, ce qui justifie l'échantillon de 2 000.
  Voir [`reports/limites.md`](reports/limites.md) §4 — tarifs volatils, tier gratuit,
  API Batch et argument RGPD.
- **85 % des textes contiennent un masquage `XXXX`.** Il détruit les entités
  nommées et handicape les deux approches pour la même raison — ce n'est donc pas
  un argument en faveur de l'une ou de l'autre. Aucun retrait n'est appliqué en
  amont : l'IDF du `TfidfVectorizer` neutralise `xxxx` tout seul (présent dans
  85 % des documents, IDF ≈ 0,16), et le texte brut reste nécessaire au LLM.

---

## Configuration

Tous les paramètres sont dans [`src/config.py`](src/config.py) — **aucun chemin
ni aucune constante en dur ailleurs**, ni dans les modules ni dans le notebook.

| Constante | Valeur | Justification |
|---|---:|---|
| `RANDOM_SEED` | 42 | reproductibilité |
| `TEST_SIZE` | 0.2 | split stratifié aléatoire, pas temporel¹ |
| `MIN_TEXT_LENGTH` | 20 | coûte 147 lignes (0,04 %) ; le p5 est à 132 caractères |
| `MIN_SAMPLES_PER_CLASS` | 1000 | garde-fou ; ne supprime rien aujourd'hui² |
| `LLM_EVAL_SAMPLE_SIZE` | 2000 | arbitrage H.1 |
| `LLM_EVAL_MIN_PER_CLASS` | 50 | plancher, stratégie B3 |
| `SAMPLE_FRACTION` | 1.0 | < 1.0 pour itérer vite en dev³ |

¹ Un split temporel serait invalide : après fusion, `Vehicle loan or lease`
n'existe qu'à partir d'avril 2017 (le référentiel CFPB a basculé le 21-24 avril
2017) et se retrouverait entièrement d'un seul côté de la coupure. Pour la même
raison, **`date_received` ne doit jamais servir de variable explicative** — elle
prédirait cette classe par pur artefact administratif. Elle est conservée comme
métadonnée d'audit.

² La plus petite classe compte 5 698 lignes, soit 5,7 × le seuil. Le garde-fou
est maintenu et journalise « 0 classe sous le seuil » : le CFPB publie en
continu, et une classe devenue trop rare doit être signalée plutôt que de passer
silencieusement dans le split.

³ Une valeur < 1.0 suffixe **tous** les fichiers produits (`train_dev5.csv`,
`test_sample_2000_dev5.csv`, `split_metadata_dev5.json`…) : il est impossible
d'écraser les fichiers de référence avec un corpus partiel. Et
`load_eval_sample()` refuse de servir un échantillon de développement sans
`autoriser_dev=True`, pour qu'aucune campagne d'appels d'API ne parte dessus par
accident.

### Étiquettes

Le fichier brut superpose deux référentiels CFPB successifs, la bascule étant
datée du **21-24 avril 2017**. `config.LABEL_MAPPING` fusionne les 16 libellés
retenus vers 9 classes sémantiques ; `config.LABELS_CFPB_OFFICIAL` donne la
correspondance vers les libellés officiels, pour tester les deux formulations
dans le prompt à l'étape 2. `config.CLASS_ORDER` fige l'ordre des classes, afin
que toutes les matrices de confusion et tous les tableaux du projet se lisent de
la même façon.

---

## Étape 2 (LLM) — clé d'API et choix du modèle

```bash
cp .env.example .env      # puis renseigner MISTRAL_API_KEY
```

`.env` est dans le `.gitignore` et ne doit jamais être commité.

Le catalogue de modèles et leurs tarifs vivent dans `config.MODELS_PRICING` (dix
modèles, cinq fournisseurs). `metrics.compare_models_cost(n)` produit le tableau
comparatif, avec et sans cache de préfixe :

```python
from src.metrics import compare_models_cost, estimate_cost
compare_models_cost(2000)                                   # tableau complet
estimate_cost(1000, "mistral-small-4", cached_prefix_tokens=260)
```

⚠️ **Tarifs volatils** — relevés le 11/08/2026, ils ont bougé deux fois en six semaines.
`claude-sonnet-5` bénéficie d'un tarif d'introduction **expirant le 31/08/2026** ;
`estimate_cost()` bascule automatiquement sur le tarif standard (+50 %) et lève un
avertissement passé cette date. Les revérifier avant toute projection présentée au
client.

---

## Errata — 2026-08-19 (étape 2)

Les chiffres de coût de ce README datent de l'étape 1 et reposaient sur des
**hypothèses non mesurées**. L'étape 2 les a mesurés sur 200 appels réels. Le
corps du document n'est pas réécrit — c'est un livrable daté.

| affirmation du README | statut |
|---|---|
| « fourchette de **19 $** à **1 031 $** par an » | **1 017 $** après mesure des tokens ; l'ordre de grandeur et la conclusion sont inchangés |
| « préfixe constant **~260 tokens** » | mesuré à **252** (`mistral-small`) et **240** (`ministral-3b`) |
| « le cache réduit la facture de **~43 %** » | **borne théorique**. Mesuré : **36,8 %** en régime établi, 20,6 % au démarrage à froid |

Le cache ne couvre jamais tout le préfixe (plafond structurel à 224 tokens sur
252) et son activation dépend du réchauffement (55 % sur les 20 premiers appels,
~98 % ensuite). Le détail est dans l'errata de
[`reports/limites.md`](reports/limites.md).

**La conclusion du projet est inchangée** : le coût n'est pas le facteur
discriminant, la latence l'est.

Source unique de vérité pour tout chiffre de coût : `src/config.py`
(`MODELS_PRICING`, `PRICING_CHECKED_ON`, `CACHE_MESURE`).
