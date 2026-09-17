"""Configuration centrale du projet ZenAssist.

Ce module est la SEULE source de verite pour les chemins, les constantes et le
referentiel d'etiquettes. Aucun chemin en dur ne doit apparaitre ailleurs dans
le projet : ni dans les modules, ni dans les notebooks.

Toutes les valeurs numeriques ci-dessous sont issues du diagnostic exploratoire
(`reports/diagnostic.md`) et de la simulation d'echantillonnage
(`reports/h1_sampling_simulation.md`). Les references de section y renvoient.
"""

from pathlib import Path

# ===========================================================================
# 1. Chemins
# ===========================================================================
# Racine du projet : ce fichier est dans <racine>/src/, donc deux niveaux au-dessus.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Fichiers
RAW_FILE = RAW_DIR / "dataset.csv"
TRAIN_FILE = PROCESSED_DIR / "train.csv"
TEST_FILE = PROCESSED_DIR / "test.csv"
SPLIT_METADATA_FILE = PROCESSED_DIR / "split_metadata.json"
CLEANING_REPORT_FILE = PROCESSED_DIR / "cleaning_report.json"


def ensure_dirs() -> None:
    """Cree les repertoires de sortie s'ils n'existent pas. Idempotent."""
    for d in (PROCESSED_DIR, REPORTS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)


# ===========================================================================
# 2. Reproductibilite
# ===========================================================================
RANDOM_SEED = 42
TEST_SIZE = 0.2

# Fraction du corpus utilisee, UNIQUEMENT pour iterer vite en developpement.
# Appliquee AVANT le split, de facon stratifiee (arbitrage H.5).
# Toute valeur < 1.0 fait suffixer les fichiers produits par "_sample" : il est
# ainsi impossible de livrer par erreur un modele entraine sur un corpus partiel.
SAMPLE_FRACTION = 1.0

# ===========================================================================
# 3. Colonnes
# ===========================================================================
# --- colonnes du fichier brut (chargement restreint : ~500 Mo au lieu de ~5 Go)
RAW_ID_COL = "Complaint ID"
RAW_DATE_COL = "Date received"
RAW_TEXT_COL = "Consumer Claim"
RAW_LABEL_COL = "Tag"
RAW_USECOLS = [RAW_ID_COL, RAW_DATE_COL, RAW_TEXT_COL, RAW_LABEL_COL]

# --- colonnes des fichiers produits
ID_COL = "complaint_id"
DATE_COL = "date_received"
TEXT_COL = "text"
LABEL_COL = "label"
WEIGHT_COL = "sampling_weight"
STRATE_COL = "strate"

OUTPUT_COLS = [ID_COL, DATE_COL, TEXT_COL, LABEL_COL]

# ATTENTION : `date_received` est conserve comme METADONNEE D'AUDIT UNIQUEMENT.
# Ne JAMAIS l'utiliser comme variable explicative. Apres fusion des libelles,
# la classe "Vehicle loan or lease" n'existe qu'a partir d'avril 2017 (le
# referentiel CFPB a bascule le 21-24 avril 2017, cf. diagnostic F.2/F.4) : la
# date la predirait a 100 % de rappel par pur artefact administratif.
FEATURE_COLS = [TEXT_COL]

# ===========================================================================
# 4. Nettoyage
# ===========================================================================
# Longueur minimale d'un texte, en caracteres (diagnostic E.1).
# Coute 148 lignes (0,039 %) et elimine des textes structurellement non
# classables ("Account is fraud", "Needs to be removed"). Le p5 du corpus est a
# 132 caracteres : le seuil est tres loin de la masse de la distribution.
MIN_TEXT_LENGTH = 20

# Effectif minimal par classe (diagnostic E.2).
# Apres fusion, la plus petite classe compte 5 704 lignes, soit 5,7x le seuil :
# ce garde-fou ne supprime RIEN aujourd'hui. Il est conserve pour signaler une
# classe devenue trop rare si le dataset est rafraichi (le CFPB publie en
# continu) ou si le mapping evolue.
MIN_SAMPLES_PER_CLASS = 1000

# ===========================================================================
# 5. Referentiel d'etiquettes
# ===========================================================================
# Le fichier brut superpose DEUX referentiels CFPB successifs, bascule datee du
# 21-24 avril 2017 (diagnostic F.2). Distinguer "Credit card" de "Credit card or
# prepaid card" reviendrait a demander au modele de deviner la date de depot, pas
# le sujet. On fusionne donc 16 libelles vers 9 classes semantiques.

LABEL_MAPPING = {
    # --- signalement credit
    "Credit reporting, credit repair services, or other personal consumer reports": "Credit reporting",
    "Credit reporting": "Credit reporting",
    # --- cartes
    "Credit card or prepaid card": "Credit card or prepaid card",
    "Credit card": "Credit card or prepaid card",
    "Prepaid card": "Credit card or prepaid card",
    # --- compte bancaire
    "Bank account or service": "Bank account or service",
    "Checking or savings account": "Bank account or service",
    # --- transfert d'argent
    "Money transfer, virtual currency, or money service": "Money transfer or virtual currency",
    "Money transfers": "Money transfer or virtual currency",
    "Virtual currency": "Money transfer or virtual currency",
    # --- pret court terme
    "Payday loan, title loan, or personal loan": "Payday, title or personal loan",
    "Payday loan": "Payday, title or personal loan",
    # --- inchanges d'un referentiel a l'autre (traversent la bascule sans interruption)
    "Debt collection": "Debt collection",
    "Mortgage": "Mortgage",
    "Student loan": "Student loan",
    "Vehicle loan or lease": "Vehicle loan or lease",
}

# Libelles exclus du corpus (diagnostic E.3 / F.4).
EXCLUDED_LABELS = {
    # Correspondance NON INJECTIVE : l'ancien "Consumer Loan" s'est eclate en
    # DEUX classes du nouveau referentiel ("Vehicle loan or lease" ET "Payday
    # loan, title loan, or personal loan"), toutes deux apparues le meme jour.
    # Tout mapping introduirait du bruit d'etiquetage. 9 474 lignes.
    "Consumer Loan",
    # Categorie fourre-tout sans definition semantique, 292 lignes, sous le
    # seuil MIN_SAMPLES_PER_CLASS.
    "Other financial service",
}

# Les 9 classes finales, dans l'ordre decroissant d'effectif (diagnostic F.7).
# Cet ordre est FIGE : il sert d'ordre des axes de la matrice de confusion et
# d'ordre des colonnes des tableaux comparatifs, pour que toutes les figures du
# projet se lisent de la meme facon.
CLASS_ORDER = [
    "Credit reporting",
    "Debt collection",
    "Mortgage",
    "Credit card or prepaid card",
    "Bank account or service",
    "Student loan",
    "Money transfer or virtual currency",
    "Payday, title or personal loan",
    "Vehicle loan or lease",
]

# Correspondance libelle court -> libelle CFPB officiel (arbitrage H.4).
# Les libelles courts sont utilises dans les CSV et les figures : ils rendent la
# matrice de confusion lisible et economisent des tokens dans le prompt. Ce
# second dictionnaire permet de tester la formulation officielle a l'etape 2,
# sans toucher aux donnees.
LABELS_CFPB_OFFICIAL = {
    "Credit reporting": "Credit reporting, credit repair services, or other personal consumer reports",
    "Debt collection": "Debt collection",
    "Mortgage": "Mortgage",
    "Credit card or prepaid card": "Credit card or prepaid card",
    # Le libelle court reprend l'ancienne appellation ; l'officiel en vigueur
    # depuis avril 2017 est "Checking or savings account".
    "Bank account or service": "Checking or savings account",
    "Student loan": "Student loan",
    "Money transfer or virtual currency": "Money transfer, virtual currency, or money service",
    "Payday, title or personal loan": "Payday loan, title loan, or personal loan",
    "Vehicle loan or lease": "Vehicle loan or lease",
}

# ===========================================================================
# 6. Echantillon d'evaluation LLM  (strategie B3)
# ===========================================================================
# Arbitrage H.1, tranche par simulation Monte-Carlo (2 000 replicats, 4 scenarios
# de matrice de confusion) : cf. reports/h1_sampling_simulation.md.
#
# B3 = allocation a PLANCHER : 50 items minimum par classe, solde reparti
# proportionnellement a la population du test.
#
# Pourquoi pas un simple echantillon proportionnel (strategie A2) ?
#   - RMSE du F1-macro          : 0,0123 (B3) contre 0,0153 (A2), -20 %
#   - RMSE du F1 de la classe la plus faible : 0,048 contre 0,063, -23 %
#   - ecart minimal detectable entre deux approches : 3,1 points contre 4,0
#   - cout LLM identique (n = 2 000 dans les deux cas)
# B3 n'est jamais battu par A2 sur aucun des quatre scenarios testes.
#
# CONTREPARTIE NON NEGOCIABLE : l'allocation a plancher sur-echantillonne les
# classes rares. Toute metrique calculee sur cet echantillon SANS la ponderation
# de Horvitz-Thompson (poids w_c = N_c / n_c) est biaisee de +0,022 sur le
# F1-macro et de +0,29 sur la precision de "Vehicle loan or lease".
LLM_EVAL_SAMPLE_SIZE = 2000
LLM_EVAL_MIN_PER_CLASS = 50

EVAL_SAMPLE_FILE = PROCESSED_DIR / f"test_sample_{LLM_EVAL_SAMPLE_SIZE}.csv"
EVAL_SAMPLE_README_FILE = PROCESSED_DIR / f"test_sample_{LLM_EVAL_SAMPLE_SIZE}.README.txt"

# Ligne de commentaire ecrite EN TETE du CSV d'evaluation (garde-fou 6).
# Elle est volontairement depourvue de virgule. Effet mesure d'un `pd.read_csv()`
# nu sur ce fichier : pandas prend la ligne d'avertissement pour l'en-tete et
# produit un DataFrame a UNE seule colonne (les six champs reels partent en
# index a cinq niveaux). Toute tentative d'acceder a `label` ou a
# `sampling_weight` leve alors un KeyError, et `df.columns` affiche
# l'avertissement en clair. L'echec est donc bruyant et immediat -- mais ce
# n'est pas une ParserError : le fichier se "lit" sans exception, il est
# seulement inexploitable. Le seul chargement supporte reste
# `data_prep.load_eval_sample()`.
EVAL_SAMPLE_HEADER_COMMENT = (
    "# ATTENTION - echantillon stratifie a plancher (strategie B3). "
    "Toute metrique calculee sans la colonne sampling_weight est biaisee "
    "de +0.022 sur le F1-macro et de +0.29 sur la precision de la classe "
    "Vehicle loan or lease. Chargement unique supporte : "
    "src.data_prep.load_eval_sample() qui retourne le tuple (df weights)."
)

# Allocation de reference issue de la simulation, sur la distribution attendue du
# test (diagnostic F.7). L'allocation reelle est RECALCULEE par le pipeline a
# partir du test effectivement produit ; cette table sert de controle : un ecart
# important signalerait que le nettoyage a change de comportement.
EVAL_ALLOCATION_REFERENCE = {
    "Credit reporting": 521,
    "Debt collection": 418,
    "Mortgage": 282,
    "Credit card or prepaid card": 231,
    "Bank account or service": 171,
    "Student loan": 145,
    "Money transfer or virtual currency": 80,
    "Payday, title or personal loan": 77,
    "Vehicle loan or lease": 75,
}

# ===========================================================================
# 7. Metriques et couts
# ===========================================================================
# F1-macro : metrique principale. Justification dans le notebook d'exploration,
# section "Choix des metriques" — le corpus presente un ratio majoritaire /
# minoritaire de 18,9, et les classes rares comptent autant que les autres pour
# le client (une reclamation mal routee coute la meme chose quelle que soit sa
# categorie).
PRIMARY_METRIC = "f1_macro"

# Latence : on retient le p95, pas la moyenne. La moyenne est ecrasee par la
# masse des appels rapides et masque la queue de distribution, qui est
# precisement ce que l'utilisateur percoit comme "le service rame".
LATENCY_PERCENTILES = (50, 95)

# Bootstrap : tout resultat affiche porte son intervalle de confiance.
BOOTSTRAP_N = 1000
BOOTSTRAP_ALPHA = 0.05

# ===========================================================================
# 7bis. ############  HYPOTHESES PROVISOIRES  ############
#       A REMPLACER PAR DES MESURES REELLES A L'ETAPE 2
# ===========================================================================
# AUCUNE des constantes de ce bloc n'est mesuree sur le systeme reel. Ce sont
# des estimations construites avant tout appel d'API, utilisees pour dimensionner
# le projet. Elles sont suffisantes pour conclure que le cout n'est pas le
# facteur discriminant (cf. diagnostic D.3), mais elles ne doivent PAS figurer
# telles quelles dans la recommandation finale au client.
#
# Ce qui les remplacera : chaque reponse de l'API Mistral porte un champ `usage`
# donnant les comptes de tokens reels (prompt_tokens, completion_tokens). Des la
# premiere campagne de l'etape 2, agreger ces comptes et substituer les valeurs
# ci-dessous. Voir reports/limites.md, section 4.
#
# `metrics.estimate_cost()` retourne une cle `hypotheses` valant True tant que
# ces valeurs par defaut sont utilisees, et False des qu'un appelant fournit des
# mesures : aucun chiffre de cout ne peut donc etre presente sans que son statut
# soit explicite.
# ---------------------------------------------------------------------------

# --- tarification -----------------------------------------------------------
# ORIGINE : pages tarifaires publiques des fournisseurs, consultees le
#           2026-08-11 (sources listees dans reports/limites.md, section 4).
# STATUT  : ces tarifs bougent vite. Deux mouvements sur les six dernieres
#           semaines : OpenAI a baisse gpt-5.6-luna de 80 % le 30/07, DeepSeek a
#           annonce une hausse le 06/08 sans en preciser ni la date ni le
#           montant. A reverifier avant toute projection chiffree presentee au
#           client.
# Releve le 2026-08-11, RECONFIRME INCHANGE le 2026-08-19 sur deux sources
# concordantes (mistral.ai/pricing/api et docs.mistral.ai/inference/pricing)
# pour les deux modeles compares a l'etape 2 :
#   Mistral Small 4  : 0,15 / 0,60 $ par M de tokens, cache a 0,015 $ (-90 %)
#   Ministral 3 3B   : 0,10 / 0,10 $ par M de tokens, cache -90 %
# Controle croise : Mistral Large 3 releve a 0,50 / 1,50 $, conforme.
PRICING_CHECKED_ON = "2026-08-19"

# ###################################################################
# CE DICTIONNAIRE EST UNE GRILLE TARIFAIRE, PAS UN REGISTRE
# D'IDENTIFIANTS D'API. NE JAMAIS PASSER UNE CLE D'ICI A L'API.
# ###################################################################
# Constat verifie le 2026-08-19 par `client.models.list()` : AUCUNE des cles
# ci-dessous n'est un identifiant accepte par l'API Mistral. "mistral-small-4"
# et "ministral-3b" n'existent pas cote API ; les identifiants reels sont
# "mistral-small-latest" et "ministral-3b-latest".
#
# Les cles sont des noms COMMERCIAUX, choisis pour la lisibilite des tableaux de
# cout. Les identifiants d'appel vivent dans `LLM_MODELES_A_COMPARER` (section
# 8), qui porte la correspondance et la version resolue de chaque alias.
#
# Sans cette note, quelqu'un finira par ecrire
# `client.chat.complete(model=cfg.DEFAULT_MODEL)` : l'API repondrait par une
# 400 sur un nom de modele inconnu, au mieux immediatement, au pire au milieu
# d'une campagne.
#
# Tarifs en DOLLARS PAR MILLION DE TOKENS.
#
# `remise_cache` : reduction appliquee a la part d'entree servie depuis le cache
#     de prefixe. Le prompt de l'etape 2 aura un prefixe constant (instruction +
#     liste des 9 etiquettes, ~260 tokens) eligible chez tous les fournisseurs.
#     None = non verifie pour ce fournisseur ; le cache n'est alors PAS modelise
#     plutot que suppose (cf. gemini-3.5-flash).
# `cache_automatique` : True si le fournisseur cache sans declaration explicite.
#     VERIFIE le 2026-08-19 pour Mistral : le cache de prefixe s'active seul,
#     sans parametre d'activation, et `usage.prompt_tokens_details.cached_tokens`
#     remonte la part servie. Ce champ valait False par erreur jusqu'a cette date.
#     Un champ documentaire faux est plus dangereux qu'un champ absent : il est
#     cru sans etre reverifie. `tests/test_llm_config.py` en controle desormais
#     la coherence pour TOUS les fournisseurs, pas seulement DeepSeek.
# `tarif_provisoire_jusquau` / `tarif_apres` : tarif promotionnel a duree limitee.
MODELS_PRICING = {
    "ministral-3b": {
        "fournisseur": "Mistral", "input_per_1m": 0.10, "output_per_1m": 0.10,
        "remise_cache": 0.90, "cache_automatique": True,
        "tarif_provisoire_jusquau": None, "tarif_apres": None,
    },
    "deepseek-v4-flash": {
        "fournisseur": "DeepSeek", "input_per_1m": 0.14, "output_per_1m": 0.28,
        "remise_cache": 0.98, "cache_automatique": True,
        "tarif_provisoire_jusquau": None, "tarif_apres": None,
    },
    "mistral-small-4": {
        "fournisseur": "Mistral", "input_per_1m": 0.15, "output_per_1m": 0.60,
        "remise_cache": 0.90, "cache_automatique": True,
        "tarif_provisoire_jusquau": None, "tarif_apres": None,
    },
    "gpt-5.6-luna": {
        "fournisseur": "OpenAI", "input_per_1m": 0.20, "output_per_1m": 1.20,
        "remise_cache": 0.90, "cache_automatique": False,
        "tarif_provisoire_jusquau": None, "tarif_apres": None,
    },
    "mistral-large-3": {
        "fournisseur": "Mistral", "input_per_1m": 0.50, "output_per_1m": 1.50,
        "remise_cache": 0.90, "cache_automatique": True,
        "tarif_provisoire_jusquau": None, "tarif_apres": None,
    },
    "gemini-3.5-flash": {
        "fournisseur": "Google", "input_per_1m": 0.75, "output_per_1m": 4.50,
        # Remise de cache NON VERIFIEE pour ce fournisseur au 2026-08-11 : on ne
        # la modelise pas plutot que de la supposer identique aux autres.
        "remise_cache": None, "cache_automatique": False,
        "tarif_provisoire_jusquau": None, "tarif_apres": None,
    },
    "claude-haiku-4.5": {
        "fournisseur": "Anthropic", "input_per_1m": 1.00, "output_per_1m": 5.00,
        "remise_cache": 0.90, "cache_automatique": False,
        "tarif_provisoire_jusquau": None, "tarif_apres": None,
    },
    "claude-sonnet-5": {
        "fournisseur": "Anthropic", "input_per_1m": 2.00, "output_per_1m": 10.00,
        "remise_cache": 0.90, "cache_automatique": False,
        # TARIF D'INTRODUCTION. Le tarif standard (+50 %) s'applique au 01/09/2026.
        # Une projection annuelle batie sur 2/10 serait fausse des le mois prochain.
        "tarif_provisoire_jusquau": "2026-08-31",
        "tarif_apres": {"input_per_1m": 3.00, "output_per_1m": 15.00},
    },
    "claude-opus-5": {
        "fournisseur": "Anthropic", "input_per_1m": 5.00, "output_per_1m": 25.00,
        "remise_cache": 0.90, "cache_automatique": False,
        "tarif_provisoire_jusquau": None, "tarif_apres": None,
    },
    "gpt-5.6-sol": {
        "fournisseur": "OpenAI", "input_per_1m": 5.00, "output_per_1m": 30.00,
        "remise_cache": 0.90, "cache_automatique": False,
        "tarif_provisoire_jusquau": None, "tarif_apres": None,
    },
}

# Modele de reference pour les projections de l'etape 2.
DEFAULT_MODEL = "mistral-small-4"

# Taille du prefixe constant du prompt, eligible au cache : instruction + liste
# des 9 etiquettes. C'est le levier de cout le plus important du projet, devant
# le choix du modele lui-meme.
#
# GRANULARITE DU CACHE : 64 TOKENS (docs Mistral, verifie le 2026-08-19).
# Le cache travaille par blocs de 64 tokens, et un prompt de moins de 64 tokens
# n'obtient JAMAIS de hit. Deux consequences :
#   - nos ~260 tokens de prefixe representent 4 blocs pleins : on est
#     confortablement au-dessus du seuil, le cache peut s'activer ;
#   - un prefixe raccourci "pour economiser des tokens" serait contre-productif
#     sous 64 tokens, puisqu'il perdrait la remise de 90 % sur la totalite.
# La part reellement servie est MESUREE par appel via
# `usage.prompt_tokens_details.cached_tokens`, jamais estimee depuis cette
# constante : celle-ci ne sert qu'aux projections faites avant mesure.
# MESURE le 2026-08-19 : 252 tokens pour mistral-small, 240 pour ministral-3b
# (ordonnee a l'origine de la regression). L'ecart de 12 tokens entre les deux
# est reel -- surcout de gabarit de conversation -- d'ou une valeur PAR MODELE.
# Cette constante globale conserve la valeur du modele par defaut, pour les
# appelants qui ne precisent pas de modele.
CACHED_PREFIX_TOKENS = 252

CACHED_PREFIX_TOKENS_PAR_MODELE = {
    "mistral-small-4": 252,
    "ministral-3b": 240,
    # Les autres modeles n'ont PAS ete mesures : leur tokenizer differe et le
    # gabarit de conversation aussi. Y appliquer 252 serait une extrapolation.
}

# ---------------------------------------------------------------------------
# CACHE REELLEMENT OBSERVE  (mesure du 2026-08-19, 40 appels)
# ---------------------------------------------------------------------------
# DECOUVERTE DE LA PHASE 2. `metrics.estimate_cost()` modelise implicitement
# une activation du cache a 100 % et une couverture totale du prefixe. La
# mesure dit tout autre chose :
#
#   - la COUVERTURE est PARTIELLE et CONSTANTE : 224 tokens sur un prefixe de
#     252 chez mistral-small (89 %), 128 sur 240 chez ministral (53 %). Ce
#     plafond est structurel et ne bouge jamais.
#   - l'ACTIVATION depend du RECHAUFFEMENT : 55 % sur les 20 premiers appels
#     chez mistral-small, ~98 % une fois le prefixe etabli.
#
# Consequence : l'economie reelle depend du regime. Sur 20 appels a froid elle
# n'est que de 20 %, ce qui a d'abord fait croire a une projection optimiste
# d'un facteur 2. Sur une campagne longue a prefixe constant, elle approche la
# borne theorique -- mais sans jamais l'atteindre, la couverture plafonnant a
# 89 % (mistral-small) et 53 % (ministral).
#
# La valeur theorique reste le PLAFOND atteignable (avec `prompt_cache_key`,
# cf. reports/leviers_optimisation.md) : on ne la remplace pas, on affiche les
# deux bornes cote a cote.
CACHE_MESURE = {
    # DEUX REGIMES, mesures sur 5 passes de 20 appels (2026-08-19).
    #
    # La COUVERTURE est structurellement CONSTANTE : le cache sert toujours
    # exactement 224 tokens chez mistral-small et 128 chez ministral, quel que
    # soit l'appel. C'est une quantification par blocs, pas une variable.
    #
    # Seule l'ACTIVATION varie, et elle MONTE avec le nombre d'appels partageant
    # le prefixe : 55 % sur les 20 premiers appels, ~98 % ensuite. Le cache se
    # rechauffe. Une campagne de 2 000 appels a prefixe constant passe donc
    # l'essentiel de son temps en regime etabli, et c'est ce regime qu'il faut
    # projeter -- pas le demarrage a froid.
    "mistral-small-4": {
        "taux_activation": 0.98,            # regime etabli (passes 3-5)
        "taux_activation_demarrage": 11 / 20,   # 20 premiers appels
        "couverture_prefixe": 224 / 252,    # constante mesuree
        "tokens_caches_par_appel": 220.3,   # regime etabli
        "n_appels": 100,
    },
    "ministral-3b": {
        "taux_activation": 1.00,
        "taux_activation_demarrage": 18 / 20,
        "couverture_prefixe": 128 / 240,
        "tokens_caches_par_appel": 128.0,
        "n_appels": 100,
    },
}
CACHE_MESURE_LE = "2026-08-19"

# --- volumes de tokens ------------------------------------------------------
# ORIGINE : approximation tokens ~ mots x 1,3 appliquee au corpus nettoye. Le
#           facteur 1,3 est une regle empirique pour un texte anglais et un
#           tokenizer BPE ; il n'a PAS ete verifie sur le tokenizer de Mistral.
#           Le corpus est par ailleurs atypique : 85 % des textes contiennent du
#           masquage XXXX, qui se tokenise mal et pourrait faire deriver le
#           facteur reel a la hausse.
# STATUT  : estimation. 257 est la valeur la plus susceptible d'etre fausse de
#           ce bloc.
# ###########  MESURE REELLE -- remplace l'hypothese du 2026-08-11  ###########
# MESURE le 2026-08-19 sur 40 appels reels (20 reclamations x 2 modeles), par
# regression `tokens_in ~ nombre de mots` sur des textes de 24 a 534 mots :
# R2 = 0,9962, pente 1,189 token/mot (et non 1,300 comme suppose).
# La pente mesuree est appliquee a la distribution de longueur des 2 000 lignes
# reelles de l'echantillon d'evaluation, d'ou 245 (et non 257).
#
# PORTEE : TOKENIZER MISTRAL UNIQUEMENT. Cette valeur ne transfere pas a
# Anthropic, OpenAI, Google ni DeepSeek, qui ont leurs propres tokenizers.
# Cf. `FOURNISSEURS_TOKENS_MESURES`.
AVG_COMPLAINT_TOKENS = 245        # texte de la reclamation, tronque a 1 000 mots
AVG_COMPLAINT_TOKENS_MESURE_LE = "2026-08-19"

# Ratio mesure tokens/mot (tokenizer Mistral). L'ancienne regle empirique de
# 1,300 surestimait de 8,5 %.
TOKENS_PAR_MOT_MESURE = 1.189

# Fournisseurs pour lesquels les VOLUMES de tokens sont mesures. Pour tous les
# autres, les volumes restent des hypotheses : un flag `hypotheses` global
# unique ferait passer 8 modeles sur 10 pour mesures.
FOURNISSEURS_TOKENS_MESURES = frozenset({"Mistral"})

# ORIGINE : comptage a la main d'un prompt type non encore ecrit.
# STATUT  : estimation grossiere ; sera exact des que le prompt sera fige.
# MESURE le 2026-08-19 : le prefixe complet (instruction + 9 libelles) vaut 252
# tokens chez mistral-small. La repartition 200/60 entre les deux blocs reste
# indicative -- seul leur TOTAL a ete mesure, l'API ne les facture pas separement.
PROMPT_INSTRUCTION_TOKENS = 194   # role, consigne, format de sortie attendu
PROMPT_LABELS_TOKENS = 58         # liste des 9 etiquettes  (194 + 58 = 252)
# MESURE : 9,7 tokens en moyenne chez mistral-small, 9,2 chez ministral, sur
# une sortie {"label": "..."} . Maximum observe : 13, tres loin des 50 autorises.
AVG_OUTPUT_TOKENS = 10            # l'etiquette seule, en sortie

# ############  FIN DES HYPOTHESES PROVISOIRES  ############

# Troncature appliquee UNIQUEMENT a la construction du prompt LLM (etape 2),
# jamais dans data_prep (arbitrage H.3). Elle ne touche que 0,82 % des textes
# mais plafonne le pire cas a ~1 300 tokens au lieu de 8 200.
LLM_MAX_WORDS = 1000

# Volume de reference pour la projection de cout au client.
DAILY_COMPLAINTS = 1000


# ===========================================================================
# 8. Etape 2 : approche LLM
# ===========================================================================
# Bloc AJOUTE a l'etape 2. Rien au-dessus n'a ete modifie : les constantes de
# l'etape 1 restent telles qu'elles ont ete validees, et `metrics.py` comme
# `data_prep.py` sont figes.

# --- sorties ----------------------------------------------------------------
# TOUT ce qui contient du texte de reclamation vit sous data/llm/, deja couvert
# par la regle `data/` du .gitignore. `reports/` ne recoit que des agregats.
LLM_DIR = DATA_DIR / "llm"


def ensure_llm_dirs() -> None:
    """Cree data/llm/. Fonction distincte de `ensure_dirs()` a dessein : on ne
    modifie pas le comportement d'une fonction appelee par le pipeline fige."""
    LLM_DIR.mkdir(parents=True, exist_ok=True)


# --- etiquette hors referentiel ---------------------------------------------
# Statut rendu par le parsing STRICT quand la reponse du modele n'est pas
# exploitable. Compte comme une ERREUR dans le F1-macro : un modele qui ne
# repond pas au format demande n'a pas classe la reclamation, et le masquer
# derriere une classe de repli embellirait le score.
#
# NE DOIT JAMAIS ENTRER DANS `CLASS_ORDER` : cet ordre sert d'axes aux matrices
# de confusion et de colonnes aux tableaux comparatifs de l'etape 4. Y ajouter
# une dixieme categorie rendrait les figures LLM et ML non superposables.
PARSE_ERROR = "PARSE_ERROR"

assert PARSE_ERROR not in CLASS_ORDER, (
    "PARSE_ERROR ne doit jamais figurer dans CLASS_ORDER : il n'est pas une "
    "classe du referentiel mais un constat d'echec de format."
)

# --- parametres d'appel -----------------------------------------------------
LLM_TEMPERATURE = 0.0

# La sortie attendue ({"label": "..."}) fait une dizaine de tokens. Un
# depassement de ce plafond est EN SOI un signal de non-conformite au format,
# a journaliser comme tel plutot qu'a corriger en relevant la limite.
LLM_MAX_TOKENS = 50

# Tier gratuit Mistral : ~1 requete par seconde. Configurable, car une cle
# payante leve cette contrainte et diviserait d'autant la duree de campagne.
LLM_REQUETES_PAR_SECONDE = 1.0
LLM_MAX_TENTATIVES = 5

# Garde-fou de depense. La campagne s'ARRETE (sans perdre le deja-ecrit, grace
# a l'ecriture incrementale) si le cout reel cumule depasse ce plafond.
LLM_BUDGET_MAX_USD = 1.0

# --- identifiants de modele -------------------------------------------------
# ATTENTION : DEUX ESPACES DE NOMS DISTINCTS, VOLONTAIREMENT NON FUSIONNES.
#
#   - les CLES de `MODELS_PRICING` ("mistral-small-4", "ministral-3b") servent a
#     la TARIFICATION ;
#   - les identifiants acceptes par l'API Mistral servent aux APPELS.
#
# Rien ne garantit qu'ils coincident, et le seul appel verifie a ce jour
# (`scratch/test_api.py`, `.env.example`) utilise "mistral-small-latest", qui
# n'est PAS une cle de MODELS_PRICING. L'ecart est donc AVERE pour au moins un
# modele.
#
# Ce dictionnaire n'est PAS une correction : c'est la liste des candidats a
# VERIFIER contre l'API en debut d'etape 2. Tant que `verifie` vaut False,
# l'identifiant n'a pas ete confronte a l'API et ne doit pas etre presente
# comme acquis.
# VERIFIE le 2026-08-19 par `client.models.list()`. Les deux alias existent, et
# l'API expose elle-meme leur cible et leur description : la correspondance
# ci-dessous est CONSTATEE, pas deduite d'une ressemblance de nom.
#
# ON APPELLE L'ALIAS, PAS LA VERSION FIGEE. C'est ce qu'un client utiliserait en
# production, et l'evaluation doit porter sur ce qu'obtient quelqu'un qui appelle
# l'API normalement. Mais la version REELLEMENT servie est consignee a chaque
# appel (`modele_resolu`, lu dans la reponse) : si Mistral fait pointer l'alias
# vers une autre version en cours d'etape 2, les premieres mesures deviendraient
# incomparables aux suivantes sans que rien ne le signale.
# `llm_eval.verifie_derive_version()` compare les versions entre campagnes.
LLM_MODELES_A_COMPARER = {
    "mistral-small-4": {
        "id_api_candidat": "mistral-small-latest",
        "verifie": True,
        "verifie_le": "2026-08-19",
        "version_resolue": "mistral-small-2603",
        "description_api": "Mistral Small 4.",
        "note": "l'API confirme que l'alias pointe sur Mistral Small 4 (v26.03), "
                "soit exactement ce que MODELS_PRICING nomme 'mistral-small-4'.",
    },
    "ministral-3b": {
        "id_api_candidat": "ministral-3b-latest",
        "verifie": True,
        "verifie_le": "2026-08-19",
        "version_resolue": "ministral-3b-2512",
        "description_api": "Ministral 3 (a.k.a. Tinystral) 3B Instruct.",
        "note": "candidat suppose par symetrie a la phase 1, CONFIRME depuis : "
                "l'alias existe et pointe sur Ministral 3 3B (v25.12).",
    },
}

# --- jeu d'iteration de la phase 2 ------------------------------------------
# 20 lignes du TRAIN, tirage stratifie deterministe, au moins 2 par classe.
# On itere sur le train et JAMAIS sur `test_sample_2000.csv` : ajuster un prompt
# en regardant ses erreurs sur l'echantillon d'evaluation revient a l'optimiser
# pour ces lignes precises et gonfle le score final -- meme logique que le
# dedoublonnage avant le split.
LLM_ITERATION_SIZE = 20
LLM_ITERATION_MIN_PER_CLASS = 2
LLM_ITERATION_FILE = PROCESSED_DIR / f"train_iteration_{LLM_ITERATION_SIZE}.csv"

# --- jeu de SELECTION des variantes de prompt (phase 3) ---------------------
# 20 exemples ne departagent pas des variantes : l'IC a 95 % du F1-macro y fait
# +/- 20 points (mesure du 2026-08-19). A 200 lignes il tombe a +/- 6,3 points.
#
# Tire du TRAIN, avec une graine DISTINCTE de celle des 20, sans recouvrement ni
# avec `train_iteration_20.csv` ni avec `test_sample_2000.csv`. Le jeu des 20
# reste le support de la lecture QUALITATIVE des erreurs ; celui-ci sert au
# chiffre.
LLM_SELECTION_SIZE = 200
LLM_SELECTION_MIN_PER_CLASS = 12
LLM_SELECTION_SEED = 1337          # distincte de RANDOM_SEED (42), a dessein
LLM_SELECTION_FILE = PROCESSED_DIR / f"train_selection_{LLM_SELECTION_SIZE}.csv"

# Exemples du few-shot (variante v6). Tires du train, HORS des deux jeux
# ci-dessus : un exemple qui figurerait dans le jeu de selection donnerait au
# few-shot une reponse qu'il a deja vue, et le gain mesure serait un artefact.
LLM_FEWSHOT_PAR_CLASSE = 1
LLM_FEWSHOT_SEED = 7
LLM_FEWSHOT_FILE = PROCESSED_DIR / "train_fewshot_examples.csv"

# --- SECOND jeu de SELECTION (etape 2, extension -- lot 1) -------------------
# DIMENSIONNE, et non choisi : 1 835 lignes est le n qui donne 80 % de chances
# de detecter un gain d'exactitude de 0,02 sous McNemar exact, correction de
# Holm, famille de TROIS comparaisons, avec un taux de discordance de 6,75 %
# (moyenne des b+c observes en phase 3 hors v3). A n = 200 la puissance pour ce
# meme ecart valait 0,053 -- a peine au-dessus du risque de premiere espece.
#
# ALLOCATION PROPORTIONNELLE AU TRAIN, SANS PLANCHER, contrairement aux 200.
# Le plancher de 12 servait a garantir un effectif minimal aux classes rares sur
# un petit jeu ; a 1 835 lignes la classe la plus rare est deja largement
# pourvue par le prorata, et le plancher ne ferait que deformer la population
# sans contrepartie. Arrondi par la methode des PLUS FORTS RESTES
# (`data_prep._repartition_plus_forts_restes`), qui totalise exactement n.
#
# Graine DISTINCTE de 1337 (jeu des 200) et de 7 (few-shot v6) : reutiliser une
# graine sur le meme corpus trie de la meme facon reselectionnerait
# preferentiellement les memes lignes.
LLM_SELECTION2_SIZE = 1835
LLM_SELECTION2_SEED = 2027
LLM_SELECTION2_FILE = PROCESSED_DIR / "train_selection_1835.csv"

# --- exemples few-shot de SECONDE generation (variantes v7 et v8) ------------
# Deux bras qui ne different QUE par le critere de retenue, pour que l'ecart
# entre eux soit attribuable :
#   v7_fewshot_court  : premier candidat tire de chaque classe, aucun autre
#                       critere. Bras "plafond de longueur seul".
#   v8_fewshot_filtre : premier candidat de chaque classe que mistral-small-4,
#                       interroge en zero-shot avec le prefixe v1_zeroshot
#                       exact a temperature 0, etiquette correctement et de
#                       facon IDENTIQUE sur 3 passes.
# Critere de v8 fige AVANT tirage : un candidat que le modele classe mal ou
# instablement sans aide est un candidat dont l'etiquette ne se deduit pas du
# texte seul.
LLM_FEWSHOT2_SEED = 4021
LLM_FEWSHOT2_CANDIDATS_PAR_CLASSE = 5
LLM_FEWSHOT2_PASSES_JUGE = 3
LLM_FEWSHOT2_CANDIDATS_FILE = PROCESSED_DIR / "train_fewshot2_candidats.csv"
LLM_FEWSHOT_V7_FILE = PROCESSED_DIR / "train_fewshot_v7.csv"
LLM_FEWSHOT_V8_FILE = PROCESSED_DIR / "train_fewshot_v8.csv"

# Plafond de longueur des exemples v7/v8, EN MOTS. Determine par la mesure
# (section 3 du lot 1) et non par decret : le prefixe de v6 pese 2 519 mots
# contre 180 pour v1, et `llm_prompts.tronque()` ne s'applique JAMAIS au
# prefixe -- seulement au message `user`.
#
# 120 RETENU le 2026-09-03. v7 est le bras "plafond de longueur seul" : sa
# fonction est d'isoler l'effet de la longueur PAR CONTRASTE avec v6. Le
# contraste doit donc etre franc. Mesure sur les 283 220 lignes du train
# restant, prefixe pire cas (9 exemples exactement au plafond) :
#     plafond 200 -> 2 689 tokens, soit -17,5 % contre v6 (3 261)
#     plafond 120 -> 1 833 tokens, soit -44,0 % contre v6
# A -17,5 % un ecart mesure ne serait pas attribuable a la longueur.
#
# LIMITE ASSUMEE, a consigner dans tout rapport qui exploite v7 ou v8 : a
# 120 mots, 7 classes sur 9 voient leurs exemples tires SOUS leur mediane de
# longueur, et Mortgage (mediane 215 mots) sous son premier tiers. 43,5 % de la
# population est eligible. Un exemple few-shot n'a pas vocation a etre
# representatif de la longueur typique d'une reclamation, mais le biais est
# reel et il est ecrit ici plutot que decouvert plus tard.
LLM_FEWSHOT2_MAX_WORDS = 120

# --- styles de prompt -------------------------------------------------------
# Un style n'est FIGE qu'apres mise au point sur le jeu d'iteration. Le runner
# REFUSE de demarrer sur `test_sample_2000.csv` avec un style absent de cet
# ensemble : c'est le garde-fou qui empeche de depenser 2 000 appels sur un
# prompt encore en cours de reglage.
# FIGE le 2026-08-19 a l'issue de la phase 3. Cinq modifications independantes
# de v1 (regle produit, definitions, libelles officiels, francais, few-shot) ont
# ete mesurees sur 200 lignes du train : AUCUNE ne produit de gain distinguable
# au test de McNemar apparie sous correction de Holm-Bonferroni. La variante la
# plus simple l'emporte donc a performance non distinguable.
#
# v1_zeroshot ne doit plus etre modifiee jusqu'a la fin de l'etape 2 : son
# prefixe est epingle par empreinte dans tests/test_llm_prompts.py.
LLM_STYLES_FIGES: set[str] = {"v1_zeroshot"}
