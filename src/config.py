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
PRICING_CHECKED_ON = "2026-08-11"

# Tarifs en DOLLARS PAR MILLION DE TOKENS.
#
# `remise_cache` : reduction appliquee a la part d'entree servie depuis le cache
#     de prefixe. Le prompt de l'etape 2 aura un prefixe constant (instruction +
#     liste des 9 etiquettes, ~260 tokens) eligible chez tous les fournisseurs.
#     None = non verifie pour ce fournisseur ; le cache n'est alors PAS modelise
#     plutot que suppose (cf. gemini-3.5-flash).
# `cache_automatique` : True si le fournisseur cache sans declaration explicite.
# `tarif_provisoire_jusquau` / `tarif_apres` : tarif promotionnel a duree limitee.
MODELS_PRICING = {
    "ministral-3b": {
        "fournisseur": "Mistral", "input_per_1m": 0.10, "output_per_1m": 0.10,
        "remise_cache": 0.90, "cache_automatique": False,
        "tarif_provisoire_jusquau": None, "tarif_apres": None,
    },
    "deepseek-v4-flash": {
        "fournisseur": "DeepSeek", "input_per_1m": 0.14, "output_per_1m": 0.28,
        "remise_cache": 0.98, "cache_automatique": True,
        "tarif_provisoire_jusquau": None, "tarif_apres": None,
    },
    "mistral-small-4": {
        "fournisseur": "Mistral", "input_per_1m": 0.15, "output_per_1m": 0.60,
        "remise_cache": 0.90, "cache_automatique": False,
        "tarif_provisoire_jusquau": None, "tarif_apres": None,
    },
    "gpt-5.6-luna": {
        "fournisseur": "OpenAI", "input_per_1m": 0.20, "output_per_1m": 1.20,
        "remise_cache": 0.90, "cache_automatique": False,
        "tarif_provisoire_jusquau": None, "tarif_apres": None,
    },
    "mistral-large-3": {
        "fournisseur": "Mistral", "input_per_1m": 0.50, "output_per_1m": 1.50,
        "remise_cache": 0.90, "cache_automatique": False,
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
CACHED_PREFIX_TOKENS = 260

# --- volumes de tokens ------------------------------------------------------
# ORIGINE : approximation tokens ~ mots x 1,3 appliquee au corpus nettoye. Le
#           facteur 1,3 est une regle empirique pour un texte anglais et un
#           tokenizer BPE ; il n'a PAS ete verifie sur le tokenizer de Mistral.
#           Le corpus est par ailleurs atypique : 85 % des textes contiennent du
#           masquage XXXX, qui se tokenise mal et pourrait faire deriver le
#           facteur reel a la hausse.
# STATUT  : estimation. 257 est la valeur la plus susceptible d'etre fausse de
#           ce bloc.
AVG_COMPLAINT_TOKENS = 257        # texte de la reclamation, tronque a 1 000 mots

# ORIGINE : comptage a la main d'un prompt type non encore ecrit.
# STATUT  : estimation grossiere ; sera exact des que le prompt sera fige.
PROMPT_INSTRUCTION_TOKENS = 200   # role, consigne, format de sortie attendu
PROMPT_LABELS_TOKENS = 60         # liste des 9 etiquettes
AVG_OUTPUT_TOKENS = 8             # l'etiquette seule, en sortie

# ############  FIN DES HYPOTHESES PROVISOIRES  ############

# Troncature appliquee UNIQUEMENT a la construction du prompt LLM (etape 2),
# jamais dans data_prep (arbitrage H.3). Elle ne touche que 0,82 % des textes
# mais plafonne le pire cas a ~1 300 tokens au lieu de 8 200.
LLM_MAX_WORDS = 1000

# Volume de reference pour la projection de cout au client.
DAILY_COMPLAINTS = 1000
