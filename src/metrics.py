"""Metriques partagees par l'etape 2 (LLM) et l'etape 3 (ML classique).

============================================================================
REGLE DE GEL
----------------------------------------------------------------------------
Ce module est gele depuis la validation de l'etape 1. Seules les modifications
SANS effet sur une valeur calculee sont autorisees (docstrings, commentaires,
annotations de type). Toute autre modification exige une revalidation croisee
des etapes 2 et 3.

Cette regle est OPPOSABLE : `tests/test_metrics_gel.py` rejoue un jeu fige et
compare les sorties a des valeurs de reference. Une modification de logique
deguisee en correction de commentaire y echoue immediatement.
============================================================================

Les deux approches doivent etre mesurees avec exactement le meme code, sinon la
comparaison finale ne vaut rien.

METRIQUE PRINCIPALE : F1-macro
------------------------------
Le corpus presente un ratio majoritaire / minoritaire de 18,9 apres fusion des
libelles. L'accuracy y est trompeuse : un modele qui predirait systematiquement
"Credit reporting" obtiendrait deja 30 % d'accuracy sans rien comprendre. Le
F1-macro pondere les 9 classes a egalite, ce qui correspond au besoin metier :
une reclamation mal routee coute la meme chose au client quelle que soit sa
categorie.

LATENCE : p95, pas la moyenne
-----------------------------
La moyenne est ecrasee par la masse des appels rapides et masque la queue de
distribution — precisement ce que l'utilisateur percoit comme "le service rame".
Le p95 repond a la question qui compte : combien de temps attendent les 5 % les
moins bien servis.

PONDERATION
-----------
L'echantillon d'evaluation LLM est stratifie A PLANCHER (strategie B3), donc
non representatif de la distribution du test. Ses poids de Horvitz-Thompson
ramenent les metriques a la population de reference. Deux filets, dans cet
ordre :

  1. `sample_weight` est un argument nomme SANS VALEUR PAR DEFAUT dans
     `evaluate()` et `plot_confusion_matrix()`. L'omettre leve un `TypeError`
     immediat et inconditionnel. Passer `None` reste possible, mais devient une
     declaration d'intention explicite plutot qu'un defaut subi.
  2. Si `source` est fourni, qu'il porte une colonne de poids et que
     `sample_weight` vaut `None`, une `ValueError` est levee. C'est le filet qui
     rattrape un `None` passe a tort.

Dans les deux cas une exception, jamais un avertissement : un warning se perd
dans la sortie d'un notebook, et l'erreur induite atteint +0,29 sur la precision
d'une classe.
"""

from __future__ import annotations

import time
import warnings
from contextlib import ContextDecorator
from datetime import date
from typing import Any, Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from . import config as cfg

# Rampe sequentielle bleue, une seule teinte, du clair au fonce.
# Une echelle sequentielle encode une magnitude : jamais un arc-en-ciel, qui
# suggererait des categories la ou il n'y a qu'un continuum.
_BLEU = ["#f4f8fe", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
         "#256abf", "#184f95", "#0d366b"]
CMAP_CONFUSION = LinearSegmentedColormap.from_list("zen_bleu", _BLEU)

_SURFACE, _ENCRE, _ENCRE2, _ATTENUE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"


# ===========================================================================
# Chronometre de latence
# ===========================================================================
class Timer(ContextDecorator):
    """Chronometre reutilisable : une mesure par appel individuel.

    Chaque entree/sortie du bloc `with` ajoute UNE latence a `latencies`. On ne
    mesure JAMAIS un temps total que l'on diviserait par le nombre d'appels :
    cette moyenne masquerait la dispersion, or c'est precisement la dispersion
    qui interesse le client (cf. le choix du p95 plutot que de la moyenne).

    EXECUTION PARALLELISEE
    ----------------------
    Si l'etape 2 parallelise les appels a l'API (plusieurs requetes en vol
    simultanement), la latence enregistree ici reste celle de l'APPEL UNITAIRE,
    mesuree de son envoi a sa reponse. C'est volontaire : c'est la latence
    percue par un utilisateur qui soumet une reclamation, et c'est elle qui doit
    figurer dans la recommandation. Le debit global du batch est une grandeur
    differente, a mesurer separement si besoin — ne pas confondre les deux.

    Exemple
    -------
    >>> chrono = Timer()
    >>> for texte in textes:
    ...     with chrono:
    ...         prediction = appelle_le_llm(texte)
    >>> resultats = evaluate(y_true, y_pred, latencies=chrono.latencies)
    """

    def __init__(self) -> None:
        self.latencies: list[float] = []
        self._debut: float | None = None

    def __enter__(self) -> "Timer":
        self._debut = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> bool:
        if self._debut is None:  # pragma: no cover - garde defensive
            raise RuntimeError("Timer.__exit__ appele sans __enter__.")
        self.latencies.append(time.perf_counter() - self._debut)
        self._debut = None
        return False

    def reset(self) -> None:
        self.latencies.clear()

    @property
    def n(self) -> int:
        return len(self.latencies)

    def summary(self) -> dict[str, float]:
        return latency_stats(self.latencies)


def latency_stats(latencies: Sequence[float] | None) -> dict[str, float]:
    """Statistiques de latence, en secondes. Le p95 est la valeur a retenir."""
    if latencies is None or len(latencies) == 0:
        return {}
    a = np.asarray(latencies, dtype=float)
    return {
        "latence_n": int(a.size),
        "latence_moyenne_s": float(a.mean()),
        "latence_p50_s": float(np.percentile(a, 50)),
        "latence_p95_s": float(np.percentile(a, 95)),
        "latence_max_s": float(a.max()),
    }


# ===========================================================================
# Evaluation
# ===========================================================================
def _verifie_ponderation(source: Any, sample_weight: Any) -> None:
    """Garde-fou : leve si des poids existent mais ne sont pas utilises.

    Une exception, pas un avertissement. L'erreur induite par un oubli est de
    +0,022 sur le F1-macro et de +0,29 sur la precision de la plus petite
    classe : elle changerait la conclusion du projet.
    """
    if sample_weight is not None or source is None:
        return
    colonnes = getattr(source, "columns", None)
    if colonnes is not None and cfg.WEIGHT_COL in colonnes:
        raise ValueError(
            f"Le DataFrame source contient la colonne '{cfg.WEIGHT_COL}' mais "
            f"aucun `sample_weight` n'a ete fourni.\n"
            f"Cet echantillon est stratifie a plancher : sans ponderation, le "
            f"F1-macro est surestime de +0,022 et la precision de "
            f"'Vehicle loan or lease' de +0,29.\n"
            f"Utiliser :  df, weights = load_eval_sample()  puis  "
            f"evaluate(df['{cfg.LABEL_COL}'], y_pred, sample_weight=weights)"
        )


def evaluate(
    y_true: Sequence,
    y_pred: Sequence,
    latencies: Sequence[float] | None = None,
    labels: Sequence[str] | None = None,
    *,
    sample_weight: Sequence[float] | None,
    source: pd.DataFrame | None = None,
    strata: str | Sequence | None = "auto",
    with_ci: bool = True,
    n_boot: int = cfg.BOOTSTRAP_N,
) -> dict[str, Any]:
    """Evalue une serie de predictions et retourne toutes les metriques.

    Parameters
    ----------
    y_true, y_pred : sequences d'etiquettes (libelles courts de CLASS_ORDER).
    latencies : latences par appel, en secondes (cf. `Timer`). Optionnel.
    labels : ordre des classes. Par defaut `config.CLASS_ORDER`, fige pour que
        toutes les matrices de confusion du projet se lisent pareil.
    sample_weight : poids de Horvitz-Thompson. **Argument nomme SANS VALEUR PAR
        DEFAUT** : il doit etre fourni explicitement a chaque appel, quitte a
        passer `None`. Un oubli leve un `TypeError` immediat et inconditionnel,
        au lieu de produire silencieusement une metrique fausse. Passer `None`
        est une declaration d'intention ("cette evaluation n'est pas ponderee"),
        pas un defaut subi.
    source : DataFrame d'ou proviennent les predictions. S'il contient une
        colonne de poids et que `sample_weight` vaut `None`, une exception est
        levee : c'est le second filet, qui rattrape un `None` passe a tort.
    strata : plan de rééchantillonnage du bootstrap. C'est un CHOIX
        STATISTIQUE, pas un detail d'implementation, donc il est expose :
          - `"auto"` (defaut) : `y_true` si des poids sont fournis, `None`
            sinon. La presence de poids signale un plan de sondage stratifie a
            effectifs fixes ; le bootstrap doit alors reproduire ce plan.
          - `None` : rééchantillonnage uniforme sur tout l'echantillon.
          - une sequence : strates explicites, alignees sur `y_true`.
        Le mode retenu est journalise dans la cle `bootstrap_strata`.
    with_ci : ajoute l'intervalle de confiance bootstrap du F1-macro. Le passer
        a `False` sur le test complet : sur 70 863 lignes l'intervalle fait
        ~0,005 de large, il n'apprend rien et coute plusieurs minutes. L'IC a du
        sens sur les 2 000 lignes de l'echantillon d'evaluation, pas sur la
        population.

    Returns
    -------
    dict contenant notamment `accuracy`, `f1_macro`, `f1_weighted`,
    `precision_macro`, `recall_macro`, `classification_report`,
    `confusion_matrix`, `weighted`, `bootstrap_strata`, et les statistiques de
    latence si fournies.
    """
    _verifie_ponderation(source, sample_weight)

    labels = list(labels) if labels is not None else list(cfg.CLASS_ORDER)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true ({len(y_true)}) et y_pred ({len(y_pred)}) "
                         f"n'ont pas la meme longueur.")
    w = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
    if w is not None and len(w) != len(y_true):
        raise ValueError(f"sample_weight ({len(w)}) n'a pas la longueur de "
                         f"y_true ({len(y_true)}).")

    commun = dict(labels=labels, sample_weight=w, zero_division=0)
    resultats: dict[str, Any] = {
        "n": int(len(y_true)),
        "weighted": w is not None,
        "accuracy": float(accuracy_score(y_true, y_pred, sample_weight=w)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", **commun)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", **commun)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", **commun)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", **commun)),
        "classification_report": classification_report(
            y_true, y_pred, labels=labels, sample_weight=w,
            zero_division=0, output_dict=True),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=labels, sample_weight=w).tolist(),
        "labels": labels,
    }

    # --- plan de rééchantillonnage du bootstrap ----------------------------
    if isinstance(strata, str):
        if strata != "auto":
            raise ValueError(f"strata='{strata}' inconnu : attendu 'auto', None "
                             f"ou une sequence de strates.")
        # La presence de poids signale un plan de sondage stratifie a effectifs
        # fixes par classe : le bootstrap doit reproduire ce plan.
        strates = y_true if w is not None else None
        mode = "y_true (auto)" if w is not None else "aucune (auto)"
    elif strata is None:
        strates, mode = None, "aucune"
    else:
        strates = np.asarray(strata)
        if len(strates) != len(y_true):
            raise ValueError(f"strata ({len(strates)}) n'a pas la longueur de "
                             f"y_true ({len(y_true)}).")
        mode = "explicite"
    resultats["bootstrap_strata"] = mode

    if with_ci:
        valeur, bas, haut = bootstrap_ci(
            y_true, y_pred, sample_weight=w, strata=strates, n_boot=n_boot)
        resultats["f1_macro_ci"] = {"valeur": valeur, "bas": bas, "haut": haut,
                                    "niveau": 1 - cfg.BOOTSTRAP_ALPHA,
                                    "strata": mode}

    resultats.update(latency_stats(latencies))
    return resultats


def f1_macro(y_true, y_pred, sample_weight=None) -> float:
    """F1-macro, signature compatible avec `bootstrap_ci(metric_fn=...)`."""
    return float(f1_score(y_true, y_pred, average="macro",
                          labels=list(cfg.CLASS_ORDER),
                          sample_weight=sample_weight, zero_division=0))


def bootstrap_ci(
    y_true: Sequence,
    y_pred: Sequence,
    metric_fn: Callable[..., float] = f1_macro,
    n_boot: int = cfg.BOOTSTRAP_N,
    alpha: float = cfg.BOOTSTRAP_ALPHA,
    *,
    sample_weight: Sequence[float] | None = None,
    strata: Sequence | None = None,
    random_state: int = cfg.RANDOM_SEED,
) -> tuple[float, float, float]:
    """Intervalle de confiance bootstrap par percentiles.

    Returns
    -------
    (valeur, borne_basse, borne_haute)
        `valeur` est la metrique calculee sur l'echantillon complet ; les bornes
        sont les percentiles alpha/2 et 1-alpha/2 des rééchantillonnages.

    REECHANTILLONNAGE INTRA-STRATES
    -------------------------------
    Quand `strata` est fourni (typiquement `y_true`), le tirage se fait AVEC
    REMISE A L'INTERIEUR de chaque classe, en conservant l'effectif de chacune.
    C'est indispensable ici : un bootstrap uniforme ferait varier le nombre
    d'items des classes rares d'un tirage a l'autre, ce qui ajouterait une
    variance qui n'existe pas dans le plan de sondage reel (les n_c sont fixes
    par construction) et sous-estimerait la precision reelle sur ces classes.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    w = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
    rng = np.random.default_rng(random_state)

    valeur = float(metric_fn(y_true, y_pred, sample_weight=w))

    if strata is None:
        groupes = [np.arange(len(y_true))]
    else:
        strates = np.asarray(strata)
        groupes = [np.flatnonzero(strates == s) for s in pd.unique(strates)]

    tirages = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(g, size=g.size, replace=True)
                              for g in groupes])
        tirages[b] = metric_fn(y_true[idx], y_pred[idx],
                               sample_weight=None if w is None else w[idx])

    bas, haut = np.percentile(tirages, [alpha / 2 * 100, (1 - alpha / 2) * 100])
    return valeur, float(bas), float(haut)


# ===========================================================================
# Cout
# ===========================================================================
def estimate_cost(
    n_predictions: int,
    model: str = cfg.DEFAULT_MODEL,
    avg_tokens: float | None = None,
    *,
    avg_output_tokens: float = cfg.AVG_OUTPUT_TOKENS,
    prompt_overhead_tokens: float | None = None,
    cached_prefix_tokens: float = 0,
    date_reference: date | None = None,
) -> dict[str, Any]:
    """Projette le cout d'une campagne de predictions pour un modele donne.

    Parameters
    ----------
    n_predictions : nombre de reclamations a classer.
    model : cle de `config.MODELS_PRICING`. Defaut `config.DEFAULT_MODEL`.
    avg_tokens : tokens d'entree moyens par appel, HORS surcout de prompt. Par
        defaut `config.AVG_COMPLAINT_TOKENS`. Aucun chiffre n'est recopie ici :
        cette valeur est passee de 257 (hypothese mots x 1,3) a 245 (mesure du
        2026-08-19 sur le tokenizer Mistral), et une docstring portant le nombre
        en dur aurait menti des ce jour-la.
    prompt_overhead_tokens : instruction + liste des 9 etiquettes. Par defaut
        260 tokens (200 + 60), hypothese documentee dans le diagnostic D.3.
    cached_prefix_tokens : nombre de tokens du prefixe constant servi depuis le
        cache. Le passer a `config.CACHED_PREFIX_TOKENS` (260) modelise le cache
        de prefixe. **C'est le levier de cout le plus important du projet**,
        devant le choix du modele : la part cachee est facturee avec la remise
        du fournisseur (90 % chez Mistral, Anthropic et OpenAI 5.4-5.6 ; 98 %
        chez DeepSeek, qui cache automatiquement).
        Simplification assumee : le surcout d'ecriture du cache a la premiere
        requete est neglige, car amorti sur des milliers d'appels il pese moins
        de 0,1 %.
    date_reference : date servant a verifier l'expiration d'un tarif
        promotionnel. Defaut : aujourd'hui. Parametrable pour les tests.

    Returns
    -------
    dict avec le cout total, le cout pour 1 000 predictions, la projection
    annuelle au volume de reference du client (1 000 reclamations par jour), et
    les cles de statut **`hypotheses`**, **`model`**, **`tarif_verifie_le`** et
    **`avertissements`**.

    `hypotheses` vaut `True` tant que les valeurs par defaut de `config.py` sont
    utilisees — c'est-a-dire tant qu'aucun chiffre ne provient d'une mesure
    reelle — et `False` des que l'appelant fournit a la fois `avg_tokens` et
    `prompt_overhead_tokens` mesures. Aucun cout ne doit etre presente au client
    sans que ce statut soit indique : a l'etape 2, le champ `usage` des reponses
    de l'API donne les comptes de tokens exacts (cf. reports/limites.md, § 4).

    Warns
    -----
    UserWarning
        Si le modele choisi beneficie d'un tarif promotionnel deja expire a
        `date_reference`. Le cout est alors calcule au tarif standard.

    Note : le cout n'est PAS le facteur discriminant de ce projet. L'ecart entre
    le modele le moins cher et le plus cher du catalogue va de ~19 $ a ~1 031 $
    par an pour 1 000 reclamations quotidiennes — deux ordres de grandeur, mais
    tous negligeables devant le cout du traitement manuel. C'est la latence
    (une vingtaine d'heures pour les 70 863 lignes du test complet) qui
    contraint reellement l'approche LLM.
    """
    if model not in cfg.MODELS_PRICING:
        raise KeyError(f"modele '{model}' inconnu. Disponibles : "
                       f"{sorted(cfg.MODELS_PRICING)}")
    tarif = dict(cfg.MODELS_PRICING[model])
    avertissements: list[str] = []

    # --- expiration d'un tarif promotionnel --------------------------------
    aujourd_hui = date_reference or date.today()
    fin = tarif.get("tarif_provisoire_jusquau")
    if fin:
        fin_date = date.fromisoformat(fin)
        if aujourd_hui > fin_date:
            msg = (f"'{model}' : le tarif d'introduction "
                   f"({tarif['input_per_1m']}/{tarif['output_per_1m']} $ par M de "
                   f"tokens) a expire le {fin}. Calcul effectue au tarif standard "
                   f"({tarif['tarif_apres']['input_per_1m']}/"
                   f"{tarif['tarif_apres']['output_per_1m']}).")
            warnings.warn(msg, UserWarning, stacklevel=2)
            avertissements.append(msg)
            tarif.update(tarif["tarif_apres"])
        else:
            msg = (f"'{model}' : tarif d'introduction, valable jusqu'au {fin} "
                   f"puis {tarif['tarif_apres']['input_per_1m']}/"
                   f"{tarif['tarif_apres']['output_per_1m']} $ par M de tokens. "
                   f"Ne pas batir de projection annuelle dessus.")
            avertissements.append(msg)

    tokens_mesures = avg_tokens is not None and prompt_overhead_tokens is not None
    avg_tokens = cfg.AVG_COMPLAINT_TOKENS if avg_tokens is None else avg_tokens
    if prompt_overhead_tokens is None:
        prompt_overhead_tokens = cfg.PROMPT_INSTRUCTION_TOKENS + cfg.PROMPT_LABELS_TOKENS

    tokens_par_appel = avg_tokens + prompt_overhead_tokens
    if cached_prefix_tokens > tokens_par_appel:
        raise ValueError(f"cached_prefix_tokens ({cached_prefix_tokens}) depasse "
                         f"le total d'entree par appel ({tokens_par_appel}).")

    # --- part cachee / part pleine tarif -----------------------------------
    remise = tarif["remise_cache"]
    cache_modelise = bool(cached_prefix_tokens) and remise is not None
    if cached_prefix_tokens and remise is None:
        msg = (f"'{model}' : remise de cache NON RELEVEE de notre cote pour ce "
               f"fournisseur — ce n'est pas une absence de cache chez lui. Le cache "
               f"n'est pas modelise, donc le cout affiche est MAJORE et n'est pas "
               f"comparable en l'etat aux modeles dont la remise est connue.")
        avertissements.append(msg)
    caches = cached_prefix_tokens if cache_modelise else 0
    pleins = tokens_par_appel - caches

    tokens_in = tokens_par_appel * n_predictions
    tokens_out = avg_output_tokens * n_predictions
    cout_in = (pleins * n_predictions / 1e6 * tarif["input_per_1m"]
               + caches * n_predictions / 1e6 * tarif["input_per_1m"] * (1 - (remise or 0)))
    cout_out = tokens_out / 1e6 * tarif["output_per_1m"]
    total = cout_in + cout_out
    par_pred = total / n_predictions if n_predictions else 0.0

    return {
        "n_predictions": int(n_predictions),
        "model": model,
        "fournisseur": tarif["fournisseur"],
        "input_per_1m": tarif["input_per_1m"],
        "output_per_1m": tarif["output_per_1m"],
        # True = chiffres issus des hypotheses provisoires de config.py, pas de
        # mesures reelles. A afficher tel quel a cote de tout cout presente.
        "hypotheses": not tokens_mesures,
        "tarif_verifie_le": cfg.PRICING_CHECKED_ON,
        "cache_modelise": cache_modelise,
        "tokens_caches_par_appel": float(caches),
        "remise_cache": remise,
        "avertissements": avertissements,
        "tokens_entree_par_appel": float(tokens_par_appel),
        "tokens_entree_total": float(tokens_in),
        "tokens_sortie_total": float(tokens_out),
        "cout_entree_usd": round(cout_in, 6),
        "cout_sortie_usd": round(cout_out, 6),
        "cout_total_usd": round(total, 6),
        "cout_par_prediction_usd": round(par_pred, 8),
        "cout_pour_1000_predictions_usd": round(par_pred * 1000, 6),
        "cout_annuel_usd": round(par_pred * cfg.DAILY_COMPLAINTS * 365, 2),
    }


def compare_models_cost(
    n_predictions: int,
    models: Sequence[str] | None = None,
    *,
    cached_prefix_tokens: float = cfg.CACHED_PREFIX_TOKENS,
    date_reference: date | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Tableau comparatif du cout de tous les modeles du catalogue.

    Support de la restitution sur le cout d'exploitation. Chaque modele est chiffre
    deux fois — sans cache de prefixe et avec — pour montrer que **le cache est
    un levier plus important que le choix du modele** sur une tache ou le
    prefixe constant (260 tokens) pese la moitie de l'entree.

    Parameters
    ----------
    n_predictions : volume de la campagne a chiffrer.
    models : sous-ensemble a comparer. Par defaut tout `config.MODELS_PRICING`,
        trie du moins cher au plus cher.
    cached_prefix_tokens : taille du prefixe cachable, pour la colonne « avec
        cache ». Defaut `config.CACHED_PREFIX_TOKENS`.

    Returns
    -------
    DataFrame indexe par modele, trie par cout croissant.
    """
    models = list(models) if models is not None else list(cfg.MODELS_PRICING)
    lignes = {}
    for m in models:
        with warnings.catch_warnings():
            # Les avertissements de tarif expire sont recuperes dans la colonne
            # `note` : on ne veut pas dix warnings a l'ecran pour un tableau.
            warnings.simplefilter("ignore", UserWarning)
            sans = estimate_cost(n_predictions, m, cached_prefix_tokens=0,
                                 date_reference=date_reference, **kwargs)
            avec = estimate_cost(n_predictions, m,
                                 cached_prefix_tokens=cached_prefix_tokens,
                                 date_reference=date_reference, **kwargs)
        economie = (1 - avec["cout_total_usd"] / sans["cout_total_usd"]) * 100 \
            if sans["cout_total_usd"] else 0.0
        lignes[m] = {
            "fournisseur": sans["fournisseur"],
            "$/M entrée": sans["input_per_1m"],
            "$/M sortie": sans["output_per_1m"],
            f"coût {n_predictions} préd. ($)": round(sans["cout_total_usd"], 4),
            "avec cache ($)": round(avec["cout_total_usd"], 4),
            "économie cache (%)": round(economie, 1),
            "$/an sans cache": sans["cout_annuel_usd"],
            "$/an avec cache": avec["cout_annuel_usd"],
            "note": " ".join(sans["avertissements"])[:70] or "",
        }
    tab = pd.DataFrame(lignes).T
    ordre = f"coût {n_predictions} préd. ($)"
    return tab.sort_values(ordre)


# ===========================================================================
# Figures
# ===========================================================================
def plot_confusion_matrix(
    y_true: Sequence,
    y_pred: Sequence,
    labels: Sequence[str] | None = None,
    save_path=None,
    *,
    sample_weight: Sequence[float] | None,
    source: pd.DataFrame | None = None,
    normalize: bool = True,
    title: str = "Matrice de confusion",
    figsize: tuple[float, float] = (10.5, 8.5),
):
    """Trace la matrice de confusion et l'enregistre.

    Par defaut normalisee par LIGNE : chaque case se lit "quel pourcentage des
    reclamations de la classe X ont ete classees en Y", c'est-a-dire le rappel
    sur la diagonale. C'est la lecture utile pour reperer les paires de classes
    confondues — avec des effectifs allant de 1 140 a 21 547, une matrice en
    valeurs brutes ne montrerait que la taille des classes.

    `sample_weight` est un argument nomme SANS VALEUR PAR DEFAUT, pour la meme
    raison que dans `evaluate()` : une matrice de confusion tracee sur
    l'echantillon a plancher sans ponderation donnerait une diagonale
    visuellement flatteuse sur les classes rares, qui est un artefact du plan de
    sondage. Passer `None` explicitement quand l'evaluation n'est pas ponderee.
    """
    _verifie_ponderation(source, sample_weight)
    labels = list(labels) if labels is not None else list(cfg.CLASS_ORDER)

    w = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
    cm = confusion_matrix(y_true, y_pred, labels=labels, sample_weight=w).astype(float)
    if normalize:
        totaux = cm.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            affichee = np.where(totaux > 0, cm / totaux * 100, 0.0)
        fmt, unite = "{:.0f}", " (% de la ligne)"
    else:
        affichee, fmt, unite = cm, "{:.0f}", " (effectifs)"

    fig, ax = plt.subplots(figsize=figsize, facecolor=_SURFACE)
    ax.set_facecolor(_SURFACE)
    im = ax.imshow(affichee, cmap=CMAP_CONFUSION, vmin=0,
                   vmax=affichee.max() if affichee.max() else 1)

    seuil = affichee.max() * 0.55 if affichee.max() else 1
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = affichee[i, j]
            if normalize and v < 0.5:
                continue  # on n'encombre pas la figure avec des zeros
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=9,
                    color="#ffffff" if v > seuil else _ENCRE2,
                    fontweight="bold" if i == j else "normal")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9, color=_ENCRE)
    ax.set_yticklabels(labels, fontsize=9, color=_ENCRE)
    ax.set_xlabel("Classe predite", fontsize=10, color=_ENCRE2, labelpad=10)
    ax.set_ylabel("Classe reelle", fontsize=10, color=_ENCRE2, labelpad=10)
    ax.set_title(title + unite, fontsize=12.5, color=_ENCRE, pad=14,
                 loc="left", fontweight="bold")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    # Filet clair entre les cases : elles restent distinctes sans quadrillage.
    ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color=_SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)

    barre = fig.colorbar(im, ax=ax, fraction=0.042, pad=0.03)
    barre.outline.set_visible(False)
    barre.ax.tick_params(labelsize=8.5, length=0, colors=_ATTENUE)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=160, bbox_inches="tight", facecolor=_SURFACE)
    return fig, ax


# ===========================================================================
# Comparaison
# ===========================================================================
def compare_results(dict_of_results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Tableau comparatif des approches, pret a reprendre dans la restitution client.

    Parameters
    ----------
    dict_of_results : {nom de l'approche: dict retourne par `evaluate`}.

    Raises
    ------
    ValueError
        Si le dictionnaire melange des resultats ponderes et non ponderes.
        Comparer un F1-macro pondere a un F1-macro brut n'a aucun sens : l'ecart
        systematique entre les deux est de +0,022, du meme ordre que l'ecart
        qu'on cherche a mesurer entre les deux approches.
        Meme raison pour des tailles d'echantillon differentes : la comparaison
        n'est valide que sur les MEMES lignes.
    """
    if not dict_of_results:
        raise ValueError("Aucun resultat a comparer.")

    ponderations = {nom: r.get("weighted") for nom, r in dict_of_results.items()}
    if len(set(ponderations.values())) > 1:
        raise ValueError(
            "Melange de resultats ponderes et non ponderes : "
            f"{ponderations}.\nL'ecart systematique introduit (+0,022 sur le "
            "F1-macro) est du meme ordre que l'ecart mesure entre les deux "
            "approches. Evaluer toutes les approches sur le meme echantillon, "
            "avec les memes poids."
        )
    tailles = {nom: r.get("n") for nom, r in dict_of_results.items()}
    if len(set(tailles.values())) > 1:
        raise ValueError(
            f"Tailles d'echantillon differentes : {tailles}. La comparaison "
            "n'est valide que sur les memes lignes."
        )

    lignes = {}
    for nom, r in dict_of_results.items():
        ligne = {
            "n": r.get("n"),
            "accuracy": r.get("accuracy"),
            "F1-macro": r.get("f1_macro"),
            "F1-weighted": r.get("f1_weighted"),
            "precision macro": r.get("precision_macro"),
            "rappel macro": r.get("recall_macro"),
        }
        ci = r.get("f1_macro_ci")
        if ci:
            ligne["F1-macro IC 95 %"] = f"[{ci['bas']:.3f} ; {ci['haut']:.3f}]"
        for cle, titre in [("latence_p50_s", "latence p50 (s)"),
                           ("latence_p95_s", "latence p95 (s)"),
                           ("latence_max_s", "latence max (s)")]:
            if cle in r:
                ligne[titre] = r[cle]
        if "cout_pour_1000_predictions_usd" in r:
            ligne["cout / 1000 pred. ($)"] = r["cout_pour_1000_predictions_usd"]
        # F1 par classe : c'est la ou les deux approches different vraiment,
        # le F1-macro global pouvant les departager de moins d'un point.
        rapport = r.get("classification_report", {})
        for classe in cfg.CLASS_ORDER:
            if classe in rapport:
                ligne[f"F1 · {classe}"] = rapport[classe]["f1-score"]
        lignes[nom] = ligne

    tab = pd.DataFrame(lignes).T
    numeriques = tab.select_dtypes(include=[np.number]).columns
    tab[numeriques] = tab[numeriques].astype(float).round(4)
    return tab


def per_class_table(resultats: dict[str, Any]) -> pd.DataFrame:
    """Precision / rappel / F1 / support par classe, dans l'ordre fige."""
    rapport = resultats["classification_report"]
    lignes = {
        classe: {
            "precision": rapport[classe]["precision"],
            "rappel": rapport[classe]["recall"],
            "f1": rapport[classe]["f1-score"],
            "support": rapport[classe]["support"],
        }
        for classe in cfg.CLASS_ORDER if classe in rapport
    }
    return pd.DataFrame(lignes).T.round(4)
