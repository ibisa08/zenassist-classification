"""Preparation des donnees ZenAssist : chargement, nettoyage, split.

Le pipeline suit un ORDRE IMPOSE, chaque etape etant justifiee dans
`reports/diagnostic.md` :

    1.  tri par Complaint ID              (rend le dedoublonnage deterministe)
    2.  filtre texte ou etiquette manquants
    3.  exclusion de Consumer Loan et Other financial service
    4.  fusion des 16 libelles vers 9 classes
    5.  normalisation des espaces UNIQUEMENT
    6.  suppression des textes < MIN_TEXT_LENGTH
    7.  resolution des etiquettes contradictoires par vote majoritaire
    8.  dedoublonnage exact sur le texte
    9.  garde-fou MIN_SAMPLES_PER_CLASS
    10. split stratifie train / test + echantillon d'evaluation LLM

Deux contraintes d'ordre sont structurantes et ne doivent pas etre inversees :
  - le VOTE MAJORITAIRE precede le DEDOUBLONNAGE, sinon l'etiquette conservee
    pour un texte apparaissant 36 fois serait celle de la premiere ligne
    rencontree et non celle de la majorite ;
  - le DEDOUBLONNAGE precede le SPLIT, sinon les courriers types de credit
    repair (11 354 textes dupliques, cf. diagnostic A.2) se retrouveraient
    simultanement dans le train et le test, et le score de test serait
    artificiellement gonfle.

Le texte n'est PAS mis en minuscules et la ponctuation n'est PAS retiree : le
texte brut est necessaire au LLM de l'etape 2. Le pretraitement lourd
(lowercase, stop-words, n-grammes) appartient au pipeline TF-IDF de l'etape 3.

CHARGEMENT DE L'ECHANTILLON D'EVALUATION
----------------------------------------
`load_eval_sample()` est le SEUL moyen supporte de lire
`data/processed/test_sample_2000.csv`. Il retourne le tuple `(df, weights)`.

Cet echantillon est stratifie A PLANCHER, donc toute metrique calculee sans la
ponderation de Horvitz-Thompson est biaisee de +0,022 sur le F1-macro et de
+0,29 sur la precision de "Vehicle loan or lease". Deux garde-fous rendent
l'oubli difficile :
  - le retour est un TUPLE, donc un `df = load_eval_sample()` suivi d'un
    `df["label"]` echoue immediatement ;
  - le CSV porte une ligne d'avertissement en tete, ce qui fait produire a un
    `pd.read_csv()` nu un DataFrame a une seule colonne dont l'en-tete EST
    l'avertissement : tout acces a `label` ou `sampling_weight` leve un
    KeyError. Attention, le fichier se "lit" sans exception -- il est
    seulement inexploitable ; ce n'est pas une ParserError.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config as cfg

logger = logging.getLogger(__name__)


# ===========================================================================
# Chargement
# ===========================================================================
def load_raw(path=None, nrows: int | None = None) -> pd.DataFrame:
    """Charge le fichier brut, restreint aux 4 colonnes utiles.

    Le fichier fait 639 Mo et 15 colonnes ; n'en charger que 4 fait passer
    l'empreinte memoire de ~5 Go a ~500 Mo. Les 11 colonnes ecartees sont soit
    des metadonnees post-traitement (donc une fuite de donnees : la reponse de
    l'entreprise est connue APRES la classification), soit non pertinentes pour
    classer un texte, soit constantes sur le perimetre (`Submitted via` vaut
    "Web" pour 100 % des lignes ayant un texte).

    L'encodage utf-8 a ete valide par decodage incremental sur l'integralite du
    fichier, pas seulement sur l'en-tete.
    """
    path = cfg.RAW_FILE if path is None else path
    logger.info("Chargement de %s (colonnes : %s)", path, ", ".join(cfg.RAW_USECOLS))
    df = pd.read_csv(
        path,
        sep=",",
        encoding="utf-8",
        usecols=cfg.RAW_USECOLS,
        nrows=nrows,
        low_memory=False,
    )
    logger.info("Fichier brut charge : %s lignes", f"{len(df):,}")
    return df


def load_split() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recharge les fichiers train et test produits par le pipeline."""
    train_file, test_file = _fichiers_split()
    return (
        pd.read_csv(train_file, encoding="utf-8"),
        pd.read_csv(test_file, encoding="utf-8"),
    )


def load_eval_sample(autoriser_dev: bool = False) -> tuple[pd.DataFrame, pd.Series]:
    """Charge l'echantillon d'evaluation LLM. SEUL moyen supporte de le lire.

    Parameters
    ----------
    autoriser_dev : echappement explicite pour travailler sur un echantillon de
        developpement. Par defaut `False`, et la fonction LEVE si
        `config.SAMPLE_FRACTION < 1.0`.

    Returns
    -------
    (df, weights) : tuple
        `df` contient les colonnes texte, etiquette, `strate` et
        `sampling_weight`. `weights` est la serie des poids de
        Horvitz-Thompson, a passer telle quelle a `metrics.evaluate(...,
        sample_weight=weights)`.

    Le retour sous forme de TUPLE est delibere : il rend impossible un
    `df = load_eval_sample()` suivi d'un `f1_score()` non pondere, puisque le
    tuple echouerait des la premiere indexation de colonne.

    Contournement volontairement rendu penible : lire le fichier avec un
    `pd.read_csv()` nu donne un DataFrame a une seule colonne (la ligne
    d'avertissement sert d'en-tete), donc un KeyError des le premier acces.

    Raises
    ------
    RuntimeError
        Si `config.SAMPLE_FRACTION < 1.0` et que `autoriser_dev` vaut `False`.
        C'est le seul endroit du projet ou une erreur coute de l'argent et des
        heures d'attente : lancer 2 000 appels d'API sur un echantillon de
        developpement est irrattrapable une fois les appels partis.
    """
    if cfg.SAMPLE_FRACTION < 1.0 and not autoriser_dev:
        raise RuntimeError(
            f"config.SAMPLE_FRACTION vaut {cfg.SAMPLE_FRACTION}, donc "
            f"l'echantillon disponible est un echantillon de DEVELOPPEMENT, "
            f"tire sur {cfg.SAMPLE_FRACTION:.0%} du corpus.\n"
            f"Lancer une campagne d'appels API dessus coute de l'argent et "
            f"plusieurs heures d'attente, pour un resultat inexploitable.\n"
            f"Corriger ainsi :\n"
            f"  1. remettre SAMPLE_FRACTION = 1.0 dans src/config.py\n"
            f"  2. relancer le pipeline :  python -m src.data_prep\n"
            f"  3. rappeler load_eval_sample()\n"
            f"Si l'intention est bien de travailler sur l'echantillon de "
            f"developpement, l'ecrire explicitement : "
            f"load_eval_sample(autoriser_dev=True)."
        )

    path = _chemins_sortie()["eval_csv"]
    if not path.exists():
        raise FileNotFoundError(
            f"{path} est absent. Lancer d'abord : python -m src.data_prep"
        )

    with path.open("r", encoding="utf-8") as f:
        premiere_ligne = f.readline()
    if not premiere_ligne.startswith("#"):
        raise ValueError(
            f"{path} ne commence pas par la ligne d'avertissement attendue. "
            "Le fichier a probablement ete reecrit par un outil tiers ; "
            "relancer le pipeline avant de l'utiliser."
        )

    df = pd.read_csv(path, encoding="utf-8", skiprows=1)
    if cfg.WEIGHT_COL not in df.columns:
        raise ValueError(
            f"{path} ne contient pas la colonne '{cfg.WEIGHT_COL}'. "
            "Toute metrique calculee sans ces poids serait biaisee."
        )
    logger.info(
        "Echantillon d'evaluation charge : %s lignes, %s classes, poids de %.1f a %.1f",
        len(df), df[cfg.LABEL_COL].nunique(),
        df[cfg.WEIGHT_COL].min(), df[cfg.WEIGHT_COL].max(),
    )
    return df, df[cfg.WEIGHT_COL]


# ===========================================================================
# Nettoyage
# ===========================================================================
def _normalise_espaces(s: pd.Series) -> pd.Series:
    """Strip + collapse des espaces. RIEN D'AUTRE.

    Pas de mise en minuscules, pas de retrait de ponctuation, pas de
    suppression du masquage XXXX : le texte brut est l'entree du LLM, et
    l'IDF du TfidfVectorizer neutralisera "xxxx" tout seul (il est present dans
    85 % des documents, son IDF vaut ~0,16).
    """
    return s.astype(str).str.replace(r"\s+", " ", regex=True).str.strip()


def _resout_contradictions(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Etape 7 : meme texte, etiquettes differentes -> vote majoritaire.

    201 textes du corpus portent au moins deux etiquettes differentes (168
    apres fusion des libelles). Ce sont surtout des courriers types de credit
    repair diffuses en masse : le meme courrier invoquant la FCRA a ete classe
    35 fois en "Credit reporting" et 1 fois en "Debt collection" par les agents
    du CFPB.

    Regle appliquee (arbitrage H.2) :
      - majorite STRICTE  -> toutes les lignes du texte recoivent l'etiquette
        majoritaire ;
      - EGALITE parfaite  -> le texte est supprime entierement, car aucune
        etiquette n'est defendable.
    """
    texte, label = cfg.TEXT_COL, cfg.LABEL_COL

    doublons = df.loc[df.duplicated(subset=[texte], keep=False), [texte, label]]
    if doublons.empty:
        return df, {"textes_contradictoires": 0, "textes_resolus_par_majorite": 0,
                    "lignes_reetiquetees": 0, "textes_supprimes_pour_egalite": 0,
                    "lignes_supprimees": 0}

    # Comptage (texte, etiquette), trie par effectif decroissant puis par
    # etiquette : l'ordre est ainsi totalement deterministe.
    cnt = (doublons.groupby([texte, label], sort=False)
                   .size().reset_index(name="n")
                   .sort_values([texte, "n", label], ascending=[True, False, True]))
    cnt["rang"] = cnt.groupby(texte).cumcount()

    premier = cnt[cnt["rang"] == 0].set_index(texte)
    second = cnt[cnt["rang"] == 1].set_index(texte)["n"]
    # Un texte n'est contradictoire que s'il porte au moins deux etiquettes.
    contradictoires = premier.loc[premier.index.isin(second.index)]
    n2 = second.reindex(contradictoires.index)

    majoritaires = contradictoires[contradictoires["n"] > n2]
    egalites = contradictoires.index.difference(majoritaires.index)

    out = df.copy()
    # --- resolution par majorite : on reecrit l'etiquette, on ne supprime rien
    if len(majoritaires):
        remplacement = majoritaires[label]
        masque = out[texte].isin(remplacement.index)
        n_reetiquetees = int(masque.sum())
        out.loc[masque, label] = out.loc[masque, texte].map(remplacement)
    else:
        n_reetiquetees = 0

    # --- egalite : suppression complete du texte
    masque_eg = out[texte].isin(egalites)
    n_supprimees = int(masque_eg.sum())
    out = out.loc[~masque_eg]

    rapport = {
        "textes_contradictoires": int(len(contradictoires)),
        "textes_resolus_par_majorite": int(len(majoritaires)),
        "lignes_reetiquetees": n_reetiquetees,
        "textes_supprimes_pour_egalite": int(len(egalites)),
        "lignes_supprimees": n_supprimees,
    }
    logger.info(
        "  contradictions : %d textes -> %d resolus par majorite (%d lignes "
        "reetiquetees), %d supprimes pour egalite (%d lignes)",
        rapport["textes_contradictoires"], rapport["textes_resolus_par_majorite"],
        rapport["lignes_reetiquetees"], rapport["textes_supprimes_pour_egalite"],
        rapport["lignes_supprimees"],
    )
    return out, rapport


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Nettoie le corpus brut. Fonction pure : aucune ecriture disque.

    Parameters
    ----------
    df : DataFrame issu de `load_raw()`.

    Returns
    -------
    (df_clean, rapport)
        `df_clean` porte les colonnes `complaint_id`, `date_received`, `text`,
        `label`, triees par `complaint_id`.
        `rapport` est un dict ordonne {etape: {supprimees, restantes}} qui
        reproduit le tableau F.6 du diagnostic, plus une cle `contradictions`
        detaillant l'etape 7 et une cle `resume`.
    """
    rapport: "OrderedDict[str, Any]" = OrderedDict()
    n = len(df)
    rapport["0_depart"] = {"supprimees": 0, "restantes": n}

    # --- 1. tri par Complaint ID -------------------------------------------
    # Indispensable a l'idempotence : `drop_duplicates(keep="first")` depend de
    # l'ordre des lignes. Sans tri prealable, l'etiquette conservee parmi 36
    # copies d'un meme courrier varierait d'une execution a l'autre.
    d = df.rename(columns={
        cfg.RAW_ID_COL: cfg.ID_COL,
        cfg.RAW_DATE_COL: cfg.DATE_COL,
        cfg.RAW_TEXT_COL: cfg.TEXT_COL,
        cfg.RAW_LABEL_COL: cfg.LABEL_COL,
    }).sort_values(cfg.ID_COL, kind="mergesort").reset_index(drop=True)
    rapport["1_tri_complaint_id"] = {"supprimees": 0, "restantes": len(d)}

    # --- 2. texte ou etiquette manquants ------------------------------------
    avant = len(d)
    d = d[d[cfg.TEXT_COL].notna() & d[cfg.LABEL_COL].notna()]
    d = d[d[cfg.LABEL_COL].astype(str).str.strip() != ""]
    rapport["2_texte_ou_label_manquant"] = {"supprimees": avant - len(d), "restantes": len(d)}

    # --- 3. exclusion des libelles non mappables ---------------------------
    avant = len(d)
    # `sorted` : EXCLUDED_LABELS est un set, dont l'ordre d'iteration varie d'un
    # processus a l'autre. Sans tri, l'ordre des cles de ce sous-dict changerait
    # entre deux executions et cleaning_report.json ne serait pas idempotent
    # (les CSV, eux, ne sont pas concernes).
    exclus = {lib: int((d[cfg.LABEL_COL] == lib).sum())
              for lib in sorted(cfg.EXCLUDED_LABELS)}
    d = d[~d[cfg.LABEL_COL].isin(cfg.EXCLUDED_LABELS)]
    rapport["3_exclusion_classes"] = {
        "supprimees": avant - len(d), "restantes": len(d), "detail": exclus,
    }

    # --- 4. fusion des libelles (16 -> 9) ----------------------------------
    inconnus = set(d[cfg.LABEL_COL].unique()) - set(cfg.LABEL_MAPPING)
    if inconnus:
        raise ValueError(
            f"Libelles absents de LABEL_MAPPING : {sorted(inconnus)}. "
            "Le referentiel CFPB a peut-etre evolue : mettre a jour config.LABEL_MAPPING."
        )
    avant = len(d)
    n_libelles_avant = d[cfg.LABEL_COL].nunique()
    d[cfg.LABEL_COL] = d[cfg.LABEL_COL].map(cfg.LABEL_MAPPING)
    rapport["4_mapping_libelles"] = {
        "supprimees": avant - len(d), "restantes": len(d),
        "libelles_avant": int(n_libelles_avant), "classes_apres": int(d[cfg.LABEL_COL].nunique()),
    }

    # --- 5. normalisation des espaces --------------------------------------
    avant = len(d)
    d[cfg.TEXT_COL] = _normalise_espaces(d[cfg.TEXT_COL])
    d = d[d[cfg.TEXT_COL] != ""]
    rapport["5_normalisation_espaces"] = {"supprimees": avant - len(d), "restantes": len(d)}

    # --- 6. longueur minimale ----------------------------------------------
    avant = len(d)
    d = d[d[cfg.TEXT_COL].str.len() >= cfg.MIN_TEXT_LENGTH]
    rapport["6_texte_trop_court"] = {
        "supprimees": avant - len(d), "restantes": len(d), "seuil": cfg.MIN_TEXT_LENGTH,
    }

    # --- 7. contradictions d'etiquetage (AVANT le dedoublonnage) -----------
    avant = len(d)
    d, rap_contra = _resout_contradictions(d)
    rapport["7_contradictions_egalite"] = {"supprimees": avant - len(d), "restantes": len(d)}
    rapport["contradictions"] = rap_contra

    # --- 8. dedoublonnage exact sur le texte (AVANT le split) --------------
    avant = len(d)
    d = d.drop_duplicates(subset=[cfg.TEXT_COL], keep="first")
    rapport["8_doublons_texte"] = {"supprimees": avant - len(d), "restantes": len(d)}

    d = d[cfg.OUTPUT_COLS].reset_index(drop=True)
    rapport["resume"] = {
        "lignes_initiales": n,
        "lignes_finales": len(d),
        "taux_conservation_pct": round(len(d) / n * 100, 2),
        "nb_classes": int(d[cfg.LABEL_COL].nunique()),
    }
    logger.info("Nettoyage : %s -> %s lignes (%.2f %%), %d classes",
                f"{n:,}", f"{len(d):,}", rapport["resume"]["taux_conservation_pct"],
                rapport["resume"]["nb_classes"])
    return d, rapport


def rapport_nettoyage_df(rapport: dict[str, Any]) -> pd.DataFrame:
    """Rend le rapport de `clean()` sous forme de tableau (pour le notebook)."""
    lignes = [
        {"etape": k, "supprimees": v["supprimees"], "restantes": v["restantes"]}
        for k, v in rapport.items()
        if isinstance(v, dict) and "restantes" in v
    ]
    return pd.DataFrame(lignes)


def handle_rare_classes(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Etape 9 : garde-fou sur les classes trop peu representees.

    Apres fusion, la plus petite classe compte ~5 700 lignes pour un seuil de
    1 000 : cette fonction ne supprime RIEN sur le dataset actuel, et c'est
    normal. Elle est conservee et journalise explicitement "0 classe sous le
    seuil" plutot que d'etre omise : le CFPB publie en continu, et une classe
    devenue trop rare doit etre signalee au lieu de passer silencieusement dans
    le split, ou elle produirait un F1-macro instable que personne ne verrait.
    """
    effectifs = df[cfg.LABEL_COL].value_counts()
    rares = effectifs[effectifs < cfg.MIN_SAMPLES_PER_CLASS]

    out = df[~df[cfg.LABEL_COL].isin(rares.index)] if len(rares) else df
    rapport = {
        "seuil": cfg.MIN_SAMPLES_PER_CLASS,
        "classes_avant": int(len(effectifs)),
        "classes_supprimees": {k: int(v) for k, v in rares.items()},
        "nb_classes_sous_seuil": int(len(rares)),
        "lignes_supprimees": int(len(df) - len(out)),
        "classes_apres": int(out[cfg.LABEL_COL].nunique()),
        "effectifs_finaux": {k: int(v) for k, v in
                             out[cfg.LABEL_COL].value_counts().items()},
    }
    if len(rares):
        logger.warning("Classes sous le seuil de %d : %s",
                       cfg.MIN_SAMPLES_PER_CLASS, dict(rares))
    else:
        logger.info("Garde-fou classes rares : 0 classe sous le seuil de %d "
                    "(la plus petite compte %s lignes)",
                    cfg.MIN_SAMPLES_PER_CLASS, f"{effectifs.min():,}")
    return out.reset_index(drop=True), rapport


# ===========================================================================
# Echantillon d'evaluation LLM  (strategie B3)
# ===========================================================================
def _repartition_plus_forts_restes(poids: pd.Series, total: int) -> pd.Series:
    """Repartit `total` unites proportionnellement a `poids`, sans perte.

    Methode des plus forts restes : la somme retournee vaut exactement `total`.
    """
    exact = poids / poids.sum() * total
    base = exact.astype(int)
    reste = total - int(base.sum())
    if reste > 0:
        ordre = (exact - base).sort_values(ascending=False, kind="mergesort").index
        base.loc[ordre[:reste]] += 1
    return base


def allocation_plancher(
    effectifs: pd.Series,
    n_total: int = cfg.LLM_EVAL_SAMPLE_SIZE,
    plancher: int = cfg.LLM_EVAL_MIN_PER_CLASS,
) -> pd.Series:
    """Allocation B3 : `plancher` items par classe, solde reparti au prorata.

    C'est la strategie retenue par la simulation de `reports/
    h1_sampling_simulation.md`. Elle sur-echantillonne volontairement les
    classes rares : sans elle, "Vehicle loan or lease" ne serait represente que
    par 32 items sur 2 000, et l'intervalle de confiance a 95 % de son F1
    couvrirait 25 points.

    Le sur-echantillonnage rend la ponderation de Horvitz-Thompson OBLIGATOIRE
    en aval : c'est le prix a payer, et il est explicite.
    """
    effectifs = effectifs.reindex(cfg.CLASS_ORDER).dropna().astype(int)
    if n_total > int(effectifs.sum()):
        raise ValueError(f"n_total={n_total} depasse la population disponible "
                         f"({effectifs.sum()}).")

    # Plancher, borne par ce qui est reellement disponible dans chaque classe.
    base = pd.Series(plancher, index=effectifs.index).clip(upper=effectifs)
    solde = n_total - int(base.sum())
    if solde < 0:
        raise ValueError(f"Le plancher de {plancher} par classe depasse deja "
                         f"n_total={n_total} pour {len(effectifs)} classes.")

    alloc = base + _repartition_plus_forts_restes(effectifs, solde)
    # Aucune classe ne peut etre sur-tiree au-dela de son effectif reel.
    depassement = (alloc - effectifs).clip(lower=0)
    if depassement.any():
        alloc -= depassement
        alloc += _repartition_plus_forts_restes(
            (effectifs - alloc).clip(lower=0), int(depassement.sum()))
    return alloc.astype(int)


def _construit_echantillon_eval(
    test: pd.DataFrame, n_total: int = cfg.LLM_EVAL_SAMPLE_SIZE,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Tire l'echantillon B3 dans le test et calcule les poids HT."""
    effectifs = test[cfg.LABEL_COL].value_counts()
    plancher = min(cfg.LLM_EVAL_MIN_PER_CLASS, n_total // len(cfg.CLASS_ORDER))
    alloc = allocation_plancher(effectifs, n_total, plancher)

    # Tirage classe par classe, dans l'ordre fige de CLASS_ORDER : l'ordre des
    # operations est deterministe, donc le fichier produit l'est aussi.
    morceaux = [
        test[test[cfg.LABEL_COL] == classe].sample(
            n=int(alloc[classe]), random_state=cfg.RANDOM_SEED)
        for classe in alloc.index
    ]
    ech = (pd.concat(morceaux)
             .sort_values(cfg.ID_COL, kind="mergesort")
             .reset_index(drop=True))

    # Poids de Horvitz-Thompson : w_c = N_c / n_c, ou N_c est l'effectif de la
    # classe c dans le TEST COMPLET et n_c le nombre d'items tires. Ils
    # ramenent l'echantillon a la structure de la population, donc les
    # metriques ponderees estiment sans biais celles du test complet.
    poids = (effectifs.reindex(alloc.index) / alloc).astype(float)
    ech[cfg.STRATE_COL] = ech[cfg.LABEL_COL]
    ech[cfg.WEIGHT_COL] = ech[cfg.LABEL_COL].map(poids)
    ech = ech[cfg.OUTPUT_COLS + [cfg.STRATE_COL, cfg.WEIGHT_COL]]
    return ech, alloc, poids


def construit_echantillon_alternatif(seed: int) -> tuple[pd.DataFrame, Any]:
    """Produit un SECOND echantillon B3, meme allocation, tirage different.

    Pourquoi
    --------
    `test_sample_2000.csv` est un tirage UNIQUE (seed 42). La simulation de
    `reports/h1_sampling_simulation.md` portait sur la DISTRIBUTION des tirages
    possibles : elle dit que l'ecart-type du F1-macro estime vaut 0,012, pas que
    ce tirage-ci est bien place. On ne sait pas si l'echantillon retenu est
    favorable ou defavorable a l'une des deux approches.

    Cette fonction fabrique un second echantillon avec exactement la MEME
    allocation par classe (donc les memes poids de Horvitz-Thompson) et une
    seed differente. Reevaluer dessus, une fois le prompt fige, coute ~0,11 $ et
    fournit une preuve EMPIRIQUE de stabilite, plus convaincante qu'un
    intervalle bootstrap : l'IC mesure la variabilite interne d'un echantillon,
    le second tirage mesure la variabilite entre echantillons.

    Elle n'est volontairement PAS appelee par `main()` : c'est un outil de
    verification pour l'etape 2, pas une etape du pipeline.

    Parameters
    ----------
    seed : graine du second tirage. Doit differer de `config.RANDOM_SEED`.

    Returns
    -------
    (echantillon, chemin) : le DataFrame produit et le fichier ecrit,
    `test_sample_2000_seed<seed>.csv`.
    """
    if seed == cfg.RANDOM_SEED:
        raise ValueError(
            f"seed={seed} est deja la seed du pipeline (config.RANDOM_SEED). "
            f"Le second tirage serait identique au premier et ne prouverait rien."
        )
    _, test_file = _fichiers_split()
    if not test_file.exists():
        raise FileNotFoundError(
            f"{test_file} est absent. Lancer d'abord : python -m src.data_prep")

    test = pd.read_csv(test_file, encoding="utf-8")
    effectifs = test[cfg.LABEL_COL].value_counts()
    # MEME allocation que l'echantillon de reference : seul le tirage change.
    alloc = allocation_plancher(effectifs)
    poids = (effectifs.reindex(alloc.index) / alloc).astype(float)

    morceaux = [test[test[cfg.LABEL_COL] == classe].sample(
        n=int(alloc[classe]), random_state=seed) for classe in alloc.index]
    ech = (pd.concat(morceaux).sort_values(cfg.ID_COL, kind="mergesort")
             .reset_index(drop=True))
    ech[cfg.STRATE_COL] = ech[cfg.LABEL_COL]
    ech[cfg.WEIGHT_COL] = ech[cfg.LABEL_COL].map(poids)
    ech = ech[cfg.OUTPUT_COLS + [cfg.STRATE_COL, cfg.WEIGHT_COL]]

    base = _chemins_sortie()["eval_csv"]
    chemin = base.with_name(f"{base.stem}_seed{seed}.csv")
    _ecrit_echantillon_eval(ech, chemin)

    reference = set(pd.read_csv(base, encoding="utf-8", skiprows=1)[cfg.ID_COL])
    recouvrement = len(reference & set(ech[cfg.ID_COL])) / len(ech) * 100
    logger.info("Ecrit %s (%s lignes, seed %d, recouvrement avec l'echantillon "
                "de reference : %.1f %%)", chemin.name, f"{len(ech):,}", seed,
                recouvrement)
    return ech, chemin


def _ecrit_echantillon_eval(ech: pd.DataFrame, path) -> None:
    """Ecrit le CSV precede de sa ligne d'avertissement (garde-fou 6)."""
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write(cfg.EVAL_SAMPLE_HEADER_COMMENT + "\n")
        ech.to_csv(f, index=False)


def _ecrit_readme_eval(alloc: pd.Series, poids: pd.Series, effectifs: pd.Series,
                       path=None) -> None:
    """Note d'avertissement autonome, a cote du CSV d'evaluation."""
    path = cfg.EVAL_SAMPLE_README_FILE if path is None else path
    lignes = [
        "ECHANTILLON D'EVALUATION LLM - %s" % path.name.replace(".README.txt", ".csv"),
        "=" * 72,
        "",
        "Strategie B3 : echantillon stratifie A PLANCHER (%d items minimum par"
        % cfg.LLM_EVAL_MIN_PER_CLASS,
        "classe, solde reparti proportionnellement a la population du test).",
        "Arbitrage documente dans reports/h1_sampling_simulation.md.",
        "",
        "*** AVERTISSEMENT ***",
        "",
        "Cet echantillon N'EST PAS representatif de la distribution du test : les",
        "classes rares y sont volontairement sur-echantillonnees. Toute metrique",
        "calculee sans la colonne `sampling_weight` est BIAISEE :",
        "",
        "    +0.022  sur le F1-macro",
        "    +0.29   sur la precision de 'Vehicle loan or lease'",
        "    -0.082  sur la precision de 'Credit reporting'",
        "",
        "Ces chiffres sont mesures, pas estimes (2 000 replicats Monte-Carlo).",
        "",
        "CHARGEMENT",
        "-" * 72,
        "    from src.data_prep import load_eval_sample",
        "    df, weights = load_eval_sample()",
        "    resultats = evaluate(df['label'], y_pred, sample_weight=weights)",
        "",
        "C'est le seul moyen supporte. Un pd.read_csv() nu sur ce fichier rend",
        "un DataFrame a une seule colonne (la ligne d'avertissement ci-dessus",
        "sert d'en-tete) : tout acces a 'label' ou 'sampling_weight' leve alors",
        "un KeyError. Le fichier se lit sans exception, mais il est inexploitable.",
        "",
        "ALLOCATION ET POIDS",
        "-" * 72,
        f"{'classe':<38}{'N_test':>9}{'n_tire':>9}{'f_i (pour mille)':>19}{'poids w_c':>12}",
    ]
    for classe in alloc.index:
        f_i = alloc[classe] / effectifs[classe] * 1000
        lignes.append(f"{classe:<38}{effectifs[classe]:>9,}{alloc[classe]:>9,}"
                      f"{f_i:>19.1f}{poids[classe]:>12.2f}")
    lignes += ["", f"TOTAL : {int(alloc.sum()):,} lignes tirees sur "
                   f"{int(effectifs.sum()):,} lignes de test.", ""]
    path.write_text("\n".join(lignes), encoding="utf-8")


# ===========================================================================
# Split
# ===========================================================================
def _suffixe_dev() -> str:
    """Suffixe applique a TOUS les fichiers produits quand SAMPLE_FRACTION < 1.

    Un corpus partiel ne doit JAMAIS ecraser les fichiers de reference : il
    serait sinon possible de livrer un modele entraine sur un echantillon de
    developpement sans que rien ne le signale (arbitrage H.5). Le suffixe
    s'applique aussi a l'echantillon d'evaluation et aux metadonnees, sans quoi
    le garde-fou serait inoperant sur les deux fichiers les plus sensibles.
    """
    if cfg.SAMPLE_FRACTION >= 1.0:
        return ""
    return f"_dev{int(round(cfg.SAMPLE_FRACTION * 100))}"


def _chemins_sortie() -> dict[str, Any]:
    """Tous les chemins produits par le split, suffixes si besoin."""
    s = _suffixe_dev()
    if not s:
        return {"train": cfg.TRAIN_FILE, "test": cfg.TEST_FILE,
                "eval_csv": cfg.EVAL_SAMPLE_FILE,
                "eval_readme": cfg.EVAL_SAMPLE_README_FILE,
                "metadata": cfg.SPLIT_METADATA_FILE}
    d = cfg.PROCESSED_DIR
    return {
        "train": d / f"train{s}.csv",
        "test": d / f"test{s}.csv",
        "eval_csv": d / f"{cfg.EVAL_SAMPLE_FILE.stem}{s}.csv",
        "eval_readme": d / f"{cfg.EVAL_SAMPLE_FILE.stem}{s}.README.txt",
        "metadata": d / f"{cfg.SPLIT_METADATA_FILE.stem}{s}.json",
    }


def _fichiers_split() -> tuple:
    """Chemins des fichiers train/test courants."""
    c = _chemins_sortie()
    return c["train"], c["test"]


def _sha256(path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for bloc in iter(lambda: f.read(1 << 20), b""):
            h.update(bloc)
    return h.hexdigest()


def split_and_save(df: pd.DataFrame) -> dict[str, Any]:
    """Etape 10 : split stratifie, ecriture des fichiers, metadonnees.

    Le split est ALEATOIRE stratifie, pas temporel. Un split temporel serait
    ici invalide : apres fusion, "Vehicle loan or lease" n'existe qu'a partir
    d'avril 2017 (le referentiel a bascule le 21-24 avril 2017) et se
    retrouverait entierement d'un seul cote de la coupure.
    """
    cfg.ensure_dirs()

    # --- sous-echantillonnage de developpement, applique AVANT le split -----
    if cfg.SAMPLE_FRACTION < 1.0:
        logger.warning(
            "SAMPLE_FRACTION=%.2f : corpus reduit pour le developpement. "
            "Les fichiers produits seront suffixes.", cfg.SAMPLE_FRACTION)
        df, _ = train_test_split(
            df, train_size=cfg.SAMPLE_FRACTION, stratify=df[cfg.LABEL_COL],
            random_state=cfg.RANDOM_SEED)
        df = df.sort_values(cfg.ID_COL, kind="mergesort").reset_index(drop=True)

    train, test = train_test_split(
        df,
        test_size=cfg.TEST_SIZE,
        stratify=df[cfg.LABEL_COL],
        random_state=cfg.RANDOM_SEED,
        shuffle=True,
    )
    train = train.sort_values(cfg.ID_COL, kind="mergesort").reset_index(drop=True)
    test = test.sort_values(cfg.ID_COL, kind="mergesort").reset_index(drop=True)

    chemins = _chemins_sortie()
    train_file, test_file = chemins["train"], chemins["test"]
    train.to_csv(train_file, index=False, encoding="utf-8")
    test.to_csv(test_file, index=False, encoding="utf-8")
    logger.info("Ecrit %s (%s lignes) et %s (%s lignes)",
                train_file.name, f"{len(train):,}", test_file.name, f"{len(test):,}")

    # --- echantillon d'evaluation LLM --------------------------------------
    # En mode developpement, le test peut etre plus petit que la taille cible :
    # on reduit l'echantillon plutot que d'echouer, en le signalant.
    taille_eval = min(cfg.LLM_EVAL_SAMPLE_SIZE, len(test))
    if taille_eval < cfg.LLM_EVAL_SAMPLE_SIZE:
        logger.warning("Echantillon d'evaluation reduit a %d lignes (test de %d "
                       "lignes seulement). Valable en developpement uniquement.",
                       taille_eval, len(test))
    ech, alloc, poids = _construit_echantillon_eval(test, taille_eval)
    _ecrit_echantillon_eval(ech, chemins["eval_csv"])
    effectifs_test = test[cfg.LABEL_COL].value_counts().reindex(alloc.index)
    _ecrit_readme_eval(alloc, poids, effectifs_test, chemins["eval_readme"])
    logger.info("Ecrit %s (%s lignes, plancher %d/classe)",
                chemins["eval_csv"].name, f"{len(ech):,}", cfg.LLM_EVAL_MIN_PER_CLASS)

    ecart_reference = {
        c: int(alloc[c] - cfg.EVAL_ALLOCATION_REFERENCE[c])
        for c in alloc.index if c in cfg.EVAL_ALLOCATION_REFERENCE
    }
    if any(abs(v) > 5 for v in ecart_reference.values()):
        logger.warning("L'allocation s'ecarte de la reference de la simulation : %s",
                       {k: v for k, v in ecart_reference.items() if abs(v) > 5})

    # --- metadonnees --------------------------------------------------------
    def distribution(d: pd.DataFrame) -> dict[str, dict[str, float]]:
        vc = d[cfg.LABEL_COL].value_counts().reindex(cfg.CLASS_ORDER)
        return {c: {"n": int(vc[c]), "pct": round(vc[c] / len(d) * 100, 3)}
                for c in cfg.CLASS_ORDER}

    metadata = {
        # `generated_at` est le SEUL champ non deterministe du pipeline. Les
        # empreintes sha256 ci-dessous permettent de verifier l'idempotence :
        # deux executions successives doivent produire les memes empreintes.
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "random_seed": cfg.RANDOM_SEED,
        "test_size": cfg.TEST_SIZE,
        "sample_fraction": cfg.SAMPLE_FRACTION,
        "min_text_length": cfg.MIN_TEXT_LENGTH,
        "min_samples_per_class": cfg.MIN_SAMPLES_PER_CLASS,
        "n_classes": len(cfg.CLASS_ORDER),
        "class_order": cfg.CLASS_ORDER,
        "tailles": {"total": len(df), "train": len(train), "test": len(test),
                    "eval_sample": len(ech)},
        "distribution": {"corpus": distribution(df), "train": distribution(train),
                         "test": distribution(test),
                         "eval_sample": distribution(ech)},
        "echantillon_evaluation": {
            "strategie": "B3 - stratification a plancher avec ponderation Horvitz-Thompson",
            "justification": "reports/h1_sampling_simulation.md",
            "taille": int(cfg.LLM_EVAL_SAMPLE_SIZE),
            "plancher_par_classe": int(cfg.LLM_EVAL_MIN_PER_CLASS),
            "ponderation_obligatoire": True,
            "biais_si_non_pondere": {
                "f1_macro": 0.022,
                "precision_vehicle_loan_or_lease": 0.29,
                "precision_credit_reporting": -0.082,
            },
            "par_classe": {
                c: {
                    "N_test": int(effectifs_test[c]),
                    "n_tire": int(alloc[c]),
                    "taux_sondage_pour_mille": round(alloc[c] / effectifs_test[c] * 1000, 2),
                    "poids_ht": round(float(poids[c]), 4),
                }
                for c in alloc.index
            },
            "ecart_a_l_allocation_de_reference": ecart_reference,
        },
        "fichiers": {},
    }
    for nom, path, n_entete in [("train", train_file, 1), ("test", test_file, 1),
                                # le CSV d'evaluation porte une ligne de
                                # commentaire en plus de son en-tete
                                ("eval_sample", chemins["eval_csv"], 2)]:
        n_lignes = sum(1 for _ in path.open(encoding="utf-8")) - n_entete
        metadata["fichiers"][nom] = {"chemin": str(path.relative_to(cfg.PROJECT_ROOT)),
                                     "lignes_de_donnees": int(n_lignes),
                                     "sha256": _sha256(path)}

    chemins["metadata"].write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Ecrit %s", chemins["metadata"].name)
    return metadata


def verifie_stratification(metadata: dict[str, Any], tolerance_pct: float = 0.5) -> pd.DataFrame:
    """Compare la distribution des classes entre corpus, train et test.

    La stratification est respectee si l'ecart en points de pourcentage reste
    sous `tolerance_pct` pour chaque classe.
    """
    dist = metadata["distribution"]
    tab = pd.DataFrame({
        "corpus_pct": {c: dist["corpus"][c]["pct"] for c in cfg.CLASS_ORDER},
        "train_pct": {c: dist["train"][c]["pct"] for c in cfg.CLASS_ORDER},
        "test_pct": {c: dist["test"][c]["pct"] for c in cfg.CLASS_ORDER},
    })
    tab["ecart_train"] = (tab.train_pct - tab.corpus_pct).round(3)
    tab["ecart_test"] = (tab.test_pct - tab.corpus_pct).round(3)
    tab["ok"] = (tab[["ecart_train", "ecart_test"]].abs().max(axis=1) < tolerance_pct)
    return tab


# ===========================================================================
# Pipeline complet
# ===========================================================================
def main() -> dict[str, Any]:
    """Execute le pipeline de bout en bout. IDEMPOTENT.

    Relancer cette fonction reproduit exactement les memes fichiers CSV : le
    tri par `complaint_id`, la seed fixe et l'allocation deterministe de
    l'echantillon garantissent des empreintes sha256 identiques. Seul le champ
    `generated_at` de `split_metadata.json` change d'une execution a l'autre.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S")
    cfg.ensure_dirs()

    brut = load_raw()
    propre, rapport = clean(brut)
    propre, rapport_rares = handle_rare_classes(propre)
    rapport["9_classes_rares"] = {
        "supprimees": rapport_rares["lignes_supprimees"], "restantes": len(propre),
    }
    rapport["classes_rares"] = rapport_rares

    cfg.CLEANING_REPORT_FILE.write_text(
        json.dumps(rapport, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Ecrit %s", cfg.CLEANING_REPORT_FILE.name)

    metadata = split_and_save(propre)

    print("\n" + "=" * 74)
    print("RAPPORT DE NETTOYAGE")
    print("=" * 74)
    print(rapport_nettoyage_df(rapport).to_string(index=False))
    print("\n" + "=" * 74)
    print("DISTRIBUTION FINALE")
    print("=" * 74)
    dist = metadata["distribution"]
    print(pd.DataFrame({
        "corpus": {c: dist["corpus"][c]["n"] for c in cfg.CLASS_ORDER},
        "%": {c: dist["corpus"][c]["pct"] for c in cfg.CLASS_ORDER},
        "train": {c: dist["train"][c]["n"] for c in cfg.CLASS_ORDER},
        "test": {c: dist["test"][c]["n"] for c in cfg.CLASS_ORDER},
        "eval_2000": {c: dist["eval_sample"][c]["n"] for c in cfg.CLASS_ORDER},
    }).to_string())
    ratio = max(d["n"] for d in dist["corpus"].values()) / min(
        d["n"] for d in dist["corpus"].values())
    print(f"\nratio majoritaire / minoritaire = {ratio:.1f}")
    print("\n" + "=" * 74)
    print("VERIFICATION DE LA STRATIFICATION")
    print("=" * 74)
    print(verifie_stratification(metadata).to_string())
    return metadata


if __name__ == "__main__":
    main()
