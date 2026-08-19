"""Metriques de l'approche LLM (etape 2).

CE MODULE NE CALCULE AUCUNE METRIQUE LUI-MEME. Il relit un journal de campagne,
l'aligne sur la source, et DELEGUE a `metrics.py`, qui reste fige. C'est la
condition pour que le LLM et le ML de l'etape 3 soient mesures par le meme code :
la moindre reimplementation d'un F1 ici viderait la comparaison finale de son
sens.

Ce que ce module ajoute, et que `metrics.py` ne peut pas connaitre :

  - l'ALIGNEMENT sur les `complaint_id`, et le constat des manquants ;
  - le COUT REEL, agrege depuis les `usage` de chaque appel, jamais depuis les
    hypotheses de `config` ;
  - les taux propres au LLM : non-conformite de format, echec HTTP, part des
    tokens servis par le cache de prefixe.

DEUX ECHECS DE NATURE DIFFERENTE
--------------------------------
Ils ne doivent pas etre confondus, et ce module les separe :

  - PARSE_ERROR : le modele a repondu, mais hors format. C'est une DEFAILLANCE
    DU MODELE, comptee comme erreur dans le F1-macro (decision de l'etape 2).
  - erreur HTTP : aucune reponse apres epuisement des tentatives. C'est une
    DEFAILLANCE D'INFRASTRUCTURE. Par defaut ces lignes sont EXCLUES du F1 --
    y compris elles mesurerait la fiabilite du reseau, pas la qualite du
    modele -- mais leur taux est rapporte, et le nombre de lignes reellement
    evaluees est toujours affiche a cote du nombre attendu.
    `erreurs_http="compter_comme_erreur"` permet l'autre convention.

CONTRAINTE IMPOSEE A L'ETAPE 3 -- A LIRE AVANT DE COMMENCER LE ML
-----------------------------------------------------------------
Exclure les erreurs HTTP change le NOMBRE DE LIGNES evaluees. Si le LLM perd
3 appels sur 2 000 et que le ML est evalue sur les 2 000, `compare_results()`
refusera l'alignement -- a juste titre : la comparaison n'est valide que sur les
memes lignes.

L'etape 3 DOIT donc restreindre son evaluation au sous-ensemble reellement
evalue par le LLM. `exporte_ids_evalues()` ecrit cette liste ; `charge_ids_evalues()`
la relit. Ce n'est pas un detail d'implementation a decouvrir au moment de la
comparaison : c'est une contrainte de conception de l'etape 3.

Avec 5 tentatives et un backoff exponentiel, ce taux devrait etre proche de
zero. S'il ne l'est pas, ce n'est PAS un incident a rattraper discretement mais
un RESULTAT : une approche dont 1 % des appels echouent malgre 5 tentatives pose
un probleme de disponibilite que la recommandation doit porter, au meme titre
que sa latence.

LIMITE CONNUE DE LA MATRICE DE CONFUSION
----------------------------------------
`metrics.evaluate()` construit la matrice avec `labels=CLASS_ORDER`. Une
prediction `PARSE_ERROR` n'appartenant a aucune de ces 9 classes, sklearn EXCLUT
l'echantillon de la matrice : la somme de la matrice vaut alors n moins le
nombre de PARSE_ERROR. Verifie empiriquement.

Le F1-macro, lui, est correct : la classe vraie perd bien un vrai positif, son
rappel baisse. Seule la FIGURE est incomplete. `n_hors_matrice` remonte cet
ecart pour qu'il soit legende a l'etape 4 plutot que decouvert en la lisant.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from . import config as cfg
from . import metrics as mt
from .llm_runner import charge_resultats

# ===========================================================================
# Alignement
# ===========================================================================
def _selectionne_passe(journal: list[dict], execution: int | None) -> list[dict]:
    """Restreint le journal a une seule passe d'execution.

    Un journal de test de determinisme contient les MEMES lignes cinq fois.
    Les evaluer ensemble donnerait un n quintuple et un F1 sans signification :
    on impose donc de choisir une passe.
    """
    passes = sorted({e.get("execution", 1) for e in journal})
    if execution is None:
        if len(passes) > 1:
            execution = passes[0]
            print(f"  [eval] {len(passes)} passes presentes {passes} ; "
                  f"evaluation de la passe {execution}. "
                  f"Utiliser analyse_determinisme() pour les comparer.")
        else:
            execution = passes[0] if passes else 1
    return [e for e in journal if e.get("execution", 1) == execution]


def aligne(journal: list[dict], source_df: pd.DataFrame
           ) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Aligne le journal sur la source par `complaint_id`.

    L'alignement se fait par jointure explicite sur l'identifiant, jamais par
    position : rien ne garantit que le JSONL soit dans l'ordre de la source
    (reprise, reordonnancement, campagne en plusieurs fois).

    Retourne (df_aligne, diagnostic) ou `diagnostic` recense les identifiants
    attendus mais absents, et les identifiants journalises hors source.
    """
    attendus = [str(i) for i in source_df[cfg.ID_COL]]
    jrn = pd.DataFrame(journal)
    if jrn.empty:
        raise ValueError("Journal vide : aucune prediction a evaluer.")

    jrn[cfg.ID_COL] = jrn[cfg.ID_COL].astype(str)
    doublons = jrn[cfg.ID_COL].duplicated().sum()
    if doublons:
        raise ValueError(
            f"{doublons} identifiant(s) en double dans le journal pour une meme "
            f"passe. Evaluer des doublons fausserait le n et les poids. "
            f"Verifier le fichier ou preciser `execution`."
        )

    src = source_df.copy()
    src[cfg.ID_COL] = src[cfg.ID_COL].astype(str)

    presents = set(jrn[cfg.ID_COL])
    manquants = [i for i in attendus if i not in presents]
    hors_source = sorted(presents - set(attendus))

    # Jointure a GAUCHE sur la source : l'ordre de la source fait foi, ce qui
    # garantit que les poids HT restent alignes sur les predictions.
    fusion = src.merge(jrn, on=cfg.ID_COL, how="inner", suffixes=("", "_jrn"))

    diagnostic = {
        "n_attendu": len(attendus),
        "n_journalise": len(jrn),
        "n_aligne": len(fusion),
        "ids_manquants": manquants,
        "n_manquants": len(manquants),
        "ids_hors_source": hors_source,
    }
    return fusion, diagnostic


# ===========================================================================
# Agregats propres au LLM
# ===========================================================================
def _somme(serie: pd.Series) -> float | None:
    """Somme en ignorant les absents. `None` si tout est absent."""
    valeurs = pd.to_numeric(serie, errors="coerce").dropna()
    return float(valeurs.sum()) if len(valeurs) else None


def agrege_usage(df: pd.DataFrame) -> dict[str, Any]:
    """Agregats de tokens, de cout et de cache, tous issus de `usage`.

    Aucune valeur de `config` n'intervient : c'est precisement l'hypothese
    `AVG_COMPLAINT_TOKENS` que ces mesures ont remplace (257 -> 245 le
    2026-08-19).
    """
    tokens_in = _somme(df["tokens_in"]) if "tokens_in" in df else None
    tokens_out = _somme(df["tokens_out"]) if "tokens_out" in df else None
    caches = _somme(df["tokens_caches"]) if "tokens_caches" in df else None

    couts = pd.to_numeric(df.get("cout_usd"), errors="coerce") if "cout_usd" in df else pd.Series(dtype=float)
    cout_calculable = len(couts.dropna()) == len(df) and len(df) > 0
    cout_total = float(couts.dropna().sum()) if len(couts.dropna()) else None

    n = len(df)
    agg: dict[str, Any] = {
        "tokens_in_total": tokens_in,
        "tokens_out_total": tokens_out,
        "tokens_caches_total": caches,
        "tokens_in_moyen": (tokens_in / n) if tokens_in is not None and n else None,
        "tokens_out_moyen": (tokens_out / n) if tokens_out is not None and n else None,
        "cout_reel_usd": cout_total,
        "cout_calculable": cout_calculable,
    }
    if cout_total is not None and n:
        # Cle lue telle quelle par `metrics.compare_results()`.
        agg["cout_pour_1000_predictions_usd"] = cout_total / n * 1000

    # Part des tokens d'entree servie depuis le cache de prefixe. C'est la
    # MESURE qui a corrige la projection de cache du notebook : celle-ci
    # supposait une activation totale, alors que la couverture plafonne et que
    # l'activation depend du rechauffement du cache.
    if tokens_in and caches is not None:
        agg["taux_tokens_caches"] = caches / tokens_in
    else:
        agg["taux_tokens_caches"] = None

    if "service_tier" in df:
        tiers = df["service_tier"].dropna().unique().tolist()
        agg["service_tiers"] = tiers
    return agg


def agrege_latences(df: pd.DataFrame) -> dict[str, Any]:
    """Latences par appel unitaire, sous leurs DEUX formes.

    `latence_s` (inference seule) est ce qui se compare a l'etape 3 et ce qui
    doit figurer dans la recommandation : c'est la vitesse du modele.
    `latence_avec_attentes_s` inclut le limiteur de debit et les backoffs :
    c'est ce que subit reellement un utilisateur sur le tier courant.

    Les deux sont rapportees parce que le client subira les deux, et que les
    confondre ferait passer une contrainte tarifaire pour une lenteur du modele.
    """
    out: dict[str, Any] = {}
    if "latence_s" in df:
        lat = pd.to_numeric(df["latence_s"], errors="coerce").dropna()
        out.update(mt.latency_stats(lat.tolist()))
    if "latence_avec_attentes_s" in df:
        lat2 = pd.to_numeric(df["latence_avec_attentes_s"], errors="coerce").dropna()
        stats2 = mt.latency_stats(lat2.tolist())
        out.update({f"attentes_{k}": v for k, v in stats2.items()})
    return out


def agrege_conformite(df: pd.DataFrame) -> dict[str, Any]:
    """Taux de non-conformite de format et d'echec HTTP, separement."""
    n = len(df)
    statuts = df["statut_parsing"] if "statut_parsing" in df else pd.Series(dtype=str)
    erreurs_http = df["erreur_http"].notna() if "erreur_http" in df else pd.Series([False] * n)

    n_http = int(erreurs_http.sum())
    # Une ligne en echec HTTP n'a pas de reponse : son statut de parsing
    # ("vide") ne dit rien du modele. On ne la compte pas comme non-conformite
    # de format, sans quoi une panne reseau se lirait comme un defaut du prompt.
    conformite = statuts[~erreurs_http.values] if n else statuts
    n_format = int((conformite != "ok").sum())

    return {
        "n_appels": n,
        "n_erreurs_http": n_http,
        "taux_erreur_http": (n_http / n) if n else 0.0,
        "n_parse_error": n_format,
        "taux_parse_error": (n_format / len(conformite)) if len(conformite) else 0.0,
        "statuts_parsing": {s: int(c) for s, c in statuts.value_counts().items()},
    }


# ===========================================================================
# Evaluation d'une campagne
# ===========================================================================
def evalue_campagne(chemin_jsonl: str | Path,
                    source_df: pd.DataFrame,
                    *,
                    poids: pd.Series | None = None,
                    execution: int | None = None,
                    erreurs_http: str = "exclure",
                    with_ci: bool = True,
                    labels: Iterable[str] | None = None,
                    verbeux: bool = True) -> dict[str, Any]:
    """Evalue un journal de campagne. Retourne un dict pret pour `compare_results`.

    Parameters
    ----------
    source_df : lignes attendues, portant `complaint_id` et l'etiquette vraie.
        Si elle porte `sampling_weight`, les poids de Horvitz-Thompson sont
        extraits et transmis a `metrics.evaluate()`.
    poids : poids explicites. A fournir quand `load_eval_sample()` les rend
        separement. Ils sont REORDONNES sur l'alignement, jamais supposes dans
        le bon ordre.
    erreurs_http : "exclure" (defaut) ou "compter_comme_erreur".

    Le garde-fou de ponderation de `metrics.py` n'est JAMAIS contourne : la
    sous-table reellement evaluee est passee en `source=`, donc un oubli de
    poids sur un echantillon pondere leve toujours.
    """
    if erreurs_http not in ("exclure", "compter_comme_erreur"):
        raise ValueError("erreurs_http doit valoir 'exclure' ou "
                         "'compter_comme_erreur'.")

    journal = charge_resultats(chemin_jsonl)
    if not journal:
        raise ValueError(f"Aucun enregistrement lisible dans {chemin_jsonl}.")
    journal = _selectionne_passe(journal, execution)

    df, diagnostic = aligne(journal, source_df)

    # --- poids, reordonnes sur l'alignement --------------------------------
    if poids is not None:
        serie = pd.Series(list(poids), index=[str(i) for i in source_df[cfg.ID_COL]])
        df = df.copy()
        df[cfg.WEIGHT_COL] = df[cfg.ID_COL].map(serie).values

    # --- agregats, calcules AVANT tout filtrage ----------------------------
    conformite = agrege_conformite(df)
    usage = agrege_usage(df)
    latences = agrege_latences(df)

    # --- perimetre du F1 ---------------------------------------------------
    if erreurs_http == "exclure" and "erreur_http" in df:
        df_eval = df[df["erreur_http"].isna()].copy()
    else:
        df_eval = df.copy()
        if "erreur_http" in df_eval:
            perdus = df_eval["erreur_http"].notna()
            df_eval.loc[perdus, "label_predit"] = cfg.PARSE_ERROR

    if df_eval.empty:
        raise ValueError("Aucune ligne evaluable : tous les appels ont echoue.")

    y_true = df_eval["label_vrai"].tolist()
    y_pred = df_eval["label_predit"].tolist()
    w = df_eval[cfg.WEIGHT_COL] if cfg.WEIGHT_COL in df_eval.columns else None

    resultats = mt.evaluate(
        y_true, y_pred,
        latencies=pd.to_numeric(df_eval.get("latence_s"), errors="coerce")
                    .dropna().tolist() if "latence_s" in df_eval else None,
        labels=list(labels) if labels is not None else None,
        sample_weight=None if w is None else w.tolist(),
        source=df_eval,          # garde-fou de ponderation actif
        with_ci=with_ci,
    )

    # --- enrichissement ----------------------------------------------------
    resultats.update(usage)
    resultats.update(conformite)
    resultats.update({k: v for k, v in latences.items()
                      if k.startswith("attentes_")})
    resultats["alignement"] = diagnostic
    # Identifiants REELLEMENT evalues : c'est sur eux que porte la verification
    # "memes lignes" de compare_campagnes(), pas sur un simple compte.
    resultats["_ids_evalues"] = df_eval[cfg.ID_COL].tolist()
    # Conserves pour la FIGURE etendue uniquement (cf. figure_confusion_etendue).
    # Prefixes par "_" : ce ne sont pas des metriques, et rien dans le tableau
    # comparatif ne doit les lire.
    resultats["_y_true"] = y_true
    resultats["_y_pred"] = y_pred
    resultats["_poids"] = None if w is None else w.tolist()
    resultats["erreurs_http_traitement"] = erreurs_http
    resultats["n_evalue"] = len(df_eval)
    resultats["n_attendu"] = diagnostic["n_attendu"]
    resultats["campagne"] = str(chemin_jsonl)
    if "style" in df:
        resultats["styles"] = sorted(df["style"].dropna().unique().tolist())
    if "modele" in df:
        resultats["modeles"] = sorted(df["modele"].dropna().unique().tolist())

    # Ecart entre le n evalue et la somme de la matrice de confusion : ce sont
    # les PARSE_ERROR, que sklearn exclut de la matrice faute de colonne.
    somme_matrice = int(np.asarray(resultats["confusion_matrix"]).sum()) \
        if w is None else None
    resultats["n_hors_matrice"] = (
        None if somme_matrice is None else len(df_eval) - somme_matrice
    )

    if verbeux:
        _affiche(resultats, diagnostic)
    return resultats


def _affiche(r: dict[str, Any], diag: dict[str, Any]) -> None:
    """Compte rendu lisible d'une campagne."""
    print(f"\n{'=' * 70}")
    print(f"CAMPAGNE : {Path(r['campagne']).name}")
    print(f"{'=' * 70}")
    print(f"  lignes attendues       : {diag['n_attendu']}")
    print(f"  lignes journalisees    : {diag['n_journalise']}")
    print(f"  lignes evaluees        : {r['n_evalue']}"
          f"   (erreurs HTTP : {r['erreurs_http_traitement']})")
    if diag["n_manquants"]:
        apercu = ", ".join(diag["ids_manquants"][:5])
        suite = " ..." if diag["n_manquants"] > 5 else ""
        print(f"  ATTENTION {diag['n_manquants']} id(s) ABSENT(S) du journal : "
              f"{apercu}{suite}")
    if diag["ids_hors_source"]:
        print(f"  ATTENTION {len(diag['ids_hors_source'])} id(s) journalise(s) "
              f"hors source (ignores).")

    print(f"\n  F1-macro               : {r['f1_macro']:.4f}"
          + (f"   IC 95 % [{r['f1_macro_ci']['bas']:.4f} ; "
             f"{r['f1_macro_ci']['haut']:.4f}]" if r.get("f1_macro_ci") else ""))
    print(f"  accuracy               : {r['accuracy']:.4f}")
    print(f"  pondere (HT)           : {r['weighted']}")

    print(f"\n  taux PARSE_ERROR       : {r['taux_parse_error']:.1%} "
          f"({r['n_parse_error']}/{r['n_appels'] - r['n_erreurs_http']})")
    print(f"  taux erreur HTTP       : {r['taux_erreur_http']:.1%} "
          f"({r['n_erreurs_http']}/{r['n_appels']})")
    if r.get("statuts_parsing"):
        print(f"  detail des statuts     : {r['statuts_parsing']}")
    if r.get("n_hors_matrice"):
        print(f"  hors matrice confusion : {r['n_hors_matrice']} "
              f"(PARSE_ERROR, exclus par sklearn faute de colonne)")

    if r.get("latence_p50_s") is not None:
        print(f"\n  latence modele    p50/p95/max : "
              f"{r['latence_p50_s']:.2f} / {r['latence_p95_s']:.2f} / "
              f"{r['latence_max_s']:.2f} s")
    if r.get("attentes_latence_p50_s") is not None:
        print(f"  latence + attentes p50/p95/max : "
              f"{r['attentes_latence_p50_s']:.2f} / "
              f"{r['attentes_latence_p95_s']:.2f} / "
              f"{r['attentes_latence_max_s']:.2f} s")

    if r.get("tokens_in_moyen") is not None:
        # `tokens_in` inclut le PREFIXE ; `AVG_COMPLAINT_TOKENS` ne couvre que
        # le TEXTE. Les comparer directement afficherait un ecart de +260 tokens
        # qui n'est que la longueur du prefixe, et ferait croire a une mesure
        # deux fois superieure a la prevision.
        modeles = r.get("modeles") or []
        cle = next((c for c in cfg.CACHED_PREFIX_TOKENS_PAR_MODELE
                    if any(c.split("-")[0] in m for m in modeles)), None)
        prefixe = cfg.CACHED_PREFIX_TOKENS_PAR_MODELE.get(
            cle, cfg.CACHED_PREFIX_TOKENS)
        texte = r["tokens_in_moyen"] - prefixe
        ecart = texte - cfg.AVG_COMPLAINT_TOKENS
        print(f"\n  tokens d'entree moyens : {r['tokens_in_moyen']:.1f}"
              f"   (prefixe {prefixe} + texte {texte:.1f})")
        print(f"  texte seul vs mesure du 2026-08-19 : {texte:.1f} contre "
              f"{cfg.AVG_COMPLAINT_TOKENS}, ecart {ecart:+.1f} "
              f"({ecart / cfg.AVG_COMPLAINT_TOKENS:+.1%})")
    if r.get("taux_tokens_caches") is not None:
        print(f"  tokens servis en cache : {r['taux_tokens_caches']:.1%}")
    if r.get("cout_calculable"):
        print(f"  cout REEL              : {r['cout_reel_usd']:.6f} $ "
              f"({r.get('cout_pour_1000_predictions_usd', 0):.4f} $ / 1000)")
    else:
        print("  cout                   : NON CALCULABLE (tarif inconnu)")


# ===========================================================================
# Comparaison multi-campagnes
# ===========================================================================
def compare_campagnes(campagnes: dict[str, Any],
                      source_df: pd.DataFrame | None = None,
                      *,
                      poids: pd.Series | None = None,
                      exiger_memes_lignes: bool = True,
                      **kwargs) -> pd.DataFrame:
    """Compare plusieurs campagnes cote a cote, avec le F1 par classe.

    Parameters
    ----------
    campagnes : {nom: chemin JSONL} ou {nom: dict deja retourne par
        `evalue_campagne`}. Les deux formes se melangent.
    exiger_memes_lignes : verifie que toutes les campagnes portent EXACTEMENT
        les memes identifiants. `metrics.compare_results()` verifie deja
        l'egalite des n, mais deux campagnes de meme taille peuvent porter des
        lignes differentes -- et la comparaison serait alors sans valeur.

    Le tableau est produit par `metrics.compare_results()` : aucune metrique
    n'est recalculee ici.
    """
    resultats: dict[str, dict] = {}
    for nom, valeur in campagnes.items():
        if isinstance(valeur, dict):
            resultats[nom] = valeur
        else:
            if source_df is None:
                raise ValueError(
                    f"'{nom}' est un chemin : `source_df` est necessaire pour "
                    f"l'evaluer. Fournir la source ou passer un resultat deja "
                    f"calcule."
                )
            resultats[nom] = evalue_campagne(valeur, source_df, poids=poids,
                                             **kwargs)

    if exiger_memes_lignes:
        _verifie_memes_lignes(resultats)

    tableau = mt.compare_results(resultats)

    # Colonnes propres au LLM, absentes du tableau generique de metrics.py.
    for cle, titre in [("taux_parse_error", "PARSE_ERROR"),
                       ("taux_erreur_http", "erreur HTTP"),
                       ("taux_tokens_caches", "tokens en cache"),
                       ("tokens_in_moyen", "tokens entree moy."),
                       ("cout_reel_usd", "cout reel ($)")]:
        if any(cle in r for r in resultats.values()):
            tableau[titre] = [resultats[nom].get(cle) for nom in tableau.index]
    return tableau


def _verifie_memes_lignes(resultats: dict[str, dict]) -> None:
    """Leve si deux campagnes ne portent pas EXACTEMENT les memes identifiants.

    Comparer les JEUX D'IDENTIFIANTS, pas seulement leur cardinal : deux
    campagnes de meme taille peuvent porter des lignes differentes si l'une a
    perdu un appel et l'autre un autre. Le tableau melangerait alors l'effet du
    prompt et l'effet de l'echantillon.

    Les resultats sans clef d'alignement (une evaluation ML de l'etape 3, par
    exemple) sont ignores : ils ne viennent pas d'un journal LLM.
    """
    jeux = {nom: set(r["_ids_evalues"])
            for nom, r in resultats.items() if "_ids_evalues" in r}
    if len(jeux) < 2:
        return

    reference_nom, reference = next(iter(jeux.items()))
    for nom, ids in jeux.items():
        if ids == reference:
            continue
        manquants = sorted(reference - ids)[:5]
        en_trop = sorted(ids - reference)[:5]
        raise ValueError(
            f"'{nom}' et '{reference_nom}' ne portent pas les memes lignes "
            f"({len(ids)} contre {len(reference)}).\n"
            f"  absents de '{nom}'  : {manquants or 'aucun'}\n"
            f"  absents de '{reference_nom}' : {en_trop or 'aucun'}\n"
            f"Comparer des prompts sur des lignes differentes melange l'effet "
            f"du prompt et l'effet de l'echantillon. Reprendre les campagnes "
            f"incompletes (la reprise ne redepense rien) ou restreindre "
            f"explicitement a l'intersection."
        )

    incomplets = {nom: r["alignement"]["n_manquants"]
                  for nom, r in resultats.items()
                  if r.get("alignement", {}).get("n_manquants")}
    if incomplets:
        print(f"  [comparaison] campagnes incompletes mais alignees sur les "
              f"memes lignes : {incomplets}")


# ===========================================================================
# Determinisme
# ===========================================================================
def analyse_determinisme(chemin_jsonl: str | Path,
                         verbeux: bool = True) -> dict[str, Any]:
    """Mesure la stabilite des reponses sur plusieurs passes des MEMES lignes.

    Temperature 0 ne garantit PAS le determinisme. Le ML de l'etape 3 l'est par
    construction ; le LLM ne l'est pas necessairement. C'est un critere qui
    pesera dans la recommandation, ZenAssist n'ayant aucune competence IA en
    interne pour diagnostiquer une reponse qui change sans raison.

    Ce module MESURE et n'INTERPRETE PAS : 100 % de stabilite retire un argument
    contre le LLM, moins de 100 % en fournit un. Les deux sont des resultats.
    """
    journal = charge_resultats(chemin_jsonl)
    if not journal:
        raise ValueError(f"Aucun enregistrement lisible dans {chemin_jsonl}.")

    df = pd.DataFrame(journal)
    df[cfg.ID_COL] = df[cfg.ID_COL].astype(str)
    passes = sorted(df.get("execution", pd.Series([1] * len(df))).unique().tolist())

    lignes, instables = [], []
    for cid, groupe in df.groupby(cfg.ID_COL):
        predits = groupe["label_predit"].tolist()
        distincts = sorted(set(predits))
        stable = len(distincts) == 1
        lat = pd.to_numeric(groupe.get("latence_s"), errors="coerce").dropna()
        info = {
            cfg.ID_COL: cid,
            "n_passes": len(predits),
            "stable": stable,
            "labels_concurrents": {l: predits.count(l) for l in distincts},
            "latence_min_s": float(lat.min()) if len(lat) else None,
            "latence_max_s": float(lat.max()) if len(lat) else None,
            "latence_ecart_type_s": float(lat.std(ddof=0)) if len(lat) > 1 else None,
        }
        lignes.append(info)
        if not stable:
            instables.append(info)

    n = len(lignes)
    complets = [l for l in lignes if l["n_passes"] == len(passes)]
    resultat = {
        "n_exemples": n,
        "n_passes": len(passes),
        "passes": passes,
        "n_exemples_complets": len(complets),
        "n_stables": sum(1 for l in lignes if l["stable"]),
        "taux_stabilite": (sum(1 for l in lignes if l["stable"]) / n) if n else None,
        "instables": instables,
        "detail": lignes,
    }

    lat_tous = pd.to_numeric(df.get("latence_s"), errors="coerce").dropna()
    ecarts = [l["latence_ecart_type_s"] for l in lignes
              if l["latence_ecart_type_s"] is not None]
    resultat["latence_ecart_type_median_s"] = (
        float(np.median(ecarts)) if ecarts else None)
    resultat["latence_globale"] = mt.latency_stats(lat_tous.tolist())

    if verbeux:
        print(f"\n{'=' * 70}")
        print(f"DETERMINISME : {Path(str(chemin_jsonl)).name}")
        print(f"{'=' * 70}")
        print(f"  {n} exemple(s), {len(passes)} passe(s) {passes}")
        if len(complets) != n:
            print(f"  ATTENTION {n - len(complets)} exemple(s) n'ont pas toutes "
                  f"les passes -- stabilite calculee sur ce qui existe.")
        if resultat["taux_stabilite"] is not None:
            print(f"  reponses identiques sur toutes les passes : "
                  f"{resultat['n_stables']}/{n} "
                  f"({resultat['taux_stabilite']:.1%})")
        for i in instables:
            print(f"    - {i[cfg.ID_COL]} : {i['labels_concurrents']}")
        if resultat["latence_ecart_type_median_s"] is not None:
            print(f"  ecart-type median de latence sur un meme texte : "
                  f"{resultat['latence_ecart_type_median_s']:.3f} s")
    return resultat


# ===========================================================================
# Sous-ensemble reellement evalue -- contrat avec l'etape 3
# ===========================================================================
def ids_evalues(resultats: dict[str, Any]) -> list[str]:
    """Identifiants REELLEMENT entres dans le calcul du F1 d'une campagne.

    Ce n'est pas la meme chose que les identifiants de la source : les appels
    perdus apres epuisement des tentatives en sont absents (convention
    `erreurs_http="exclure"`).
    """
    if "_ids_evalues" not in resultats:
        raise KeyError(
            "Ce dictionnaire ne provient pas de `evalue_campagne()` : il ne "
            "porte pas la liste des identifiants evalues."
        )
    return list(resultats["_ids_evalues"])


def exporte_ids_evalues(resultats: dict[str, Any],
                        chemin: str | Path | None = None) -> Path:
    """Ecrit la liste des identifiants evalues, pour que l'ETAPE 3 s'y restreigne.

    POURQUOI CE FICHIER EXISTE
    --------------------------
    Le LLM peut perdre des lignes (echec HTTP apres 5 tentatives) ; le ML, lui,
    n'en perd aucune. Evaluer le ML sur 2 000 lignes et le LLM sur 1 997
    produirait deux F1 non comparables -- et `metrics.compare_results()` le
    refuserait, a juste titre.

    L'etape 3 doit donc charger ce fichier et restreindre son evaluation a ces
    identifiants EXACTS. C'est une contrainte de conception, pas un ajustement
    de derniere minute.

    Le fichier ne contient QUE des identifiants : aucun texte de reclamation. Il
    est neanmoins ecrit sous `data/llm/` par defaut, `reports/` etant reserve
    aux agregats.
    """
    cfg.ensure_llm_dirs()
    if chemin is None:
        nom = Path(resultats.get("campagne", "campagne")).stem
        chemin = cfg.LLM_DIR / f"{nom}_ids_evalues.json"
    chemin = Path(chemin)

    charge = {
        "campagne": resultats.get("campagne"),
        "styles": resultats.get("styles"),
        "modeles": resultats.get("modeles"),
        "n_attendu": resultats.get("n_attendu"),
        "n_evalue": resultats.get("n_evalue"),
        "n_erreurs_http": resultats.get("n_erreurs_http"),
        "erreurs_http_traitement": resultats.get("erreurs_http_traitement"),
        "avertissement": (
            "L'etape 3 doit restreindre son evaluation a ces identifiants "
            "exacts. Comparer un ML evalue sur toutes les lignes a un LLM "
            "evalue sur un sous-ensemble n'a pas de sens, et "
            "metrics.compare_results() le refusera."
        ),
        "ids_evalues": ids_evalues(resultats),
    }
    chemin.write_text(json.dumps(charge, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    print(f"  [export] {len(charge['ids_evalues'])} identifiant(s) evalue(s) "
          f"-> {chemin}")
    return chemin


def charge_ids_evalues(chemin: str | Path) -> list[str]:
    """Relit un export d'identifiants. Point d'entree de l'etape 3."""
    charge = json.loads(Path(chemin).read_text(encoding="utf-8"))
    return [str(i) for i in charge["ids_evalues"]]


def restreint_au_sous_ensemble(df: pd.DataFrame,
                               ids: Iterable[str]) -> pd.DataFrame:
    """Restreint un DataFrame aux identifiants donnes, dans l'ordre du DataFrame.

    Destine a l'etape 3. Leve si un identifiant attendu manque, plutot que de
    rendre silencieusement un sous-ensemble plus petit que demande.
    """
    ids = [str(i) for i in ids]
    serie = df[cfg.ID_COL].astype(str)
    manquants = sorted(set(ids) - set(serie))
    if manquants:
        raise ValueError(
            f"{len(manquants)} identifiant(s) evalue(s) par le LLM sont absents "
            f"du DataFrame fourni : {manquants[:5]}. La comparaison ne peut pas "
            f"porter sur les memes lignes."
        )
    return df[serie.isin(ids)].copy()


# ===========================================================================
# Figure etendue -- AFFICHAGE UNIQUEMENT
# ===========================================================================
def figure_confusion_etendue(resultats: dict[str, Any],
                             save_path: str | Path | None = None,
                             *,
                             title: str = "Matrice de confusion — LLM",
                             normalize: bool = True):
    """Matrice de confusion avec une COLONNE supplementaire pour PARSE_ERROR.

    CE CHEMIN EST STRICTEMENT SEPARE DU CALCUL DES METRIQUES
    --------------------------------------------------------
    Le referentiel etendu (`CLASS_ORDER + [PARSE_ERROR]`) sert UNIQUEMENT a
    cette figure. Le F1-macro reste calcule sur les 9 classes, par
    `evalue_campagne()`, sans que PARSE_ERROR y figure jamais.

    La raison est arithmetique : le F1-macro est une moyenne NON PONDEREE sur
    les classes. Ajouter PARSE_ERROR au calcul creerait une dixieme classe
    fictive, dont le F1 vaudrait toujours 0 (aucune ligne n'a PARSE_ERROR pour
    etiquette vraie), et diviserait mecaniquement le F1-macro par 10/9 -- soit
    -11 % sans qu'aucune prediction ait change. Les deux chemins doivent donc
    rester separes, et ils le sont : cette fonction ne rend aucune metrique.

    Ce que la figure montre, et que le F1 ne dit pas : QUELLES classes vraies
    engendrent des reponses hors format. Une non-conformite concentree sur une
    ou deux classes ambigues est un signal de prompt ; repartie uniformement,
    c'est un signal de modele. Les deux orientent la phase 3 differemment.

    La LIGNE PARSE_ERROR est vide, et c'est correct : aucune reclamation n'a
    PARSE_ERROR pour etiquette de reference.
    """
    if "_y_true" not in resultats:
        raise KeyError(
            "Ce dictionnaire ne provient pas de `evalue_campagne()`. "
            "La figure etendue a besoin des vecteurs de predictions."
        )

    labels_etendus = list(cfg.CLASS_ORDER) + [cfg.PARSE_ERROR]
    poids = resultats.get("_poids")

    # `sample_weight` est nomme sans defaut dans metrics.py : on le passe
    # explicitement, y compris a None. Le garde-fou reste donc actif.
    fig, ax = mt.plot_confusion_matrix(
        resultats["_y_true"], resultats["_y_pred"],
        labels=labels_etendus,
        save_path=save_path,
        sample_weight=poids,
        source=None,
        normalize=normalize,
        title=title,
    )
    return fig, ax


def hors_format_par_classe(resultats: dict[str, Any]) -> pd.DataFrame:
    """Ventilation des PARSE_ERROR par classe VRAIE. Complement chiffre de la figure.

    Repond a "quelles classes engendrent des reponses hors format", sans avoir a
    lire une figure. Ne rend, la encore, aucune metrique de qualite.
    """
    if "_y_true" not in resultats:
        raise KeyError("Ce dictionnaire ne provient pas de `evalue_campagne()`.")

    df = pd.DataFrame({"vrai": resultats["_y_true"],
                       "predit": resultats["_y_pred"]})
    total = df.groupby("vrai").size()
    hors = df[df["predit"] == cfg.PARSE_ERROR].groupby("vrai").size()

    tab = pd.DataFrame({
        "n": total,
        "hors_format": hors.reindex(total.index).fillna(0).astype(int),
    })
    tab["taux"] = (tab["hors_format"] / tab["n"]).round(4)
    return tab.reindex([c for c in cfg.CLASS_ORDER if c in tab.index])


# ===========================================================================
# Derive de version derriere un alias
# ===========================================================================
def versions_servies(chemin_jsonl: str | Path) -> dict[str, set]:
    """Versions REELLEMENT servies, par alias appele, dans un journal.

    Retourne {alias: {versions resolues}}. Plus d'une version pour un meme
    alias signale un repointage EN COURS de campagne.
    """
    par_alias: dict[str, set] = {}
    for e in charge_resultats(chemin_jsonl):
        alias, resolue = e.get("modele"), e.get("modele_resolu")
        if alias is None or resolue is None:
            continue          # appel perdu : aucune version servie
        par_alias.setdefault(alias, set()).add(resolue)
    return par_alias


def verifie_derive_version(campagnes: dict[str, str | Path],
                           strict: bool = False) -> dict[str, Any]:
    """Compare, POUR CHAQUE ALIAS, la version servie d'une campagne a l'autre.

    POURQUOI
    --------
    On appelle un alias (`mistral-small-latest`), pas une version figee, parce
    que c'est ce qu'un client fait en production. Mais le fournisseur peut
    repointer l'alias a tout moment. Une campagne menee avant et une campagne
    menee apres seraient alors comparees comme si elles portaient sur le meme
    modele -- et un ecart de F1 du au changement de modele serait attribue au
    prompt. Rien dans les donnees ne le signalerait.

    La comparaison se fait ALIAS PAR ALIAS. Deux alias differents servant deux
    versions differentes est le cas NORMAL (c'est toute la comparaison
    mistral-small / ministral) : ce n'est pas une derive.

    strict : lever au lieu d'avertir. Par defaut on avertit, car une derive
    detectee n'invalide pas forcement le travail -- elle doit surtout etre
    CONNUE et documentee dans la restitution.
    """
    observe: dict[str, dict[str, set]] = {}
    for nom, chemin in campagnes.items():
        observe[nom] = versions_servies(chemin)

    alias_tous = sorted({a for v in observe.values() for a in v})
    derives, interne = [], []

    for alias in alias_tous:
        par_campagne = {nom: v[alias] for nom, v in observe.items() if alias in v}
        for nom, versions in par_campagne.items():
            if len(versions) > 1:
                interne.append({"campagne": nom, "alias": alias,
                                "versions": sorted(versions)})
        vues = {v for versions in par_campagne.values() for v in versions}
        if len(vues) > 1:
            derives.append({"alias": alias, "versions": sorted(vues),
                            "par_campagne": {n: sorted(v)
                                             for n, v in par_campagne.items()}})

    # GARDE-FOU DE REGRESSION. Si `modele_resolu` est egal a l'alias, c'est que
    # la version n'a pas ete resolue : le champ `model` de la reponse renvoie
    # l'alias demande, et s'y fier rendrait toute cette fonction inoperante --
    # on comparerait des alias a des alias et aucune derive ne serait jamais
    # detectee. Le bug a existe, il est corrige, ce controle empeche son retour.
    non_resolus = [{"campagne": nom, "alias": a}
                   for nom, d in observe.items()
                   for a, versions in d.items() if a in versions]

    rapport = {"alias_suivis": alias_tous, "derives": derives,
               "derives_intra_campagne": interne,
               "versions_non_resolues": non_resolus,
               "stable": not derives and not interne,
               "detail": {n: {a: sorted(v) for a, v in d.items()}
                          for n, d in observe.items()}}

    if non_resolus:
        print("\n  ATTENTION : version non resolue (l'alias a ete journalise "
              "tel quel).\n  La detection de derive est INOPERANTE sur ces "
              f"campagnes : {non_resolus}")

    if not rapport["stable"]:
        lignes = ["", "=" * 70,
                  "DERIVE DE VERSION DETECTEE -- LES CAMPAGNES NE SONT PAS "
                  "COMPARABLES EN L'ETAT", "=" * 70]
        for d in derives:
            lignes.append(f"  alias {d['alias']} a servi {d['versions']}")
            for n, v in d["par_campagne"].items():
                lignes.append(f"      {n} : {v}")
        for i in interne:
            lignes.append(f"  campagne {i['campagne']} : l'alias {i['alias']} a "
                          f"change EN COURS DE ROUTE {i['versions']}")
        lignes.append("  Un ecart de F1 entre ces campagnes melangerait l'effet "
                      "du prompt et un changement de modele.")
        message = "\n".join(lignes)
        if strict:
            raise RuntimeError(message)
        print(message)
    else:
        print(f"  [versions] stable : {ana_resume(observe)}")
    return rapport


def ana_resume(observe: dict[str, dict[str, set]]) -> str:
    """Resume compact des versions servies, pour l'affichage."""
    paires = sorted({(a, v) for d in observe.values()
                     for a, vs in d.items() for v in vs})
    return ", ".join(f"{a} -> {v}" for a, v in paires) or "aucune version relevee"


# ===========================================================================
# Recapitulatif de facturation
# ===========================================================================
def recapitulatif_facturation(chemins: dict[str, str | Path] | str | Path,
                              verbeux: bool = True) -> dict[str, Any]:
    """Recapitulatif a confronter au tableau de bord Mistral.

    COMPTE TOUS LES APPELS EMIS, sans exception : toutes les passes du test de
    determinisme, et les tentatives ayant abouti a une reponse. C'est
    volontairement un perimetre DIFFERENT de celui de `evalue_campagne()`, qui
    ne retient qu'une passe et exclut les appels perdus.

    Confondre les deux perimetres serait l'erreur a ne pas commettre : le
    fournisseur facture tout ce qui a ete servi, y compris les 4 passes
    supplementaires du test de determinisme, qui ne comptent dans aucun F1.

    Un ecart notable avec la console signalerait soit une erreur de comptage,
    soit une tarification differente de `MODELS_PRICING` -- dans les deux cas il
    faut le savoir AVANT la campagne sur les 2 000, pas apres.
    """
    if isinstance(chemins, (str, Path)):
        chemins = {Path(chemins).stem: chemins}

    lignes, total = [], {"n_appels": 0, "tokens_in": 0, "tokens_out": 0,
                         "tokens_caches": 0, "cout": 0.0}
    for nom, chemin in chemins.items():
        journal = charge_resultats(chemin)
        if not journal:
            continue
        df = pd.DataFrame(journal)
        aboutis = df[df["erreur_http"].isna()] if "erreur_http" in df else df

        entree = {
            "campagne": nom,
            "n_appels_emis": len(df),
            "n_appels_factures": len(aboutis),
            "n_erreurs_http": int(len(df) - len(aboutis)),
            "passes": sorted(df["execution"].unique().tolist())
                      if "execution" in df else [1],
            "alias": sorted(df["modele"].dropna().unique().tolist())
                     if "modele" in df else [],
            "versions_servies": sorted(df["modele_resolu"].dropna().unique().tolist())
                                if "modele_resolu" in df else [],
            "tokens_in": int(_somme(aboutis["tokens_in"]) or 0),
            "tokens_out": int(_somme(aboutis["tokens_out"]) or 0),
            "tokens_caches": int(_somme(aboutis["tokens_caches"]) or 0),
            "cout_calcule_usd": float(_somme(aboutis["cout_usd"]) or 0.0),
        }
        entree["tokens_in_non_caches"] = entree["tokens_in"] - entree["tokens_caches"]
        entree["taux_cache"] = (entree["tokens_caches"] / entree["tokens_in"]
                                if entree["tokens_in"] else None)
        lignes.append(entree)
        total["n_appels"] += entree["n_appels_factures"]
        for c in ("tokens_in", "tokens_out", "tokens_caches"):
            total[c] += entree[c]
        total["cout"] += entree["cout_calcule_usd"]

    total["tokens_in_non_caches"] = total["tokens_in"] - total["tokens_caches"]
    total["taux_cache"] = (total["tokens_caches"] / total["tokens_in"]
                           if total["tokens_in"] else None)
    rapport = {"campagnes": lignes, "total": total}

    if verbeux:
        print(f"\n{'=' * 70}")
        print("RECAPITULATIF DE FACTURATION -- a comparer a la console Mistral")
        print(f"{'=' * 70}")
        for e in lignes:
            print(f"\n  {e['campagne']}")
            print(f"    alias appele(s)      : {', '.join(e['alias']) or '-'}")
            print(f"    version(s) servie(s) : {', '.join(e['versions_servies']) or '-'}")
            print(f"    passes               : {e['passes']}")
            print(f"    appels emis          : {e['n_appels_emis']}"
                  f"   (factures : {e['n_appels_factures']}, "
                  f"echecs HTTP : {e['n_erreurs_http']})")
            print(f"    tokens entree        : {e['tokens_in']:,}"
                  f"   dont {e['tokens_caches']:,} en cache"
                  + (f" ({e['taux_cache']:.1%})" if e["taux_cache"] is not None else ""))
            print(f"    tokens sortie        : {e['tokens_out']:,}")
            print(f"    cout calcule         : {e['cout_calcule_usd']:.6f} $")
        t = total
        print(f"\n  {'-' * 66}")
        print(f"  TOTAL  {t['n_appels']} appel(s) factures")
        print(f"    tokens entree : {t['tokens_in']:,} "
              f"({t['tokens_in_non_caches']:,} pleins + {t['tokens_caches']:,} caches"
              + (f", {t['taux_cache']:.1%} en cache)" if t["taux_cache"] is not None else ")"))
        print(f"    tokens sortie : {t['tokens_out']:,}")
        print(f"    COUT CALCULE  : {t['cout']:.6f} $")
        print(f"\n  A confronter au tableau de bord Mistral. Un ecart signale "
              f"une erreur\n  de comptage ou un tarif different de "
              f"MODELS_PRICING (releve le {cfg.PRICING_CHECKED_ON}).")
    return rapport


# ===========================================================================
# Comparaison APPARIEE de deux campagnes -- test de McNemar
# ===========================================================================
def mcnemar(res_a: dict[str, Any], res_b: dict[str, Any],
            nom_a: str = "A", nom_b: str = "B",
            verbeux: bool = True) -> dict[str, Any]:
    """Test de McNemar exact sur deux campagnes evaluees sur les MEMES lignes.

    POURQUOI CE TEST PLUTOT QU'UNE DIFFERENCE DE F1
    -----------------------------------------------
    Comparer deux F1-macro et leurs intervalles de confiance revient a traiter
    les deux campagnes comme independantes. Elles ne le sont pas : ce sont les
    MEMES reclamations, et les cas faciles sont faciles pour les deux modeles.
    Cette information commune est du bruit partage, que la comparaison
    non appariee laisse dans les deux intervalles -- lesquels se recouvrent
    alors meme quand un modele domine systematiquement l'autre.

    McNemar ne regarde QUE les paires discordantes : les lignes ou un modele a
    raison et l'autre tort. C'est la seule information qui discrimine, et c'est
    ce qui donne au test sa puissance a effectif egal.

    Version EXACTE (binomiale), pas l'approximation du chi2 : avec quelques
    paires discordantes seulement, l'approximation est trompeuse.

    Returns
    -------
    dict avec `b` et `c` (paires discordantes dans chaque sens), `p_valeur`
    bilaterale exacte, et `significatif` au seuil de 5 %.
    """
    from math import comb

    ids_a, ids_b = ids_evalues(res_a), ids_evalues(res_b)
    if set(ids_a) != set(ids_b):
        raise ValueError(
            "McNemar exige les MEMES lignes dans les deux campagnes "
            f"({len(ids_a)} contre {len(ids_b)}, "
            f"{len(set(ids_a) ^ set(ids_b))} identifiant(s) non communs). "
            "Un test apparie sur des lignes differentes n'a pas de sens."
        )

    ja = {i: (v == p) for i, v, p in zip(ids_a, res_a["_y_true"], res_a["_y_pred"])}
    jb = {i: (v == p) for i, v, p in zip(ids_b, res_b["_y_true"], res_b["_y_pred"])}

    b = sum(1 for i in ja if ja[i] and not jb[i])   # A juste, B faux
    c = sum(1 for i in ja if not ja[i] and jb[i])   # A faux, B juste
    n_d = b + c

    if n_d == 0:
        p = 1.0
    else:
        k = min(b, c)
        queue = sum(comb(n_d, i) for i in range(k + 1)) / (2 ** n_d)
        p = min(1.0, 2 * queue)

    resultat = {
        "n_paires": len(ids_a),
        "b_" + nom_a: b, "c_" + nom_b: c,
        "b": b, "c": c, "n_discordantes": n_d,
        "accord": sum(1 for i in ja if ja[i] == jb[i]),
        "p_valeur": p,
        "significatif_5pct": p < 0.05,
        "sens": (nom_a if b > c else nom_b if c > b else "aucun"),
    }
    if verbeux:
        print(f"\n{'=' * 70}")
        print(f"McNEMAR EXACT — {nom_a} vs {nom_b}  (n={len(ids_a)} lignes appariees)")
        print(f"{'=' * 70}")
        print(f"  {nom_a} juste / {nom_b} faux : {b}")
        print(f"  {nom_a} faux  / {nom_b} juste : {c}")
        print(f"  paires discordantes          : {n_d}")
        print(f"  p bilaterale exacte          : {p:.4f}")
        if resultat["significatif_5pct"]:
            print(f"  -> ecart SIGNIFICATIF au seuil de 5 %, en faveur de "
                  f"{resultat['sens']}")
        else:
            print(f"  -> NON significatif. L'ecart observe est compatible avec "
                  f"le hasard.\n     Avec {n_d} paire(s) discordante(s), aucune "
                  f"conclusion n'est possible :\n     meme un ecart systematique "
                  f"ne pourrait pas etre demontre a cet effectif.")
    return resultat


# ===========================================================================
# Projection de cout : borne theorique et valeur mesuree
# ===========================================================================
def projette_cout(n_predictions: int,
                  model: str = cfg.DEFAULT_MODEL,
                  *,
                  taux_activation_cache: float = 1.0,
                  couverture_prefixe: float = 1.0,
                  avg_tokens: float | None = None,
                  prefixe_tokens: float | None = None,
                  avg_output_tokens: float | None = None,
                  **kwargs) -> dict[str, Any]:
    """Projection de cout tenant compte du cache REELLEMENT observe.

    `metrics.estimate_cost()` n'est PAS modifie : il reste fige, et cette
    fonction l'appelle. Son parametre `cached_prefix_tokens` suffit, car le
    cout moyen ne depend que du nombre MOYEN de tokens caches par appel :

        tokens_caches_moyens = prefixe x couverture x taux_activation

    Verifie sur la mesure : 252 x (224/252) x (11/20) = 123,2 tokens, soit
    exactement les 2 464 tokens caches observes sur 20 appels.

    Parameters
    ----------
    taux_activation_cache : part des appels ou le cache s'active. **1.0 par
        defaut = BORNE THEORIQUE**, ce que modelise `estimate_cost` seul.
        La valeur mesuree chez Mistral est de 0,55 a 0,90 selon le modele.
    couverture_prefixe : part du prefixe reellement servie depuis le cache
        quand il s'active. 1.0 par defaut ; mesure 0,89 (small) et 0,53
        (ministral).

    Le defaut a 1.0 est deliberé : il preserve exactement le comportement
    anterieur, et la degradation doit etre un geste EXPLICITE de l'appelant.
    """
    tarif = cfg.MODELS_PRICING.get(model, {})
    fournisseur = tarif.get("fournisseur")
    mesure = fournisseur in cfg.FOURNISSEURS_TOKENS_MESURES

    if prefixe_tokens is None:
        prefixe_tokens = cfg.CACHED_PREFIX_TOKENS_PAR_MODELE.get(
            model, cfg.CACHED_PREFIX_TOKENS)
    caches = prefixe_tokens * couverture_prefixe * taux_activation_cache

    r = mt.estimate_cost(
        n_predictions, model,
        avg_tokens=cfg.AVG_COMPLAINT_TOKENS if avg_tokens is None else avg_tokens,
        avg_output_tokens=(cfg.AVG_OUTPUT_TOKENS if avg_output_tokens is None
                           else avg_output_tokens),
        prompt_overhead_tokens=prefixe_tokens,
        cached_prefix_tokens=caches,
        **kwargs)

    # Statut PAR FOURNISSEUR. Un flag global unique ferait passer 8 modeles sur
    # 10 pour mesures alors que seul Mistral l'a ete : les tokenizers different,
    # et un volume de tokens mesure chez Mistral ne transfere pas ailleurs.
    r["statut_tokens"] = "mesure" if mesure else "hypothese"
    r["fournisseur_mesure"] = bool(mesure)
    r["mesure_le"] = cfg.AVG_COMPLAINT_TOKENS_MESURE_LE if mesure else None
    r["taux_activation_cache"] = taux_activation_cache
    r["couverture_prefixe"] = couverture_prefixe
    r["prefixe_tokens"] = prefixe_tokens
    r["prefixe_mesure"] = model in cfg.CACHED_PREFIX_TOKENS_PAR_MODELE
    return r


def tableau_couts_deux_bornes(n_predictions: int = cfg.DAILY_COMPLAINTS,
                              modeles: Iterable[str] | None = None
                              ) -> pd.DataFrame:
    """Tableau de couts a DEUX colonnes de cache : borne theorique et mesure.

    POURQUOI DEUX COLONNES ET NON UNE SEULE CORRIGEE
    -------------------------------------------------
    La borne theorique (activation 100 %, couverture totale) reste le PLAFOND
    atteignable : c'est ce qu'un deploiement avec `prompt_cache_key` peut viser,
    et le supprimer effacerait l'interet du levier. La valeur mesuree est ce
    qu'obtient un appel par defaut, sans reglage -- la situation de ZenAssist.

    Les presenter ensemble est la seule facon honnete : une colonne unique,
    theorique, est optimiste d'un facteur ~2 ; une colonne unique, mesuree,
    ferait croire que rien de mieux n'est possible.

    Le cache mesure n'existe que pour les modeles reellement testes. Pour les
    autres, la colonne mesuree est vide -- pas extrapolee.
    """
    modeles = list(modeles) if modeles is not None else list(cfg.MODELS_PRICING)
    lignes = {}
    for m in modeles:
        theorique = projette_cout(n_predictions, m)
        sans = projette_cout(n_predictions, m, taux_activation_cache=0.0)
        obs = cfg.CACHE_MESURE.get(m)
        ligne = {
            "fournisseur": theorique["fournisseur"],
            "statut tokens": theorique["statut_tokens"],
            "$ sans cache": round(sans["cout_total_usd"], 4),
            "$ cache THEORIQUE": round(theorique["cout_total_usd"], 4),
            "economie theorique": round(
                1 - theorique["cout_total_usd"] / sans["cout_total_usd"], 3),
        }
        if obs:
            reel = projette_cout(n_predictions, m,
                                 taux_activation_cache=obs["taux_activation"],
                                 couverture_prefixe=obs["couverture_prefixe"])
            ligne["$ cache MESURE"] = round(reel["cout_total_usd"], 4)
            ligne["economie mesuree"] = round(
                1 - reel["cout_total_usd"] / sans["cout_total_usd"], 3)
        else:
            ligne["$ cache MESURE"] = None
            ligne["economie mesuree"] = None
        lignes[m] = ligne

    tab = pd.DataFrame(lignes).T
    tab.attrs["note"] = (
        f"Colonne MESURE : uniquement les modeles testes le "
        f"{cfg.CACHE_MESURE_LE} ({', '.join(cfg.CACHE_MESURE)}). "
        f"Volumes de tokens mesures pour "
        f"{', '.join(sorted(cfg.FOURNISSEURS_TOKENS_MESURES))} seulement ; "
        f"hypotheses pour les autres fournisseurs, dont les tokenizers different."
    )
    return tab


# ===========================================================================
# Correction de multiplicite -- Holm-Bonferroni
# ===========================================================================
def holm_bonferroni(p_valeurs: dict[str, float],
                    alpha: float = 0.05,
                    verbeux: bool = True) -> dict[str, Any]:
    """Procedure descendante de Holm sur une famille de comparaisons.

    POURQUOI UNE CORRECTION
    -----------------------
    Comparer 5 variantes a v1 au seuil de 5 % donne 22,6 % de risque de retenir
    au moins une variante par pur hasard. Sur un protocole arrete AVANT mesure
    et non revisable ensuite, retenir une fausse variante une fois sur quatre
    n'est pas acceptable.

    POURQUOI HOLM ET NON BONFERRONI
    -------------------------------
    Holm controle le meme risque familial (FWER) et DOMINE UNIFORMEMENT
    Bonferroni : il ne rejette jamais moins, et rejette parfois plus. Bonferroni
    compare tous les p a alpha/m ; Holm ne l'impose qu'au plus petit, puis
    relache le seuil (alpha/(m-1), alpha/(m-2), ...). Aucune raison de preferer
    la version dominee.

    PROCEDURE
    ---------
    Trier les p par ordre croissant. Comparer le i-eme (1-indexe) a
    alpha/(m-i+1). S'ARRETER AU PREMIER ECHEC : toutes les hypotheses de rang
    superieur sont non rejetees, quel que soit leur p. C'est ce qui distingue
    Holm d'une simple suite de seuils, et c'est ce qui preserve le FWER.

    NON-REJET N'EST PAS EQUIVALENCE
    -------------------------------
    Une hypothese non rejetee n'est pas demontree fausse : elle n'est pas
    demontree. Lire un non-rejet comme "cette variante ne vaut pas mieux que v1"
    est une erreur. `effet_minimal_detectable_holm()` donne ce qu'il aurait
    fallu observer pour conclure -- sans quoi un non-rejet est illisible.
    """
    m = len(p_valeurs)
    if m == 0:
        return {"m": 0, "resultats": [], "rejetees": []}

    ordonne = sorted(p_valeurs.items(), key=lambda kv: kv[1])
    resultats, arret = [], False
    for i, (nom, p) in enumerate(ordonne, start=1):
        seuil = alpha / (m - i + 1)
        if arret:
            rejet, motif = False, "non rejetee (procedure arretee au rang precedent)"
        elif p < seuil:
            rejet, motif = True, "rejetee"
        else:
            rejet, motif, arret = False, "non rejetee (arret de la procedure)", True
        resultats.append({"rang": i, "nom": nom, "p_brut": p,
                          "seuil_holm": seuil, "rejetee": rejet, "motif": motif})

    rapport = {
        "m": m, "alpha_familial": alpha, "resultats": resultats,
        "rejetees": [r["nom"] for r in resultats if r["rejetee"]],
    }
    if verbeux:
        print(f"\n{'=' * 78}")
        print(f"HOLM-BONFERRONI — {m} comparaisons, FWER controle a {alpha:.0%}")
        print(f"{'=' * 78}")
        print(f"  {'rang':>4}  {'variante':24s} {'p brut':>9}  {'seuil Holm':>11}  decision")
        for r in resultats:
            marque = "REJETEE" if r["rejetee"] else "non rejetee"
            print(f"  {r['rang']:>4}  {r['nom']:24s} {r['p_brut']:>9.4f}  "
                  f"{r['seuil_holm']:>11.5f}  {marque}")
        if not rapport["rejetees"]:
            print("\n  Aucune hypothese rejetee : aucune variante ne se distingue "
                  "de v1.\n  C'est une issue LEGITIME du protocole, pas un echec.")
    return rapport


def effet_minimal_detectable_holm(n_comparaisons: int = 5,
                                  alpha: float = 0.05,
                                  casses: Sequence[int] = (0, 2, 5, 10),
                                  n_lignes: int = 200,
                                  verbeux: bool = True) -> pd.DataFrame:
    """Ce qu'il faudrait observer pour rejeter, a chaque rang de Holm.

    Indispensable pour lire un NON-REJET. Sans cette table, "non rejetee" se lit
    a tort comme "equivalente a v1", alors que le protocole peut simplement
    manquer de puissance pour l'ecart en jeu.

    Pour chaque rang de la procedure (donc chaque seuil) et chaque nombre de
    lignes que la variante CASSE, donne le nombre de lignes qu'elle doit
    CORRIGER pour que McNemar exact franchisse le seuil.
    """
    from math import comb

    def p_exact(b: int, c: int) -> float:
        n = b + c
        if n == 0:
            return 1.0
        k = min(b, c)
        return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n))

    lignes = []
    for rang in range(1, n_comparaisons + 1):
        seuil = alpha / (n_comparaisons - rang + 1)
        for c in casses:
            b_min = None
            for b in range(c + 1, 200):          # b > c : en FAVEUR de la variante
                if p_exact(b, c) < seuil:
                    b_min = b
                    break
            lignes.append({
                "rang": rang, "seuil_holm": round(seuil, 5), "lignes_cassees": c,
                "lignes_a_corriger": b_min,
                "gain_net": None if b_min is None else b_min - c,
                "gain_net_pts": None if b_min is None
                                else round(100 * (b_min - c) / n_lignes, 1),
            })
    tab = pd.DataFrame(lignes)
    if verbeux:
        print(f"\n{'=' * 78}")
        print(f"EFFET MINIMAL DETECTABLE PAR RANG DE HOLM (n = {n_lignes} lignes)")
        print(f"{'=' * 78}")
        print("  Un NON-REJET ne signifie pas equivalence : il signifie que "
              "l'ecart,\n  s'il existe, est inferieur a ce qui figure ici.\n")
        for rang in range(1, n_comparaisons + 1):
            sous = tab[tab["rang"] == rang]
            seuil = sous["seuil_holm"].iloc[0]
            detail = "   ".join(
                f"casse {int(r.lignes_cassees)} -> corrige "
                f"{'—' if r.lignes_a_corriger is None else int(r.lignes_a_corriger)}"
                f" ({r.gain_net_pts:+.1f} pts)" if r.lignes_a_corriger else ""
                for r in sous.itertuples())
            print(f"  rang {rang} (seuil {seuil:.5f}) : {detail}")
    return tab


# ===========================================================================
# Analyse d'erreurs par paire de classes
# ===========================================================================
# Paires de confusion ANNONCEES au §C.4 du diagnostic, avant tout appel d'API.
# Les confronter aux erreurs reellement observees est le seul moyen de savoir si
# le plafond annonce etait le bon -- une prediction faite avant la mesure vaut
# infiniment plus qu'une explication trouvee apres.
PAIRES_ANNONCEES_C4 = [
    ("Credit reporting", "Debt collection"),
    ("Vehicle loan or lease", "Credit reporting"),
    ("Credit card or prepaid card", "Bank account or service"),
    ("Money transfer or virtual currency", "Bank account or service"),
    ("Payday, title or personal loan", "Debt collection"),
]


def paires_erreurs(resultats: dict[str, Any],
                   top: int = 15,
                   verbeux: bool = True) -> pd.DataFrame:
    """Distribution des erreurs par paire (classe vraie, classe predite).

    Les paires sont comptees de facon ORIENTEE (vrai -> predit) puis regroupees
    en paires NON ORIENTEES, car le §C.4 annonce des confusions bidirectionnelles.

    Le comptage est NON PONDERE : il decrit l'echantillon tel qu'il a ete tire.
    Les poids de Horvitz-Thompson servent aux metriques ramenees a la population,
    pas au denombrement des cas a lire.
    """
    df = pd.DataFrame({"vrai": resultats["_y_true"], "predit": resultats["_y_pred"]})
    err = df[df["vrai"] != df["predit"]].copy()
    if err.empty:
        return pd.DataFrame()

    err["paire"] = err.apply(
        lambda r: " ↔ ".join(sorted([r["vrai"], r["predit"]])), axis=1)
    err["orientee"] = err["vrai"] + " → " + err["predit"]

    tab = (err.groupby("paire")
              .agg(n=("paire", "size"))
              .sort_values("n", ascending=False))
    tab["part_des_erreurs"] = (tab["n"] / len(err)).round(4)

    annoncees = {" ↔ ".join(sorted(p)) for p in PAIRES_ANNONCEES_C4}
    tab["annoncee_C4"] = [p in annoncees for p in tab.index]

    if verbeux:
        n_ann = int(tab.loc[tab["annoncee_C4"], "n"].sum())
        print(f"\n{'=' * 78}")
        print(f"ERREURS PAR PAIRE DE CLASSES — {len(err)} erreurs")
        print(f"{'=' * 78}")
        print(f"  couvertes par les paires annoncees au §C.4 : {n_ann}/{len(err)} "
              f"({n_ann / len(err):.1%})\n")
        for paire, r in tab.head(top).iterrows():
            marque = "  [C.4]" if r["annoncee_C4"] else ""
            print(f"    {r['n']:4d}  ({r['part_des_erreurs']:5.1%})  {paire}{marque}")

        print(f"\n  Paires annoncees au §C.4 et leur realisation :")
        for a, b in PAIRES_ANNONCEES_C4:
            cle = " ↔ ".join(sorted([a, b]))
            n = int(tab.loc[cle, "n"]) if cle in tab.index else 0
            print(f"    {n:4d}  {cle}")

        print(f"\n  Sens des confusions (orientees), 10 premieres :")
        for o, n in err["orientee"].value_counts().head(10).items():
            print(f"    {n:4d}  {o}")
    return tab


def exporte_erreurs(resultats: dict[str, Any],
                    chemin: str | Path | None = None) -> Path:
    """Exporte les erreurs pour l'analyse croisee LLM/ML de l'etape 3.

    FORMAT ARRETE AVANT MESURE, ET VOLONTAIREMENT MINIMAL :
        complaint_id, label_vrai, label_predit
    et rien d'autre.

    PAS DE TEXTE DE RECLAMATION. Ce fichier est destine a etre relu plusieurs
    fois, par l'etape 3 puis par la restitution ; y embarquer le texte le
    transformerait en copie diffuse du corpus, alors qu'une jointure sur
    `complaint_id` le retrouve a tout moment depuis `data/processed/`.
    Pas de poids non plus : ils appartiennent a l'echantillon, pas aux erreurs,
    et les dupliquer ici creerait une seconde source de verite.

    POURQUOI CE FICHIER EXISTE
    --------------------------
    Le croisement des erreurs du LLM et de celles du ML est la SEULE mesure
    empirique du plafond de performance : une reclamation manquee par les deux
    approches, dont les mecanismes sont independants, releve probablement d'une
    ambiguite d'etiquetage plutot que d'une faiblesse de modele.

    Le protocole de ce croisement est arrete MAINTENANT, avant que les deux
    series d'erreurs soient connues. Sinon on choisirait apres coup la
    definition de "meme erreur" qui arrange la conclusion.
    """
    cfg.ensure_llm_dirs()
    if chemin is None:
        nom = Path(resultats.get("campagne", "campagne")).stem
        chemin = cfg.LLM_DIR / f"{nom}_erreurs.csv"
    chemin = Path(chemin)

    df = pd.DataFrame({
        cfg.ID_COL: [str(i) for i in resultats["_ids_evalues"]],
        "label_vrai": resultats["_y_true"],
        "label_predit": resultats["_y_pred"],
    })
    err = df[df["label_vrai"] != df["label_predit"]].copy()
    err.to_csv(chemin, index=False, encoding="utf-8")
    print(f"  [export] {len(err)} erreur(s) -> {chemin.name} "
          f"(id, vrai, predit — sans texte)")
    return chemin


def confronte_paires_c4(resultats: dict[str, Any],
                        verbeux: bool = True) -> dict[str, Any]:
    """Confronte les erreurs observees aux paires ANNONCEES au §C.4.

    DEUX REFERENCES, TOUTES DEUX FIXEES AVANT MESURE
    ------------------------------------------------
    Une part observee ne veut rien dire seule : il faut savoir a quoi la
    comparer. Les deux references ci-dessous sont arretees avant que les
    chiffres soient connus, pour qu'aucune ne puisse etre choisie apres coup.

    REFERENCE 1 — uniforme sur les paires (la reference principale).
        Sous l'hypothese ou les erreurs se repartissent uniformement entre
        toutes les paires de classes, les 5 paires annoncees captureraient
        5 / C(9,2) = 5/36 = 13,9 % des erreurs.
        Naive a dessein : elle ignore que les grosses classes produisent plus
        d'erreurs. C'est la reference LA PLUS FAVORABLE au §C.4, donc celle qui
        doit etre lue avec le plus de prudence.

    REFERENCE 2 — proportionnelle aux effectifs (reference severe).
        Modele de hasard ou la probabilite d'une paire (i, j) est proportionnelle
        a n_i x n_j, les effectifs reels de l'echantillon. Une confusion entre
        les deux plus grosses classes y est mecaniquement plus probable qu'entre
        deux classes rares. C'est la reference EXIGEANTE : si le §C.4 la bat
        aussi, sa prediction porte une information que la seule taille des
        classes n'explique pas.

    Le rapport donne les deux `lift` (part observee / part attendue). Un lift de
    1 signifie "rien de plus que le hasard sous ce modele".
    """
    from math import comb

    df = pd.DataFrame({"vrai": resultats["_y_true"], "predit": resultats["_y_pred"]})
    err = df[df["vrai"] != df["predit"]]
    if err.empty:
        return {"n_erreurs": 0}

    annoncees = {" ↔ ".join(sorted(p)) for p in PAIRES_ANNONCEES_C4}
    paires_err = err.apply(
        lambda r: " ↔ ".join(sorted([r["vrai"], r["predit"]])), axis=1)
    # Les PARSE_ERROR ne forment pas une paire de classes : exclus du calcul.
    valides = paires_err[~paires_err.str.contains(cfg.PARSE_ERROR, regex=False)]
    n_sur_annoncees = int(valides.isin(annoncees).sum())
    part_observee = n_sur_annoncees / len(valides) if len(valides) else 0.0

    n_paires_total = comb(len(cfg.CLASS_ORDER), 2)
    attendu_uniforme = len(PAIRES_ANNONCEES_C4) / n_paires_total

    # Reference 2 : poids proportionnel a n_i x n_j sur les effectifs reels.
    eff = df["vrai"].value_counts()
    total_poids, poids_annoncees = 0.0, 0.0
    for i, a in enumerate(cfg.CLASS_ORDER):
        for b in cfg.CLASS_ORDER[i + 1:]:
            w = float(eff.get(a, 0)) * float(eff.get(b, 0))
            total_poids += w
            if " ↔ ".join(sorted([a, b])) in annoncees:
                poids_annoncees += w
    attendu_effectifs = poids_annoncees / total_poids if total_poids else 0.0

    r = {
        "n_erreurs": int(len(err)),
        "n_erreurs_hors_parse_error": int(len(valides)),
        "n_sur_paires_annoncees": n_sur_annoncees,
        "part_observee": round(part_observee, 4),
        "attendu_uniforme": round(attendu_uniforme, 4),
        "lift_uniforme": round(part_observee / attendu_uniforme, 3)
                         if attendu_uniforme else None,
        "attendu_effectifs": round(attendu_effectifs, 4),
        "lift_effectifs": round(part_observee / attendu_effectifs, 3)
                          if attendu_effectifs else None,
        "n_paires_possibles": n_paires_total,
    }
    if verbeux:
        print(f"\n{'=' * 78}")
        print("CONFRONTATION AUX PAIRES ANNONCEES AU §C.4 (prediction anterieure)")
        print(f"{'=' * 78}")
        print(f"  erreurs analysees : {r['n_erreurs_hors_parse_error']} "
              f"(sur {r['n_erreurs']}, PARSE_ERROR exclus)")
        print(f"  tombant sur les 5 paires annoncees : {n_sur_annoncees} "
              f"= {part_observee:.1%}\n")
        print(f"  reference 1, uniforme sur {n_paires_total} paires : "
              f"{attendu_uniforme:.1%}   -> lift x{r['lift_uniforme']}")
        print(f"  reference 2, proportionnelle aux effectifs : "
              f"{attendu_effectifs:.1%}   -> lift x{r['lift_effectifs']}")
    return r


# ===========================================================================
# Comparabilite de deux campagnes -- A VERIFIER AVANT LA SECONDE, PAS APRES
# ===========================================================================
def verifie_source_identique(chemin_journal: str | Path,
                             source_df: pd.DataFrame,
                             poids: pd.Series | None = None,
                             verbeux: bool = True) -> dict[str, Any]:
    """Verifie qu'une campagne A VENIR portera sur les MEMES lignes qu'une deja faite.

    QUAND L'APPELER : avant de lancer la seconde campagne, jamais apres. Une
    fois les 2 000 appels partis, un desalignement coute l'argent ET le temps,
    et le McNemar apparie devient impossible -- or c'est le test qui porte toute
    la comparaison entre les deux modeles.

    CE QUI EST VERIFIE
    ------------------
      1. les identifiants du journal existant et ceux de la source sont le MEME
         ensemble, sans manquant ni surnumeraire ;
      2. la source ne contient aucun doublon d'identifiant ;
      3. si des poids sont fournis, ils sont alignes sur la source ET identiques,
         classe par classe, a ceux que porte la source.

    Le point 3 n'est pas redondant : `evalue_campagne()` reordonne les poids sur
    l'alignement, mais si la SOURCE elle-meme differe (autre tirage, autre
    allocation), les poids de Horvitz-Thompson ne ramenent plus a la meme
    population et les deux F1 ne sont plus comparables.

    Raises
    ------
    ValueError, avec le detail des identifiants en cause. Un echec BRUYANT est
    voulu : un avertissement se perdrait dans la sortie et la campagne partirait
    quand meme.
    """
    journal = charge_resultats(chemin_journal)
    if not journal:
        raise ValueError(f"{chemin_journal} est vide ou illisible : impossible "
                         f"de verifier la comparabilite.")

    ids_journal = [str(e[cfg.ID_COL]) for e in journal]
    ids_source = [str(i) for i in source_df[cfg.ID_COL]]

    doublons_j = len(ids_journal) - len(set(ids_journal))
    doublons_s = len(ids_source) - len(set(ids_source))
    if doublons_s:
        raise ValueError(f"La source contient {doublons_s} identifiant(s) en "
                         f"double : les poids seraient appliques deux fois.")

    manquants = sorted(set(ids_source) - set(ids_journal))
    en_trop = sorted(set(ids_journal) - set(ids_source))
    if manquants or en_trop:
        raise ValueError(
            f"LES DEUX CAMPAGNES NE PORTERAIENT PAS SUR LES MEMES LIGNES.\n"
            f"  presents dans la source, absents du journal : {len(manquants)}"
            f"  {manquants[:5]}\n"
            f"  presents dans le journal, absents de la source : {len(en_trop)}"
            f"  {en_trop[:5]}\n"
            f"Le McNemar apparie serait impossible et compare_results() "
            f"refuserait l'alignement. Verifier que la seconde campagne part "
            f"bien de data_prep.load_eval_sample()."
        )

    rapport = {"n": len(ids_source), "doublons_journal": doublons_j,
               "identiques": True, "poids_verifies": False}

    if poids is not None:
        if len(poids) != len(source_df):
            raise ValueError(f"poids ({len(poids)}) et source ({len(source_df)}) "
                             f"n'ont pas la meme longueur.")
        if cfg.WEIGHT_COL in source_df.columns:
            ecart = float(np.abs(np.asarray(poids, dtype=float)
                                 - source_df[cfg.WEIGHT_COL].to_numpy()).max())
            if ecart > 1e-9:
                raise ValueError(
                    f"Les poids fournis different de la colonne "
                    f"'{cfg.WEIGHT_COL}' de la source (ecart max {ecart:.6g}). "
                    f"Un poids desaligne biaise le F1-macro de +0,022 sans "
                    f"qu'aucune exception ne le signale ensuite."
                )
        rapport["poids_verifies"] = True
        rapport["somme_poids"] = float(np.asarray(poids, dtype=float).sum())

    if verbeux:
        print(f"  [comparabilite] {rapport['n']} identifiants identiques entre "
              f"{Path(str(chemin_journal)).name} et la source")
        if rapport["poids_verifies"]:
            print(f"  [comparabilite] poids alignes, somme "
                  f"{rapport['somme_poids']:,.0f} (population de reference)")
    return rapport


# ===========================================================================
# CRITERES DE LECTURE DU §C.4 -- PRE-ENREGISTRES, AVANT TOUTE MESURE
# ===========================================================================
# Fixes le 2026-08-19, campagne d'evaluation NON ENCORE ANALYSEE. Ils sont
# encodes ici plutot que rediges apres coup : un critere qu'on formule en
# regardant les chiffres n'est plus un critere, c'est une justification.
#
# Le verdict est CALCULE par `verdict_c4()`. Aucune place n'est laissee a
# l'appreciation au moment de la lecture.

# --- critere A : concentration des erreurs sur les 5 paires annoncees -------
# Lu contre la reference SEVERE (proportionnelle aux effectifs) EN PREMIER.
# La reference uniforme est favorable au §C.4 par construction -- elle ignore
# que les grosses classes produisent mecaniquement plus de confusions -- et ne
# peut donc pas, seule, valider la prediction.
SEUIL_A_CONFIRME = 1.50      # lift sur la reference severe
SEUIL_A_INFIRME = 1.00

# --- critere B : rang de la paire annoncee comme la plus couteuse -----------
# Le §C.4 classe `Credit reporting <-> Debt collection` en tete, par risque
# decroissant, et la qualifie de "confusion la plus couteuse".
PAIRE_PRINCIPALE_C4 = ("Credit reporting", "Debt collection")
RANG_B_CONFIRME = 2          # rang 1 ou 2 parmi les paires observees
RANG_B_INFIRME = 5           # au-dela du rang 5

# --- critere C : completude, paires fortes NON predites ---------------------
# Une prediction n'est pas seulement jugee sur ce qu'elle annonce, mais sur ce
# qu'elle a manque. Une paire lourde absente du §C.4 est une lacune, meme si
# les paires annoncees se realisent.
PART_C_LACUNE = 0.10         # une paire non annoncee pesant >= 10 % des erreurs
RANG_C_LACUNE = 3            # ou figurant dans les 3 premieres


def verdict_c4(resultats: dict[str, Any], verbeux: bool = True) -> dict[str, Any]:
    """Applique les criteres PRE-ENREGISTRES du §C.4. Verdict calcule, non narre.

    Rend, pour chacun des trois criteres, l'une des trois valeurs
    "confirme" / "indecis" / "infirme", plus le classement COMPLET des paires
    observees -- pas seulement celles qui etaient annoncees.
    """
    c4 = confronte_paires_c4(resultats, verbeux=False)
    tab = paires_erreurs(resultats, verbeux=False)
    if tab.empty or not c4.get("n_erreurs"):
        return {"verdict": "aucune erreur a analyser"}

    # -- A : concentration, lue contre la reference SEVERE d'abord -----------
    lift_sev = c4["lift_effectifs"]
    if lift_sev is None:
        a = "indecis"
    elif lift_sev >= SEUIL_A_CONFIRME:
        a = "confirme"
    elif lift_sev < SEUIL_A_INFIRME:
        a = "infirme"
    else:
        a = "indecis"

    # -- B : rang de la paire principale -------------------------------------
    cle_principale = " ↔ ".join(sorted(PAIRE_PRINCIPALE_C4))
    classement = list(tab.index)
    rang = classement.index(cle_principale) + 1 if cle_principale in classement else None
    if rang is None or rang > RANG_B_INFIRME:
        b = "infirme"
    elif rang <= RANG_B_CONFIRME:
        b = "confirme"
    else:
        b = "indecis"

    # -- C : lacunes, paires lourdes non annoncees ---------------------------
    non_annoncees = tab[~tab["annoncee_C4"]]
    lacunes = [
        {"paire": p, "n": int(r["n"]), "part": float(r["part_des_erreurs"]),
         "rang": classement.index(p) + 1}
        for p, r in non_annoncees.iterrows()
        if r["part_des_erreurs"] >= PART_C_LACUNE
        or classement.index(p) + 1 <= RANG_C_LACUNE
    ]
    c = "infirme" if lacunes else "confirme"

    rapport = {
        "A_concentration": a, "A_lift_severe": lift_sev,
        "A_lift_uniforme": c4["lift_uniforme"],
        "B_rang_paire_principale": rang, "B_verdict": b,
        "C_lacunes": lacunes, "C_verdict": c,
        "classement_complet": tab,
        "confrontation": c4,
    }

    if verbeux:
        print(f"\n{'=' * 78}")
        print("VERDICT §C.4 — CRITERES PRE-ENREGISTRES AVANT MESURE")
        print(f"{'=' * 78}")
        print(f"\n  A. CONCENTRATION sur les 5 paires annoncees : "
              f"{c4['part_observee']:.1%} des erreurs")
        print(f"     reference SEVERE (proportionnelle aux effectifs) : "
              f"{c4['attendu_effectifs']:.1%}  ->  lift x{lift_sev}")
        print(f"     reference uniforme sur 36 paires : "
              f"{c4['attendu_uniforme']:.1%}  ->  lift x{c4['lift_uniforme']}")
        print(f"     ATTENTION : la reference uniforme est FAVORABLE au §C.4 par")
        print(f"     construction — elle ignore que les grosses classes produisent")
        print(f"     mecaniquement plus de confusions. Elle ne peut pas valider")
        print(f"     seule la prediction.")
        print(f"     seuils pre-enregistres : confirme si lift severe >= "
              f"{SEUIL_A_CONFIRME}, infirme si < {SEUIL_A_INFIRME}")
        print(f"     -> {a.upper()}")

        print(f"\n  B. RANG de '{cle_principale}'")
        print(f"     annoncee au §C.4 comme la confusion la plus couteuse")
        print(f"     rang observe : {rang if rang else 'absente du classement'}")
        print(f"     seuils : confirme si rang <= {RANG_B_CONFIRME}, "
              f"infirme si > {RANG_B_INFIRME}")
        print(f"     -> {b.upper()}")

        print(f"\n  C. LACUNES : paires lourdes NON annoncees au §C.4")
        print(f"     seuils : part >= {PART_C_LACUNE:.0%} ou rang <= {RANG_C_LACUNE}")
        if lacunes:
            for l in lacunes:
                print(f"       rang {l['rang']:2d}  {l['n']:4d} ({l['part']:.1%})  "
                      f"{l['paire']}")
        else:
            print("       aucune")
        print(f"     -> {c.upper()}")

        print(f"\n{'-' * 78}")
        print("  CLASSEMENT COMPLET DES PAIRES OBSERVEES")
        print(f"{'-' * 78}")
        for i, (paire, r) in enumerate(tab.iterrows(), start=1):
            marque = " [C.4]" if r["annoncee_C4"] else ""
            print(f"    {i:2d}. {int(r['n']):4d}  ({r['part_des_erreurs']:5.1%})  "
                  f"{paire}{marque}")
    return rapport
