"""Execution d'une campagne de classification LLM (etape 2).

ECRITURE INCREMENTALE ET REPRISE
--------------------------------
Chaque reponse est ecrite immediatement en JSONL puis vidée sur disque
(`flush` + `fsync`). Au demarrage, le fichier existant est relu et les
`complaint_id` deja traites sont sautes.

Ce n'est pas du confort : 2 000 appels a ~1 req/s font environ 35 minutes. Une
coupure reseau, un Ctrl-C ou une fermeture de portable en cours de campagne ne
doit pas obliger a redepenser ce qui a deja ete paye. La reprise rend aussi
acceptable de lancer la campagne en plusieurs fois.

LA REPONSE BRUTE EST TOUJOURS JOURNALISEE
-----------------------------------------
Meme quand le parsing strict echoue. C'est ce qui rend possible un re-parsing
tolerant hors ligne (`llm_prompts.reparse_tolerant`) sans reappeler l'API : le
cout d'une decision de parsing severe est donc chiffrable apres coup, gratuit.

COUT REEL, PAS PROJETE
----------------------
`cout_reel()` additionne les tokens rendus par `usage`, avec la remise de cache
appliquee a la part servie depuis le cache. Aucun recours a
`config.AVG_COMPLAINT_TOKENS` : c'est precisement l'hypothese que cette etape
doit remplacer. Quand le tarif d'un modele n'est pas connu, le cout est declare
non calculable plutot qu'estime.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import date
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from . import config as cfg
from . import llm_prompts as prompts
from .metrics import latency_stats

# ===========================================================================
# Cout reel
# ===========================================================================
def _tarif_effectif(cle_tarification: str,
                    date_reference: date | None = None) -> dict[str, Any]:
    """Tarif du modele, promotion expiree prise en compte.

    Reproduit la regle de `metrics.estimate_cost` : un tarif d'introduction
    perime bascule sur le tarif standard. Aucun modele Mistral n'est concerne
    aujourd'hui, mais dupliquer la regle evite qu'un cout reel et un cout projete
    divergent silencieusement si le catalogue evolue.
    """
    tarif = dict(cfg.MODELS_PRICING[cle_tarification])
    fin = tarif.get("tarif_provisoire_jusquau")
    if fin and (date_reference or date.today()) > date.fromisoformat(fin):
        tarif.update(tarif["tarif_apres"])
    return tarif


def cout_reel(tokens_in: int, tokens_out: int, tokens_caches: int = 0,
              cle_tarification: str | None = None,
              date_reference: date | None = None) -> float | None:
    """Cout en dollars d'un appel, a partir des tokens REELS de `usage`.

    Retourne `None` si le tarif est inconnu : sans tarif, le cout est declare
    non calculable, jamais suppose.

    La part cachee est facturee avec la remise du fournisseur. Si `remise_cache`
    vaut `None` (non verifiee), le cache n'est PAS modelise et tout l'input est
    facture plein tarif -- une surestimation assumee, preferable a une remise
    inventee.
    """
    if cle_tarification is None or cle_tarification not in cfg.MODELS_PRICING:
        return None
    tarif = _tarif_effectif(cle_tarification, date_reference)

    tokens_in = tokens_in or 0
    tokens_out = tokens_out or 0
    caches = tokens_caches or 0

    remise = tarif.get("remise_cache")
    if remise is None:
        # Cache non verifie chez ce fournisseur : on ne le modelise pas du tout
        # plutot que de supposer une remise. Tout l'input part au plein tarif.
        remise = 0.0
        caches = 0

    caches = min(caches, tokens_in)
    plein = tokens_in - caches

    return (
        plein * tarif["input_per_1m"]
        + caches * tarif["input_per_1m"] * (1.0 - remise)
        + tokens_out * tarif["output_per_1m"]
    ) / 1_000_000


# ===========================================================================
# Journal JSONL
# ===========================================================================
def charge_resultats(chemin: str | Path) -> list[dict[str, Any]]:
    """Relit un JSONL de campagne. Tolere une derniere ligne tronquee.

    Une coupure pendant une ecriture peut laisser une ligne incomplete. On la
    signale et on l'ignore : elle sera rejouee par la reprise, puisque son
    `complaint_id` ne sera pas compte comme traite.
    """
    chemin = Path(chemin)
    if not chemin.exists():
        return []

    lignes: list[dict[str, Any]] = []
    with chemin.open("r", encoding="utf-8") as f:
        for numero, ligne in enumerate(f, start=1):
            ligne = ligne.strip()
            if not ligne:
                continue
            try:
                lignes.append(json.loads(ligne))
            except json.JSONDecodeError:
                print(f"  [reprise] ligne {numero} illisible (ecriture "
                      f"interrompue ?), ignoree -- elle sera rejouee.",
                      file=sys.stderr)
    return lignes


def _ecrit(f, enregistrement: dict[str, Any]) -> None:
    """Ecrit un enregistrement et le force sur disque immediatement."""
    f.write(json.dumps(enregistrement, ensure_ascii=False) + "\n")
    f.flush()
    os.fsync(f.fileno())


# ===========================================================================
# Garde-fous
# ===========================================================================
def _est_echantillon_eval(df: pd.DataFrame) -> bool:
    """Detecte l'echantillon d'evaluation par sa STRUCTURE, pas par son chemin.

    `load_eval_sample()` est le seul chargeur supporte et rend un DataFrame
    portant `sampling_weight` et `strate`. Se fier a ces colonnes plutot qu'au
    nom du fichier rend le garde-fou insensible a une copie renommee.
    """
    return cfg.WEIGHT_COL in df.columns or cfg.STRATE_COL in df.columns


def _verifie_autorisation(df: pd.DataFrame, style: str,
                          est_eval: bool | None) -> bool:
    """Leve si l'on tente une campagne d'evaluation avec un style non fige."""
    est_eval = _est_echantillon_eval(df) if est_eval is None else est_eval
    if est_eval and style not in cfg.LLM_STYLES_FIGES:
        raise RuntimeError(
            f"REFUS DE DEMARRER.\n"
            f"Cible detectee : echantillon d'EVALUATION (colonnes de "
            f"ponderation presentes).\n"
            f"Style demande  : '{style}', absent de config.LLM_STYLES_FIGES "
            f"{sorted(cfg.LLM_STYLES_FIGES) or '(vide)'}.\n\n"
            f"Mettre au point un prompt en regardant ses erreurs sur "
            f"l'echantillon d'evaluation revient a l'optimiser pour ces lignes "
            f"precises : le score final serait gonfle et non comparable au ML.\n"
            f"La mise au point se fait sur {cfg.LLM_ITERATION_FILE.name} (train).\n"
            f"Une fois le prompt arrete, l'ajouter a config.LLM_STYLES_FIGES."
        )
    return est_eval


# ===========================================================================
# Campagne
# ===========================================================================
def run_campagne(df: pd.DataFrame,
                 client,
                 style: str,
                 chemin_sortie: str | Path,
                 *,
                 max_appels: int = cfg.LLM_ITERATION_SIZE,
                 budget_max_usd: float = cfg.LLM_BUDGET_MAX_USD,
                 labels: Iterable[str] | None = None,
                 est_echantillon_eval: bool | None = None,
                 frequence_progression: int = 5,
                 execution: int = 1) -> dict[str, Any]:
    """Classe les lignes de `df` et journalise chaque appel en JSONL.

    Parameters
    ----------
    max_appels : PLAFOND DUR d'appels pour cette invocation. Vaut 20 par defaut,
        volontairement bas : depasser ce volume doit etre un geste explicite.
    budget_max_usd : la campagne s'arrete si le cout reel cumule le depasse.
        L'ecriture etant incrementale, rien de ce qui a ete paye n'est perdu.
    execution : numero de passe, journalise tel quel. Sert au test de
        determinisme, qui rejoue les MEMES lignes plusieurs fois : sans ce
        champ, la reprise sauterait la deuxieme passe.

    Returns
    -------
    dict de synthese : compte d'appels, cout reel, taux de parse_error,
    latences, et le motif d'arret.
    """
    cfg.ensure_llm_dirs()
    chemin_sortie = Path(chemin_sortie)
    # Le referentiel PRESENTE AU MODELE depend du style : v4 interroge avec les
    # libelles CFPB officiels. Les predictions sont ensuite REMAPPEES vers les
    # libelles courts avant journalisation, sans quoi elles seraient toutes
    # comptees `label_inconnu` et la variante mesuree a zero.
    labels = tuple(labels) if labels is not None else prompts.labels_du_style(style)

    est_eval = _verifie_autorisation(df, style, est_echantillon_eval)

    # --- reprise -----------------------------------------------------------
    deja = charge_resultats(chemin_sortie)
    # La cle de reprise inclut la passe : rejouer les memes lignes pour le test
    # de determinisme ne doit pas etre confondu avec un doublon a sauter.
    traites = {(str(r.get(cfg.ID_COL)), r.get("execution", 1)) for r in deja}
    if deja:
        print(f"[reprise] {len(deja)} appel(s) deja journalise(s) dans "
              f"{chemin_sortie.name} -- ils ne seront pas redepenses.")

    a_faire = [r for _, r in df.iterrows()
               if (str(r[cfg.ID_COL]), execution) not in traites]

    # --- cout deja engage, repris du journal -------------------------------
    cout_cumule = sum(
        c for c in (r.get("cout_usd") for r in deja) if isinstance(c, (int, float))
    )

    print(f"[campagne] style={style!r} modele={client.modele_api!r} "
          f"tarif={client.cle_tarification!r} passe={execution}")
    print(f"[campagne] {len(a_faire)} ligne(s) a traiter, plafond {max_appels} "
          f"appel(s), budget {budget_max_usd:.2f} $ "
          f"(deja engage {cout_cumule:.4f} $)")
    if est_eval:
        print("[campagne] cible = ECHANTILLON D'EVALUATION (style fige).")

    latences: list[float] = []
    statuts: list[str] = []
    n_appels = 0
    # Deux compteurs distincts : le cumul du JOURNAL sert au garde-fou de
    # budget (il doit compter ce qui a deja ete paye), mais le compte rendu de
    # CETTE invocation ne doit pas s'attribuer les depenses des precedentes.
    cout_cette_execution = 0.0
    motif_arret = "termine"

    with chemin_sortie.open("a", encoding="utf-8") as f:
        for ligne in a_faire:
            if n_appels >= max_appels:
                motif_arret = f"plafond de {max_appels} appels atteint"
                break
            if cout_cumule >= budget_max_usd:
                motif_arret = (f"budget de {budget_max_usd:.2f} $ atteint "
                               f"({cout_cumule:.4f} $)")
                break

            messages, tronque = prompts.construit_prompt(
                ligne[cfg.TEXT_COL], labels=labels, style=style)
            reponse = client.predict(messages)
            n_appels += 1

            label, statut = prompts.parse_reponse(reponse["reponse_brute"], labels)
            # Ramene au referentiel court fige de CLASS_ORDER. Sans effet pour
            # tous les styles sauf v4 ; PARSE_ERROR traverse inchange.
            label = prompts.remappe_vers_court(label, style)
            statuts.append(statut if reponse["erreur_http"] is None else "erreur_http")
            if reponse["latence_s"] is not None:
                latences.append(reponse["latence_s"])

            cout = cout_reel(reponse["tokens_in"], reponse["tokens_out"],
                             reponse["tokens_caches"], client.cle_tarification)
            if cout is not None:
                cout_cumule += cout
                cout_cette_execution += cout

            _ecrit(f, {
                cfg.ID_COL: str(ligne[cfg.ID_COL]),
                # L'etiquette vraie est journalisee pour rendre le JSONL
                # auto-suffisant a l'analyse d'erreurs. Le TEXTE, lui, n'est pas
                # duplique : il reste dans data/processed/, joignable par l'id.
                "label_vrai": ligne.get(cfg.LABEL_COL),
                "label_predit": label,
                "statut_parsing": statut,
                "reponse_brute": reponse["reponse_brute"],
                "style": style,
                "execution": execution,
                "texte_tronque": tronque,
                "cout_usd": cout,
                **{k: reponse[k] for k in (
                    "tokens_in", "tokens_out", "tokens_total", "tokens_caches",
                    "service_tier", "latence_s", "latence_avec_attentes_s",
                    "erreur_http", "tentatives", "modele", "modele_resolu",
                    "modele_annonce", "horodatage")},
            })

            if n_appels % frequence_progression == 0:
                _affiche_progression(n_appels, len(a_faire), cout_cumule,
                                     statuts, latences)

    synthese = {
        "n_appels": n_appels,
        "motif_arret": motif_arret,
        "cout_reel_usd": cout_cette_execution,
        "cout_cumule_journal_usd": cout_cumule,
        "cout_calculable": client.cle_tarification is not None,
        "taux_parse_error": _taux_erreur(statuts),
        "statuts": {s: statuts.count(s) for s in sorted(set(statuts))},
        "chemin": str(chemin_sortie),
        **latency_stats(latences),
    }
    print(f"\n[campagne] arret : {motif_arret}")
    print(f"[campagne] {n_appels} appel(s), cout de cette execution "
          f"{cout_cette_execution:.4f} $ (cumul du journal {cout_cumule:.4f} $), "
          f"parse_error {synthese['taux_parse_error']:.1%}")
    return synthese


def _taux_erreur(statuts: list[str]) -> float:
    """Part des appels dont le format n'est pas conforme."""
    if not statuts:
        return 0.0
    return sum(1 for s in statuts if s != "ok") / len(statuts)


def _affiche_progression(n: int, total: int, cout: float,
                         statuts: list[str], latences: list[float]) -> None:
    """Progression : avancement, cout REEL cumule, non-conformite, latence."""
    lat = latency_stats(latences)
    p50 = lat.get("latence_p50_s", float("nan"))
    p95 = lat.get("latence_p95_s", float("nan"))
    print(f"  {n:>4}/{total}  cout {cout:7.4f} $  "
          f"parse_error {_taux_erreur(statuts):5.1%}  "
          f"latence p50 {p50:5.2f}s p95 {p95:5.2f}s")


def resultats_en_df(chemin: str | Path) -> pd.DataFrame:
    """Charge un JSONL de campagne en DataFrame, pour l'analyse."""
    return pd.DataFrame(charge_resultats(chemin))
