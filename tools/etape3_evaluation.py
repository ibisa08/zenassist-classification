"""ETAPE 3, PHASE 2 — evaluation du ML et analyse croisee.

    python tools/etape3_evaluation.py

ORDRE, tel qu'arrete avant execution :
  1. LinearSVC sur les 2 000 lignes PONDEREES Horvitz-Thompson
  2. LinearSVC sur le test complet (70 863), tableau SEPARE, NON pondere
  3. McNemar apparie LinearSVC vs mistral-small sur les 2 000
  4. Analyse croisee selon reports/protocole_analyse_croisee.md et son erratum
  5. LogisticRegression et MultinomialNB sur les 2 000, analyse croisee seulement

CE QUI EST GELE ET N'EST PAS TOUCHE ICI
---------------------------------------
`src/metrics.py` et `src/data_prep.py`. Toutes les metriques passent par
`metrics.evaluate()`, avec `sample_weight` fourni explicitement a chaque appel.
`CLASS_ORDER` est inchange, pour que les matrices de confusion du ML et du LLM
se superposent.

LES MODELES SONT CHARGES DEPUIS `models/`, PAS REENTRAINES
-----------------------------------------------------------
Le chiffre publie doit venir de l'artefact effectivement livre. Reentrainer ici
produirait un modele equivalent mais non identifie : on ne saurait pas quelle
empreinte porte le resultat.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RACINE = Path(__file__).resolve().parent
while not (RACINE / "src" / "config.py").exists() and RACINE != RACINE.parent:
    RACINE = RACINE.parent
sys.path.insert(0, str(RACINE))

from src import config as cfg          # noqa: E402
from src import data_prep as dp        # noqa: E402
from src import llm_eval as lle        # noqa: E402
from src import metrics as mt          # noqa: E402

MODELS_DIR = cfg.PROJECT_ROOT / "models"
CAMPAGNE_LLM = cfg.LLM_DIR / "phase4_eval2000_mistral_small.jsonl"
CAMPAGNE_LLM_B = cfg.LLM_DIR / "phase4_eval2000_ministral_3b.jsonl"
SORTIE = cfg.REPORTS_DIR / "etape3_evaluation.json"

# Ancrage haut, mesure a l'etape 2 entre les deux LLM. Fige.
LIFT_INTRA_LLM = 3.19
# Trois classes de plus faible F1 au §C.4, arretees avant tout entrainement ML.
CLASSES_DIFFICILES = ("Vehicle loan or lease", "Payday, title or personal loan",
                      "Debt collection")


def titre(t: str, char: str = "=") -> None:
    print("\n" + char * 78 + f"\n{t}\n" + char * 78, flush=True)


# ===========================================================================
# Predictions
# ===========================================================================
def predit_avec_latence(pipe, textes) -> tuple[np.ndarray, list[float]]:
    """Une reclamation a la fois, chronometree — memes conditions qu'en phase B.

    On ne divise JAMAIS un temps de lot par le nombre de lignes : c'est la
    latence percue par un utilisateur qui soumet UNE reclamation qui interesse
    le client (cf. la docstring de `metrics.Timer`).
    """
    for t in textes[:50]:                      # prechauffage, non mesure
        pipe.predict([t])
    chrono = mt.Timer()
    preds = []
    for t in textes:
        with chrono:
            preds.append(pipe.predict([t])[0])
    return np.asarray(preds), chrono.latencies


def resultat_ml(nom, y_true, y_pred, ids, poids, source, latences=None,
                with_ci=True):
    """Emballe une evaluation ML dans la meme forme qu'une campagne LLM.

    Les cles prefixees par `_` sont celles que `llm_eval` attend pour ses outils
    apparies (McNemar, paires d'erreurs). Les reutiliser garantit que le ML et
    le LLM sont mesures par exactement le meme code — c'est la raison d'etre du
    gel de `metrics.py`.
    """
    r = mt.evaluate(y_true, y_pred, latencies=latences,
                    sample_weight=poids, source=source, with_ci=with_ci)
    r["_ids_evalues"] = [str(i) for i in ids]
    r["_y_true"] = list(y_true)
    r["_y_pred"] = list(y_pred)
    r["_poids"] = None if poids is None else list(np.asarray(poids, dtype=float))
    r["modele"] = nom
    return r


# ===========================================================================
# Ecart de F1 apparie, avec intervalle
# ===========================================================================
def ecart_f1_apparie(res_a, res_b, poids, strates, n_boot=cfg.BOOTSTRAP_N):
    """IC bootstrap de l'ECART de F1-macro entre deux modeles sur les memes lignes.

    Le plan de rééchantillonnage reproduit exactement celui de
    `metrics.bootstrap_ci` : tirage avec remise A L'INTERIEUR de chaque classe,
    effectifs conserves, meme graine. La difference tient a un seul point : les
    DEUX modeles sont evalues sur LE MEME tirage. Comparer deux IC calcules
    separement traiterait les modeles comme independants, ce qu'ils ne sont pas
    — memes reclamations, memes cas faciles.
    """
    ya = np.asarray(res_a["_y_true"]); pa = np.asarray(res_a["_y_pred"])
    pb = np.asarray(res_b["_y_pred"])
    w = None if poids is None else np.asarray(poids, dtype=float)
    obs = mt.f1_macro(ya, pa, sample_weight=w) - mt.f1_macro(ya, pb, sample_weight=w)

    rng = np.random.default_rng(cfg.RANDOM_SEED)
    st = np.asarray(strates)
    groupes = [np.flatnonzero(st == s) for s in pd.unique(st)]
    tirages = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.concatenate([rng.choice(g, size=g.size, replace=True) for g in groupes])
        ww = None if w is None else w[idx]
        tirages[i] = (mt.f1_macro(ya[idx], pa[idx], sample_weight=ww)
                      - mt.f1_macro(ya[idx], pb[idx], sample_weight=ww))
    bas, haut = np.percentile(tirages, [cfg.BOOTSTRAP_ALPHA / 2 * 100,
                                        (1 - cfg.BOOTSTRAP_ALPHA / 2) * 100])
    return {"ecart": float(obs), "ic_bas": float(bas), "ic_haut": float(haut),
            "n_boot": n_boot, "plan": "stratifie intra-classe, meme tirage pour les deux"}


# ===========================================================================
# Analyse croisee — protocole_analyse_croisee.md
# ===========================================================================
def analyse_croisee(res_a, nom_a, res_b, nom_b, verbeux=True):
    """Recouvrement des erreurs entre deux approches, protocole du 2026-08-19.

    Denombrement NON PONDERE : il s'agit de compter des cas a lire, pas
    d'estimer une grandeur de population (protocole §5, dernier point).
    """
    A = pd.DataFrame({"id": res_a["_ids_evalues"], "vrai": res_a["_y_true"],
                      "predit_a": res_a["_y_pred"]}).set_index("id")
    B = pd.DataFrame({"id": res_b["_ids_evalues"],
                      "predit_b": res_b["_y_pred"]}).set_index("id")
    d = A.join(B, how="inner")
    if len(d) != len(A) or len(d) != len(B):
        raise ValueError("Les deux approches ne portent pas sur les memes lignes.")

    # §1 — exclusion pre-specifiee des PARSE_ERROR : echecs de FORMAT, pas de
    # classification. Le ML ne peut pas en produire ; les compter gonflerait le
    # co-echec sans rapport avec la difficulte du texte.
    est_pe = (d["predit_a"] == cfg.PARSE_ERROR) | (d["predit_b"] == cfg.PARSE_ERROR)
    n_pe = int(est_pe.sum())
    d = d[~est_pe]
    N = len(d)

    err_a = d["vrai"] != d["predit_a"]
    err_b = d["vrai"] != d["predit_b"]
    inter = err_a & err_b
    oriente = inter & (d["predit_a"] == d["predit_b"])
    p_a, p_b = float(err_a.mean()), float(err_b.mean())
    attendu = N * p_a * p_b
    lift = float(inter.sum() / attendu) if attendu else float("nan")
    union = int((err_a | err_b).sum())
    jaccard = float(inter.sum() / union) if union else float("nan")
    R = lift / LIFT_INTRA_LLM

    # §4 — critere complementaire : concentration sur les paires du §C.4
    annoncees = {" ↔ ".join(sorted(p)) for p in lle.PAIRES_ANNONCEES_C4}
    def part_sur_paires(masque, col):
        s = d[masque]
        if s.empty:
            return None, 0
        paires = s.apply(lambda r: " ↔ ".join(sorted([r["vrai"], r[col]])), axis=1)
        return float(paires.isin(annoncees).mean()), len(s)
    part_com, n_com = part_sur_paires(inter, "predit_a")
    part_pa, n_pa = part_sur_paires(err_a & ~err_b, "predit_a")
    part_pb, n_pb = part_sur_paires(err_b & ~err_a, "predit_b")
    parts_propres = [x for x in (part_pa, part_pb) if x is not None]
    part_propre = float(np.mean(parts_propres)) if parts_propres else None

    # §4, point 3 — taux de rattrapage sur les 3 classes de plus faible F1.
    # Denominateur : les erreurs de B (le LLM) dont la classe VRAIE est dans les
    # trois classes, et non le total de ses erreurs.
    dur = d["vrai"].isin(CLASSES_DIFFICILES)
    den = int((err_b & dur).sum())
    num = int((err_b & dur & ~err_a).sum())
    rattrapage = num / den if den else None

    res = {"paire": f"{nom_a} / {nom_b}", "n_lignes": N, "n_parse_error_exclus": n_pe,
           "erreurs_" + nom_a: int(err_a.sum()), "erreurs_" + nom_b: int(err_b.sum()),
           "taux_erreur_a": p_a, "taux_erreur_b": p_b,
           "attendu_sous_independance": float(attendu),
           "co_echec": int(inter.sum()), "co_echec_oriente": int(oriente.sum()),
           "part_orientee": float(oriente.sum() / inter.sum()) if inter.sum() else None,
           "lift": lift, "jaccard": jaccard, "R": float(R),
           "concentration": {"communes": part_com, "n_communes": n_com,
                             "propres_a": part_pa, "n_propres_a": n_pa,
                             "propres_b": part_pb, "n_propres_b": n_pb,
                             "propres_moyenne": part_propre,
                             "ecart_points": (None if part_com is None or part_propre is None
                                              else (part_com - part_propre) * 100)},
           "rattrapage_3_classes": {"numerateur": num, "denominateur": den,
                                    "taux": rattrapage}}
    if verbeux:
        _affiche_croisee(res, nom_a, nom_b)
    return res


def _affiche_croisee(r, nom_a, nom_b):
    titre(f"ANALYSE CROISEE — {nom_a} / {nom_b}", "-")
    print(f"  lignes retenues                  : {r['n_lignes']}"
          f"   (PARSE_ERROR exclus : {r['n_parse_error_exclus']})")
    print(f"  erreurs {nom_a:<24s} : {r['erreurs_' + nom_a]:>5}  "
          f"({r['taux_erreur_a']:.2%})")
    print(f"  erreurs {nom_b:<24s} : {r['erreurs_' + nom_b]:>5}  "
          f"({r['taux_erreur_b']:.2%})")
    print(f"  attendu sous independance        : {r['attendu_sous_independance']:>8.1f}")
    print(f"  CO-ECHEC observe (definition primaire)   : {r['co_echec']:>5}")
    print(f"  co-echec ORIENTE (meme classe fausse)    : {r['co_echec_oriente']:>5}"
          + (f"  ({r['part_orientee']:.1%} des co-echecs)" if r["part_orientee"] else ""))
    print(f"  lift    : {r['lift']:.3f}      jaccard : {r['jaccard']:.3f}")
    print(f"  R = lift / {LIFT_INTRA_LLM} : {r['R']:.3f}")
    c = r["concentration"]
    print(f"\n  concentration sur les paires §C.4")
    print(f"    erreurs communes   : {c['communes']:.1%} de {c['n_communes']}"
          if c["communes"] is not None else "    erreurs communes   : —")
    print(f"    propres a {nom_a:<14s}: {c['propres_a']:.1%} de {c['n_propres_a']}"
          if c["propres_a"] is not None else f"    propres a {nom_a} : —")
    print(f"    propres a {nom_b:<14s}: {c['propres_b']:.1%} de {c['n_propres_b']}"
          if c["propres_b"] is not None else f"    propres a {nom_b} : —")
    if c["ecart_points"] is not None:
        print(f"    ecart communes - propres : {c['ecart_points']:+.1f} points "
              f"(seuil pre-fixe : 10 points)")
    ra = r["rattrapage_3_classes"]
    print(f"\n  rattrapage sur les 3 classes de plus faible F1")
    print(f"    {ra['numerateur']} / {ra['denominateur']} = "
          + (f"{ra['taux']:.1%}" if ra["taux"] is not None else "—")
          + "   (seuil pre-fixe : 30 %)")


# ===========================================================================
def main() -> int:
    sortie: dict = {"_avertissement": (
        "Tableau 1 : 2 000 lignes PONDEREES Horvitz-Thompson — SEULE comparaison "
        "valide avec le LLM. Tableau 2 : test complet NON pondere — ne se compare "
        "PAS au LLM, il montre ce que le volume d'entrainement apporte au ML.")}

    # --- lignes d'evaluation, identiques a celles du LLM --------------------
    df_eval, poids_tous = dp.load_eval_sample()
    titre("CAMPAGNE LLM DE REFERENCE — mistral-small")
    res_llm = lle.evalue_campagne(CAMPAGNE_LLM, df_eval, poids=poids_tous,
                                  with_ci=True, verbeux=False)
    ids = lle.ids_evalues(res_llm)
    print(f"  {len(ids)} ligne(s) evaluee(s) par le LLM, F1-macro pondere "
          f"{res_llm['f1_macro']:.4f}")

    # Le journal de campagne stocke `complaint_id` en TEXTE, le CSV en entier.
    # On aligne sur la forme texte, et on passe par les POSITIONS : un `merge`
    # sur des types differents joindrait a vide sans rien signaler, alors qu'une
    # indexation positionnelle est verifiable ligne a ligne (assert ci-dessous).
    cle = df_eval[cfg.ID_COL].astype(str)
    if cle.duplicated().any():
        raise ValueError("identifiants dupliques dans l'echantillon d'evaluation")
    pos = pd.Series(np.arange(len(df_eval)), index=cle).loc[ids].to_numpy()
    sub = df_eval.iloc[pos].reset_index(drop=True)
    poids = np.asarray(poids_tous, dtype=float)[pos]
    y_true = sub[cfg.LABEL_COL].to_numpy()
    textes = sub[cfg.TEXT_COL].to_numpy()
    assert list(sub[cfg.ID_COL].astype(str)) == list(ids), \
        "reordonnancement des lignes rompu"
    print(f"  memes lignes cote ML : {len(sub)}, somme des poids "
          f"{poids.sum():,.0f}")

    # --- 1 et 5 : les modeles ML sur les 2 000 pondere ----------------------
    res_ml, latences = {}, {}
    for nom in ("LinearSVC", "LogisticRegression", "MultinomialNB"):
        pipe = pickle.loads((MODELS_DIR / f"{nom}.pkl").read_bytes())
        yp, lat = predit_avec_latence(pipe, textes)
        res_ml[nom] = resultat_ml(nom, y_true, yp, ids, poids, sub, latences=lat)
        latences[nom] = mt.latency_stats(lat)
        del pipe

    titre("TABLEAU 1 — 2 000 LIGNES PONDEREES HORVITZ-THOMPSON")
    print("  Seule comparaison valide avec le LLM.\n")
    tab1 = mt.compare_results({**{n: r for n, r in res_ml.items()},
                               "mistral-small (LLM)": res_llm})
    print(tab1.to_string())
    sortie["tableau1_2000_pondere"] = json.loads(tab1.to_json(orient="index"))
    sortie["latences_2000"] = latences

    print("\n  Matrice de confusion — LinearSVC, 9 classes, ponderee")
    fig = cfg.FIGURES_DIR / "etape3_confusion_linearsvc.png"
    mt.plot_confusion_matrix(res_ml["LinearSVC"]["_y_true"],
                             res_ml["LinearSVC"]["_y_pred"],
                             sample_weight=poids,
                             title="LinearSVC — 2 000 lignes ponderees HT",
                             save_path=fig)
    print(f"    ecrite : {fig.relative_to(cfg.PROJECT_ROOT)}")
    print("    Aucune colonne PARSE_ERROR : le ML rend toujours une classe du referentiel.")

    # --- 2 : test complet, tableau SEPARE, NON pondere ----------------------
    titre("TABLEAU 2 — TEST COMPLET (70 863), NON PONDERE — SEPARE")
    print("  CE CHIFFRE NE SE COMPARE PAS AU LLM. Population, ponderation et")
    print("  effectif differents. Il montre ce que le VOLUME d'entrainement")
    print("  apporte au ML, rien d'autre.\n")
    test = pd.read_csv(dp._fichiers_split()[1], encoding="utf-8")
    pipe = pickle.loads((MODELS_DIR / "LinearSVC.pkl").read_bytes())
    yp_test = pipe.predict(test[cfg.TEXT_COL].to_numpy())   # par lot : aucune latence lue ici
    res_test = mt.evaluate(test[cfg.LABEL_COL].to_numpy(), yp_test,
                           sample_weight=None, with_ci=False)
    del pipe
    print(f"  n                 : {len(test):,}")
    print(f"  F1-macro          : {res_test['f1_macro']:.4f}")
    print(f"  accuracy          : {res_test['accuracy']:.4f}")
    print(f"  F1-pondere        : {res_test['f1_weighted']:.4f}")
    sortie["tableau2_test_complet_non_pondere"] = {
        "n": len(test), "f1_macro": res_test["f1_macro"],
        "accuracy": res_test["accuracy"], "f1_weighted": res_test["f1_weighted"],
        "avertissement": "NON PONDERE, NE SE COMPARE PAS AU LLM"}

    # --- 3 : McNemar apparie ------------------------------------------------
    titre("3. McNEMAR APPARIE — LinearSVC vs mistral-small, memes 2 000 lignes")
    mn = lle.mcnemar(res_ml["LinearSVC"], res_llm, "LinearSVC", "mistral-small")
    ec = ecart_f1_apparie(res_ml["LinearSVC"], res_llm, poids, y_true)
    print(f"\n  ecart de F1-macro pondere (LinearSVC - mistral-small)")
    print(f"    {ec['ecart']:+.4f}   IC 95 % apparie [{ec['ic_bas']:+.4f} ; {ec['ic_haut']:+.4f}]")
    print(f"    plan : {ec['plan']}")
    sortie["mcnemar_linearsvc_vs_mistral_small"] = mn
    sortie["ecart_f1_apparie"] = ec

    # --- 4 : analyse croisee ------------------------------------------------
    titre("4. ANALYSE CROISEE — protocole du 2026-08-19 et son erratum")
    sortie["analyse_croisee"] = {
        "LinearSVC_vs_mistral_small": analyse_croisee(
            res_ml["LinearSVC"], "LinearSVC", res_llm, "mistral-small"),
        "LogisticRegression_vs_mistral_small": analyse_croisee(
            res_ml["LogisticRegression"], "LogisticRegression", res_llm, "mistral-small"),
        "MultinomialNB_vs_mistral_small": analyse_croisee(
            res_ml["MultinomialNB"], "MultinomialNB", res_llm, "mistral-small"),
    }
    titre("ANCRAGE INTRA-FAMILLE ML — CONTAMINE, NON DECISIONNEL", "!")
    print("  Rapporte pour documentation seulement. MultinomialNB porte deux")
    print("  inadequations de representation mesurees (vectoriseur +0,010,")
    print("  lissage +0,042) : son recouvrement avec LogisticRegression melange")
    print("  difference d'architecture et defaut de representation, sans qu'on")
    print("  puisse les separer. Le critere R est lu SANS cet ancrage.")
    sortie["analyse_croisee"]["CONTAMINE_LogReg_vs_NB"] = analyse_croisee(
        res_ml["LogisticRegression"], "LogisticRegression",
        res_ml["MultinomialNB"], "MultinomialNB")

    SORTIE.write_text(json.dumps(sortie, indent=2, ensure_ascii=False, default=str) + "\n",
                      encoding="utf-8")
    print(f"\nEcrit : {SORTIE.relative_to(cfg.PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
