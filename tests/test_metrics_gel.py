"""Non-regression de `src/metrics.py` — LE GEL, RENDU OPPOSABLE.

`metrics.py` porte en tete la regle suivante :

    Seules les modifications SANS effet sur une valeur calculee sont
    autorisees (docstrings, commentaires, annotations de type). Toute autre
    modification exige une revalidation croisee des etapes 2 et 3.

Une regle qu'aucun test ne verifie n'est pas une regle, c'est une intention.
Ce fichier rejoue un JEU FIGE et compare les sorties a des valeurs de reference
relevees le 2026-08-19. Une modification de logique deguisee en correction de
commentaire echoue ici, immediatement et sans ambiguite.

POURQUOI CE GEL COMPTE
----------------------
Le LLM (etape 2) et le ML (etape 3) doivent etre mesures par exactement le meme
code. Si `evaluate()` change entre les deux mesures, l'ecart final melange
l'ecart entre les approches et l'ecart entre deux versions du metre. La
comparaison, qui est le livrable du projet, ne vaudrait plus rien.

DEUX STRATEGIES DE VERIFICATION, DELIBEREMENT DIFFERENTES
---------------------------------------------------------
1. `evaluate()` : valeurs de reference EN DUR. La fonction ne depend que de ses
   entrees et de `CLASS_ORDER` (fige lui aussi), donc ses sorties doivent etre
   immuables au dernier chiffre.

2. `estimate_cost()` : verification de la FORMULE par recalcul arithmetique
   independant, PAS de valeurs en dur. Cette fonction lit `config` (tarifs,
   volumes de tokens), et ces valeurs changent legitimement -- elles ont change
   le 2026-08-19. Y epingler des nombres produirait un echec a chaque mesure
   nouvelle : un test qui crie au loup a chaque changement legitime finit par
   etre desactive, et c'est ainsi qu'on perd un garde-fou.

    python tests/test_metrics_gel.py
"""

import sys
from pathlib import Path

import numpy as np

RACINE = Path(__file__).resolve().parent
while not (RACINE / "src" / "config.py").exists() and RACINE != RACINE.parent:
    RACINE = RACINE.parent
sys.path.insert(0, str(RACINE))

from src import config as cfg    # noqa: E402
from src import metrics as mt    # noqa: E402

OK, KO = "  OK  ", " ECHEC"
_RESULTATS: list[bool] = []
TOL = 1e-12


def check(nom, cond, detail=""):
    _RESULTATS.append(bool(cond))
    print(f"[{OK if cond else KO}] {nom}" + (f"   -> {detail}" if detail else ""))


def proche(a, b, tol=TOL):
    return a is not None and abs(a - b) < tol


# ---------------------------------------------------------------------------
# Jeu fige. Construit par une regle deterministe, jamais tire au hasard :
# 36 lignes couvrant les 9 classes, 6 erreurs a positions fixes, une
# non-conformite de format, des poids inegaux et des latences connues.
# ---------------------------------------------------------------------------
def jeu_fige():
    CO = cfg.CLASS_ORDER
    y_true = [CO[i % 9] for i in range(36)]
    y_pred = list(y_true)
    for i in (0, 5, 11, 17, 23, 29):
        y_pred[i] = CO[(CO.index(y_true[i]) + 3) % 9]
    y_pred[7] = cfg.PARSE_ERROR
    poids = [1.0 + (i % 4) * 7.5 for i in range(36)]
    lat = [0.1 * (1 + (i % 7)) for i in range(36)]
    return y_true, y_pred, poids, lat


# Valeurs relevees le 2026-08-19 sur metrics.py valide a l'etape 1.
REFERENCE_NON_PONDERE = {
    "n": 36,
    "accuracy": 0.8055555555555556,
    "f1_macro": 0.8156966490299823,
    "f1_weighted": 0.8156966490299823,
    "precision_macro": 0.8407407407407407,
    "recall_macro": 0.8055555555555556,
    "ci_bas": 0.6422314296088806,
    "ci_haut": 0.9166997354497354,
    "bootstrap_strata": "aucune (auto)",
}
REFERENCE_PONDERE = {
    "accuracy": 0.780045351473923,
    "f1_macro": 0.7926816390045314,
    "f1_weighted": 0.7926816390045314,
    "precision_macro": 0.839136236870227,
    "recall_macro": 0.780045351473923,
    "ci_bas": 0.6534181425207538,
    "ci_haut": 0.9066898118265571,
    "bootstrap_strata": "y_true (auto)",
}
REFERENCE_F1_CLASSE = {
    "Credit reporting": 0.857142857143, "Debt collection": 1.0,
    "Mortgage": 0.571428571429, "Credit card or prepaid card": 0.888888888889,
    "Bank account or service": 1.0, "Student loan": 0.5,
    "Money transfer or virtual currency": 1.0,
    "Payday, title or personal loan": 0.857142857143,
    "Vehicle loan or lease": 0.666666666667,
}
REFERENCE_LATENCE = {
    "latence_n": 36, "latence_moyenne_s": 0.39166666666666666,
    "latence_p50_s": 0.4, "latence_p95_s": 0.7000000000000001,
    "latence_max_s": 0.7000000000000001,
}


def main() -> int:
    _RESULTATS.clear()
    y_true, y_pred, poids, lat = jeu_fige()

    print("=" * 78)
    print("evaluate() — SORTIES IMMUABLES SUR LE JEU FIGE")
    print("=" * 78)
    r = mt.evaluate(y_true, y_pred, latencies=lat, sample_weight=None, with_ci=True)
    for cle, attendu in REFERENCE_NON_PONDERE.items():
        if cle in ("ci_bas", "ci_haut"):
            continue
        obtenu = r[cle]
        ok = (obtenu == attendu if isinstance(attendu, str)
              else proche(obtenu, attendu))
        check(f"non pondere · {cle}", ok, f"{obtenu!r} (reference {attendu!r})")
    check("non pondere · IC bootstrap bas",
          proche(r["f1_macro_ci"]["bas"], REFERENCE_NON_PONDERE["ci_bas"]),
          f"{r['f1_macro_ci']['bas']!r}")
    check("non pondere · IC bootstrap haut",
          proche(r["f1_macro_ci"]["haut"], REFERENCE_NON_PONDERE["ci_haut"]),
          f"{r['f1_macro_ci']['haut']!r}")

    rp = mt.evaluate(y_true, y_pred, latencies=lat, sample_weight=poids, with_ci=True)
    for cle, attendu in REFERENCE_PONDERE.items():
        if cle in ("ci_bas", "ci_haut"):
            continue
        obtenu = rp[cle]
        ok = (obtenu == attendu if isinstance(attendu, str)
              else proche(obtenu, attendu))
        check(f"pondere · {cle}", ok, f"{obtenu!r} (reference {attendu!r})")
    check("pondere · IC bootstrap bas",
          proche(rp["f1_macro_ci"]["bas"], REFERENCE_PONDERE["ci_bas"]))
    check("pondere · IC bootstrap haut",
          proche(rp["f1_macro_ci"]["haut"], REFERENCE_PONDERE["ci_haut"]))

    print("\n  -- F1 par classe --")
    for classe, attendu in REFERENCE_F1_CLASSE.items():
        obtenu = r["classification_report"][classe]["f1-score"]
        check(f"F1 · {classe}", proche(obtenu, attendu, 1e-9),
              f"{obtenu:.12f} (reference {attendu})")

    print("\n  -- latences --")
    for cle, attendu in REFERENCE_LATENCE.items():
        check(f"latence · {cle}", proche(r[cle], attendu), f"{r[cle]!r}")

    print("\n  -- comportements structurels --")
    check("le bootstrap est reproductible (meme graine, meme resultat)",
          mt.evaluate(y_true, y_pred, sample_weight=None,
                      with_ci=True)["f1_macro_ci"]["bas"]
          == r["f1_macro_ci"]["bas"])
    check("PARSE_ERROR reste hors du rapport par classe",
          cfg.PARSE_ERROR not in r["classification_report"])
    check("PARSE_ERROR est exclu de la matrice de confusion",
          int(np.array(r["confusion_matrix"]).sum()) == 35,
          "35 = 36 lignes moins la non-conformite")
    check("f1_macro() autonome concorde avec evaluate()",
          proche(mt.f1_macro(y_true, y_pred), r["f1_macro"]))

    print("\n" + "=" * 78)
    print("GARDE-FOUS DE PONDERATION — TOUJOURS DES EXCEPTIONS")
    print("=" * 78)
    leve = False
    try:
        mt.evaluate(y_true, y_pred)          # sample_weight omis
    except TypeError:
        leve = True
    check("sample_weight omis leve TypeError", leve)

    import pandas as pd
    src = pd.DataFrame({cfg.LABEL_COL: y_true, cfg.WEIGHT_COL: poids})
    leve = False
    try:
        mt.evaluate(y_true, y_pred, sample_weight=None, source=src)
    except ValueError:
        leve = True
    check("source ponderee sans poids leve ValueError", leve)

    print("\n" + "=" * 78)
    print("estimate_cost() — LA FORMULE, PAS LES NOMBRES")
    print("=" * 78)
    # Recalcul arithmetique independant. Aucune valeur en dur : le test suit
    # config sans jamais devenir une seconde source de verite.
    for modele in ("mistral-small-4", "ministral-3b", "claude-opus-5"):
        t = cfg.MODELS_PRICING[modele]
        n, texte, overhead, sortie, caches = 1000, 300.0, 250.0, 12.0, 100.0
        r_c = mt.estimate_cost(n, modele, avg_tokens=texte,
                               avg_output_tokens=sortie,
                               prompt_overhead_tokens=overhead,
                               cached_prefix_tokens=caches)
        pleins = texte + overhead - caches
        attendu = (pleins * n / 1e6 * t["input_per_1m"]
                   + caches * n / 1e6 * t["input_per_1m"] * (1 - t["remise_cache"])
                   + sortie * n / 1e6 * t["output_per_1m"])
        check(f"formule de cout · {modele}",
              proche(r_c["cout_total_usd"], round(attendu, 6), 1e-6),
              f"{r_c['cout_total_usd']} vs {round(attendu, 6)} recalcule")

    r_h = mt.estimate_cost(1000)
    check("flag `hypotheses` vrai sans mesure fournie", r_h["hypotheses"] is True)
    r_m = mt.estimate_cost(1000, avg_tokens=245, prompt_overhead_tokens=252)
    check("flag `hypotheses` faux quand les tokens sont fournis",
          r_m["hypotheses"] is False)
    check("cache non modelise si la remise n'est pas relevee",
          mt.estimate_cost(1000, "gemini-3.5-flash",
                           cached_prefix_tokens=250)["cache_modelise"] is False)
    leve = False
    try:
        mt.estimate_cost(1000, "modele-inexistant")
    except KeyError:
        leve = True
    check("modele inconnu leve KeyError", leve)

    print("\n" + "=" * 78)
    print(f"RESULTAT : {sum(_RESULTATS)}/{len(_RESULTATS)} verifications passees")
    if not all(_RESULTATS):
        print("\n  Une sortie de metrics.py a CHANGE. Si la modification est")
        print("  volontaire, elle sort du perimetre autorise par la regle de gel")
        print("  et impose une revalidation croisee des etapes 2 et 3.")
    print("=" * 78)
    return 0 if all(_RESULTATS) else 1


if __name__ == "__main__":
    sys.exit(main())
