"""
Phase 2bis - Simulation Monte-Carlo pour trancher H.1 (composition de
l'echantillon d'evaluation LLM).

SOURCE DE VERITE de reports/h1_sampling_simulation.md. Ne fait partie d'aucun
module du pipeline et n'est importe par aucun : il documente et reproduit un
arbitrage, il ne tourne pas en production.

Principe
--------
On simule une matrice de confusion plausible M (M[i,j] = P(predit j | vrai i)),
on l'applique a la population de test complete (70 879 lignes, distribution F.7)
pour obtenir un y_pred synthetique, puis on compare la capacite de 6 strategies
d'echantillonnage a estimer le F1-macro "vrai" de cette population.

Astuce d'implementation
-----------------------
Un tirage stratifie sans remise dans la strate c, ou les items portent des
etiquettes predites en proportions connues, suit exactement une loi
hypergeometrique multivariee. On tire donc directement les lignes de la matrice
de confusion echantillonnee (rng.multivariate_hypergeometric) au lieu de
materialiser les individus : c'est exact et environ 1000x plus rapide.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# 0. Parametres
# --------------------------------------------------------------------------
SEED = 42
N_BOOT = 2000          # nombre d'echantillons independants par strategie
# Racine du projet = le repertoire qui contient src/config.py. Aucun chemin en
# dur : ce script doit tourner depuis n'importe quel repertoire courant.
RACINE = Path(__file__).resolve().parent
while not (RACINE / "src" / "config.py").exists() and RACINE != RACINE.parent:
    RACINE = RACINE.parent
# scratch/ recueille les resultats intermediaires ; il n'est pas versionne.
OUT = RACINE / "scratch"
OUT.mkdir(exist_ok=True)

CLASSES = [
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
SHORT = ["CR", "DC", "MO", "CC", "BA", "SL", "MT", "PD", "VL"]
K = len(CLASSES)

# Effectifs du test set (20 % stratifie du corpus final de 354 395 lignes,
# cf. section F.7 du diagnostic). Total force a 70 879.
N_POP = np.array([21550, 16806, 10585, 8280, 5541, 4354, 1393, 1229, 1141])
assert N_POP.sum() == 70_879, N_POP.sum()
N_TOTAL = int(N_POP.sum())

# --------------------------------------------------------------------------
# 1. Construction de la matrice de confusion synthetique M
# --------------------------------------------------------------------------
# HYPOTHESE DE TRAVAIL, PAS UNE MESURE.
#
# (a) Diagonale (= rappel par classe). Valeurs imposees par l'enonce pour
#     Mortgage (0.88), Student loan (0.85), Debt collection (0.65) et
#     Payday (0.55). Les cinq autres sont calees a la main en suivant la
#     signature lexicale mesuree en C.3 (colonne diagonale du tableau) :
#     plus la signature propre est forte, plus le rappel est eleve.
#         CR 62.7 % -> 0.90 (classe attracteur, tres gros effectif)
#         MO 80.4 % -> 0.88 (impose)
#         SL 73.3 % -> 0.85 (impose)
#         CC 51.6 % -> 0.72
#         BA 42.9 % -> 0.70
#         MT 43.0 % -> 0.68
#         DC 27.8 % -> 0.65 (impose)
#         VL 55.1 % -> 0.60 (signature propre correcte mais forte fuite vers CR)
#         PD 24.3 % -> 0.55 (impose)
#
# (b) Hors-diagonale. Le budget d'erreur E_i = 1 - M_ii est reparti sur les
#     autres classes PROPORTIONNELLEMENT au tableau de recouvrement lexical
#     de la section C.3 (A[i][j] = % de textes de la classe i contenant le
#     vocabulaire signature de la classe j). Cette repartition n'est donc pas
#     choisie a la main : elle decoule des mesures.
#     Elle produit mecaniquement les deux proprietes demandees :
#       - la fuite dominante de la plupart des classes va vers Credit
#         reporting (colonne "credit report" de C.3, la plus forte pour
#         CR/DC/MO/CC/SL/PD/VL) ;
#       - la fuite Vehicle loan -> Credit reporting est la plus elevee de sa
#         ligne (26.7 sur un total hors-diagonale de 38.1, soit 70 % du budget
#         d'erreur de VL).
#     Deux exceptions honnetes : pour Bank account et Money transfer, la fuite
#     dominante n'est PAS Credit reporting mais respectivement Credit card
#     (9.2) et Bank account (9.1). C'est ce que disent les donnees de C.3 et
#     c'est coherent avec la paire confondable identifiee en F.9 ; on ne force
#     pas la structure vers CR contre la mesure.

# Tableau C.3 : lignes = classe reelle, colonnes = mot-cle signature
#              CR    DC    MO    CC    BA    SL    VL    PD    MT
AFFINITE = np.array([
    [62.7,  5.6,  6.0,  9.5,  0.9,  3.6,  4.4,  0.7,  0.1],  # CR
    [35.8, 27.8,  3.2,  6.9,  1.3,  2.0,  2.1,  1.6,  0.3],  # DC
    [ 9.8,  2.4, 80.4,  2.0,  2.5,  2.2,  0.9,  0.2,  0.5],  # MO
    [17.1,  2.1,  1.8, 51.6,  4.7,  0.3,  1.0,  0.2,  1.9],  # CC
    [ 3.5,  0.9,  3.6,  9.2, 42.9,  0.5,  1.5,  0.7,  3.1],  # BA
    [13.4,  4.9,  2.6,  2.3,  2.6, 73.3,  0.9,  0.5,  0.1],  # SL
    [ 0.7,  0.6,  1.6,  7.2,  9.1,  0.4,  0.9,  0.1, 43.0],  # MT
    [11.2,  4.8,  2.0,  3.8,  8.8,  0.8,  4.7, 24.3,  0.9],  # PD
    [26.7,  2.8,  1.7,  2.4,  3.2,  0.4, 55.1,  0.7,  0.2],  # VL
])
# Remise dans l'ordre des classes (le tableau C.3 a VL en 7e colonne, MT en 9e)
ORDRE_C3 = [0, 1, 2, 3, 4, 5, 8, 7, 6]  # -> CR DC MO CC BA SL MT PD VL
AFFINITE = AFFINITE[:, ORDRE_C3]

DIAG_BASE = np.array([0.90, 0.65, 0.88, 0.72, 0.70, 0.85, 0.68, 0.55, 0.60])


def construit_M(diag):
    """Construit une matrice ligne-stochastique a partir d'une diagonale."""
    diag = np.clip(np.asarray(diag, float), 0.05, 0.97)
    M = np.zeros((K, K))
    aff = AFFINITE.copy()
    np.fill_diagonal(aff, 0.0)          # on ne redistribue que hors-diagonale
    for i in range(K):
        E = 1.0 - diag[i]
        M[i] = E * aff[i] / aff[i].sum()
        M[i, i] = diag[i]
    assert np.allclose(M.sum(axis=1), 1.0)
    return M


SCENARIOS = {
    "realiste":     DIAG_BASE,
    "optimiste":    DIAG_BASE + 0.10,
    "pessimiste":   DIAG_BASE - 0.10,
    # variante ou les 3 classes rares sont particulierement mal classees
    "rares_degradees": np.where(
        np.isin(np.arange(K), [6, 7, 8]),
        np.array([0, 0, 0, 0, 0, 0, 0.40, 0.35, 0.35]),
        DIAG_BASE,
    ),
}

# --------------------------------------------------------------------------
# 2. Strategies d'echantillonnage
# --------------------------------------------------------------------------


def alloc_proportionnelle(n_total):
    """Allocation proportionnelle a la population, methode des plus forts restes."""
    exact = N_POP / N_TOTAL * n_total
    base = np.floor(exact).astype(int)
    reste = n_total - base.sum()
    if reste:
        base[np.argsort(-(exact - base))[:reste]] += 1
    return base


def alloc_plancher(n_total, plancher=50):
    """Plancher fixe par classe, le solde reparti proportionnellement."""
    base = np.full(K, plancher, dtype=int)
    solde = n_total - base.sum()
    if solde < 0:
        raise ValueError("plancher trop eleve pour n_total")
    base += alloc_proportionnelle(solde)
    return base


def alloc_equilibree(n_par_classe=111):
    return np.full(K, n_par_classe, dtype=int)


STRATEGIES = {
    "A1": dict(label="Proportionnel n=1000",              n=alloc_proportionnelle(1000), ht=False),
    "A2": dict(label="Proportionnel n=2000",              n=alloc_proportionnelle(2000), ht=False),
    "B1": dict(label="Plancher 50 n=1000 SANS repond.",   n=alloc_plancher(1000),        ht=False),
    "B2": dict(label="Plancher 50 n=1000 AVEC HT",        n=alloc_plancher(1000),        ht=True),
    "B3": dict(label="Plancher 50 n=2000 AVEC HT",        n=alloc_plancher(2000),        ht=True),
    "D":  dict(label="Equilibre 111/classe AVEC HT",      n=alloc_equilibree(111),       ht=True),
}

# --------------------------------------------------------------------------
# 3. Metriques a partir d'une matrice de comptages (eventuellement ponderee)
# --------------------------------------------------------------------------


def metriques(C):
    """C de forme (..., K, K) : comptages [vrai, predit], possiblement ponderes.

    Retourne precision, rappel, f1 par classe et f1_macro. Convention
    zero_division=0, identique a sklearn.
    """
    C = np.asarray(C, float)
    tp = np.einsum("...ii->...i", C)
    pred_tot = C.sum(axis=-2)   # somme sur la dimension "vrai" -> total predit j
    vrai_tot = C.sum(axis=-1)   # somme sur la dimension "predit" -> total vrai i
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(pred_tot > 0, tp / pred_tot, 0.0)
        rapp = np.where(vrai_tot > 0, tp / vrai_tot, 0.0)
        f1 = np.where(prec + rapp > 0, 2 * prec * rapp / (prec + rapp), 0.0)
    return prec, rapp, f1, f1.mean(axis=-1)


# --------------------------------------------------------------------------
# 4. Boucle principale
# --------------------------------------------------------------------------
resultats = []
verifs = {}
distributions = {}      # pour les graphes (scenario realiste uniquement)

for nom_sc, diag in SCENARIOS.items():
    M = construit_M(diag)

    # Population : comptages exacts K_pop[i,j] = nb d'items vrai i predits j.
    # On arrondit a l'entier en conservant le total par ligne.
    K_pop = np.zeros((K, K), dtype=int)
    for i in range(K):
        exact = M[i] * N_POP[i]
        base = np.floor(exact).astype(int)
        reste = N_POP[i] - base.sum()
        if reste:
            base[np.argsort(-(exact - base))[:reste]] += 1
        K_pop[i] = base
    assert (K_pop.sum(axis=1) == N_POP).all()

    # Valeurs "vraies" a estimer
    prec_v, rapp_v, f1_v, f1m_v = metriques(K_pop)

    if nom_sc == "realiste":
        verifs["rappel_vrai"] = dict(zip(SHORT, rapp_v.round(4)))
        verifs["precision_vraie"] = dict(zip(SHORT, prec_v.round(4)))
        verifs["f1_vrai"] = dict(zip(SHORT, f1_v.round(4)))
        verifs["f1_macro_vrai"] = float(f1m_v)
        verifs["M_realiste"] = M.round(4).tolist()

    for code, cfg in STRATEGIES.items():
        n_c = cfg["n"]
        rng = np.random.default_rng(SEED)

        # Tirage : une hypergeometrique multivariee par strate et par replicat
        C_ech = np.empty((N_BOOT, K, K), dtype=np.int64)
        for i in range(K):
            C_ech[:, i, :] = rng.multivariate_hypergeometric(
                K_pop[i], int(n_c[i]), size=N_BOOT
            )

        # Ponderation Horvitz-Thompson : poids w_i = N_i / n_i sur la strate i
        if cfg["ht"]:
            w = (N_POP / n_c).reshape(1, K, 1)
            C_use = C_ech * w
        else:
            C_use = C_ech.astype(float)

        prec_e, rapp_e, f1_e, f1m_e = metriques(C_use)

        def stats(est, vrai):
            biais = float(est.mean() - vrai)
            sd = float(est.std(ddof=1))
            rmse = float(np.sqrt(np.mean((est - vrai) ** 2)))
            lo, hi = np.percentile(est, [2.5, 97.5])
            return dict(vrai=float(vrai), moyenne=float(est.mean()), biais=biais,
                        ecart_type=sd, rmse=rmse, ic_bas=float(lo), ic_haut=float(hi))

        iVL = SHORT.index("VL")
        iCR = SHORT.index("CR")
        ligne = dict(
            scenario=nom_sc, strategie=code, label=cfg["label"],
            n_total=int(n_c.sum()), n_VL=int(n_c[iVL]), n_CR=int(n_c[iCR]),
            **{f"f1m_{k}": v for k, v in stats(f1m_e, f1m_v).items()},
            **{f"f1VL_{k}": v for k, v in stats(f1_e[:, iVL], f1_v[iVL]).items()},
            **{f"precCR_{k}": v for k, v in stats(prec_e[:, iCR], prec_v[iCR]).items()},
        )
        resultats.append(ligne)

        if nom_sc == "realiste":
            distributions[code] = dict(
                f1_macro=f1m_e, f1_VL=f1_e[:, iVL], prec_CR=prec_e[:, iCR],
                rappel_moyen=rapp_e.mean(axis=0), precision_moyenne=prec_e.mean(axis=0),
                n=n_c.tolist(),
            )
            # Nombre d'items PREDITS comme VL (question 6) - non pondere
            distributions[code]["nb_predits_VL"] = C_ech[:, :, iVL].sum(axis=1)
            distributions[code]["nb_TP_VL"] = C_ech[:, iVL, iVL]
            distributions[code]["nb_FP_VL"] = (
                C_ech[:, :, iVL].sum(axis=1) - C_ech[:, iVL, iVL]
            )

df = pd.DataFrame(resultats)
df.to_csv(OUT / "h1_resultats.csv", index=False)

np.savez(
    OUT / "h1_distributions.npz",
    **{f"{c}__{k}": np.asarray(v) for c, d in distributions.items() for k, v in d.items()},
)
json.dump(verifs, open(OUT / "h1_verifs.json", "w"), indent=1, default=float)

# --------------------------------------------------------------------------
# 5. Affichage
# --------------------------------------------------------------------------
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 60)

print("=" * 110)
print("ALLOCATION PAR STRATEGIE (nombre d'items tires par classe)")
print("=" * 110)
alloc = pd.DataFrame({c: STRATEGIES[c]["n"] for c in STRATEGIES}, index=SHORT).T
alloc["TOTAL"] = alloc.sum(axis=1)
print(alloc.to_string())
print("\npopulation :", dict(zip(SHORT, N_POP)))
print("taux de sondage f_i = n_i/N_i (x1000) :")
tx = pd.DataFrame({c: (STRATEGIES[c]["n"] / N_POP * 1000).round(2) for c in STRATEGIES}, index=SHORT).T
print(tx.to_string())

print("\n" + "=" * 110)
print("SCENARIO REALISTE - valeurs VRAIES sur les 70 879 lignes")
print("=" * 110)
print(pd.DataFrame({
    "rappel": verifs["rappel_vrai"], "precision": verifs["precision_vraie"], "f1": verifs["f1_vrai"],
}).T.to_string())
print(f"\nF1-MACRO VRAI = {verifs['f1_macro_vrai']:.5f}")

for cible, titre in [("f1m", "F1-MACRO"), ("f1VL", "F1 de Vehicle loan or lease"),
                     ("precCR", "PRECISION de Credit reporting")]:
    print("\n" + "=" * 110)
    print(f"{titre} - scenario realiste")
    print("=" * 110)
    sub = df[df.scenario == "realiste"][[
        "strategie", "label", "n_VL",
        f"{cible}_vrai", f"{cible}_moyenne", f"{cible}_biais",
        f"{cible}_ecart_type", f"{cible}_rmse", f"{cible}_ic_bas", f"{cible}_ic_haut"]]
    sub.columns = ["strat", "label", "n_VL", "vrai", "moyenne", "biais", "sd", "RMSE", "IC2.5", "IC97.5"]
    print(sub.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

print("\n" + "=" * 110)
print("ROBUSTESSE - RMSE du F1-macro par scenario")
print("=" * 110)
piv = df.pivot(index="strategie", columns="scenario", values="f1m_rmse")
piv = piv[["realiste", "optimiste", "pessimiste", "rares_degradees"]].loc[list(STRATEGIES)]
print((piv * 1000).round(2).to_string() + "   (x1000)")
print("\nBIAIS du F1-macro par scenario (x1000)")
pivb = df.pivot(index="strategie", columns="scenario", values="f1m_biais")
print((pivb[["realiste", "optimiste", "pessimiste", "rares_degradees"]].loc[list(STRATEGIES)] * 1000).round(2).to_string())

print("\n" + "=" * 110)
print("VERIFICATION 5a - LE RAPPEL EST-IL INVARIANT AU TAUX DE SONDAGE ?")
print("=" * 110)
rap = pd.DataFrame({c: distributions[c]["rappel_moyen"] for c in STRATEGIES}, index=SHORT).T
rap.loc["VRAI"] = [verifs["rappel_vrai"][s] for s in SHORT]
print(rap.round(4).to_string())
print("\necart max a la valeur vraie, par strategie :")
print((rap.drop("VRAI") - rap.loc["VRAI"]).abs().max(axis=1).round(5).to_string())

print("\n" + "=" * 110)
print("VERIFICATION 5b - SIGNE DU BIAIS SUR LA PRECISION (B1, non pondere)")
print("=" * 110)
pr = pd.DataFrame({c: distributions[c]["precision_moyenne"] for c in STRATEGIES}, index=SHORT).T
pr.loc["VRAI"] = [verifs["precision_vraie"][s] for s in SHORT]
biais_prec = (pr.drop("VRAI") - pr.loc["VRAI"])
print("biais sur la precision par classe :")
print(biais_prec.round(4).to_string())
print("\ntaux de sondage relatif f_i / f_moyen pour B1 :")
f_b1 = STRATEGIES["B1"]["n"] / N_POP
print(pd.Series((f_b1 / f_b1.mean()).round(2), index=SHORT).to_string())

print("\n" + "=" * 110)
print("QUESTION 6 - NOMBRE D'ITEMS PREDITS COMME 'Vehicle loan or lease'")
print("=" * 110)
q6 = []
for c in STRATEGIES:
    d = distributions[c]
    q6.append(dict(
        strat=c, n_VL_tires=d["n"][SHORT.index("VL")],
        predits_VL_moy=d["nb_predits_VL"].mean(),
        predits_VL_p2_5=np.percentile(d["nb_predits_VL"], 2.5),
        predits_VL_p97_5=np.percentile(d["nb_predits_VL"], 97.5),
        dont_TP=d["nb_TP_VL"].mean(), dont_FP=d["nb_FP_VL"].mean(),
        pct_zero=(d["nb_predits_VL"] == 0).mean() * 100,
    ))
print(pd.DataFrame(q6).to_string(index=False, float_format=lambda v: f"{v:.2f}"))

print("\n>>> resultats ecrits dans scratch/h1_resultats.csv et h1_distributions.npz")
