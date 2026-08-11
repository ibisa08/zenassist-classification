"""
Phase 2bis (complement) - decomposition du bruit sur F1(Vehicle loan or lease).

Question 6 demande si le denominateur de la precision (= nb d'items PREDITS VL,
que le plancher ne controle pas) est le facteur limitant. On separe donc la
variabilite du rappel (gouvernee par n_VL, controle par le plancher) de celle
de la precision (gouvernee par TP + FP, dont FP est incontrole).
"""

import numpy as np
import pandas as pd

import h1_simulation as sim

K, SHORT, N_POP, N_BOOT, SEED = sim.K, sim.SHORT, sim.N_POP, 2000, 42
iVL = SHORT.index("VL")
M = sim.construit_M(sim.DIAG_BASE)

K_pop = np.zeros((K, K), dtype=int)
for i in range(K):
    exact = M[i] * N_POP[i]
    base = np.floor(exact).astype(int)
    r = N_POP[i] - base.sum()
    if r:
        base[np.argsort(-(exact - base))[:r]] += 1
    K_pop[i] = base

prec_v, rapp_v, f1_v, _ = sim.metriques(K_pop)

lignes = []
for code, cfg in sim.STRATEGIES.items():
    n_c, rng = cfg["n"], np.random.default_rng(SEED)
    C = np.empty((N_BOOT, K, K), dtype=np.int64)
    for i in range(K):
        C[:, i, :] = rng.multivariate_hypergeometric(K_pop[i], int(n_c[i]), size=N_BOOT)
    Cu = C * (N_POP / n_c).reshape(1, K, 1) if cfg["ht"] else C.astype(float)
    prec, rapp, f1, _ = sim.metriques(Cu)

    tp, fp = C[:, iVL, iVL], C[:, :, iVL].sum(axis=1) - C[:, iVL, iVL]
    lignes.append(dict(
        strat=code, n_VL=int(n_c[iVL]),
        sd_rappel_VL=rapp[:, iVL].std(ddof=1),
        sd_precision_VL=prec[:, iVL].std(ddof=1),
        sd_f1_VL=f1[:, iVL].std(ddof=1),
        TP_moy=tp.mean(), FP_moy=fp.mean(), denom_moy=(tp + fp).mean(),
        cv_TP=tp.std(ddof=1) / tp.mean(), cv_FP=fp.std(ddof=1) / fp.mean(),
        # part de la variance de F1_VL attribuable a chaque composante
        # (approximation par la correlation empirique)
        corr_f1_rappel=np.corrcoef(f1[:, iVL], rapp[:, iVL])[0, 1],
        corr_f1_precision=np.corrcoef(f1[:, iVL], prec[:, iVL])[0, 1],
    ))

pd.set_option("display.width", 220)
print("=" * 110)
print("DECOMPOSITION DU BRUIT SUR F1(Vehicle loan or lease) - scenario realiste")
print(f"valeurs vraies : rappel={rapp_v[iVL]:.4f}  precision={prec_v[iVL]:.4f}  f1={f1_v[iVL]:.4f}")
print("=" * 110)
df = pd.DataFrame(lignes)
print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print("\nLecture : sd_rappel depend de n_VL (controle par le plancher) ;")
print("          sd_precision depend de TP+FP, dont FP n'est PAS controle par le plancher.")
df.to_csv(sim.OUT / "h1_decomposition.csv", index=False)
