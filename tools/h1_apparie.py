"""
Phase 2bis (complement) - precision de l'ECART APPARIE entre deux approches.

Motivation
----------
Le F1-macro n'est pas estime pour lui-meme : il sert a decider si le LLM est
meilleur ou moins bon que le ML. Or les deux approches sont evaluees sur LE MEME
echantillon. L'alea de tirage est donc en grande partie COMMUN aux deux
estimations et s'annule dans la difference. La quantite qui gouverne reellement
la decision est donc RMSE(delta_estime) et non RMSE(F1_macro).

Hypothese de dependance
-----------------------
On suppose les erreurs des deux modeles conditionnellement independantes sachant
la classe vraie. C'est l'hypothese la PLUS DEFAVORABLE a l'annulation de
variance : deux modeles reels se trompent sur les memes textes ambigus, ce qui
correle leurs erreurs et reduit encore la variance de la difference. Les gains
mesures ici sont donc un plancher.
"""

import numpy as np
import pandas as pd

import h1_simulation as sim   # reutilise CLASSES, N_POP, construit_M, metriques, allocations

SEED, N_BOOT, K = 4242, 2000, sim.K
SHORT, N_POP = sim.SHORT, sim.N_POP

# --------------------------------------------------------------------------
# Deux profils d'erreur contrastes, dans le sens attendu par le projet :
#   - le ML classique exploite 283 516 exemples annotes -> fort sur les classes
#     frequentes, faible sur les classes rares (peu d'exemples) ;
#   - le LLM zero/few-shot n'a pas d'exemples mais comprend la semantique ->
#     profil plus homogene, meilleur sur les classes rares.
# C'est precisement la configuration ou le F1-MACRO discrimine les deux
# approches, et ou l'estimation des classes rares pese le plus.
# --------------------------------------------------------------------------
#                        CR    DC    MO    CC    BA    SL    MT    PD    VL
DIAG_ML = np.array([0.92, 0.70, 0.90, 0.75, 0.73, 0.87, 0.55, 0.40, 0.45])
DIAG_LLM = np.array([0.85, 0.62, 0.86, 0.70, 0.68, 0.84, 0.72, 0.62, 0.65])

M_ML, M_LLM = sim.construit_M(DIAG_ML), sim.construit_M(DIAG_LLM)


def comptages_pop(M):
    C = np.zeros((K, K), dtype=int)
    for i in range(K):
        exact = M[i] * N_POP[i]
        base = np.floor(exact).astype(int)
        r = N_POP[i] - base.sum()
        if r:
            base[np.argsort(-(exact - base))[:r]] += 1
        C[i] = base
    return C


f1m_ML = sim.metriques(comptages_pop(M_ML))[3]
f1m_LLM = sim.metriques(comptages_pop(M_LLM))[3]
DELTA_VRAI = float(f1m_LLM - f1m_ML)

print("=" * 100)
print("VALEURS VRAIES SUR LES 70 879 LIGNES")
print("=" * 100)
print(f"  F1-macro ML  = {f1m_ML:.5f}")
print(f"  F1-macro LLM = {f1m_LLM:.5f}")
print(f"  DELTA VRAI (LLM - ML) = {DELTA_VRAI:+.5f}")

# --------------------------------------------------------------------------
# Population appariee : pour chaque classe vraie i, comptages joints
# sur les 81 couples (predit_LLM, predit_ML).
# --------------------------------------------------------------------------
def loi_jointe(i, rho):
    """Loi jointe (pred_LLM, pred_ML) pour la classe vraie i.

    rho = 0 : independance conditionnelle sachant la classe vraie.
    rho > 0 : erreurs correlees via une variable latente de "difficulte".
        Un item est difficile avec proba h ; s'il ne l'est pas, LES DEUX
        modeles le classent correctement ; s'il l'est, chacun se trompe selon
        son propre taux conditionnel. Les marges M_LLM[i] et M_ML[i] sont
        preservees EXACTEMENT, seule la dependance change.
        h est calibre a (1 + rho) x la plus grosse des deux erreurs.
    """
    if rho <= 0:
        return np.outer(M_LLM[i], M_ML[i])
    eL, eM = 1 - M_LLM[i, i], 1 - M_ML[i, i]
    h = min(1.0, (1 + rho) * max(eL, eM))
    a, b = 1 - eL / h, 1 - eM / h          # P(correct | difficile) pour chaque modele
    qL, qM = M_LLM[i].copy(), M_ML[i].copy()
    qL[i] = qM[i] = 0
    qL, qM = qL / qL.sum(), qM / qM.sum()  # loi des fuites, hors diagonale
    pL, pM = (1 - a) * qL, (1 - b) * qM
    pL[i], pM[i] = a, b
    joint = h * np.outer(pL, pM)
    joint[i, i] += 1 - h
    return joint


def population_jointe(rho):
    J = np.zeros((K, K * K), dtype=int)
    for i in range(K):
        exact = loi_jointe(i, rho).ravel() * N_POP[i]
        base = np.floor(exact).astype(int)
        r = N_POP[i] - base.sum()
        if r:
            base[np.argsort(-(exact - base))[:r]] += 1
        J[i] = base
    assert (J.sum(axis=1) == N_POP).all()
    # verification des marges
    Jr = J.reshape(K, K, K)
    assert np.allclose(Jr.sum(axis=2) / N_POP[:, None], M_LLM, atol=2e-3)
    assert np.allclose(Jr.sum(axis=1) / N_POP[:, None], M_ML, atol=2e-3)
    return J

STRATS = {
    "A1": dict(label="Proportionnel n=1000",       n=sim.alloc_proportionnelle(1000), ht=False),
    "A2": dict(label="Proportionnel n=2000",       n=sim.alloc_proportionnelle(2000), ht=False),
    "B2": dict(label="Plancher 50 n=1000 + HT",    n=sim.alloc_plancher(1000),        ht=True),
    "B3": dict(label="Plancher 50 n=2000 + HT",    n=sim.alloc_plancher(2000),        ht=True),
    "D":  dict(label="Equilibre 111/classe + HT",  n=sim.alloc_equilibree(111),       ht=True),
}

RHOS = {0.0: "erreurs independantes", 0.6: "erreurs correlees (realiste)"}

lignes = []
for rho, nom_rho in RHOS.items():
    JOINT = population_jointe(rho)
    for code, cfg in STRATS.items():
        n_c, rng = cfg["n"], np.random.default_rng(SEED)
        J = np.empty((N_BOOT, K, K * K), dtype=np.int64)
        for i in range(K):
            J[:, i, :] = rng.multivariate_hypergeometric(JOINT[i], int(n_c[i]), size=N_BOOT)
        J = J.reshape(N_BOOT, K, K, K)      # (replicat, vrai, pred_LLM, pred_ML)

        C_llm = J.sum(axis=3).astype(float)  # marginalise sur le ML
        C_ml = J.sum(axis=2).astype(float)   # marginalise sur le LLM
        if cfg["ht"]:
            w = (N_POP / n_c).reshape(1, K, 1)
            C_llm, C_ml = C_llm * w, C_ml * w

        est_llm = sim.metriques(C_llm)[3]
        est_ml = sim.metriques(C_ml)[3]
        delta = est_llm - est_ml

        def st(e, v):
            return dict(biais=e.mean() - v, sd=e.std(ddof=1),
                        rmse=np.sqrt(np.mean((e - v) ** 2)))

        s_llm, s_ml, s_d = st(est_llm, f1m_LLM), st(est_ml, f1m_ML), st(delta, DELTA_VRAI)
        lignes.append(dict(
            correlation=nom_rho, strat=code, label=cfg["label"], n=int(n_c.sum()),
            rmse_LLM=s_llm["rmse"], rmse_ML=s_ml["rmse"],
            sd_delta=s_d["sd"], rmse_delta=s_d["rmse"], biais_delta=s_d["biais"],
            # facteur d'annulation : sd(delta) / sqrt(sd_LLM^2 + sd_ML^2)
            annulation=s_d["sd"] / np.sqrt(s_llm["sd"] ** 2 + s_ml["sd"] ** 2),
            # taux de conclusions correctes sur le SIGNE de l'ecart
            pct_signe_ok=(np.sign(delta) == np.sign(DELTA_VRAI)).mean() * 100,
            ic_bas=np.percentile(delta, 2.5), ic_haut=np.percentile(delta, 97.5),
        ))

df = pd.DataFrame(lignes)
for nom_rho in RHOS.values():
    print("\n" + "=" * 105)
    print(f"ECART APPARIE - {nom_rho.upper()} (delta vrai = {DELTA_VRAI:+.4f})")
    print("=" * 105)
    print(df[df.correlation == nom_rho].drop(columns="correlation")
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print("\nannulation < 1 => l'appariement reduit la variance de la difference")
df.to_csv(sim.OUT / "h1_apparie.csv", index=False)
print("\n>>> ecrit dans scratch/h1_apparie.csv")
