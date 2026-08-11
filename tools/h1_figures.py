"""Phase 2bis - graphes du rapport H.1. Ecrit dans reports/figures/."""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# Racine du projet = le repertoire qui contient src/config.py. Aucun chemin en
# dur : ce script doit tourner depuis n'importe quel repertoire courant.
RACINE = Path(__file__).resolve().parent
while not (RACINE / "src" / "config.py").exists() and RACINE != RACINE.parent:
    RACINE = RACINE.parent
SCR = RACINE / "scratch"
FIG = RACINE / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# --- palette validee (validate_palette.js, mode light : ALL CHECKS PASS) ------
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXE = "#e1e0d9", "#c3c2b7"
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": AXE, "axes.linewidth": 0.8,
    "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.titlesize": 10, "axes.labelsize": 9,
    "grid.color": GRID, "grid.linewidth": 0.8,
    "savefig.facecolor": SURFACE, "savefig.bbox": "tight", "savefig.dpi": 160,
})


def epure(ax, axe_x=True):
    """Chrome recessif : pas de cadre, grille discrete sur un seul axe."""
    for s in ("top", "right", "left" if axe_x else "bottom"):
        ax.spines[s].set_visible(False)
    ax.grid(True, axis="x" if axe_x else "y", zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


res = pd.read_csv(SCR / "h1_resultats.csv")
dec = pd.read_csv(SCR / "h1_decomposition.csv")
npz = np.load(SCR / "h1_distributions.npz")

ORDRE = ["A1", "A2", "B1", "B2", "B3", "D"]
TITRES = {
    "A1": "A1 · proportionnel n=1000", "A2": "A2 · proportionnel n=2000",
    "B1": "B1 · plancher 50, n=1000, SANS repond.", "B2": "B2 · plancher 50, n=1000, HT",
    "B3": "B3 · plancher 50, n=2000, HT", "D": "D · equilibre 111/classe, HT",
}
VRAI = float(res.loc[res.scenario == "realiste", "f1m_vrai"].iloc[0])

# ============================================================================
# FIGURE 1 - distributions echantillonnales du F1-macro (petits multiples)
# ============================================================================
fig, axes = plt.subplots(2, 3, figsize=(13, 6.4), sharex=True, sharey=True)
bins = np.linspace(0.655, 0.795, 61)

for ax, code in zip(axes.ravel(), ORDRE):
    est = npz[f"{code}__f1_macro"]
    r = res[(res.scenario == "realiste") & (res.strategie == code)].iloc[0]
    biaise = code == "B1"
    ax.hist(est, bins=bins, color=S2 if biaise else S1, alpha=0.85, zorder=2)
    ax.axvline(VRAI, color=INK, lw=2, zorder=4)
    ax.axvline(est.mean(), color=INK, lw=1.6, ls=(0, (4, 3)), zorder=4)
    epure(ax)
    ax.set_title(TITRES[code], color=INK, loc="left", pad=8,
                 fontweight="bold" if code in ("A2", "B3") else "normal")
    ax.text(0.03, 0.93, f"RMSE {r.f1m_rmse*1000:.1f}e-3\nbiais {r.f1m_biais*1000:+.1f}e-3",
            transform=ax.transAxes, va="top", ha="left", fontsize=9,
            color="#b04a1c" if biaise else INK2,
            fontweight="bold" if biaise else "normal")
    ax.set_yticks([])

axes[0, 0].annotate("F1-macro vrai\n= 0.7247", xy=(VRAI, 150), xytext=(0.688, 205),
                    fontsize=8.5, color=INK2, ha="right",
                    arrowprops=dict(arrowstyle="-", color=AXE, lw=1))
for ax in axes[1]:
    ax.set_xlabel("F1-macro estime")
fig.suptitle("Distribution du F1-macro estime sur 2 000 echantillons independants",
             x=0.008, ha="left", fontsize=13, fontweight="bold", color=INK, y=1.015)
fig.text(0.008, 0.975,
         "Trait plein = valeur vraie sur les 70 879 lignes du test  ·  trait pointille = moyenne des estimations  ·  "
         "scenario realiste",
         ha="left", fontsize=9.5, color=INK2)
fig.tight_layout(rect=[0, 0, 1, 0.955])
fig.savefig(FIG / "h1_distributions_f1macro.png")
plt.close(fig)

# ============================================================================
# FIGURE 2 - RMSE du F1-macro, par strategie et par scenario (decision)
# ============================================================================
SCEN = [("realiste", "Realiste", S1), ("optimiste", "Optimiste (+0.10)", S2),
        ("pessimiste", "Pessimiste (-0.10)", S3), ("rares_degradees", "Classes rares degradees", S4)]

fig, ax = plt.subplots(figsize=(11, 5.4))
y = np.arange(len(ORDRE))[::-1]
h = 0.19
for k, (sc, lab, col) in enumerate(SCEN):
    vals = [res[(res.scenario == sc) & (res.strategie == c)].f1m_rmse.iloc[0] * 1000 for c in ORDRE]
    pos = y + (1.5 - k) * h
    ax.barh(pos, vals, height=h - 0.035, color=col, label=lab, zorder=3)
    for p, v in zip(pos, vals):                       # relief : labels visibles
        ax.text(v + 0.35, p, f"{v:.1f}", va="center", ha="left", fontsize=8.5,
                color=INK2, fontweight="medium")

ax.set_yticks(y, [TITRES[c] for c in ORDRE], color=INK, fontsize=9.5)
ax.set_xlabel("RMSE du F1-macro  (x1000 — plus bas = meilleur)")
ax.set_xlim(0, 37)
epure(ax)
ax.legend(frameon=False, ncols=4, loc="lower center", bbox_to_anchor=(0.5, -0.22),
          fontsize=9.5, labelcolor=INK2, handlelength=1.1, handleheight=1.1)
ax.axhline(y[ORDRE.index("A2")] - 0.5, color=AXE, lw=0.8, ls=":", zorder=1)
fig.suptitle("Critere de decision : RMSE du F1-macro, quatre scenarios de confusion",
             x=0.008, ha="left", fontsize=13, fontweight="bold", color=INK, y=1.03)
fig.text(0.008, 0.965,
         "B3 domine A2 dans les quatre scenarios, a cout identique (n=2000). B1, non pondere, est le seul a se degrader quand le modele empire.",
         ha="left", fontsize=9.5, color=INK2)
fig.tight_layout(rect=[0, 0, 1, 0.945])
fig.savefig(FIG / "h1_rmse_scenarios.png")
plt.close(fig)

# ============================================================================
# FIGURE 3 - le mecanisme : biais de precision (a) et bruit sur F1(VL) (b)
# ============================================================================
SHORT = ["CR", "DC", "MO", "CC", "BA", "SL", "MT", "PD", "VL"]
N_POP = np.array([21550, 16806, 10585, 8280, 5541, 4354, 1393, 1229, 1141])

fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.5, 5.2),
                               gridspec_kw={"width_ratios": [1.25, 1]})

# --- (a) biais sur la precision par classe : B1 (non pondere) vs B3 (HT)
biais_B1 = npz["B1__precision_moyenne"] - np.array(
    [0.7347, 0.9154, 0.8756, 0.7035, 0.7689, 0.8419, 0.6716, 0.6740, 0.4335])
biais_B3 = npz["B3__precision_moyenne"] - np.array(
    [0.7347, 0.9154, 0.8756, 0.7035, 0.7689, 0.8419, 0.6716, 0.6740, 0.4335])
x = np.arange(len(SHORT))
w = 0.38
axA.axhline(0, color=AXE, lw=1.2, zorder=2)
axA.bar(x - w / 2, biais_B1, w - 0.04, color=S2, label="B1 · plancher SANS ponderation", zorder=3)
axA.bar(x + w / 2, biais_B3, w - 0.04, color=S1, label="B3 · plancher AVEC ponderation HT", zorder=3)
for xi, v in zip(x - w / 2, biais_B1):
    axA.text(xi, v + (0.012 if v >= 0 else -0.012), f"{v:+.2f}", ha="center",
             va="bottom" if v >= 0 else "top", fontsize=8, color=INK2)
axA.set_xticks(x, SHORT, color=INK)
axA.set_ylabel("Biais sur la precision par classe")
axA.set_ylim(-0.145, 0.36)
epure(axA, axe_x=False)
axA.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper left",
           handlelength=1.1, handleheight=1.1)
axA.set_title("(a)  Sans ponderation, le biais suit le taux de sondage relatif",
              loc="left", color=INK, fontweight="bold", pad=10)
axA.text(0.62, -0.155, "MT, PD, VL : classes sur-echantillonnees par le plancher",
         transform=axA.transAxes, fontsize=8.5, color=MUTED, ha="left")

# --- (b) bruit sur F1(VL) : rappel (controle) vs precision (non controle)
# B1 est exclu : son estimateur etant biaise, son ecart-type n'est pas
# comparable a celui des estimateurs sans biais.
ORDRE_B = ["A1", "A2", "B2", "B3", "D"]
d = dec.set_index("strat").loc[ORDRE_B]
xb = np.arange(len(ORDRE_B))
axB.plot(xb, d.sd_rappel_VL, marker="o", ms=8, lw=2, color=S1,
         label="rappel(VL) — pilote par n(VL)", zorder=4)
axB.plot(xb, d.sd_precision_VL, marker="s", ms=8, lw=2, color=S2,
         label="precision(VL) — pilotee par les faux positifs", zorder=4)
axB.plot(xb, d.sd_f1_VL, marker="D", ms=8, lw=2.4, color=INK,
         label="F1(VL)", zorder=5)
for xi, v in zip(xb, d.sd_f1_VL):
    axB.text(xi, v - 0.006, f"{v:.3f}", ha="center", va="top", fontsize=8.5,
             color=INK2, zorder=6)
axB.set_xticks(xb, [f"{c}\nn(VL)={int(n)}" for c, n in zip(ORDRE_B, d.n_VL)],
               color=INK, fontsize=9)
axB.set_ylabel("Ecart-type sur 2 000 echantillons")
axB.set_ylim(0, 0.16)
axB.set_xlim(-0.45, 4.75)
epure(axB, axe_x=False)
axB.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper left",
           handlelength=1.4, handleheight=1.1, borderaxespad=0.2)
axB.set_title("(b)  Au-dela de B3, augmenter n(VL) degrade le F1(VL)",
              loc="left", color=INK, fontweight="bold", pad=10)
axB.annotate("D : n(VL)=111, mais les classes\nqui produisent les faux positifs\nsont trop peu tirees",
             xy=(4, d.sd_precision_VL.iloc[-1] + 0.002), xytext=(2.35, 0.145),
             fontsize=8.5, color="#b04a1c", ha="left", va="top",
             arrowprops=dict(arrowstyle="->", color="#b04a1c", lw=1.2,
                             connectionstyle="arc3,rad=-0.15"))

fig.suptitle("Mecanisme : ce que le plancher controle, et ce qu'il ne controle pas",
             x=0.008, ha="left", fontsize=13, fontweight="bold", color=INK, y=1.045)
fig.text(0.008, 0.975,
         "Le plancher fixe n(VL) — donc le rappel. Il ne controle pas les faux positifs venus des autres classes, "
         "qui gouvernent la precision.",
         ha="left", fontsize=9.5, color=INK2)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(FIG / "h1_mecanisme.png")
plt.close(fig)

print("figures ecrites :")
for p in sorted(FIG.glob("h1_*.png")):
    print("  ", p.name, f"{p.stat().st_size/1024:.0f} Ko")
