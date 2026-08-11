"""Produit les mesures chiffrees de reports/limites.md.

SOURCE DE VERITE du rapport : toutes les valeurs qui y figurent sortent de ce
script. Le relancer apres un changement de pipeline permet de verifier
qu'elles sont toujours exactes.

    python tools/limites_analyse.py

Deux volets independants :
  A. representativite du corpus (lignes AVEC texte vs SANS texte)
  B. bornes du plafond de performance (contradictions d'etiquetage)
"""

import re
import sys

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

# Racine du projet = le repertoire qui contient src/config.py. Aucun chemin en
# dur : ce script doit tourner depuis n'importe quel repertoire courant.
RACINE = Path(__file__).resolve().parent
while not (RACINE / "src" / "config.py").exists() and RACINE != RACINE.parent:
    RACINE = RACINE.parent
sys.path.insert(0, str(RACINE))
from src import config as cfg

PATH = cfg.RAW_FILE
pd.set_option("display.width", 240)
pd.set_option("display.max_columns", 40)


def titre(t, n=1):
    marque = "=" if n == 1 else "-"
    print("\n" + marque * 96)
    print(t)
    print(marque * 96)


# ===========================================================================
# VOLET A - REPRESENTATIVITE
# ===========================================================================
COLS_A = ["Submitted via", "Consumer consent provided?", "Tag", "State",
          "Company", "Date received", "Consumer Claim"]

df = pd.read_csv(PATH, usecols=COLS_A, encoding="utf-8", low_memory=False)
df["avec_texte"] = df["Consumer Claim"].notna()
df = df.drop(columns=["Consumer Claim"])          # on libere le gros de la RAM
df["annee"] = pd.to_datetime(df["Date received"], format="%m/%d/%Y").dt.year

N = len(df)
n_avec = int(df["avec_texte"].sum())
n_sans = N - n_avec
titre("A. REPRESENTATIVITE DU CORPUS")
print(f"population totale : {N:,}")
print(f"  AVEC texte : {n_avec:,}  ({n_avec/N*100:.2f} %)")
print(f"  SANS texte : {n_sans:,}  ({n_sans/N*100:.2f} %)")

# --- a) canal de depot ------------------------------------------------------
titre("a) Submitted via", 2)
ct = pd.crosstab(df["Submitted via"], df["avec_texte"])
ct.columns = ["sans_texte", "avec_texte"] if False in ct.columns else ct.columns
ct = ct.rename(columns={False: "sans_texte", True: "avec_texte"})
ct["total"] = ct.sum(axis=1)
ct["pct_du_flux"] = (ct.total / N * 100).round(2)
ct["pct_avec_texte"] = (ct.avec_texte / ct.total * 100).round(2)
ct = ct.sort_values("total", ascending=False)
print(ct.to_string())
hors_perimetre = ct.loc[ct.avec_texte == 0, "total"].sum()
print(f"\ncanaux ne produisant JAMAIS de narratif : "
      f"{list(ct.index[ct.avec_texte == 0])}")
print(f"part du flux structurellement hors perimetre : "
      f"{hors_perimetre:,} lignes = {hors_perimetre/N*100:.2f} %")
web = ct.loc["Web"]
print(f"canal Web : {web.total:,} lignes ({web.pct_du_flux} % du flux), "
      f"dont {web.pct_avec_texte} % avec narratif")

# --- b) distribution des Tag ------------------------------------------------
titre("b) Distribution des Tag (16 libelles bruts) avec / sans texte", 2)
ctag = pd.crosstab(df["Tag"], df["avec_texte"]).rename(
    columns={False: "sans_texte", True: "avec_texte"})
tab = pd.DataFrame({
    "n_avec": ctag["avec_texte"], "n_sans": ctag["sans_texte"],
    "pct_avec": (ctag["avec_texte"] / n_avec * 100).round(2),
    "pct_sans": (ctag["sans_texte"] / n_sans * 100).round(2),
})
tab["ecart_pp"] = (tab.pct_avec - tab.pct_sans).round(2)
tab["taux_narratif_pct"] = (ctag["avec_texte"] /
                            (ctag["avec_texte"] + ctag["sans_texte"]) * 100).round(2)
tab = tab.sort_values("ecart_pp", ascending=False)
print(tab.to_string())

chi2, p, ddl, _ = chi2_contingency(ctag.values)
v_cramer = np.sqrt(chi2 / (ctag.values.sum() * (min(ctag.shape) - 1)))
print(f"\nkhi-deux d'independance Tag x presence de texte :")
print(f"  chi2 = {chi2:,.0f}   ddl = {ddl}   p = {p:.3g}   V de Cramer = {v_cramer:.4f}")
print(f"  ecart absolu maximal : {tab.ecart_pp.abs().max():.2f} points de %")
print(f"  somme des ecarts positifs (= distance de variation totale) : "
      f"{tab.ecart_pp[tab.ecart_pp > 0].sum():.2f} points")

# --- c) consentement --------------------------------------------------------
titre("c) Consumer consent provided? x presence de texte", 2)
cons = df["Consumer consent provided?"].fillna("(non renseigne)")
cc = pd.crosstab(cons, df["avec_texte"]).rename(
    columns={False: "sans_texte", True: "avec_texte"})
cc["total"] = cc.sum(axis=1)
cc["pct_avec_texte"] = (cc.avec_texte / cc.total * 100).round(2)
print(cc.sort_values("total", ascending=False).to_string())
consent_sans_texte = int(cc.loc[cc.index.str.contains("Consent provided", case=False),
                                "sans_texte"].sum()) if any(
    cc.index.str.contains("Consent provided", case=False)) else 0
print(f"\nlignes AVEC texte mais consentement non renseigne : "
      f"{int(cc.loc['(non renseigne)', 'avec_texte']):,}")
print(f"lignes SANS texte et consentement explicitement refuse : "
      f"{int(cc.loc['Consent not provided', 'sans_texte']):,}"
      if "Consent not provided" in cc.index else "")

# --- d) evolution temporelle ------------------------------------------------
titre("d) Part de lignes avec narratif, par annee", 2)
an = df.groupby("annee")["avec_texte"].agg(["sum", "count"])
an.columns = ["avec_texte", "total"]
an["sans_texte"] = an.total - an.avec_texte
an["pct_avec_texte"] = (an.avec_texte / an.total * 100).round(2)
print(an.to_string())

# --- e) State et Company ----------------------------------------------------
titre("e) State - top 10 et ecarts", 2)


def compare_top(col, k=10):
    a = df.loc[df.avec_texte, col].value_counts(normalize=True) * 100
    s = df.loc[~df.avec_texte, col].value_counts(normalize=True) * 100
    top = (a.head(k).index.union(s.head(k).index, sort=False))
    t = pd.DataFrame({"pct_avec": a.reindex(top), "pct_sans": s.reindex(top)}).fillna(0)
    t["ecart_pp"] = (t.pct_avec - t.pct_sans).round(2)
    t["rang_avec"] = a.rank(ascending=False).reindex(top).astype("Int64")
    t["rang_sans"] = s.rank(ascending=False).reindex(top).astype("Int64")
    return t.sort_values("pct_avec", ascending=False).round(2)


print(compare_top("State").to_string())
titre("e) Company - top 10 et ecarts", 2)
print(compare_top("Company", 10).to_string())

del df

# ===========================================================================
# VOLET B - PLAFOND DE PERFORMANCE
# ===========================================================================
titre("B. BORNES DU PLAFOND DE PERFORMANCE")

d = pd.read_csv(PATH, usecols=[cfg.RAW_ID_COL, cfg.RAW_TEXT_COL, cfg.RAW_LABEL_COL],
                encoding="utf-8", low_memory=False)
d = d.rename(columns={cfg.RAW_ID_COL: "id", cfg.RAW_TEXT_COL: "texte",
                      cfg.RAW_LABEL_COL: "label"})
d = d[d.texte.notna() & d.label.notna()].sort_values("id", kind="mergesort")
d = d[~d.label.isin(cfg.EXCLUDED_LABELS)]
d["label"] = d.label.map(cfg.LABEL_MAPPING)
d["texte"] = d.texte.astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
d = d[d.texte.str.len() >= cfg.MIN_TEXT_LENGTH]
print(f"population de reference (avant vote majoritaire et dedoublonnage) : {len(d):,}")

PONCT = re.compile(r"[^a-z0-9 ]+")


def borne_inferieure(cle: pd.Series, nom: str, base: pd.DataFrame):
    """Erreurs necessaires : dans un groupe de textes identiques selon `cle`,
    un classifieur deterministe rend une seule etiquette ; toutes les lignes
    portant une autre etiquette sont fatalement fausses."""
    g = base.groupby(cle)["label"]
    taille = g.size()
    modal = g.agg(lambda s: s.value_counts().iloc[0])
    n_lab = g.nunique()

    conflit = n_lab > 1
    erreurs = int((taille[conflit] - modal[conflit]).sum())
    print(f"\n--- {nom} ---")
    print(f"  groupes distincts                : {len(taille):,}")
    print(f"  groupes de taille > 1            : {int((taille > 1).sum()):,}")
    print(f"  groupes a etiquettes CONTRADICTOIRES : {int(conflit.sum()):,}")
    print(f"  lignes dans ces groupes          : {int(taille[conflit].sum()):,}")
    print(f"  ERREURS NECESSAIRES              : {erreurs:,}")
    print(f"  borne inferieure du taux d'erreur : {erreurs/len(base)*100:.4f} %")
    print(f"  -> plafond d'accuracy correspondant : {100-erreurs/len(base)*100:.4f} %")
    return {"groupes_conflit": int(conflit.sum()), "lignes": int(taille[conflit].sum()),
            "erreurs": erreurs, "taux_pct": erreurs / len(base) * 100}

titre("a/b) Bornes inferieures dures, par definition de 'meme texte'", 2)
res = {}
res["exact"] = borne_inferieure(d.texte, "a) TEXTE EXACT (apres normalisation des espaces)", d)

norm = d.texte.str.lower()
res["casse"] = borne_inferieure(norm, "b1) insensible a la CASSE", d)

norm2 = PONCT.sub(" ", "").join([]) if False else norm.str.replace(PONCT, " ", regex=True)
norm2 = norm2.str.replace(r"\s+", " ", regex=True).str.strip()
res["ponctuation"] = borne_inferieure(norm2, "b2) casse + ponctuation retiree", d)

prefixe = norm2.str[:200]
res["prefixe200"] = borne_inferieure(prefixe, "b3) 200 premiers caracteres normalises", d)

titre("Synthese des bornes", 2)
synth = pd.DataFrame(res).T
synth["taux_pct"] = synth.taux_pct.round(4)
print(synth.to_string())

# --- c) courriers types FCRA ------------------------------------------------
titre("c) Courriers types citant la FCRA", 2)
FCRA = r"fair credit reporting act|15 u\.?s\.?c|section 609|section 611"
est_fcra = d.texte.str.lower().str.contains(FCRA, regex=True, na=False)
sub = d[est_fcra]
print(f"textes citant la FCRA / 15 USC / section 609-611 : {len(sub):,} "
      f"({len(sub)/len(d)*100:.2f} % du corpus)")
print("\nrepartition de leurs etiquettes :")
vc = sub.label.value_counts()
print(pd.DataFrame({"n": vc, "%": (vc / len(sub) * 100).round(2)}).to_string())

# dispersion par modele de lettre (groupes de prefixe identique)
pref_fcra = norm2[est_fcra].str[:200]
gf = sub.groupby(pref_fcra)["label"]
tailles = gf.size()
nlab = gf.nunique()
modeles = tailles[tailles >= 5]           # un "modele" = lettre vue >= 5 fois
print(f"\nmodeles de lettre (prefixe identique, vus >= 5 fois) : {len(modeles):,}")
print(f"  lignes couvertes : {int(modeles.sum()):,}")
disp = nlab.reindex(modeles.index).value_counts().sort_index()
print("\nnombre de categories differentes par modele de lettre :")
print(pd.DataFrame({"nb_modeles": disp,
                    "%": (disp / len(modeles) * 100).round(1)}).to_string())

multi = modeles.index[nlab.reindex(modeles.index) > 1]
if len(multi):
    print(f"\nmodeles classes dans PLUSIEURS categories : {len(multi):,} "
          f"({len(multi)/len(modeles)*100:.1f} %), "
          f"{int(tailles.reindex(multi).sum()):,} lignes")
    print("\n3 exemples (repartition des etiquettes) :")
    for p in tailles.reindex(multi).sort_values(ascending=False).head(3).index:
        rep = sub.loc[pref_fcra == p, "label"].value_counts()
        print(f"\n  n={int(rep.sum())} | " + " | ".join(f"{k}: {v}" for k, v in rep.items()))
        print(f'    "{p[:150]}..."')

titre("FIN")
