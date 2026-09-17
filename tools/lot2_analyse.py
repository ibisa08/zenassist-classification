"""Mesure comparative du lot 2 : 4 variantes sur les memes 1 835 lignes.

    python tools/lot2_analyse.py

Produit `reports/etape2_lot2_resultats.md`. AUCUN appel API : tout est recalcule
sur les JSONL de `data/llm/lot2_*.jsonl`.

Le gate est celui PRE-ENREGISTRE dans `reports/protocole_lot2.md`, ecrit avant
les campagnes. Ce script ne le choisit pas, il l'applique.
"""

import json
import sys
from pathlib import Path

import pandas as pd

RACINE = Path(__file__).resolve().parent
while not (RACINE / "src" / "config.py").exists() and RACINE != RACINE.parent:
    RACINE = RACINE.parent
sys.path.insert(0, str(RACINE))

from src import config as cfg          # noqa: E402
from src import llm_eval as lle        # noqa: E402
from src import llm_prompts as lp      # noqa: E402
from src import metrics as mt          # noqa: E402

PROTOCOLE = cfg.REPORTS_DIR / "protocole_lot2.md"
SORTIE = cfg.REPORTS_DIR / "etape2_lot2_resultats.md"
N_BOOT, SEED = 10_000, 42

VARIANTES = [("v1_zeroshot", "lot2_v1.jsonl"), ("v6_fewshot", "lot2_v6.jsonl"),
             ("v7_fewshot_court", "lot2_v7.jsonl"),
             ("v8_fewshot_filtre", "lot2_v8.jsonl")]

# Famille PRE-ENREGISTREE. (nom, reference, variante)
FAMILLE = [("H1", "v1_zeroshot", "v7_fewshot_court"),
           ("H2", "v1_zeroshot", "v8_fewshot_filtre"),
           ("H3", "v7_fewshot_court", "v8_fewshot_filtre")]


def main() -> int:
    if not PROTOCOLE.exists():
        print(f"ARRET : {PROTOCOLE} absent."); return 1

    src = pd.read_csv(cfg.LLM_SELECTION2_FILE, encoding="utf-8")
    src[cfg.ID_COL] = src[cfg.ID_COL].astype(str)

    # --- chargement et evaluation ------------------------------------------
    res = {}
    for nom, fichier in VARIANTES:
        chemin = cfg.LLM_DIR / fichier
        if not chemin.exists():
            print(f"ARRET : {chemin} absent."); return 1
        res[nom] = lle.evalue_campagne(chemin, src, with_ci=False, verbeux=False)
        print(f"[charge] {nom:20s} n={res[nom]['n']:>5}  "
              f"F1={res[nom]['f1_macro']:.4f}")

    # --- MEMES LIGNES : bloquant -------------------------------------------
    try:
        lle._verifie_memes_lignes(res)
        print("\n[verif] les 4 campagnes portent EXACTEMENT les memes lignes.")
    except ValueError as e:
        print(f"\nARRET : {e}"); return 2
    n_lignes = len(res["v1_zeroshot"]["_ids_evalues"])
    if n_lignes != cfg.LLM_SELECTION2_SIZE:
        print(f"ARRET : {n_lignes} lignes evaluees au lieu de "
              f"{cfg.LLM_SELECTION2_SIZE}."); return 2

    # --- IC bootstrap 10 000 tirages, graine 42 ----------------------------
    for nom in res:
        v, bas, haut = mt.bootstrap_ci(
            res[nom]["_y_true"], res[nom]["_y_pred"], metric_fn=mt.f1_macro,
            n_boot=N_BOOT, strata=res[nom]["_y_true"], random_state=SEED)
        res[nom]["ci"] = (v, bas, haut)
        print(f"[IC] {nom:20s} {v:.4f} [{bas:.4f} ; {haut:.4f}]")

    # --- McNemar sur la famille pre-enregistree ----------------------------
    tests = {}
    for h, ref, var in FAMILLE:
        mn = lle.mcnemar(res[ref], res[var], ref, var, verbeux=False)
        ja = {i: (v == p) for i, v, p in zip(res[ref]["_ids_evalues"],
                                             res[ref]["_y_true"], res[ref]["_y_pred"])}
        jb = {i: (v == p) for i, v, p in zip(res[var]["_ids_evalues"],
                                             res[var]["_y_true"], res[var]["_y_pred"])}
        mn["accord_justes"] = sum(1 for i in ja if ja[i] and jb[i])
        mn["accord_fautes"] = sum(1 for i in ja if not ja[i] and not jb[i])
        mn["reference"], mn["variante"] = ref, var
        tests[h] = mn
        print(f"[McNemar] {h} {var} vs {ref} : b={mn['b']} c={mn['c']} "
              f"p={mn['p_valeur']:.5f}")

    holm = lle.holm_bonferroni({h: tests[h]["p_valeur"] for h in tests},
                               alpha=0.05, verbeux=False)
    for r in holm["resultats"]:
        t = tests[r["nom"]]
        # Critere PRE-ENREGISTRE : p <= seuil ET c > b (la variante corrige
        # plus de lignes qu'elle n'en casse). Cf. protocole_lot2.md section 3.
        favorable = t["c"] > t["b"]
        t["rang"], t["seuil"] = r["rang"], r["seuil_holm"]
        t["sous_seuil"] = r["rejetee"]
        t["decision"] = ("rejetee EN FAVEUR de la variante" if r["rejetee"] and favorable
                         else "rejetee EN DEFAVEUR de la variante" if r["rejetee"]
                         else r["motif"])

    _ecrit_rapport(res, tests, holm, n_lignes)
    print(f"\n-> {SORTIE}")
    return 0


def _ecrit_rapport(res, tests, holm, n_lignes):
    L = list(cfg.CLASS_ORDER)
    noms = [n for n, _ in VARIANTES]

    def pref(nom):
        p = lp.prefixe_fige(nom, lp.labels_du_style(nom))
        return len(p.split())

    o = []
    o.append("# Étape 2, extension — lot 2 : résultats bruts\n")
    o.append(f"Mesure du 2026-09-10 sur les **{n_lignes:,} lignes** de "
             f"`data/processed/train_selection_1835.csv`".replace(",", " ") +
             ", évaluées par les quatre variantes.\n")
    o.append("Aucune interprétation, aucun choix de variante. Les chiffres, "
             "et le gate tel qu'il a été pré-enregistré.\n")

    # --- rappel du pre-enregistrement --------------------------------------
    o.append("## 1. Pré-enregistrement rappelé\n")
    o.append(f"Écrit dans [`reports/protocole_lot2.md`](protocole_lot2.md) "
             f"**avant** toute campagne.\n")
    o.append("| hypothèse | référence | variante |")
    o.append("|---|---|---|")
    for h, ref, var in FAMILLE:
        o.append(f"| **{h}** | `{ref}` | `{var}` |")
    o.append("")
    o.append("Gate : McNemar exact bilatéral, correction de Holm-Bonferroni, "
             "FWER 5 %, famille de **m = 3**. Seuils par rang : "
             "0,0166667 / 0,0250000 / 0,0500000. Procédure arrêtée au premier "
             "échec.\n")
    o.append("Critère de rejet en faveur de la variante : "
             "`p_brut <= seuil_Holm` **et** `c > b`, avec `b` = référence juste "
             "/ variante fausse (lignes **cassées**) et `c` = référence fausse / "
             "variante juste (lignes **corrigées**), convention de "
             "`src/llm_eval.py:1010-1011`.\n")
    o.append("`v6_fewshot` est mesurée pour référence descriptive et "
             "**n'appartient pas à la famille** : l'y ajouter aurait durci les "
             "seuils des trois hypothèses testées.\n")

    # --- tableau des 4 variantes -------------------------------------------
    o.append("## 2. Les quatre variantes\n")
    o.append("| | " + " | ".join(f"`{n}`" for n in noms) + " |")
    o.append("|---|" + "---|" * len(noms))
    def ligne(lib, f):
        o.append(f"| {lib} | " + " | ".join(f(res[n]) for n in noms) + " |")
    o.append(f"| préfixe (mots) | " + " | ".join(f"{pref(n):,}".replace(",", " ")
                                                 for n in noms) + " |")
    ligne("lignes évaluées", lambda r: f"{r['n']:,}".replace(",", " "))
    ligne("**F1-macro**", lambda r: f"**{r['f1_macro']:.4f}**")
    ligne("IC 95 % (bootstrap)", lambda r: f"[{r['ci'][1]:.4f} ; {r['ci'][2]:.4f}]")
    ligne("exactitude", lambda r: f"{r['accuracy']:.4f}")
    ligne("F1 pondéré", lambda r: f"{r['f1_weighted']:.4f}")
    ligne("parse_error (n)", lambda r: f"{r['n_parse_error']}")
    ligne("parse_error (taux)", lambda r: f"{r['taux_parse_error']:.2%}")
    ligne("`label_inconnu`", lambda r: f"{r['statuts_parsing'].get('label_inconnu', 0)}")
    ligne("`json_invalide`", lambda r: f"{r['statuts_parsing'].get('json_invalide', 0)}")
    ligne("`vide`", lambda r: f"{r['statuts_parsing'].get('vide', 0)}")
    ligne("erreurs HTTP", lambda r: f"{r['n_erreurs_http']}")
    ligne("tokens_in moyens", lambda r: f"{r['tokens_in_moyen']:,.1f}".replace(",", " "))
    ligne("tokens_caches moyens",
          lambda r: f"{r['tokens_caches_total'] / r['n_appels']:,.1f}".replace(",", " "))
    ligne("coût total", lambda r: f"{r['cout_reel_usd']:.4f} $")
    ligne("latence p50 (s)", lambda r: f"{r['latence_p50_s']:.3f}")
    ligne("latence p95 (s)", lambda r: f"{r['latence_p95_s']:.3f}")
    o.append("")
    o.append(f"IC 95 % : bootstrap par percentiles, "
             "**" + f"{N_BOOT:,}".replace(",", " ") + " tirages**, "
             f"graine **{SEED}**, rééchantillonnage stratifié sur la classe "
             f"vraie (`src/metrics.py:299`).\n")

    # --- F1 par classe -----------------------------------------------------
    o.append("## 3. F1 par classe\n")
    o.append("Descriptif. Le jeu est alloué **proportionnellement au train et "
             "sans plancher** : la classe la plus rare n'y compte que 29 lignes, "
             "et son F1 n'est pas exploitable.\n")
    o.append("| classe | n | " + " | ".join(f"`{n}`" for n in noms) + " |")
    o.append("|---|---:|" + "---|" * len(noms))
    for c in L:
        n_c = int(res[noms[0]]["classification_report"][c]["support"])
        cells = " | ".join(f"{res[n]['classification_report'][c]['f1-score']:.4f}"
                           for n in noms)
        o.append(f"| {c} | {n_c} | {cells} |")
    o.append("")

    # --- McNemar -----------------------------------------------------------
    o.append("## 4. Les trois comparaisons appariées\n")
    o.append("| | H1 — `v7` vs `v1` | H2 — `v8` vs `v1` | H3 — `v8` vs `v7` |")
    o.append("|---|---|---|---|")
    def lm(lib, f):
        o.append(f"| {lib} | " + " | ".join(f(tests[h]) for h in ("H1","H2","H3")) + " |")
    lm("accord — justes des deux", lambda t: f"{t['accord_justes']:,}".replace(",", " "))
    lm("accord — fautes des deux", lambda t: f"{t['accord_fautes']:,}".replace(",", " "))
    lm("**b** (variante casse)", lambda t: f"**{t['b']}**")
    lm("**c** (variante corrige)", lambda t: f"**{t['c']}**")
    lm("paires discordantes", lambda t: f"{t['n_discordantes']}")
    lm("bilan net (c − b)", lambda t: f"{t['c'] - t['b']:+d}")
    lm("p brut (exact bilatéral)", lambda t: f"{t['p_valeur']:.5f}")
    lm("rang Holm", lambda t: f"{t['rang']}")
    lm("seuil du rang", lambda t: f"{t['seuil']:.7f}")
    lm("p ≤ seuil ?", lambda t: "oui" if t["sous_seuil"] else "non")
    lm("c > b ?", lambda t: "oui" if t["c"] > t["b"] else "non")
    lm("**décision**", lambda t: f"**{t['decision']}**")
    o.append("")
    o.append(f"Hypothèses rejetées par la procédure de Holm : "
             f"**{', '.join(holm['rejetees']) if holm['rejetees'] else 'aucune'}**.\n")

    o.append("## 5. Provenance des chiffres\n")
    o.append("| élément | source |")
    o.append("|---|---|")
    o.append("| journaux d'appels | `data/llm/lot2_v{1,6,7,8}.jsonl` |")
    o.append("| jeu évalué | `data/processed/train_selection_1835.csv` |")
    o.append("| F1-macro, exactitude, F1 par classe | `src/metrics.py:245-260` |")
    o.append("| IC bootstrap | `src/metrics.py:299` |")
    o.append("| b, c | `src/llm_eval.py:1010-1011` |")
    o.append("| p exact bilatéral | `src/llm_eval.py:1018-1019` |")
    o.append("| seuils de Holm | `src/llm_eval.py:1215` |")
    o.append("| vérification « mêmes lignes » | `src/llm_eval.py:476` |")
    o.append("| conformité de format | `src/llm_eval.py:217` |")
    o.append("| ce script | `tools/lot2_analyse.py` |")
    o.append("")
    SORTIE.write_text("\n".join(o), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
