"""Construit le SECOND jeu de SELECTION des variantes de prompt (1 835 lignes).

    python tools/build_selection_set_v2.py

POURQUOI 1 835 ET NON 200
-------------------------
Le jeu de 200 de la phase 3 n'avait que 5,3 % de chances de detecter un gain
d'exactitude de 0,02 (McNemar exact, Holm, famille de 3, discordance 6,75 %).
Les cinq non-rejets de la phase 3 BORNENT l'effet, ils ne l'annulent pas.
1 835 est le n qui porte cette puissance a 0,80 pour le meme ecart.

POURQUOI PAS DE PLANCHER, CONTRAIREMENT AUX 200
-----------------------------------------------
Le plancher de 12 par classe existait parce qu'a 200 lignes le prorata aurait
laisse les classes rares a 3 ou 4 items. A 1 835 lignes le prorata les pourvoit
seul. Un plancher deformerait la population sans rien acheter, et obligerait a
une ponderation en aval. L'allocation est donc STRICTEMENT PROPORTIONNELLE aux
effectifs du train restant.

ARRONDI : METHODE DES PLUS FORTS RESTES
---------------------------------------
`n_c = round(1835 x effectif_c / total)` ne totalise pas 1 835 en general. On
reutilise `data_prep.allocation_plancher(..., plancher=0)`, qui delegue a
`_repartition_plus_forts_restes` : parties entieres d'abord, puis les unites
restantes aux classes de plus fort reste fractionnaire. La somme vaut
EXACTEMENT n_total, et la regle est deterministe (tri stable `mergesort`).

TROIS EXCLUSIONS
----------------
  1. `train_iteration_20.csv`  -- erreurs deja lues et commentees ;
  2. `train_fewshot_examples.csv` -- exemples du prefixe de v6 ;
  3. `train_selection_200.csv` -- jeu de selection de la phase 3, deja mesure
     sur six variantes.
`test_sample_2000.csv` est disjoint par construction (train et test le sont),
mais la verification est refaite quand meme.

GRAINE DISTINCTE
----------------
`LLM_SELECTION2_SEED = 2027`, ni 1337 (les 200) ni 7 (few-shot v6) : reutiliser
une graine sur le meme corpus trie de la meme facon reselectionnerait
preferentiellement les memes lignes.
"""

import json
import sys
from pathlib import Path

import pandas as pd

RACINE = Path(__file__).resolve().parent
while not (RACINE / "src" / "config.py").exists() and RACINE != RACINE.parent:
    RACINE = RACINE.parent
sys.path.insert(0, str(RACINE))

from src import config as cfg      # noqa: E402
from src import data_prep as dp    # noqa: E402

# Seuil d'ALERTE, pas de blocage. ARBITRAGE TRANCHE le 2026-09-03 : PAS de
# plancher. 30 etait une garde arbitraire posee a la construction du script, pas
# un seuil statistique -- rien ne le derive d'une puissance ou d'une largeur
# d'intervalle. L'allocation reste STRICTEMENT PROPORTIONNELLE, et la classe la
# plus rare (Vehicle loan or lease) recoit 29 lignes.
#
# CONSEQUENCE, ecrite dans le meta.json : ce jeu ne permet PAS de rapporter un
# F1 PAR CLASSE fiable pour les classes rares. Ce n'est pas son role -- il sert
# a departager des variantes de prompt par un test APPARIE sur l'ensemble des
# lignes, ou seules comptent les paires discordantes.
MIN_PAR_CLASSE_ALERTE = 30


def main() -> int:
    if cfg.SAMPLE_FRACTION < 1.0:
        raise RuntimeError("SAMPLE_FRACTION < 1 : corpus de developpement.")

    print(f"lecture de {dp._fichiers_split()[0].name} ...")
    train = pd.read_csv(dp._fichiers_split()[0], encoding="utf-8")
    train[cfg.ID_COL] = train[cfg.ID_COL].astype(str)
    train = train.sort_values(cfg.ID_COL).reset_index(drop=True)
    print(f"  {len(train):,} lignes")

    # --- exclusions ---------------------------------------------------------
    exclus, detail = set(), {}
    for nom, chemin in [("train_iteration_20", cfg.LLM_ITERATION_FILE),
                        ("train_fewshot_examples", cfg.LLM_FEWSHOT_FILE),
                        ("train_selection_200", cfg.LLM_SELECTION_FILE)]:
        ids = set(pd.read_csv(chemin, encoding="utf-8")[cfg.ID_COL].astype(str))
        detail[nom] = len(ids)
        exclus |= ids
        print(f"  {len(ids):>5} ligne(s) exclues : {chemin.name}")
    print(f"  {len(exclus):>5} identifiant(s) exclus au total (union)")

    dispo = train[~train[cfg.ID_COL].isin(exclus)]
    print(f"  {len(dispo):,} lignes disponibles")

    # --- allocation PROPORTIONNELLE, plancher = 0 ---------------------------
    effectifs = dispo[cfg.LABEL_COL].value_counts()
    alloc = dp.allocation_plancher(effectifs,
                                   n_total=cfg.LLM_SELECTION2_SIZE,
                                   plancher=0)

    # Residu d'arrondi : ce que les plus forts restes ont ajoute a chaque classe
    # par rapport a la troncature de la part exacte. Documente, pas cache.
    exact = effectifs.reindex(cfg.CLASS_ORDER).astype(float) \
        / int(effectifs.sum()) * cfg.LLM_SELECTION2_SIZE
    residu = (alloc - exact.astype(int)).astype(int)

    print(f"\n{'=' * 78}")
    print(f"ALLOCATION PROPORTIONNELLE (plus forts restes) — n = "
          f"{cfg.LLM_SELECTION2_SIZE}")
    print(f"{'=' * 78}")
    print(f"  {'classe':36s} {'train dispo':>11} {'part':>7} {'exacte':>9} "
          f"{'alloue':>7} {'residu':>7}")
    for c in cfg.CLASS_ORDER:
        print(f"  {c:36s} {int(effectifs[c]):>11,} "
              f"{effectifs[c]/int(effectifs.sum()):>7.2%} {exact[c]:>9.2f} "
              f"{int(alloc[c]):>7} {int(residu[c]):>+7}")
    print(f"  {'TOTAL':36s} {int(effectifs.sum()):>11,} {1.0:>7.2%} "
          f"{exact.sum():>9.2f} {int(alloc.sum()):>7} {int(residu.sum()):>+7}")

    if int(alloc.sum()) != cfg.LLM_SELECTION2_SIZE:
        print(f"\n  ECHEC : la somme vaut {int(alloc.sum())} et non "
              f"{cfg.LLM_SELECTION2_SIZE}.")
        return 1

    # --- ALERTE (non bloquante) sur les classes peu pourvues ----------------
    maigres = {c: int(n) for c, n in alloc.items() if n < MIN_PAR_CLASSE_ALERTE}
    if maigres:
        print(f"\n  AVERTISSEMENT : {len(maigres)} classe(s) sous "
              f"{MIN_PAR_CLASSE_ALERTE} lignes :")
        for c, n in maigres.items():
            print(f"    {c:36s} {n}")
        print("  Accepte a dessein : allocation strictement proportionnelle,\n"
              "  sans plancher. Le F1 PAR CLASSE de ces classes n'est pas\n"
              "  exploitable sur ce jeu ; le test apparie, si.")

    # --- tirage, classe par classe dans l'ordre fige de CLASS_ORDER ---------
    sel = (pd.concat([dispo[dispo[cfg.LABEL_COL] == c].sample(
                          n=int(alloc[c]), random_state=cfg.LLM_SELECTION2_SEED)
                      for c in cfg.CLASS_ORDER if int(alloc[c]) > 0])
             .sort_values([cfg.LABEL_COL, cfg.ID_COL], kind="mergesort")
             .reset_index(drop=True))
    sel[cfg.OUTPUT_COLS].to_csv(cfg.LLM_SELECTION2_FILE, index=False,
                                encoding="utf-8")

    (cfg.LLM_SELECTION2_FILE.with_suffix(".meta.json")).write_text(json.dumps({
        "role": "second jeu de selection des variantes de prompt, etape 2 "
                "extension lot 1",
        "provenance": "TRAIN uniquement",
        "seed": cfg.LLM_SELECTION2_SEED,
        "n": int(len(sel)),
        "dimensionnement": (
            "n = 1835 : puissance 0,80 pour un gain d'exactitude de 0,02 sous "
            "McNemar exact, correction de Holm, famille de TROIS comparaisons, "
            "taux de discordance suppose 6,75 % (moyenne des b+c observes en "
            "phase 3 hors v3). A n = 200 la puissance valait 0,053."
        ),
        "allocation_regle": "proportionnelle au train restant, SANS plancher ; "
                            "arrondi par plus forts restes",
        "plancher": None,
        "plancher_arbitrage": (
            "PAS de plancher, tranche le 2026-09-03. Le seuil de 30 envisage "
            "etait une garde arbitraire, pas un seuil statistique. L'allocation "
            "reste strictement proportionnelle au train."
        ),
        "classe_la_plus_rare": {
            "classe": min(cfg.CLASS_ORDER, key=lambda c: int(alloc[c])),
            "n": int(min(int(alloc[c]) for c in cfg.CLASS_ORDER)),
        },
        "limite_f1_par_classe": (
            "La classe la plus rare recoit 29 lignes. Ce jeu ne permet donc PAS "
            "de rapporter un F1 PAR CLASSE fiable pour les classes rares : "
            "l'intervalle de confiance d'un F1 estime sur 29 items est trop "
            "large pour etre informatif. Ce n'est pas le role du jeu, qui sert a "
            "departager des variantes par un test APPARIE (McNemar) sur "
            "l'ensemble des lignes, ou seules comptent les paires discordantes."
        ),
        "allocation": {c: int(alloc[c]) for c in cfg.CLASS_ORDER},
        "residu_arrondi": {c: int(residu[c]) for c in cfg.CLASS_ORDER},
        "exclusions": {**detail,
                       "test_sample_2000.csv": "disjoint par construction"},
        "avertissement": (
            "Jeu PROPORTIONNEL et NON PONDERE : le F1-macro qu'on y calcule sert "
            "a CLASSER des variantes entre elles, pas a predire le F1 sur "
            "test_sample_2000.csv, qui est pondere par Horvitz-Thompson."
        ),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    # --- verifications ------------------------------------------------------
    print(f"\n{'=' * 78}\nJEU DE SELECTION : {cfg.LLM_SELECTION2_FILE.name}\n{'=' * 78}")
    rep = sel[cfg.LABEL_COL].value_counts().reindex(cfg.CLASS_ORDER)
    mots = sel[cfg.TEXT_COL].str.split().str.len()
    print(f"  longueur en mots : min {mots.min()}, mediane {int(mots.median())}, "
          f"max {mots.max()}")
    print(f"  textes > {cfg.LLM_MAX_WORDS} mots (tronques a l'appel) : "
          f"{int((mots > cfg.LLM_MAX_WORDS).sum())}")
    print(f"  masquage XXXX : {int(sel[cfg.TEXT_COL].str.contains('XXXX').sum())}"
          f"/{len(sel)}")

    ok = True
    df_eval, _ = dp.load_eval_sample()
    ids_sel = set(sel[cfg.ID_COL].astype(str))
    for nom, autres in [
            ("test_sample_2000", set(df_eval[cfg.ID_COL].astype(str))),
            ("train_iteration_20",
             set(pd.read_csv(cfg.LLM_ITERATION_FILE)[cfg.ID_COL].astype(str))),
            ("train_fewshot_examples",
             set(pd.read_csv(cfg.LLM_FEWSHOT_FILE)[cfg.ID_COL].astype(str))),
            ("train_selection_200",
             set(pd.read_csv(cfg.LLM_SELECTION_FILE)[cfg.ID_COL].astype(str)))]:
        fuite = ids_sel & autres
        if fuite:
            print(f"  ECHEC : {len(fuite)} identifiant(s) partage(s) avec {nom} "
                  f"-- exemple(s) : {sorted(fuite)[:5]}")
            ok = False
        else:
            print(f"  aucun recouvrement avec {nom}")
    if len(sel) != cfg.LLM_SELECTION2_SIZE:
        print(f"  ECHEC : {len(sel)} lignes"); ok = False
    if len(ids_sel) != len(sel):
        print(f"  ECHEC : identifiants dupliques"); ok = False
    if not rep.equals(alloc.reindex(cfg.CLASS_ORDER)):
        print(f"  ECHEC : le tire ne suit pas l'allocation"); ok = False
    else:
        print(f"  repartition conforme a l'allocation")

    print(f"\n  -> {cfg.LLM_SELECTION2_FILE}")
    print(f"  -> {cfg.LLM_SELECTION2_FILE.with_suffix('.meta.json')}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
