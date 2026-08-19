"""Construit le jeu de SELECTION des variantes de prompt (phase 3).

    python tools/build_selection_set.py

POURQUOI UN SECOND JEU, PLUS GRAND
----------------------------------
Le jeu d'iteration de 20 lignes ne peut pas departager des variantes : l'IC a
95 % du F1-macro y fait +/- 20 points (simulation du 2026-08-19). A 200 lignes
il tombe a +/- 6,3 points. Les 20 restent le support de la lecture QUALITATIVE
des erreurs, ou lire 20 textes a la main a du sens ; le chiffre vient des 200.

TROIS EXIGENCES DE NON-RECOUVREMENT
-----------------------------------
  1. hors `test_sample_2000.csv` -- garanti par construction (train et test sont
     disjoints), verifie quand meme ;
  2. hors `train_iteration_20.csv` -- sinon les erreurs deja lues et
     commentees pesertaient dans le chiffre de selection ;
  3. les exemples du few-shot (v6) sont hors des deux -- sinon v6 repondrait
     sur des lignes qu'il a vues, et son gain serait un artefact.

GRAINE DISTINCTE
----------------
`LLM_SELECTION_SEED = 1337`, et non `RANDOM_SEED = 42`. Reutiliser la meme
graine sur le meme corpus trie de la meme facon reselectionnerait
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


def main() -> int:
    if cfg.SAMPLE_FRACTION < 1.0:
        raise RuntimeError("SAMPLE_FRACTION < 1 : corpus de developpement.")

    print(f"lecture de {dp._fichiers_split()[0].name} ...")
    train = pd.read_csv(dp._fichiers_split()[0], encoding="utf-8")
    train[cfg.ID_COL] = train[cfg.ID_COL].astype(str)
    train = train.sort_values(cfg.ID_COL).reset_index(drop=True)
    print(f"  {len(train):,} lignes")

    iteration = pd.read_csv(cfg.LLM_ITERATION_FILE, encoding="utf-8")
    deja = set(iteration[cfg.ID_COL].astype(str))
    print(f"  {len(deja)} ligne(s) du jeu d'iteration exclues")

    dispo = train[~train[cfg.ID_COL].isin(deja)]

    # --- few-shot d'abord, pour l'exclure ensuite du jeu de selection -------
    # Construction explicite classe par classe : meme forme que le jeu
    # d'iteration, et insensible aux evolutions de `groupby.apply`.
    fs = (pd.concat([dispo[dispo[cfg.LABEL_COL] == c].sample(
                        n=cfg.LLM_FEWSHOT_PAR_CLASSE,
                        random_state=cfg.LLM_FEWSHOT_SEED)
                     for c in cfg.CLASS_ORDER])
            .sort_values(cfg.LABEL_COL).reset_index(drop=True))
    fs[cfg.OUTPUT_COLS].to_csv(cfg.LLM_FEWSHOT_FILE, index=False, encoding="utf-8")
    print(f"  {len(fs)} exemple(s) few-shot mis de cote (1 par classe)")

    dispo = dispo[~dispo[cfg.ID_COL].isin(set(fs[cfg.ID_COL]))]

    # --- jeu de selection ---------------------------------------------------
    alloc = dp.allocation_plancher(dispo[cfg.LABEL_COL].value_counts(),
                                   n_total=cfg.LLM_SELECTION_SIZE,
                                   plancher=cfg.LLM_SELECTION_MIN_PER_CLASS)
    sel = (pd.concat([dispo[dispo[cfg.LABEL_COL] == c].sample(
                          n=int(n), random_state=cfg.LLM_SELECTION_SEED)
                      for c, n in alloc.items() if n > 0])
             .sort_values([cfg.LABEL_COL, cfg.ID_COL]).reset_index(drop=True))
    sel[cfg.OUTPUT_COLS].to_csv(cfg.LLM_SELECTION_FILE, index=False,
                                encoding="utf-8")

    (cfg.LLM_SELECTION_FILE.with_suffix(".meta.json")).write_text(json.dumps({
        "role": "jeu de selection des variantes de prompt, etape 2 phase 3",
        "provenance": "TRAIN uniquement",
        "seed": cfg.LLM_SELECTION_SEED,
        "n": int(len(sel)),
        "plancher_par_classe": cfg.LLM_SELECTION_MIN_PER_CLASS,
        "allocation": {k: int(v) for k, v in alloc.items()},
        "exclusions": ["train_iteration_20.csv", "train_fewshot_examples.csv",
                       "test_sample_2000.csv (disjoint par construction)"],
        "avertissement": (
            "Jeu STRATIFIE A PLANCHER et NON PONDERE : le F1-macro qu'on y "
            "calcule sert a CLASSER des variantes entre elles, pas a predire le "
            "F1 sur test_sample_2000.csv, qui est pondere par Horvitz-Thompson."
        ),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    # --- verifications ------------------------------------------------------
    print(f"\n{'='*70}\nJEU DE SELECTION : {cfg.LLM_SELECTION_FILE.name}\n{'='*70}")
    rep = sel[cfg.LABEL_COL].value_counts().reindex(cfg.CLASS_ORDER)
    for c, n in rep.items():
        print(f"    {c:36s} {int(n)}")
    mots = sel[cfg.TEXT_COL].str.split().str.len()
    print(f"\n  longueur en mots : min {mots.min()}, mediane {int(mots.median())}, "
          f"max {mots.max()}")
    print(f"  textes > {cfg.LLM_MAX_WORDS} mots (tronques) : "
          f"{int((mots > cfg.LLM_MAX_WORDS).sum())}")
    print(f"  masquage XXXX : {int(sel[cfg.TEXT_COL].str.contains('XXXX').sum())}"
          f"/{len(sel)}")

    ok = True
    df_eval, _ = dp.load_eval_sample()
    for nom, autres in [("test_sample_2000", set(df_eval[cfg.ID_COL].astype(str))),
                        ("train_iteration_20", deja),
                        ("few-shot", set(fs[cfg.ID_COL]))]:
        fuite = set(sel[cfg.ID_COL].astype(str)) & autres
        if fuite:
            print(f"  ECHEC : {len(fuite)} ligne(s) partagee(s) avec {nom}"); ok = False
        else:
            print(f"  aucun recouvrement avec {nom}")
    if len(sel) != cfg.LLM_SELECTION_SIZE:
        print(f"  ECHEC : {len(sel)} lignes"); ok = False
    if int(rep.min()) < cfg.LLM_SELECTION_MIN_PER_CLASS:
        print(f"  ECHEC : plancher non respecte"); ok = False

    print(f"\n  -> {cfg.LLM_SELECTION_FILE}")
    print(f"  -> {cfg.LLM_FEWSHOT_FILE}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
