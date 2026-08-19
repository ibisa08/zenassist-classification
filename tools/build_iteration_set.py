"""Construit le jeu d'iteration de la phase 2 : 20 lignes du TRAIN.

    python tools/build_iteration_set.py

POURQUOI LE TRAIN, ET JAMAIS test_sample_2000.csv
-------------------------------------------------
Regler un prompt en regardant ses erreurs sur l'echantillon d'evaluation revient
a l'optimiser pour ces lignes precises. Le score final serait gonfle et ne
mesurerait plus rien -- meme logique que le dedoublonnage avant le split.

CE JEU EST FIGE POUR TOUTE L'ETAPE 2
------------------------------------
C'est ce qui rend les variantes de prompt comparables entre elles : si le jeu
changeait d'une variante a l'autre, on ne saurait pas si l'ecart vient du prompt
ou des lignes. Le tirage est donc DETERMINISTE (seed 42) et le fichier n'est
reecrit que si on le supprime explicitement.

ALLOCATION
----------
On reutilise `data_prep.allocation_plancher()`, la fonction deja validee a
l'etape 1, avec un plancher de 2 au lieu de 50. Reecrire une allocation ici
introduirait une seconde logique de tirage a maintenir, et rien ne garantirait
qu'elle se comporte comme celle de l'echantillon d'evaluation.
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


def construit(force: bool = False) -> pd.DataFrame:
    if cfg.SAMPLE_FRACTION < 1.0:
        raise RuntimeError(
            f"config.SAMPLE_FRACTION vaut {cfg.SAMPLE_FRACTION} : le train "
            f"disponible est un echantillon de DEVELOPPEMENT. Le jeu "
            f"d'iteration doit venir du corpus complet."
        )
    if cfg.LLM_ITERATION_FILE.exists() and not force:
        print(f"{cfg.LLM_ITERATION_FILE.name} existe deja -- il est FIGE pour "
              f"toute l'etape 2, on ne le retire pas.")
        return pd.read_csv(cfg.LLM_ITERATION_FILE, encoding="utf-8")

    chemin_train = dp._fichiers_split()[0]
    print(f"lecture de {chemin_train.name} ...")
    train = pd.read_csv(chemin_train, encoding="utf-8")
    print(f"  {len(train):,} lignes, {train[cfg.LABEL_COL].nunique()} classes")

    # Tri prealable : le tirage ne doit pas dependre de l'ordre du fichier.
    train = train.sort_values(cfg.ID_COL).reset_index(drop=True)

    effectifs = train[cfg.LABEL_COL].value_counts()
    alloc = dp.allocation_plancher(effectifs,
                                   n_total=cfg.LLM_ITERATION_SIZE,
                                   plancher=cfg.LLM_ITERATION_MIN_PER_CLASS)

    morceaux = [
        train[train[cfg.LABEL_COL] == classe].sample(
            n=int(n), random_state=cfg.RANDOM_SEED)
        for classe, n in alloc.items() if n > 0
    ]
    ech = (pd.concat(morceaux)
             .sort_values([cfg.LABEL_COL, cfg.ID_COL])
             .reset_index(drop=True))

    cfg.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    ech[cfg.OUTPUT_COLS].to_csv(cfg.LLM_ITERATION_FILE, index=False,
                                encoding="utf-8")

    meta = {
        "role": "jeu d'iteration des prompts, etape 2 phase 2",
        "provenance": "TRAIN uniquement -- jamais test_sample_2000.csv",
        "seed": cfg.RANDOM_SEED,
        "n": int(len(ech)),
        "plancher_par_classe": cfg.LLM_ITERATION_MIN_PER_CLASS,
        "allocation": {k: int(v) for k, v in alloc.items()},
        "avertissement": (
            "FIGE pour toute l'etape 2. Changer ces lignes rendrait les "
            "variantes de prompt non comparables entre elles."
        ),
    }
    (cfg.LLM_ITERATION_FILE.with_suffix(".meta.json")).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return ech


def main() -> int:
    ech = construit(force="--force" in sys.argv)

    print(f"\n{'=' * 70}")
    print(f"JEU D'ITERATION : {cfg.LLM_ITERATION_FILE.name}")
    print(f"{'=' * 70}")
    print(f"  {len(ech)} lignes, seed {cfg.RANDOM_SEED}")
    repartition = ech[cfg.LABEL_COL].value_counts().reindex(cfg.CLASS_ORDER)
    for classe, n in repartition.items():
        print(f"    {classe:36s} {int(n)}")

    mots = ech[cfg.TEXT_COL].str.split().str.len()
    print(f"\n  longueur en mots : min {mots.min()}, mediane {int(mots.median())}, "
          f"max {mots.max()}")
    print(f"  textes depassant {cfg.LLM_MAX_WORDS} mots (tronques au prompt) : "
          f"{int((mots > cfg.LLM_MAX_WORDS).sum())}")
    print(f"  textes contenant du masquage XXXX : "
          f"{int(ech[cfg.TEXT_COL].str.contains('XXXX').sum())}/{len(ech)}")

    ok = True
    if int(repartition.min()) < cfg.LLM_ITERATION_MIN_PER_CLASS:
        print("  ECHEC : plancher par classe non respecte."); ok = False
    if len(ech) != cfg.LLM_ITERATION_SIZE:
        print(f"  ECHEC : {len(ech)} lignes au lieu de "
              f"{cfg.LLM_ITERATION_SIZE}."); ok = False

    # Garde-fou : aucune ligne ne doit provenir de l'echantillon d'evaluation.
    df_eval, _ = dp.load_eval_sample()
    fuite = set(ech[cfg.ID_COL].astype(str)) & set(df_eval[cfg.ID_COL].astype(str))
    if fuite:
        print(f"  ECHEC : {len(fuite)} ligne(s) partagee(s) avec "
              f"l'echantillon d'evaluation."); ok = False
    else:
        print(f"\n  aucune ligne partagee avec test_sample_2000.csv (verifie)")

    print(f"\n  -> {cfg.LLM_ITERATION_FILE}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
