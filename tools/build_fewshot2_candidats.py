"""Tire les CANDIDATS des jeux d'exemples de seconde generation (v7 / v8).

    python tools/build_fewshot2_candidats.py

CE QUE CE SCRIPT FAIT, ET CE QU'IL NE FAIT PAS
----------------------------------------------
Il tire 5 candidats par classe et s'arrete la. Il ne choisit PAS les exemples :
  - v7 prendra le PREMIER candidat de chaque classe, dans l'ordre du tirage,
    sans autre critere ;
  - v8 prendra le premier qui passe le filtre du juge (3 passes concordantes et
    correctes en zero-shot), decide par `tools/build_fewshot2_jeux.py`.
Separer le tirage du filtrage garantit que les deux bras partagent EXACTEMENT
la meme population de depart : l'ecart entre v7 et v8 est alors imputable au
seul critere de retenue, pas a deux tirages differents.

L'ORDRE DU TIRAGE EST UNE DONNEE, PAS UN DETAIL
-----------------------------------------------
"Premier candidat" n'a de sens que si l'ordre est stable et auditable. Il est
donc materialise par une colonne `rang_tirage` (1 a 5 par classe) ecrite dans le
CSV, et non laisse a l'ordre des lignes -- qu'un tri ou une relecture pourrait
changer sans que rien ne le signale.

QUATRE EXCLUSIONS
-----------------
  1. train_iteration_20.csv
  2. train_fewshot_examples.csv   (exemples du prefixe de v6)
  3. train_selection_200.csv      (jeu de selection de la phase 3)
  4. train_selection_1835.csv     (jeu de selection du lot 1)
La 4e est la plus importante : un exemple present dans le jeu de mesure
donnerait a v7/v8 une reponse deja vue, et leur gain serait un artefact.

PLAFOND DE LONGUEUR
-------------------
`cfg.LLM_FEWSHOT2_MAX_WORDS` mots au maximum. Rappel de la raison :
`llm_prompts.tronque()` ne s'applique JAMAIS au prefixe, seulement au message
`user`. Un exemple long gonfle donc le prefixe de TOUS les appels de la
campagne, sans plafond d'aucune sorte.
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
    if cfg.LLM_FEWSHOT2_MAX_WORDS is None:
        raise RuntimeError("LLM_FEWSHOT2_MAX_WORDS non renseigne : le plafond "
                           "doit etre valide avant le tirage.")

    print(f"lecture de {dp._fichiers_split()[0].name} ...")
    train = pd.read_csv(dp._fichiers_split()[0], encoding="utf-8")
    train[cfg.ID_COL] = train[cfg.ID_COL].astype(str)
    train = train.sort_values(cfg.ID_COL).reset_index(drop=True)
    print(f"  {len(train):,} lignes")

    # --- exclusions ---------------------------------------------------------
    exclus, detail = set(), {}
    for nom, chemin in [("train_iteration_20", cfg.LLM_ITERATION_FILE),
                        ("train_fewshot_examples", cfg.LLM_FEWSHOT_FILE),
                        ("train_selection_200", cfg.LLM_SELECTION_FILE),
                        ("train_selection_1835", cfg.LLM_SELECTION2_FILE)]:
        ids = set(pd.read_csv(chemin, encoding="utf-8")[cfg.ID_COL].astype(str))
        detail[nom] = len(ids)
        exclus |= ids
        print(f"  {len(ids):>5} ligne(s) exclues : {chemin.name}")
    print(f"  {len(exclus):>5} identifiant(s) exclus au total (union)")

    dispo = train[~train[cfg.ID_COL].isin(exclus)].copy()
    dispo["mots"] = dispo[cfg.TEXT_COL].str.split().str.len()
    avant = len(dispo)
    dispo = dispo[dispo["mots"] <= cfg.LLM_FEWSHOT2_MAX_WORDS]
    print(f"  {avant:,} lignes hors exclusions, dont {len(dispo):,} a "
          f"<= {cfg.LLM_FEWSHOT2_MAX_WORDS} mots "
          f"({len(dispo)/avant:.1%})")

    # --- tirage, boucle EXPLICITE sur CLASS_ORDER ---------------------------
    morceaux = []
    for classe in cfg.CLASS_ORDER:
        sous = dispo[dispo[cfg.LABEL_COL] == classe]
        if len(sous) < cfg.LLM_FEWSHOT2_CANDIDATS_PAR_CLASSE:
            print(f"  ECHEC : {classe} n'offre que {len(sous)} candidat(s) "
                  f"eligible(s) pour {cfg.LLM_FEWSHOT2_CANDIDATS_PAR_CLASSE} "
                  f"demandes.")
            return 1
        tire = sous.sample(n=cfg.LLM_FEWSHOT2_CANDIDATS_PAR_CLASSE,
                           random_state=cfg.LLM_FEWSHOT2_SEED)
        # L'ordre rendu par .sample() EST l'ordre du tirage : on le fige en
        # colonne avant toute concatenation ou tri ulterieur.
        tire = tire.assign(rang_tirage=range(1, len(tire) + 1))
        morceaux.append(tire)

    cand = pd.concat(morceaux).reset_index(drop=True)
    colonnes = cfg.OUTPUT_COLS + ["rang_tirage"]
    cand[colonnes].to_csv(cfg.LLM_FEWSHOT2_CANDIDATS_FILE, index=False,
                          encoding="utf-8")

    (cfg.LLM_FEWSHOT2_CANDIDATS_FILE.with_suffix(".meta.json")).write_text(
        json.dumps({
            "role": "candidats des jeux d'exemples few-shot v7 et v8, "
                    "etape 2 extension lot 1",
            "provenance": "TRAIN uniquement",
            "seed": cfg.LLM_FEWSHOT2_SEED,
            "candidats_par_classe": cfg.LLM_FEWSHOT2_CANDIDATS_PAR_CLASSE,
            "n": int(len(cand)),
            "plafond_mots": cfg.LLM_FEWSHOT2_MAX_WORDS,
            "plafond_motif": (
                "Contraste de taille de prefixe avec v6 : 1 833 tokens pire cas "
                "contre 3 261 pour v6, soit -44 %. A 200 mots le contraste "
                "n'aurait valu que -17,5 %, insuffisant pour qu'un ecart mesure "
                "soit attribuable a la longueur."
            ),
            "plafond_limite": (
                "BIAIS DE LONGUEUR ASSUME. A 120 mots, 43,5 % de la population "
                "du train est eligible, et 7 classes sur 9 voient leurs exemples "
                "tires SOUS leur mediane de longueur ; Mortgage (mediane 215 "
                "mots) sous son premier tiers. Un exemple few-shot n'a pas "
                "vocation a etre representatif de la longueur typique d'une "
                "reclamation, mais ce biais doit etre rappele partout ou v7 ou "
                "v8 est exploite."
            ),
            "ordre_tirage": (
                "La colonne `rang_tirage` fige l'ordre rendu par .sample(). "
                "v7 retient le rang 1 de chaque classe ; v8 le plus petit rang "
                "qui passe le filtre du juge. Sans cette colonne, "
                "'premier candidat' dependrait de l'ordre des lignes."
            ),
            "exclusions": {**detail,
                           "test_sample_2000.csv": "disjoint par construction"},
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    # --- verifications ------------------------------------------------------
    print(f"\n{'=' * 78}\nCANDIDATS : {cfg.LLM_FEWSHOT2_CANDIDATS_FILE.name}\n{'=' * 78}")
    ok = True
    ids_cand = set(cand[cfg.ID_COL].astype(str))
    df_eval, _ = dp.load_eval_sample()
    for nom, chemin in [("test_sample_2000", None),
                        ("train_iteration_20", cfg.LLM_ITERATION_FILE),
                        ("train_fewshot_examples", cfg.LLM_FEWSHOT_FILE),
                        ("train_selection_200", cfg.LLM_SELECTION_FILE),
                        ("train_selection_1835", cfg.LLM_SELECTION2_FILE)]:
        autres = (set(df_eval[cfg.ID_COL].astype(str)) if chemin is None
                  else set(pd.read_csv(chemin, encoding="utf-8")[cfg.ID_COL].astype(str)))
        fuite = ids_cand & autres
        if fuite:
            print(f"  ECHEC : {len(fuite)} identifiant(s) partage(s) avec {nom} "
                  f"-- {sorted(fuite)[:5]}")
            ok = False
        else:
            print(f"  aucun recouvrement avec {nom}")
    if len(ids_cand) != len(cand):
        print("  ECHEC : identifiants dupliques"); ok = False
    else:
        print("  aucun identifiant duplique")
    trop_long = cand[cand["mots"] > cfg.LLM_FEWSHOT2_MAX_WORDS]
    if len(trop_long):
        print(f"  ECHEC : {len(trop_long)} candidat(s) au-dessus du plafond"); ok = False
    else:
        print(f"  tous les candidats a <= {cfg.LLM_FEWSHOT2_MAX_WORDS} mots")
    attendu = len(cfg.CLASS_ORDER) * cfg.LLM_FEWSHOT2_CANDIDATS_PAR_CLASSE
    if len(cand) != attendu:
        print(f"  ECHEC : {len(cand)} candidats au lieu de {attendu}"); ok = False
    else:
        print(f"  {len(cand)} candidats, "
              f"{cfg.LLM_FEWSHOT2_CANDIDATS_PAR_CLASSE} par classe")

    print(f"\n  {'classe':36s} {'rang':>4} {'id':>10} {'mots':>5}")
    for c in cfg.CLASS_ORDER:
        for _, r in cand[cand[cfg.LABEL_COL] == c].iterrows():
            print(f"  {c if r['rang_tirage'] == 1 else '':36s} "
                  f"{int(r['rang_tirage']):>4} {r[cfg.ID_COL]:>10} {int(r['mots']):>5}")

    print(f"\n  -> {cfg.LLM_FEWSHOT2_CANDIDATS_FILE}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
