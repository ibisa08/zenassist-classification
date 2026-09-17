"""Construit les deux jeux d'exemples few-shot de seconde generation.

    python tools/build_fewshot2_jeux.py

DEUX BRAS QUI NE DIFFERENT QUE PAR LE CRITERE DE RETENUE
--------------------------------------------------------
Meme tirage, meme plafond de longueur, meme population : `v7` et `v8` partent
des MEMES 45 candidats de `train_fewshot2_candidats.csv`. Seul le critere
change, ce qui rend l'ecart entre les deux attribuable.

  v7_fewshot_court   rang 1 de chaque classe. Aucun autre critere.
                     Bras "plafond de longueur seul".

  v8_fewshot_filtre  plus petit rang de chaque classe que mistral-small-4,
                     interroge EN ZERO-SHOT avec le prefixe `v1_zeroshot`
                     exact, temperature 0, etiquette correctement et de facon
                     IDENTIQUE sur 3 passes.

CRITERE DE v8, FIGE AVANT LE TIRAGE
-----------------------------------
Un candidat que le modele classe mal, ou instablement, SANS AIDE, est un
candidat dont l'etiquette ne se deduit pas du texte seul. Le donner en exemple
revient a demander au modele d'inferer une correspondance que le texte ne
porte pas.

Le critere exige les DEUX proprietes, et il n'est pas assoupli en cours de
route : si aucun des 5 candidats d'une classe ne passe, le script S'ARRETE. Il
ne retombe PAS sur le candidat de v7 -- ce serait fabriquer silencieusement un
troisieme bras, ni v7 ni v8.

LES 429 NE SONT PAS DES REPONSES
--------------------------------
Le lot 0 a essuye des HTTP 429 a 1 req/s. La cadence est ramenee a 0,33 req/s.
Un appel qui echoue est journalise par le runner avec `erreur_http` renseigne ;
`_purge_echecs()` retire ces lignes du JSONL pour que la reprise du runner les
REJOUE au lieu de les compter comme traitees. Un 429 compte comme une passe
manquante, jamais comme un desaccord du modele.
"""

import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

RACINE = Path(__file__).resolve().parent
while not (RACINE / "src" / "config.py").exists() and RACINE != RACINE.parent:
    RACINE = RACINE.parent
sys.path.insert(0, str(RACINE))

from src import config as cfg          # noqa: E402
from src import llm_runner as runner   # noqa: E402
from src.llm_client import MistralClient  # noqa: E402

JOURNAL = cfg.LLM_DIR / "fewshot2_juge.jsonl"
DEBIT_JUGE = 0.33          # req/s -- 1,0 a produit des 429 au lot 0
MAX_CYCLES_REJEU = 5


def _purge_echecs(chemin: Path) -> int:
    """Retire du JSONL les appels en erreur, pour que la reprise les rejoue."""
    if not chemin.exists():
        return 0
    lignes = [json.loads(l) for l in chemin.read_text(encoding="utf-8").splitlines() if l.strip()]
    gardees = [l for l in lignes if not l.get("erreur_http")]
    n = len(lignes) - len(gardees)
    if n:
        chemin.write_text("".join(json.dumps(l, ensure_ascii=False) + "\n"
                                  for l in gardees), encoding="utf-8")
    return n


def _juge(cand: pd.DataFrame) -> pd.DataFrame:
    """3 passes zero-shot sur les 45 candidats. Rend le journal en DataFrame."""
    client = MistralClient(
        modele_api=cfg.LLM_MODELES_A_COMPARER["mistral-small-4"]["id_api_candidat"],
        cle_tarification="mistral-small-4",
        temperature=0.0,
        requetes_par_seconde=DEBIT_JUGE)
    attendu = len(cand) * cfg.LLM_FEWSHOT2_PASSES_JUGE

    for cycle in range(1, MAX_CYCLES_REJEU + 1):
        purges = _purge_echecs(JOURNAL)
        if purges:
            print(f"\n[rejeu {cycle}] {purges} appel(s) en erreur retire(s) du "
                  f"journal : ils vont etre rejoues.")
        for passe in range(1, cfg.LLM_FEWSHOT2_PASSES_JUGE + 1):
            runner.run_campagne(
                cand, client, style="v1_zeroshot", chemin_sortie=JOURNAL,
                max_appels=len(cand) + 5,
                budget_max_usd=cfg.LLM_BUDGET_MAX_USD,
                est_echantillon_eval=False,
                execution=passe)
        journal = [json.loads(l) for l in JOURNAL.read_text(encoding="utf-8").splitlines() if l.strip()]
        restants = sum(1 for l in journal if l.get("erreur_http"))
        if not restants and len(journal) == attendu:
            break
        print(f"\n[rejeu] {restants} appel(s) encore en erreur, "
              f"{len(journal)}/{attendu} journalise(s).")
    else:
        print(f"\nARRET : appels en erreur persistants apres "
              f"{MAX_CYCLES_REJEU} cycles.")
        sys.exit(3)

    df = pd.DataFrame([json.loads(l) for l in
                       JOURNAL.read_text(encoding="utf-8").splitlines() if l.strip()])
    df[cfg.ID_COL] = df[cfg.ID_COL].astype(str)
    return df


def _ecrit_meta(chemin, nom, critere, choix, cand, journal=None, cout=None):
    """Ecrit le .meta.json d'un jeu d'exemples."""
    chemin.with_suffix(".meta.json").write_text(json.dumps({
        "role": f"exemples few-shot du style {nom}, etape 2 extension lot 1",
        "provenance": "TRAIN uniquement, via train_fewshot2_candidats.csv",
        "critere_de_retenue": critere,
        "seed_du_tirage": cfg.LLM_FEWSHOT2_SEED,
        "plafond_mots": cfg.LLM_FEWSHOT2_MAX_WORDS,
        "exemples": choix,
        "mots_par_exemple": {
            c: int(cand.loc[cand[cfg.ID_COL] == choix[c]["id"], "mots"].iloc[0])
            for c in cfg.CLASS_ORDER},
        "plafond_limite": (
            "BIAIS DE LONGUEUR ASSUME. A 120 mots, 43,5 % de la population du "
            "train est eligible, et 7 classes sur 9 voient leurs exemples tires "
            "SOUS leur mediane de longueur ; Mortgage (mediane 215 mots) sous "
            "son premier tiers. Un exemple few-shot n'a pas vocation a etre "
            "representatif de la longueur typique d'une reclamation, mais ce "
            "biais doit etre rappele partout ou v7 ou v8 est exploite."
        ),
        "journal_du_juge": journal,
        "cout_juge_usd": cout,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    if cfg.LLM_FEWSHOT2_MAX_WORDS is None:
        raise RuntimeError("LLM_FEWSHOT2_MAX_WORDS non renseigne.")

    cand = pd.read_csv(cfg.LLM_FEWSHOT2_CANDIDATS_FILE, encoding="utf-8")
    cand[cfg.ID_COL] = cand[cfg.ID_COL].astype(str)
    cand["mots"] = cand[cfg.TEXT_COL].str.split().str.len()
    print(f"{len(cand)} candidats lus depuis "
          f"{cfg.LLM_FEWSHOT2_CANDIDATS_FILE.name}")

    # =======================================================================
    # v7 : rang 1 de chaque classe, aucun autre critere
    # =======================================================================
    v7 = (cand[cand["rang_tirage"] == 1]
          .set_index(cfg.LABEL_COL).loc[cfg.CLASS_ORDER].reset_index())
    if len(v7) != len(cfg.CLASS_ORDER):
        print(f"ECHEC : {len(v7)} exemples pour v7"); return 1
    v7[cfg.OUTPUT_COLS].to_csv(cfg.LLM_FEWSHOT_V7_FILE, index=False,
                               encoding="utf-8")
    # Le meta de v7 est ecrit ICI, avant le juge. v7 ne depend d'AUCUN appel
    # API : le retenir jusqu'apres le filtre de v8 rendrait un livrable complet
    # otage d'une panne de quota qui ne le concerne pas.
    _ecrit_meta(cfg.LLM_FEWSHOT_V7_FILE, "v7_fewshot_court",
                "rang 1 du tirage pour chaque classe, aucun autre critere",
                {c: {"id": v7.loc[v7[cfg.LABEL_COL] == c, cfg.ID_COL].iloc[0],
                     "rang": 1} for c in cfg.CLASS_ORDER},
                cand)
    print(f"  -> {cfg.LLM_FEWSHOT_V7_FILE} (+ meta)")

    # =======================================================================
    # v8 : filtre du juge
    # =======================================================================
    print(f"\n{'=' * 78}")
    print(f"JUGE — {len(cand)} candidats x {cfg.LLM_FEWSHOT2_PASSES_JUGE} passes "
          f"= {len(cand) * cfg.LLM_FEWSHOT2_PASSES_JUGE} appels, "
          f"{DEBIT_JUGE} req/s")
    print(f"{'=' * 78}")
    journal = _juge(cand)

    verdicts = {}
    for cid, sous in journal.groupby(cfg.ID_COL):
        preds = sous.sort_values("execution")["label_predit"].tolist()
        vrai = sous["label_vrai"].iloc[0]
        verdicts[cid] = {
            "passes": preds,
            "stable": len(set(preds)) == 1,
            "correct": len(set(preds)) == 1 and preds[0] == vrai,
        }

    retenus, ecartes, bloquantes = {}, [], []
    for c in cfg.CLASS_ORDER:
        sous = cand[cand[cfg.LABEL_COL] == c].sort_values("rang_tirage")
        choisi = None
        for _, r in sous.iterrows():
            v = verdicts[r[cfg.ID_COL]]
            if v["correct"] and choisi is None:
                choisi = r
            elif choisi is None:
                ecartes.append({"classe": c, "rang": int(r["rang_tirage"]),
                                "id": r[cfg.ID_COL], "mots": int(r["mots"]),
                                "passes": v["passes"], "stable": v["stable"]})
        if choisi is None:
            bloquantes.append(c)
        else:
            retenus[c] = choisi

    if bloquantes:
        print(f"\n{'=' * 78}")
        print(f"ARRET : aucun des {cfg.LLM_FEWSHOT2_CANDIDATS_PAR_CLASSE} "
              f"candidats ne passe le filtre pour {len(bloquantes)} classe(s) :")
        for c in bloquantes:
            print(f"    {c}")
        print("  Le critere n'est PAS assoupli et v8 ne retombe PAS sur le\n"
              "  candidat de v7. train_fewshot_v8.csv n'a pas ete ecrit.")
        print(f"{'=' * 78}")
        return 2

    v8 = pd.DataFrame([retenus[c] for c in cfg.CLASS_ORDER])
    v8[cfg.OUTPUT_COLS].to_csv(cfg.LLM_FEWSHOT_V8_FILE, index=False,
                               encoding="utf-8")

    # =======================================================================
    # meta de v8 (celui de v7 a deja ete ecrit, il ne depend pas du juge)
    # =======================================================================
    cout = float(journal["cout_usd"].fillna(0).sum())
    _ecrit_meta(cfg.LLM_FEWSHOT_V8_FILE, "v8_fewshot_filtre",
                f"plus petit rang dont mistral-small-4 en zero-shot (prefixe "
                f"v1_zeroshot exact, temperature 0) rend l'etiquette vraie de "
                f"facon identique sur {cfg.LLM_FEWSHOT2_PASSES_JUGE} passes",
                {c: {"id": retenus[c][cfg.ID_COL],
                     "rang": int(retenus[c]["rang_tirage"])}
                 for c in cfg.CLASS_ORDER},
                cand, journal=str(JOURNAL), cout=round(cout, 6))

    # =======================================================================
    # compte rendu
    # =======================================================================
    print(f"\n{'=' * 90}")
    print("VERDICT DU JUGE, PAR CANDIDAT")
    print(f"{'=' * 90}")
    print(f"  {'classe':30s} {'rg':>2} {'id':>10} {'mots':>5} {'stable':>7} "
          f"{'correct':>8}  passes")
    for c in cfg.CLASS_ORDER:
        for _, r in cand[cand[cfg.LABEL_COL] == c].sort_values("rang_tirage").iterrows():
            v = verdicts[r[cfg.ID_COL]]
            marque = ""
            if retenus[c][cfg.ID_COL] == r[cfg.ID_COL]:
                marque = "  <- v8"
            if int(r["rang_tirage"]) == 1:
                marque += "  <- v7"
            print(f"  {c if r['rang_tirage'] == 1 else '':30s} "
                  f"{int(r['rang_tirage']):>2} {r[cfg.ID_COL]:>10} "
                  f"{int(r['mots']):>5} {str(v['stable']):>7} "
                  f"{str(v['correct']):>8}  "
                  f"{'/'.join(sorted(set(v['passes'])))[:34]}{marque}")

    n_ok = sum(1 for v in verdicts.values() if v["correct"])
    n_stable = sum(1 for v in verdicts.values() if v["stable"])
    print(f"\n  candidats stables sur 3 passes : {n_stable}/{len(cand)}")
    print(f"  candidats stables ET corrects   : {n_ok}/{len(cand)}")
    print(f"  appels du juge : {len(journal)}, cout {cout:.6f} $")
    print(f"\n  -> {cfg.LLM_FEWSHOT_V7_FILE}")
    print(f"  -> {cfg.LLM_FEWSHOT_V8_FILE}")
    print(f"  -> {JOURNAL}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
