"""Lance les quatre campagnes du lot 2 sur le jeu de selection de 1 835 lignes.

    python tools/run_lot2_campagnes.py [style ...]

Sans argument : les quatre campagnes dans l'ordre v1, v6, v7, v8.
Avec arguments : seulement les styles nommes (reprise ciblee).

POINT D'ARRET STRICT ENTRE CAMPAGNES
------------------------------------
Apres chaque campagne, trois conditions sont exigees avant de lancer la
suivante :
  1. 1 835 lignes journalisees ;
  2. 0 erreur HTTP non rejouee ;
  3. taux de parse_error < 5 %.
Si l'une echoue, le script S'ARRETE. Il n'enchaine pas : une campagne
incomplete rendrait le test apparie impossible, et l'enchainement masquerait
le probleme derriere six heures d'appels supplementaires.

LES 429 SONT REJOUES, PAS COMPTES
---------------------------------
Le runner journalise un appel echoue avec `erreur_http` renseigne, et sa
reprise est indexee sur (complaint_id, execution) : sans purge, un 429 serait
compte comme une ligne traitee. `_purge_echecs()` retire ces enregistrements
pour que la reprise les REJOUE. Un 429 est une passe manquante, jamais une
reponse du modele.

PRE-ENREGISTREMENT
------------------
Le script REFUSE de demarrer si `reports/protocole_lot2.md` est absent : le
critere de decision doit etre ecrit avant que la moindre metrique soit
observable.
"""

import json
import sys
import time
from pathlib import Path

import pandas as pd

RACINE = Path(__file__).resolve().parent
while not (RACINE / "src" / "config.py").exists() and RACINE != RACINE.parent:
    RACINE = RACINE.parent
sys.path.insert(0, str(RACINE))

from src import config as cfg              # noqa: E402
from src import llm_prompts as prompts     # noqa: E402
from src import llm_runner as runner       # noqa: E402
from src.llm_client import MistralClient   # noqa: E402

PROTOCOLE = cfg.REPORTS_DIR / "protocole_lot2.md" \
    if hasattr(cfg, "REPORTS_DIR") else RACINE / "reports" / "protocole_lot2.md"

CAMPAGNES = [("v1_zeroshot",       cfg.LLM_DIR / "lot2_v1.jsonl"),
             ("v6_fewshot",        cfg.LLM_DIR / "lot2_v6.jsonl"),
             ("v7_fewshot_court",  cfg.LLM_DIR / "lot2_v7.jsonl"),
             ("v8_fewshot_filtre", cfg.LLM_DIR / "lot2_v8.jsonl")]

DEBIT = 0.33
MAX_CYCLES = 40
SEUIL_PARSE_ERROR = 0.05


def _purge_echecs(chemin: Path) -> int:
    if not chemin.exists():
        return 0
    lignes = [json.loads(l) for l in chemin.read_text(encoding="utf-8").splitlines() if l.strip()]
    gardees = [l for l in lignes if not l.get("erreur_http")]
    n = len(lignes) - len(gardees)
    if n:
        chemin.write_text("".join(json.dumps(l, ensure_ascii=False) + "\n"
                                  for l in gardees), encoding="utf-8")
    return n


def _etat(chemin: Path) -> dict:
    if not chemin.exists():
        return {"n": 0, "erreurs": 0, "parse_error": 0.0, "cout": 0.0, "ids": set()}
    rs = [json.loads(l) for l in chemin.read_text(encoding="utf-8").splitlines() if l.strip()]
    ok = [r for r in rs if not r.get("erreur_http")]
    return {
        "n": len(ok),
        "erreurs": len(rs) - len(ok),
        "parse_error": (sum(1 for r in ok if r["statut_parsing"] != "ok") / len(ok)) if ok else 0.0,
        "cout": sum(r.get("cout_usd") or 0 for r in ok),
        "ids": {str(r[cfg.ID_COL]) for r in ok},
    }


def campagne(style: str, chemin: Path, df: pd.DataFrame) -> bool:
    """Mene une campagne a terme. Rend True si les 3 conditions sont remplies."""
    print(f"\n{'#' * 78}\n# CAMPAGNE {style}  ->  {chemin.name}\n{'#' * 78}")
    if style not in prompts._INSTRUCTIONS:
        print(f"ARRET : style '{style}' absent de llm_prompts._INSTRUCTIONS.")
        return False
    prefixe = prompts.prefixe_fige(style, prompts.labels_du_style(style))
    print(f"[prefixe] {len(prefixe.split()):,} mots, {len(prefixe):,} caracteres")

    client = MistralClient(
        modele_api=cfg.LLM_MODELES_A_COMPARER["mistral-small-4"]["id_api_candidat"],
        cle_tarification="mistral-small-4",
        temperature=0.0,
        requetes_par_seconde=DEBIT)

    for cycle in range(1, MAX_CYCLES + 1):
        purges = _purge_echecs(chemin)
        if purges:
            print(f"[cycle {cycle}] {purges} appel(s) en erreur purge(s), a rejouer.")
        etat = _etat(chemin)
        if etat["n"] >= len(df) and etat["erreurs"] == 0:
            break
        runner.run_campagne(df, client, style=style, chemin_sortie=chemin,
                            max_appels=len(df) + 10,
                            budget_max_usd=cfg.LLM_BUDGET_MAX_USD,
                            est_echantillon_eval=False,
                            frequence_progression=100,
                            execution=1)
        etat = _etat(chemin)
        print(f"[cycle {cycle}] {etat['n']}/{len(df)} reponses, "
              f"{etat['erreurs']} erreur(s), {etat['cout']:.4f} $")
        if etat["n"] >= len(df) and etat["erreurs"] == 0:
            break
        # Quota epuise : inutile de marteler l'API.
        if etat["erreurs"] and cycle < MAX_CYCLES:
            time.sleep(30)
    else:
        print(f"ARRET : {MAX_CYCLES} cycles epuises sans campagne complete.")
        return False

    # --- point d'arret strict ----------------------------------------------
    etat = _etat(chemin)
    print(f"\n{'=' * 78}\nPOINT D'ARRET — {style}\n{'=' * 78}")
    conditions = [
        (f"1 835 lignes journalisees", etat["n"] == len(df), f"{etat['n']}/{len(df)}"),
        ("0 erreur HTTP non rejouee", etat["erreurs"] == 0, f"{etat['erreurs']} erreur(s)"),
        (f"parse_error < {SEUIL_PARSE_ERROR:.0%}",
         etat["parse_error"] < SEUIL_PARSE_ERROR, f"{etat['parse_error']:.2%}"),
        ("identifiants uniques", len(etat["ids"]) == etat["n"],
         f"{len(etat['ids'])} distincts pour {etat['n']} lignes"),
    ]
    ok = True
    for libelle, verdict, detail in conditions:
        print(f"  [{'OK  ' if verdict else 'ECHEC'}] {libelle:32s} {detail}")
        ok &= verdict
    print(f"  cout de la campagne : {etat['cout']:.4f} $")
    return ok


def main(argv: list[str]) -> int:
    if not PROTOCOLE.exists():
        print(f"ARRET : {PROTOCOLE} absent. Le pre-enregistrement doit etre "
              f"ecrit AVANT toute campagne.")
        return 1
    print(f"[pre-enregistrement] {PROTOCOLE.name} present "
          f"({PROTOCOLE.stat().st_size:,} octets)")

    df = pd.read_csv(cfg.LLM_SELECTION2_FILE, encoding="utf-8")
    df[cfg.ID_COL] = df[cfg.ID_COL].astype(str)
    print(f"[jeu] {cfg.LLM_SELECTION2_FILE.name} : {len(df):,} lignes")

    voulus = set(argv) if argv else None
    for style, chemin in CAMPAGNES:
        if voulus is not None and style not in voulus:
            continue
        if not campagne(style, chemin, df):
            print(f"\nARRET DE LA SEQUENCE apres {style}. "
                  f"Les campagnes suivantes ne sont PAS lancees.")
            return 2
    print(f"\n{'=' * 78}\nLES 4 CAMPAGNES SONT COMPLETES.\n{'=' * 78}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
