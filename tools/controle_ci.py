"""Controles de reproductibilite executes par l'integration continue.

Deux sous-commandes, deux questions distinctes, deux moments differents du
pipeline :

    python tools/controle_ci.py donnees --rapport sorties/controle_donnees.json
    python tools/controle_ci.py modele  --rapport sorties/controle_modele.json

`donnees` (critere C0) repond a « le corpus reconstruit est-il bit-a-bit celui
de la reference ? ». Il tourne APRES `src.data_prep` et AVANT tout
entrainement : si la chaine de donnees a bouge, comparer des modeles n'a aucun
sens, et le job doit s'arreter la.

`modele` (criteres C1/C2/C3) repond a « le modele reentraine se comporte-t-il
comme la reference ? ». Il compare les PREDICTIONS, pas seulement les scores :
deux modeles peuvent afficher le meme F1-macro en se trompant sur des lignes
differentes.

CE QUI EST DECISIONNEL, ET CE QUI NE L'EST PAS
----------------------------------------------
Decisionnel : C0, puis C1 ou C2 (voir reports/protocole_alignement_env.md).
Information : la version de scikit-learn installee et l'empreinte de contenu.
L'empreinte ne vaut qu'a version de scikit-learn egale — elle ne peut donc pas
servir de critere entre versions, et un rapport qui la presenterait comme tel
ferait echouer la CI a la premiere montee de version legitime.

Les seuils ne sont PAS ecrits ici : ils sont lus dans `ci/reference.json`, qui
est l'unique source de verite du protocole. Un seuil en double finit par
diverger.

Code de sortie : 0 si conforme, 1 sinon.
"""
from __future__ import annotations

import argparse
import gzip
import json
import platform
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parent
while not (RACINE / "src" / "config.py").exists() and RACINE != RACINE.parent:
    RACINE = RACINE.parent
sys.path.insert(0, str(RACINE))
sys.path.insert(0, str(RACINE / "tools"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy  # noqa: E402
import sklearn  # noqa: E402

import export_modele as ex  # noqa: E402
from src import config as cfg  # noqa: E402
from src.data_prep import load_eval_sample, load_split  # noqa: E402
from src.metrics import f1_macro  # noqa: E402

REFERENCE = RACINE / "ci" / "reference.json"


def charge_reference() -> dict:
    if not REFERENCE.exists():
        raise SystemExit(f"Reference absente : {REFERENCE}")
    return json.loads(REFERENCE.read_text(encoding="utf-8"))


def affiche(nom: str, valeur: bool) -> bool:
    print(f"{nom}={valeur}")
    return valeur


def _affichable(chemin: Path) -> str:
    """Chemin relatif a la racine quand c'est possible, sinon tel quel.

    Un rapport ecrit hors du depot est un usage legitime en local ; il ne doit
    pas faire echouer le controle sur une erreur de mise en forme.
    """
    try:
        return str(chemin.relative_to(RACINE))
    except ValueError:
        return str(chemin)


def ecrit_rapport(chemin: Path, rapport: dict) -> None:
    chemin.parent.mkdir(parents=True, exist_ok=True)
    chemin.write_text(json.dumps(rapport, indent=2, ensure_ascii=False) + "\n",
                      encoding="utf-8")
    print(f"rapport : {_affichable(chemin)}")


def environnement() -> dict:
    return {
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
        "plateforme": platform.system(),
    }


# ===========================================================================
# C0 — chaine de donnees
# ===========================================================================
def commande_donnees(rapport_path: Path) -> int:
    ref = charge_reference()
    attendus = ref["donnees"]

    fichiers = [
        ("dataset_csv", cfg.RAW_FILE, attendus["dataset_csv_sha256"]),
        ("train", cfg.TRAIN_FILE, attendus["train_sha256"]),
        ("test", cfg.TEST_FILE, attendus["test_sha256"]),
        ("eval_sample", cfg.EVAL_SAMPLE_FILE, attendus["eval_sample_sha256"]),
    ]

    details, tous = {}, True
    for nom, chemin, attendu in fichiers:
        if not chemin.exists():
            details[nom] = {"chemin": str(chemin.relative_to(RACINE)),
                            "present": False, "attendu": attendu,
                            "obtenu": None, "egal": False}
            tous = False
            print(f"  {nom} : ABSENT ({chemin.relative_to(RACINE)})")
            continue
        obtenu = ex.sha256_fichier(chemin)
        egal = obtenu == attendu
        tous &= egal
        details[nom] = {"chemin": str(chemin.relative_to(RACINE)),
                        "present": True, "attendu": attendu,
                        "obtenu": obtenu, "egal": egal}
        print(f"  {nom} : EGAL={egal}  ({chemin.relative_to(RACINE)})")

    c0 = affiche("C0", tous)
    ecrit_rapport(rapport_path, {
        "controle": "donnees",
        "protocole": ref["protocole"],
        "release_donnees": attendus["release_donnees"],
        "environnement": environnement(),
        "fichiers": details,
        "C0": c0,
    })
    return 0 if c0 else 1


# ===========================================================================
# C1 / C2 / C3 — modele
# ===========================================================================
def _reference_predictions(chemin: Path) -> pd.DataFrame:
    with gzip.open(chemin, "rt", encoding="utf-8") as f:
        return pd.read_csv(f)


def commande_modele(rapport_path: Path) -> int:
    ref = charge_reference()
    seuils = ref["seuils"]
    accord_min = seuils["accord_min"]
    delta_f1_max = seuils["delta_f1_max"]
    print(f"seuils lus dans {REFERENCE.relative_to(RACINE)} : "
          f"accord_min={accord_min} delta_f1_max={delta_f1_max}")

    version_egale = affiche("VERSION_REFERENCE_EGALE",
                            sklearn.__version__ == ref["modele"]["scikit_learn"])

    pkl = ex.MODELS_DIR / "LinearSVC.pkl"
    if not pkl.exists():
        raise SystemExit(f"Modele absent : {pkl}")
    import pickle
    with open(pkl, "rb") as f:
        pipe = pickle.load(f)
    sha_pickle = ex.sha256_fichier(pkl)
    empreinte = ex.empreinte_contenu(pipe)
    empreinte_egale = affiche("EMPREINTE_EGALE",
                              empreinte == ref["modele"]["empreinte_contenu"])

    _, test = load_split()
    echantillon, poids = load_eval_sample()

    jeux = {
        "test": (test, None, "f1_macro"),
        "sample2000": (echantillon, poids.to_numpy(), "f1_macro_pondere_ht"),
    }

    details, ordre_global, identiques = {}, True, {}
    for cle, (df, w, cle_f1) in jeux.items():
        attendu = ref["predictions"][cle]
        ref_df = _reference_predictions(RACINE / attendu["fichier"])

        # Alignement : par identifiant si la reference en porte un, sinon par
        # position. Un taux d'accord calcule sur deux ordres differents est un
        # chiffre sans signification, donc l'alignement est CONTROLE, pas
        # suppose.
        if cfg.ID_COL in ref_df.columns and cfg.ID_COL in df.columns:
            mode = cfg.ID_COL
            ordre = (len(ref_df) == len(df)
                     and bool((ref_df[cfg.ID_COL].to_numpy()
                               == df[cfg.ID_COL].to_numpy()).all()))
        else:
            mode = "index"
            ordre = len(ref_df) == len(df)
        ordre_global &= ordre

        y_ref = ref_df["prediction"].astype(str).to_numpy()
        y_new = pipe.predict(df[cfg.TEXT_COL].to_numpy()).astype(str)
        y_true = df[cfg.LABEL_COL].astype(str).to_numpy()

        n = int(len(df))
        divergences = int((y_ref != y_new).sum()) if ordre else n
        accord = (n - divergences) / n
        identiques[cle] = ordre and divergences == 0

        f1_ref = attendu[cle_f1]
        f1_new = f1_macro(y_true, y_new, sample_weight=w)
        details[cle] = {
            "alignement": mode,
            "ordre_aligne": ordre,
            "n": n,
            "n_reference": int(len(ref_df)),
            "divergences": divergences,
            "taux_accord": accord,
            "identique": identiques[cle],
            "f1_reference": f1_ref,
            "f1_obtenu": f1_new,
            "delta_f1": f1_new - f1_ref,
        }
        print(f"  {cle} : n={n} divergences={divergences} "
              f"accord={accord:.6f} delta_f1={f1_new - f1_ref:+.6f}")

    affiche("ORDRE_ALIGNE", ordre_global)

    c1 = ordre_global and all(identiques.values())
    deltas_ok = all(abs(d["delta_f1"]) <= delta_f1_max for d in details.values())
    c2 = (not c1) and ordre_global \
        and details["test"]["taux_accord"] >= accord_min and deltas_ok
    verdict = "C1" if c1 else ("C2" if c2 else "C3")

    affiche("C1", c1)
    affiche("C2", c2)
    print(f"VERDICT={verdict}")

    ecrit_rapport(rapport_path, {
        "controle": "modele",
        "protocole": ref["protocole"],
        "environnement": environnement(),
        "version_reference": ref["modele"]["scikit_learn"],
        "VERSION_REFERENCE_EGALE": version_egale,
        "modele": {
            "sha256_pickle": sha_pickle,
            "sha256_pickle_reference": ref["modele"]["sha256_pickle"],
            "empreinte_contenu": empreinte,
            "empreinte_contenu_reference": ref["modele"]["empreinte_contenu"],
            "EMPREINTE_EGALE": empreinte_egale,
        },
        "seuils": seuils,
        "jeux": details,
        "ORDRE_ALIGNE": ordre_global,
        "C1": c1,
        "C2": c2,
        "VERDICT": verdict,
    })
    return 1 if verdict == "C3" else 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sous = p.add_subparsers(dest="commande", required=True)
    for nom in ("donnees", "modele"):
        sp = sous.add_parser(nom)
        sp.add_argument("--rapport", required=True,
                        help="chemin du rapport JSON a ecrire")
    a = p.parse_args(argv)

    chemin = Path(a.rapport)
    if not chemin.is_absolute():
        chemin = RACINE / chemin
    if a.commande == "donnees":
        return commande_donnees(chemin)
    return commande_modele(chemin)


if __name__ == "__main__":
    raise SystemExit(main())
