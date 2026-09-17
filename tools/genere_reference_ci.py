"""Fige la reference de reproductibilite utilisee par l'integration continue.

Ce script ne s'execute PAS en CI : il produit les fichiers que la CI compare.
Il est lance a la main, sur la machine qui detient le modele de reference, et
ses sorties (`ci/`) sont versionnees.

CE QU'IL FIGE
-------------
  - les predictions du modele de reference sur `test.csv` (70 863 lignes) et
    sur l'echantillon d'evaluation de 2 000 lignes ;
  - les empreintes qui identifient le modele, les donnees et le jeu de
    donnees brut ;
  - les seuils d'acceptation C1/C2 du protocole.

POURQUOI DES PREDICTIONS, ET PAS SEULEMENT UN F1
------------------------------------------------
Deux modeles differents peuvent afficher le meme F1-macro a la quatrieme
decimale en se trompant sur des lignes differentes. Le F1 est une statistique
de la sortie ; la sortie elle-meme est ce qui doit etre compare. Les fichiers
de predictions pesent quelques centaines de kilo-octets compresses, ce qui
est le prix d'un controle qui ne peut pas passer par accident.

DETERMINISME DU .GZ
-------------------
`gzip` inscrit par defaut l'horodatage de compression dans l'en-tete : deux
compressions du meme contenu donneraient deux fichiers differents, et chaque
regeneration polluerait le diff. `mtime=0` supprime ce champ.

    <chemin du .venv>/bin/python tools/genere_reference_ci.py
"""
from __future__ import annotations

import gzip
import hashlib
import json
import pickle
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parent
while not (RACINE / "src" / "config.py").exists() and RACINE != RACINE.parent:
    RACINE = RACINE.parent
sys.path.insert(0, str(RACINE))
sys.path.insert(0, str(RACINE / "tools"))

import pandas as pd  # noqa: E402
import sklearn  # noqa: E402

import export_modele as ex  # noqa: E402
from src import config as cfg  # noqa: E402
from src.data_prep import load_eval_sample, load_split  # noqa: E402
from src.metrics import f1_macro  # noqa: E402

CI_DIR = RACINE / "ci"

# Empreinte du modele de reference, etablie par reports/resultats_alignement_env.md.
SHA_PICKLE_REFERENCE = (
    "52504691ce1e914e2b939f81ceb24ffbee22609f47f060d646130cb4660bfb9e")

# Empreinte du jeu de donnees brut publie en release `data-v1`. C'est la seule
# valeur en dur de ce fichier : elle designe un artefact externe au depot, dont
# rien ici ne permet de la deriver.
SHA_DATASET_PUBLIE = (
    "80db47ac8e817bfd1f0588f785a847f03d5f3052f244c0e47bdf8f312e7b52bc")
RELEASE_DONNEES = "data-v1"

SEUIL_ACCORD_MIN = 0.999
SEUIL_DELTA_F1_MAX = 0.002


def sha256_liste(predictions) -> str:
    """Empreinte de la SUITE des predictions, dans l'ordre de chargement."""
    return hashlib.sha256(
        "\n".join(map(str, predictions)).encode("utf-8")).hexdigest()


def verifie_modele() -> str:
    pkl = ex.MODELS_DIR / "LinearSVC.pkl"
    if not pkl.exists():
        raise SystemExit(f"Modele absent : {pkl}")
    obtenu = ex.sha256_fichier(pkl)
    if obtenu != SHA_PICKLE_REFERENCE:
        raise SystemExit(
            "Le modele present n'est pas le modele de reference.\n"
            f"  attendu : {SHA_PICKLE_REFERENCE}\n"
            f"  obtenu  : {obtenu}\n"
            "Regenerer la reference a partir d'un autre modele exige de "
            "rejouer le protocole : reports/protocole_alignement_env.md.")
    return obtenu


def verifie_donnees() -> dict[str, str]:
    """sha256 des trois fichiers de data/processed, contre split_metadata.json."""
    meta = json.loads(cfg.SPLIT_METADATA_FILE.read_text(encoding="utf-8"))
    shas, ecarts = {}, []
    for nom in ("train", "test", "eval_sample"):
        info = meta["fichiers"][nom]
        attendu = info["sha256"]
        obtenu = ex.sha256_fichier(RACINE / info["chemin"])
        shas[nom] = obtenu
        if obtenu != attendu:
            ecarts.append(f"  {info['chemin']} : attendu {attendu}, obtenu {obtenu}")
    if ecarts:
        raise SystemExit(
            "Fichiers de data/processed non conformes a split_metadata.json :\n"
            + "\n".join(ecarts)
            + "\nRejouer `python -m src.data_prep` sous l'environnement declare.")
    return shas


def ecrit_gz(chemin: Path, df: pd.DataFrame) -> None:
    """CSV compresse de facon deterministe : aucun horodatage dans l'en-tete."""
    donnees = df.to_csv(index=False).encode("utf-8")
    with open(chemin, "wb") as brut:
        with gzip.GzipFile(filename="", mode="wb", fileobj=brut, mtime=0) as gz:
            gz.write(donnees)


def main() -> int:
    ex.verifie_version_sklearn()
    sha_pickle = verifie_modele()
    shas_donnees = verifie_donnees()

    with open(ex.MODELS_DIR / "LinearSVC.pkl", "rb") as f:
        pipe = pickle.load(f)

    _, test = load_split()
    echantillon, poids = load_eval_sample()

    CI_DIR.mkdir(parents=True, exist_ok=True)
    predictions = {}
    for cle, df, fichier, w in (
        ("test", test, "reference_predictions_test.csv.gz", None),
        ("sample2000", echantillon, "reference_predictions_sample2000.csv.gz",
         poids),
    ):
        pred = pipe.predict(df[cfg.TEXT_COL].to_numpy())
        colonnes = {}
        if cfg.ID_COL in df.columns:
            colonnes[cfg.ID_COL] = df[cfg.ID_COL].to_numpy()
        else:
            colonnes["ligne"] = range(len(df))
        colonnes["prediction"] = pred
        ecrit_gz(CI_DIR / fichier, pd.DataFrame(colonnes))
        predictions[cle] = {
            "fichier": f"ci/{fichier}",
            "n": int(len(df)),
            "sha256_liste": sha256_liste(pred),
            "f1": f1_macro(df[cfg.LABEL_COL].to_numpy(), pred,
                           sample_weight=None if w is None
                           else w.to_numpy()),
        }

    reference = {
        "protocole": "reports/protocole_alignement_env.md",
        "modele": {
            "nom": "LinearSVC",
            "sha256_pickle": sha_pickle,
            "scikit_learn": sklearn.__version__,
            "empreinte_contenu": ex.empreinte_contenu(pipe),
        },
        "donnees": {
            "dataset_csv_sha256": SHA_DATASET_PUBLIE,
            "release_donnees": RELEASE_DONNEES,
            "train_sha256": shas_donnees["train"],
            "test_sha256": shas_donnees["test"],
            "eval_sample_sha256": shas_donnees["eval_sample"],
        },
        "predictions": {
            "test": {
                "fichier": predictions["test"]["fichier"],
                "n": predictions["test"]["n"],
                "sha256_liste": predictions["test"]["sha256_liste"],
                "f1_macro": predictions["test"]["f1"],
            },
            "sample2000": {
                "fichier": predictions["sample2000"]["fichier"],
                "n": predictions["sample2000"]["n"],
                "sha256_liste": predictions["sample2000"]["sha256_liste"],
                "f1_macro_pondere_ht": predictions["sample2000"]["f1"],
            },
        },
        "seuils": {
            "accord_min": SEUIL_ACCORD_MIN,
            "delta_f1_max": SEUIL_DELTA_F1_MAX,
        },
    }
    (CI_DIR / "reference.json").write_text(
        json.dumps(reference, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")

    print("  ci/reference.json ecrit")
    for cle in ("test", "sample2000"):
        f = RACINE / predictions[cle]["fichier"]
        print(f"  {predictions[cle]['fichier']}  "
              f"{f.stat().st_size / 1024:.1f} Ko  n={predictions[cle]['n']}")
    print(f"  empreinte du modele : {reference['modele']['empreinte_contenu']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
