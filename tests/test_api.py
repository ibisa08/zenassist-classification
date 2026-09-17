"""Verifications du service HTTP `api/`.

CE QUI EST VERIFIE
------------------
  1. le service charge son modele au demarrage et dit lequel sur `/health` ;
  2. les etiquettes rendues par `/tags` sont EXACTEMENT celles du pipeline
     appele directement -- l'API ne doit rien ajouter ni retrancher au
     modele ;
  3. les entrees invalides sont refusees en 422, y compris les cas qui
     « marcheraient » techniquement : une chaine d'espaces, un entier ;
  4. les GARDE-FOUS DU CHARGEMENT echouent bien quand ils doivent echouer --
     un pickle dont l'empreinte ne correspond pas, une version de
     scikit-learn differente. C'est la partie qui compte : un controle qu'on
     ne voit jamais refuser n'a jamais ete verifie ;
  5. les journaux ne contiennent pas le texte des reclamations.

Aucune ecriture hors d'un dossier temporaire, supprime a la fin.

    <chemin du .venv>/bin/python tests/test_api.py
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path

RACINE = Path(__file__).resolve().parent
while not (RACINE / "src" / "config.py").exists() and RACINE != RACINE.parent:
    RACINE = RACINE.parent
sys.path.insert(0, str(RACINE))

from fastapi import FastAPI                     # noqa: E402
from fastapi.testclient import TestClient       # noqa: E402

from api.main import LONGUEUR_MAX_RECLAMATION, app, lifespan  # noqa: E402
from api.modele import (ModeleNonConforme, charge_modele,     # noqa: E402
                        chemins, sha256_fichier)
from src import config as cfg                   # noqa: E402
from src.data_prep import load_eval_sample      # noqa: E402

OK, KO = "  OK  ", " ECHEC"
_RESULTATS: list[bool] = []
N_TEXTES = 20


def check(nom, cond, detail=""):
    _RESULTATS.append(bool(cond))
    print(f"[{OK if cond else KO}] {nom}" + (f"   -> {detail}" if detail else ""))


class JournalCapture(logging.Handler):
    """Retient les lignes formatees du journal de l'API."""

    def __init__(self):
        super().__init__()
        self.lignes: list[str] = []

    def emit(self, record):
        self.lignes.append(self.format(record))


def _charge_avec(pkl: Path, meta: Path | None = None):
    """Appelle `charge_modele()` avec des chemins imposes, puis restaure."""
    avant = {c: os.environ.get(c)
             for c in ("ZENASSIST_MODELE_PKL", "ZENASSIST_MODELE_META")}
    try:
        os.environ["ZENASSIST_MODELE_PKL"] = str(pkl)
        if meta is None:
            os.environ.pop("ZENASSIST_MODELE_META", None)
        else:
            os.environ["ZENASSIST_MODELE_META"] = str(meta)
        return charge_modele(), None
    except Exception as e:                       # noqa: BLE001 — on la rapporte
        return None, e
    finally:
        for cle, valeur in avant.items():
            if valeur is None:
                os.environ.pop(cle, None)
            else:
                os.environ[cle] = valeur


def _demarrage_echoue(pkl: Path, meta: Path | None = None):
    """True si le lifespan refuse de demarrer avec ces chemins."""
    avant = {c: os.environ.get(c)
             for c in ("ZENASSIST_MODELE_PKL", "ZENASSIST_MODELE_META")}
    try:
        os.environ["ZENASSIST_MODELE_PKL"] = str(pkl)
        if meta is None:
            os.environ.pop("ZENASSIST_MODELE_META", None)
        else:
            os.environ["ZENASSIST_MODELE_META"] = str(meta)
        try:
            with TestClient(FastAPI(lifespan=lifespan)):
                return False, None
        except Exception as e:                   # noqa: BLE001
            return True, e
    finally:
        for cle, valeur in avant.items():
            if valeur is None:
                os.environ.pop(cle, None)
            else:
                os.environ[cle] = valeur


def main() -> int:
    pkl, meta_path = chemins()
    if not pkl.exists():
        print(f"IGNORE : {pkl.name} absent (models/ est exclu du depot).")
        print("Recuperer le modele depuis une release, ou le reconstruire avec")
        print("`python tools/export_modele.py --modele LinearSVC`.")
        return 0

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    sha_reel = sha256_fichier(pkl)

    # --- 1. demarrage et /health ---------------------------------------------
    with TestClient(app) as client:
        r = client.get("/health")
        check("/health repond 200", r.status_code == 200, str(r.status_code))
        etat = r.json()
        check("/health : statut ok", etat.get("statut") == "ok")
        check("/health : sha256 egal a celui des metadonnees",
              etat.get("sha256_pickle") == meta["empreintes"]["sha256_pickle"],
              etat.get("sha256_pickle", "")[:12])
        check("/health : sha256 egal a celui du fichier servi",
              etat.get("sha256_pickle") == sha_reel)
        check("/health : version de scikit-learn du modele",
              etat.get("scikit_learn") == meta["environnement"]["scikit_learn"],
              etat.get("scikit_learn"))
        check("/health : date de generation",
              etat.get("genere_le") == meta["genere_le"], etat.get("genere_le"))
        check("/health : 9 classes", etat.get("n_classes") == 9,
              str(etat.get("n_classes")))

        # --- 2. /tags == pipeline appele directement -------------------------
        modele = app.state.modele
        echantillon, _ = load_eval_sample()
        textes = echantillon[cfg.TEXT_COL].astype(str).tolist()[:N_TEXTES]

        memes, dans_referentiel, codes = 0, 0, set()
        for texte in textes:
            rep = client.post("/tags", json={"user_claim": texte})
            codes.add(rep.status_code)
            if rep.status_code != 200:
                continue
            tag = rep.json()["tag"]
            if tag == str(modele.pipeline.predict([texte])[0]):
                memes += 1
            if tag in cfg.CLASS_ORDER:
                dans_referentiel += 1

        check(f"/tags repond 200 sur les {N_TEXTES} textes", codes == {200},
              str(sorted(codes)))
        check("/tags rend exactement la prediction du pipeline",
              memes == len(textes), f"{memes}/{len(textes)}")
        check("/tags rend une classe du referentiel",
              dans_referentiel == len(textes),
              f"{dans_referentiel}/{len(textes)}")

        rep = client.post("/tags", json={"user_claim": textes[0]})
        check("/tags : nom du modele dans la reponse",
              rep.json()["modele"] == meta["modele"])
        check("/tags : version_modele = 12 premiers caracteres du sha256",
              rep.json()["version_modele"] == sha_reel[:12],
              rep.json()["version_modele"])

        # --- 3. entrees invalides --------------------------------------------
        cas = {
            "corps vide": {},
            "champ manquant": {"reclamation": "texte"},
            "user_claim entier": {"user_claim": 42},
            "user_claim nul": {"user_claim": None},
            "user_claim liste": {"user_claim": ["texte"]},
            "espaces seulement": {"user_claim": "   "},
            "chaine vide": {"user_claim": ""},
            "au-dela de la longueur maximale":
                {"user_claim": "a" * (LONGUEUR_MAX_RECLAMATION + 1)},
        }
        for nom, corps in cas.items():
            code = client.post("/tags", json=corps).status_code
            check(f"422 : {nom}", code == 422, str(code))

        code = client.post(
            "/tags",
            json={"user_claim": "a" * LONGUEUR_MAX_RECLAMATION}).status_code
        check("200 : exactement la longueur maximale", code == 200, str(code))

        # --- 5. journalisation sans le texte ---------------------------------
        capture = JournalCapture()
        capture.setFormatter(logging.Formatter("%(message)s"))
        journal = logging.getLogger("zenassist.api")
        niveau = journal.level
        journal.addHandler(capture)
        journal.setLevel(logging.INFO)
        try:
            temoin = ("Temoin unique de journalisation NEPASJOURNALISER, "
                      "mon compte bancaire a ete debite a tort.")
            rep = client.post("/tags", json={"user_claim": temoin})
        finally:
            journal.removeHandler(capture)
            journal.setLevel(niveau)

        journalise = "\n".join(capture.lignes)
        check("une ligne de journal par requete /tags",
              len(capture.lignes) >= 1, f"{len(capture.lignes)} ligne(s)")
        check("le journal ne contient pas le texte de la reclamation",
              "NEPASJOURNALISER" not in journalise and temoin not in journalise)
        check("le journal porte le tag et la duree",
              rep.json()["tag"] in journalise and "duree_ms=" in journalise,
              journalise.strip().splitlines()[-1] if capture.lignes else "")

    # --- 4. garde-fous du chargement ------------------------------------------
    # Tout se passe dans un dossier temporaire : le modele servi n'est jamais
    # touche, et on le verifie.
    temporaire = Path(tempfile.mkdtemp(prefix="zenassist_test_api_"))
    try:
        # (d) un octet modifie en fin de fichier -> empreinte differente.
        copie = temporaire / "LinearSVC.pkl"
        shutil.copy2(pkl, copie)
        shutil.copy2(meta_path, temporaire / "LinearSVC.metadata.json")
        with open(copie, "r+b") as f:
            f.seek(-1, os.SEEK_END)
            dernier = f.read(1)
            f.seek(-1, os.SEEK_END)
            f.write(bytes([dernier[0] ^ 0x01]))

        check("le pickle altere a bien une autre empreinte",
              sha256_fichier(copie) != sha_reel)

        _, erreur = _charge_avec(copie)
        check("pickle non conforme -> le chargement leve",
              isinstance(erreur, ModeleNonConforme),
              type(erreur).__name__ if erreur else "aucune exception")
        check("le message nomme les deux empreintes",
              erreur is not None and sha_reel in str(erreur)
              and sha256_fichier(copie) in str(erreur))

        echoue, err_demarrage = _demarrage_echoue(copie)
        check("pickle non conforme -> le demarrage echoue", echoue,
              type(err_demarrage).__name__ if err_demarrage else "demarre")

        # (e) version de scikit-learn attendue differente.
        meta_faux = dict(meta)
        meta_faux["environnement"] = dict(meta["environnement"])
        meta_faux["environnement"]["scikit_learn"] = "0.0.0-inexistante"
        chemin_meta_faux = temporaire / "version_differente.metadata.json"
        chemin_meta_faux.write_text(
            json.dumps(meta_faux, ensure_ascii=False), encoding="utf-8")

        _, erreur = _charge_avec(pkl, chemin_meta_faux)
        check("version de scikit-learn differente -> le chargement leve",
              isinstance(erreur, ModeleNonConforme),
              type(erreur).__name__ if erreur else "aucune exception")
        check("le message nomme la version attendue et l'installee",
              erreur is not None and "0.0.0-inexistante" in str(erreur))

        echoue, _ = _demarrage_echoue(pkl, chemin_meta_faux)
        check("version differente -> le demarrage echoue", echoue)

        check("le modele servi n'a pas ete modifie",
              sha256_fichier(pkl) == sha_reel)
        check("les metadonnees servies n'ont pas ete modifiees",
              json.loads(meta_path.read_text(encoding="utf-8")) == meta)
    finally:
        shutil.rmtree(temporaire, ignore_errors=True)

    check("le dossier temporaire est supprime", not temporaire.exists(),
          str(temporaire))

    print("\n" + "=" * 78)
    print(f"RESULTAT : {sum(_RESULTATS)}/{len(_RESULTATS)} verifications passees")
    if not all(_RESULTATS):
        print("\n  Le service ne peut pas etre mis en ligne en l'etat : soit il")
        print("  ne rend pas ce que rend le modele, soit ses garde-fous de")
        print("  chargement ne refusent plus ce qu'ils doivent refuser.")
    print("=" * 78)
    return 0 if all(_RESULTATS) else 1


if __name__ == "__main__":
    sys.exit(main())
