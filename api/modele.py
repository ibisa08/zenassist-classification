"""Chargement VERIFIE du modele servi par l'API.

Un pickle est du code : le charger, c'est executer ce qu'il contient. Le
service ne peut donc pas se contenter d'ouvrir le fichier qu'on lui designe.
Deux controles precedent `pickle.load`, et aucun des deux n'est contournable :

  1. `sha256` du fichier == `empreintes.sha256_pickle` des metadonnees. Le
     fichier livre est-il celui que le depot decrit ? C'est la meme empreinte
     que publie `sorties/SHA256SUMS` a chaque release.
  2. version de scikit-learn installee == `environnement.scikit_learn` des
     metadonnees. Un pickle scikit-learn se charge souvent sous une autre
     version, en emettant au mieux un avertissement, et rien ne garantit alors
     que les predictions soient les memes. L'erratum du 2026-09-17
     (reports/limites.md) documente ce qu'a coute une divergence de version
     passee inapercue.

Le controle 2 porte sur la version INSTALLEE, pas sur le chemin de
l'interpreteur : c'est la propriete qui compte, et elle se verifie.

CE QUI N'EST PAS VERIFIE ICI
----------------------------
L'empreinte de CONTENU (`empreintes.contenu`) n'est pas recalculee. Elle
depend de la plateforme — le controle Linux de l'integration continue l'a
montre — et ne vaut qu'a version de scikit-learn egale. Elle ne peut donc pas
servir de garde-fou au demarrage.

CONFIGURATION
-------------
    ZENASSIST_MODELE_PKL    chemin du pickle
                            (defaut : <racine>/models/<VERSION_MODELE_SERVI>/
                             LinearSVC.pkl)
    ZENASSIST_MODELE_META   chemin des metadonnees
                            (defaut : <pickle sans suffixe>.metadata.json)
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import sklearn

RACINE = Path(__file__).resolve().parent
while not (RACINE / "src" / "config.py").exists() and RACINE != RACINE.parent:
    RACINE = RACINE.parent

# Release du modele servie par defaut. Le service ne prend PAS « le dernier
# modele disponible » : changer de modele servi, c'est changer les predictions
# rendues aux clients, donc un commit explicite et revisable, pas un effet de
# bord du contenu d'un dossier.
VERSION_MODELE_SERVI = "modele-v1.0.0"

PKL_PAR_DEFAUT = RACINE / "models" / VERSION_MODELE_SERVI / "LinearSVC.pkl"

COMMANDE_RECUPERATION = (
    f"gh release download {VERSION_MODELE_SERVI} "
    "--repo ibisa08/zenassist-classification "
    "--pattern 'LinearSVC.*' --pattern SHA256SUMS "
    f"--dir models/{VERSION_MODELE_SERVI}")


class ModeleNonConforme(RuntimeError):
    """Le modele designe ne correspond pas a ses metadonnees.

    Volontairement distincte de `FileNotFoundError` : « le fichier manque » et
    « le fichier n'est pas celui annonce » n'appellent pas la meme reaction.
    """


@dataclass(frozen=True)
class ModeleServi:
    """Le modele charge et ce qui permet de dire lequel c'est."""

    pipeline: Any
    nom: str
    sha256: str
    scikit_learn: str
    classes: tuple[str, ...]
    genere_le: str

    @property
    def version_courte(self) -> str:
        """12 premiers caracteres du sha256 : de quoi identifier sans encombrer."""
        return self.sha256[:12]

    def predire(self, texte: str) -> str:
        """Etiquette predite pour UNE reclamation.

        Le modele est appele sur une liste d'un element : c'est la latence
        percue par un utilisateur qui soumet une reclamation, celle que
        l'etape 3 a mesuree. Diviser un temps de lot par le nombre de lignes
        donnerait un autre chiffre, et pas celui qui interesse le client.
        """
        return str(self.pipeline.predict([texte])[0])


def chemins() -> tuple[Path, Path]:
    """(pickle, metadonnees), d'apres l'environnement ou les valeurs par defaut."""
    pkl = Path(os.environ.get("ZENASSIST_MODELE_PKL", PKL_PAR_DEFAUT))
    meta_env = os.environ.get("ZENASSIST_MODELE_META")
    meta = Path(meta_env) if meta_env else pkl.with_suffix(".metadata.json")
    return pkl, meta


def sha256_fichier(chemin: Path) -> str:
    h = hashlib.sha256()
    with chemin.open("rb") as f:
        for bloc in iter(lambda: f.read(1 << 20), b""):
            h.update(bloc)
    return h.hexdigest()


def charge_modele() -> ModeleServi:
    """Charge le modele apres verification, ou leve.

    Aucun repli silencieux : un service qui demarre sans modele repond des
    erreurs a la premiere requete, plusieurs minutes apres que la cause est
    apparue dans les journaux.
    """
    pkl, meta_path = chemins()
    if not pkl.exists():
        raise FileNotFoundError(
            f"Modele absent : {pkl}\n"
            "`models/` est exclu du depot. Recuperer la release servie :\n"
            f"    {COMMANDE_RECUPERATION}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadonnees absentes : {meta_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    attendu = meta["empreintes"]["sha256_pickle"]
    obtenu = sha256_fichier(pkl)
    if obtenu != attendu:
        raise ModeleNonConforme(
            "Le pickle ne correspond pas a ses metadonnees.\n"
            f"  fichier             : {pkl}\n"
            f"  sha256 attendu      : {attendu}\n"
            f"  sha256 obtenu       : {obtenu}\n"
            "Le modele a ete modifie, tronque, ou ces metadonnees decrivent un "
            "autre export. Chargement refuse.")

    version_attendue = meta["environnement"]["scikit_learn"]
    if version_attendue != sklearn.__version__:
        raise ModeleNonConforme(
            "Version de scikit-learn differente de celle de l'export.\n"
            f"  attendue par le modele : {version_attendue}\n"
            f"  installee              : {sklearn.__version__}\n"
            "Les predictions ne sont plus garanties identiques a celles "
            "mesurees. Chargement refuse.")

    with pkl.open("rb") as f:
        pipeline = pickle.load(f)

    return ModeleServi(
        pipeline=pipeline,
        nom=meta["modele"],
        sha256=obtenu,
        scikit_learn=version_attendue,
        classes=tuple(meta["vocabulaire"]["classes"]),
        genere_le=meta["genere_le"],
    )
