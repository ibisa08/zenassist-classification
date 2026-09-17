"""Service HTTP de classification des reclamations.

    <chemin du .venv>/bin/python -m uvicorn api.main:app --port 8000

Une seule route utile, `POST /tags`, et une route d'etat, `GET /health`. Le
modele est charge UNE fois au demarrage : le charger par requete ajouterait
plusieurs secondes a chacune, pour un pickle de 124 Mo.

ECHOUER AU DEMARRAGE PLUTOT QU'A LA PREMIERE REQUETE
----------------------------------------------------
Si le modele ne passe pas ses controles, le `lifespan` leve et le service ne
demarre pas. C'est deliberé : un service qui demarre avec un modele nul
repondrait des erreurs 500 a l'usage, et la cause reelle serait a chercher
dans des journaux vieux de plusieurs minutes. Un demarrage refuse est visible
immediatement, par l'orchestrateur comme par la personne qui deploie.

JOURNALISATION
--------------
Une ligne par requete `/tags` : duree de prediction et etiquette rendue,
JAMAIS le texte de la reclamation. Ces textes sont des recits de clients, ils
contiennent des situations financieres personnelles ; les journaux d'un
service sont copies, agreges et conserves bien plus longtemps que prevu.
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field, field_validator

from api.modele import ModeleServi, charge_modele

# Longueur maximale acceptee pour une reclamation, en caracteres.
#
# Mesure sur data/processed/train.csv (283 449 lignes, 2026-09-17) :
#   - longueur maximale observee : 31 634 caracteres
#   - quantile 0,999             : 12 213 caracteres
# Plafond fixe au millier superieur de la valeur maximale. Il n'exclut donc
# aucune reclamation du corpus d'entrainement, tout en bornant ce qu'un
# appelant peut envoyer : sans borne, le cout d'une requete n'a pas de limite.
LONGUEUR_MAX_RECLAMATION = 32_000

journal = logging.getLogger("zenassist.api")


class Reclamation(BaseModel):
    """Corps attendu par `POST /tags`."""

    user_claim: str = Field(
        max_length=LONGUEUR_MAX_RECLAMATION,
        description="Texte libre de la reclamation, en anglais.",
        examples=["I have been overdrawn at my bank and they charged me "
                  "380 dollars in fees."],
    )

    @field_validator("user_claim")
    @classmethod
    def non_vide(cls, v: str) -> str:
        """Refuse une reclamation vide ou faite d'espaces.

        Le modele rendrait une etiquette pour une chaine vide — la classe
        majoritaire — sans rien signaler. Une reponse plausible a une entree
        vide est pire qu'une erreur.
        """
        if not v.strip():
            raise ValueError("user_claim ne peut pas etre vide ni "
                             "uniquement compose d'espaces")
        return v


class Reponse(BaseModel):
    """Corps rendu par `POST /tags`."""

    tag: str = Field(description="Categorie predite.",
                     examples=["Bank account or service"])
    modele: str = Field(description="Nom du modele.", examples=["LinearSVC"])
    version_modele: str = Field(
        description="12 premiers caracteres du sha256 du pickle servi.",
        examples=["52504691ce1e"])


class Etat(BaseModel):
    """Corps rendu par `GET /health`."""

    statut: str
    modele: str
    sha256_pickle: str
    scikit_learn: str
    genere_le: str
    n_classes: int


@asynccontextmanager
async def lifespan(app: FastAPI):
    modele = charge_modele()          # leve si le modele n'est pas conforme
    app.state.modele = modele
    journal.info("modele charge : %s %s (scikit-learn %s, %d classes)",
                 modele.nom, modele.version_courte, modele.scikit_learn,
                 len(modele.classes))
    yield


app = FastAPI(
    title="ZenAssist — classification de reclamations",
    version="1.0.0",
    description=(
        "Attribue a une reclamation client l'une des neuf categories du "
        "referentiel ZenAssist.\n\n"
        "Le modele servi est un `LinearSVC` sur TF-IDF, dont l'export est "
        "verifie au demarrage : empreinte du pickle et version de "
        "scikit-learn doivent correspondre aux metadonnees publiees avec le "
        "modele. Les criteres de reproductibilite sont decrits dans "
        "`reports/protocole_alignement_env.md`."
    ),
    lifespan=lifespan,
)


def _modele(request: Request) -> ModeleServi:
    return request.app.state.modele


@app.post("/tags", response_model=Reponse, summary="Classer une reclamation",
          description="Rend la categorie predite pour le texte fourni.")
def tags(reclamation: Reclamation, request: Request) -> Reponse:
    modele = _modele(request)
    debut = time.perf_counter()
    tag = modele.predire(reclamation.user_claim)
    duree_ms = (time.perf_counter() - debut) * 1000

    # Longueur et duree, jamais le texte.
    journal.info("/tags tag=%s duree_ms=%.2f longueur=%d",
                 tag, duree_ms, len(reclamation.user_claim))
    return Reponse(tag=tag, modele=modele.nom,
                   version_modele=modele.version_courte)


@app.get("/health", response_model=Etat, summary="Etat du service",
         description="Identifie precisement le modele servi.")
def health(request: Request) -> Etat:
    modele = _modele(request)
    return Etat(
        statut="ok",
        modele=modele.nom,
        sha256_pickle=modele.sha256,
        scikit_learn=modele.scikit_learn,
        genere_le=modele.genere_le,
        n_classes=len(modele.classes),
    )
