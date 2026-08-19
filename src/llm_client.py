"""Client Mistral instrumente (etape 2).

AUCUNE LOGIQUE DE PROMPT ICI. Ce module recoit des messages deja construits par
`llm_prompts` et rend des faits mesures. La separation n'est pas cosmetique :
elle garantit qu'un changement de prompt en phase 3 ne touche pas au code qui
mesure, donc que les variantes restent comparables.

SDK MISTRAL v2 -- PIEGE D'IMPORT
-------------------------------
Le paquet installe est `mistralai` 2.9.3. En v2 les imports du SDK ont migre de
`mistralai` vers `mistralai.client` ; `mistralai` est devenu un namespace sans
`__init__.py`.

    from mistralai import Mistral          # v1 -- ImportError
    from mistralai.client import Mistral   # v2 -- correct

La quasi-totalite des exemples en ligne sont en v1. Les API elles-memes
(`chat.complete`, streaming, embeddings) sont inchangees.

CE QUI EST MESURE, ET POURQUOI
------------------------------
Les comptes de tokens viennent TOUS de `reponse.usage`, jamais d'une estimation.
C'est ce qui a remplace `config.AVG_COMPLAINT_TOKENS` (257 -> 245 le
2026-08-19), hypothese non
verifiee (mots x 1,3, regle empirique jamais confrontee au tokenizer de Mistral,
sur un corpus dont 85 % des textes contiennent du masquage XXXX qui se tokenise
mal). Si `usage` est absent, le champ vaut `None` et le manque est visible :
mieux vaut un trou declare qu'un chiffre invente.

`tokens_caches` vient de `usage.prompt_tokens_details['cached_tokens']`. Il
permet de MESURER l'effet du cache de prefixe au lieu de l'estimer, et donc de
verifier la projection de cache du notebook. MESURE le 2026-08-19 : la
couverture plafonne a 224 tokens sur 252, et l'activation depend du
rechauffement (55 % sur 20 appels, ~98 % en regime etabli).

LATENCE
-------
`latence_s` est la duree de la SEULE tentative aboutie, hors attente du
limiteur de debit et hors backoff des tentatives ratees. C'est la latence
d'inference, celle qui se compare a l'etape 3 et qui figurera dans la
recommandation. Les attentes de protocole sont reportees separement dans
`latence_avec_attentes_s` : elles dependent du palier tarifaire, pas du modele,
et les confondre ferait passer une contrainte de tier gratuit pour une lenteur
du LLM.

Les latences des tentatives ratees sont RETIREES de la distribution : un 429
n'est pas une mesure de la vitesse du modele.

ROBUSTESSE
----------
Apres epuisement des tentatives, `predict()` RETOURNE un dictionnaire portant
`erreur_http` au lieu de lever. Un appel perdu ne doit pas tuer une campagne de
2 000 : le runner journalise l'echec, poursuit, et la reprise permettra de le
rejouer.
"""

from __future__ import annotations

import os
import random
import time
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv
from mistralai.client import Mistral
from mistralai.client.errors import MistralError

from . import config as cfg
from .metrics import Timer

# Codes qui justifient une nouvelle tentative : saturation (429) et pannes
# serveur (5xx). Une 401 (cle invalide) ou une 400 (requete malformee) ne sont
# PAS rejouees -- les repeter cinq fois ne fait que retarder le diagnostic.
_CODES_REJOUABLES = frozenset({408, 409, 429})


def resout_cle_tarification(modele_api: str) -> str | None:
    """Retrouve la cle `MODELS_PRICING` correspondant a un identifiant d'API.

    Retourne `None` si la correspondance est inconnue, et NE DEVINE PAS : sans
    cle de tarification, le runner declarera le cout comme non calculable
    plutot que d'appliquer un tarif suppose.
    """
    for cle, info in cfg.LLM_MODELES_A_COMPARER.items():
        if info["id_api_candidat"] == modele_api:
            return cle
    # L'identifiant peut aussi etre directement une cle de tarification.
    if modele_api in cfg.MODELS_PRICING:
        return modele_api
    return None


class MistralClient:
    """Enveloppe instrumentee autour de `chat.complete`.

    Parameters
    ----------
    modele_api : identifiant accepte par l'API (ex. "mistral-small-latest").
        DISTINCT d'une cle de `config.MODELS_PRICING` : cf. le commentaire de
        `config.LLM_MODELES_A_COMPARER`.
    cle_tarification : cle de `config.MODELS_PRICING` pour le calcul de cout.
        Deduite de `modele_api` si omise, et laissee a `None` si indeduisible.
    chrono : `metrics.Timer` partage. En fournir un permet d'accumuler les
        latences de plusieurs clients dans une seule distribution.
    """

    def __init__(
        self,
        modele_api: str,
        cle_tarification: str | None = None,
        temperature: float = cfg.LLM_TEMPERATURE,
        max_tokens: int = cfg.LLM_MAX_TOKENS,
        requetes_par_seconde: float = cfg.LLM_REQUETES_PAR_SECONDE,
        max_tentatives: int = cfg.LLM_MAX_TENTATIVES,
        api_key: str | None = None,
        chrono: Timer | None = None,
    ) -> None:
        load_dotenv()
        cle = api_key or os.environ.get("MISTRAL_API_KEY")
        if not cle:
            raise RuntimeError(
                "MISTRAL_API_KEY absente. Renseigner la cle dans le fichier .env "
                "(cf. .env.example). Elle ne doit JAMAIS etre ecrite en dur."
            )

        self.modele_api = modele_api
        self.cle_tarification = cle_tarification or resout_cle_tarification(modele_api)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_tentatives = max_tentatives
        self.chrono = chrono if chrono is not None else Timer()

        # Limiteur de debit : le tier gratuit est a ~1 req/s. Configurable, car
        # une cle payante leve la contrainte et raccourcit d'autant la campagne.
        self.intervalle_min_s = (
            1.0 / requetes_par_seconde if requetes_par_seconde and requetes_par_seconde > 0
            else 0.0
        )
        self._dernier_envoi: float | None = None

        self._client = Mistral(api_key=cle)
        self._version_resolue: str | None = None
        self._version_tentee = False

    # -- limiteur de debit --------------------------------------------------
    def _attend_son_tour(self) -> float:
        """Bloque le temps necessaire pour respecter le debit. Rend l'attente."""
        if self.intervalle_min_s <= 0 or self._dernier_envoi is None:
            self._dernier_envoi = time.monotonic()
            return 0.0
        reste = self.intervalle_min_s - (time.monotonic() - self._dernier_envoi)
        if reste > 0:
            time.sleep(reste)
        self._dernier_envoi = time.monotonic()
        return max(0.0, reste)

    # -- version derriere l'alias -------------------------------------------
    @property
    def version_resolue(self) -> str | None:
        """Version REELLEMENT servie derriere l'alias appele.

        ATTENTION -- PIEGE VERIFIE LE 2026-08-19 : la reponse de `chat.complete`
        renvoie dans son champ `model` l'ALIAS DEMANDE, pas la version resolue.
        Un appel a `mistral-small-latest` repond `mistral-small-latest`. S'y
        fier rendrait toute detection de derive inoperante : on comparerait des
        alias a des alias, et un repointage passerait totalement inapercu.

        La resolution passe donc par les METADONNEES : `models.retrieve(alias)`
        expose `name`, qui porte la version (`mistral-small-2603`). C'est un
        appel de metadonnees, gratuit et sans inference.

        Resolu UNE FOIS par client, puis memorise. Limite assumee : un
        repointage survenant en cours de campagne ne serait pas vu par ce
        client-la. La comparaison ENTRE campagnes, elle, reste valide, chaque
        campagne construisant son propre client.
        """
        if not self._version_tentee:
            self._version_tentee = True
            try:
                carte = self._client.models.retrieve(model_id=self.modele_api)
                self._version_resolue = getattr(carte, "name", None)
            except Exception as exc:  # noqa: BLE001
                # Une resolution impossible ne doit pas empecher la campagne :
                # on journalise None, et l'absence sera visible au recapitulatif.
                print(f"  [version] resolution impossible pour "
                      f"{self.modele_api} : {type(exc).__name__}")
                self._version_resolue = None
        return self._version_resolue

    # -- lecture de usage ---------------------------------------------------
    @staticmethod
    def _lit_usage(reponse: Any) -> dict[str, Any]:
        """Extrait les comptes de tokens. Jamais d'estimation : `None` si absent."""
        usage = getattr(reponse, "usage", None)
        if usage is None:
            return {"tokens_in": None, "tokens_out": None, "tokens_caches": None,
                    "tokens_total": None, "service_tier": None}

        details = getattr(usage, "prompt_tokens_details", None)
        if isinstance(details, dict):
            caches = details.get("cached_tokens")
        elif details is not None:
            caches = getattr(details, "cached_tokens", None)
        else:
            caches = None

        return {
            "tokens_in": getattr(usage, "prompt_tokens", None),
            "tokens_out": getattr(usage, "completion_tokens", None),
            "tokens_total": getattr(usage, "total_tokens", None),
            "tokens_caches": caches,
            "service_tier": getattr(usage, "service_tier", None),
        }

    # -- appel --------------------------------------------------------------
    def predict(self, messages: list[dict]) -> dict[str, Any]:
        """Un appel, avec limitation de debit et reprise sur erreur transitoire.

        Ne leve pas sur echec reseau ou HTTP : retourne un dictionnaire dont
        `erreur_http` est renseigne et `reponse_brute` vaut `None`.

        Returns
        -------
        dict portant `reponse_brute`, `tokens_in`, `tokens_out`,
        `tokens_caches`, `service_tier`, `latence_s`, `erreur_http`, `modele`,
        `horodatage`, plus `tentatives` et `latence_avec_attentes_s` qui isolent
        le surcout de protocole de la latence d'inference, et `modele_resolu`
        qui consigne la version derriere l'alias appele.
        """
        debut_total = time.monotonic()
        horodatage = datetime.now(timezone.utc).isoformat()
        derniere_erreur = "echec sans exception capturee"

        for tentative in range(1, self.max_tentatives + 1):
            self._attend_son_tour()
            try:
                with self.chrono:
                    reponse = self._client.chat.complete(
                        model=self.modele_api,
                        messages=messages,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                    )
            except Exception as exc:  # noqa: BLE001 -- re-trie juste apres
                # Une tentative ratee n'est pas une mesure de la vitesse du
                # modele : on la retire de la distribution de latence.
                if self.chrono.latencies:
                    self.chrono.latencies.pop()

                code = getattr(exc, "status_code", None)
                rejouable = (
                    code in _CODES_REJOUABLES
                    or (code is not None and code >= 500)
                    or (code is None and not isinstance(exc, MistralError))
                )
                derniere_erreur = f"{type(exc).__name__}: {exc}"[:500]
                if code is not None:
                    derniere_erreur = f"HTTP {code} -- {derniere_erreur}"

                if not rejouable or tentative == self.max_tentatives:
                    break

                # Backoff exponentiel + jitter. Le jitter evite que plusieurs
                # appels repartis en meme temps retapent l'API en phase.
                attente = min(2.0 ** (tentative - 1), 30.0) + random.uniform(0, 0.5)
                time.sleep(attente)
                continue

            contenu = reponse.choices[0].message.content if reponse.choices else None
            return {
                "reponse_brute": contenu,
                # Version REELLEMENT servie. Elle vient des METADONNEES, pas
                # du champ `model` de la reponse, qui se contente de renvoyer
                # l'alias demande (verifie le 2026-08-19). Cf. la propriete
                # `version_resolue` et llm_eval.verifie_derive_version().
                "modele_resolu": self.version_resolue,
                # Ce que la reponse dit d'elle-meme, conserve tel quel pour
                # l'audit : si Mistral se met un jour a renvoyer la version
                # resolue ici, l'ecart avec `modele_resolu` le montrera.
                "modele_annonce": getattr(reponse, "model", None),
                **self._lit_usage(reponse),
                "latence_s": self.chrono.latencies[-1],
                "latence_avec_attentes_s": time.monotonic() - debut_total,
                "erreur_http": None,
                "tentatives": tentative,
                "modele": self.modele_api,
                "horodatage": horodatage,
            }

        # Toutes les tentatives ont echoue : on rend l'echec, on ne leve pas.
        return {
            "reponse_brute": None,
            "modele_resolu": self.version_resolue,
            "modele_annonce": None,
            "tokens_in": None, "tokens_out": None, "tokens_total": None,
            "tokens_caches": None, "service_tier": None,
            "latence_s": None,
            "latence_avec_attentes_s": time.monotonic() - debut_total,
            "erreur_http": derniere_erreur,
            "tentatives": self.max_tentatives,
            "modele": self.modele_api,
            "horodatage": horodatage,
        }

    # -- verification des identifiants de modele ----------------------------
    def modeles_disponibles(self) -> list[str]:
        """Liste les identifiants exposes par l'API. UN APPEL RESEAU (gratuit).

        Sert a confronter `config.LLM_MODELES_A_COMPARER` a la realite. N'est
        JAMAIS appelee automatiquement : l'ecart entre cles de tarification et
        identifiants d'API doit etre CONSTATE puis signale, pas corrige en
        silence au moment ou il gene.
        """
        return sorted(m.id for m in self._client.models.list().data)
