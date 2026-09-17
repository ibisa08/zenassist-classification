"""Construction des prompts et parsing des reponses (etape 2).

Ce module ne fait AUCUN appel reseau : il transforme du texte en messages, et
des messages en etiquette. Il est donc entierement testable hors ligne, ce qui
est exactement ce que fait `tests/test_llm_prompts.py`.

CACHE DE PREFIXE
----------------
Le cache du fournisseur s'active sur le PREFIXE LITTERAL commun a deux appels
successifs. Le prompt est donc construit dans cet ordre strict :

    1. instruction systeme (role, consigne, masquage XXXX, format de sortie)
    2. liste des 9 libelles
    3. le texte de la reclamation, EN DERNIER

Les blocs 1 et 2 forment le message `system` et sont identiques A L'OCTET PRES
d'un appel a l'autre : aucune date, aucun identifiant, aucun compteur n'y est
interpole. Le bloc 3 est le message `user`, seul a varier.

C'est une contrainte de MESURE, pas une elegance : sans prefixe stable le cache
ne s'active jamais, `usage.prompt_tokens_details['cached_tokens']` reste a zero,
et la projection de cache du notebook devient invérifiable. La fonction
`prefixe_fige()` est memoisee pour que la meme chaine d'objet soit reutilisee.

INSTRUCTION EN ANGLAIS
----------------------
Le corpus et les 9 libelles sont en anglais. Melanger les langues ajouterait une
variable inutile a la comparaison. La variante francaise est un style a tester
en phase 3, pas un defaut.

PARSING STRICT
--------------
`parse_reponse()` ne rattrape RIEN : une reponse non conforme devient
`config.PARSE_ERROR`, comptee comme une erreur dans le F1-macro. C'est
deliberement severe -- un modele qui ne respecte pas le format demande n'a pas
classe la reclamation.

La reponse brute etant TOUJOURS journalisee par le runner, un re-parsing
tolerant reste possible hors ligne, sans reappel ni surcout : c'est le role de
`reparse_tolerant()`, qui n'est JAMAIS utilisee pendant une campagne. Elle sert
a chiffrer ce que la severite a coute, pour que l'arbitrage soit documente.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from . import config as cfg

# ===========================================================================
# Blocs constants du prompt
# ===========================================================================
# Ces gabarits ne contiennent AUCUN champ interpole autre que la liste des
# libelles, elle-meme figee par `config.CLASS_ORDER`. Toute interpolation
# supplementaire (date, identifiant, compteur) casserait le cache de prefixe.

_INSTRUCTION_V1_EN = """\
You are a classification system for consumer finance complaints submitted to the \
US Consumer Financial Protection Bureau.

Read the complaint provided by the user and assign it to exactly one of the \
product categories listed below.

Redaction notice: the complaints contain personal data replaced by runs of the \
letter X (for example XXXX, XX/XX/XXXX, $XXXX). This masking is normal and \
expected. Do not treat it as missing information, do not refuse to classify \
because of it, and do not comment on it.

You must pick exactly one category from the list, copied verbatim. Do not \
invent a category and do not return more than one.

Reply with a single minimal JSON object and nothing else:
{"label": "<one category, copied exactly from the list>"}

No explanation, no confidence score, no markdown code fences, no text before or \
after the JSON object."""

_INSTRUCTION_V1_FR = """\
Tu es un systeme de classification de reclamations financieres deposees aupres \
du Consumer Financial Protection Bureau americain.

Lis la reclamation fournie par l'utilisateur et attribue-lui exactement une des \
categories de produit listees ci-dessous.

Avertissement sur le masquage : les reclamations contiennent des donnees \
personnelles remplacees par des suites de la lettre X (par exemple XXXX, \
XX/XX/XXXX, $XXXX). Ce masquage est normal et attendu. Ne le considere pas \
comme une information manquante, ne refuse pas de classer a cause de lui, et ne \
le commente pas.

Tu dois choisir exactement une categorie de la liste, recopiee mot pour mot. \
N'invente pas de categorie et n'en rends pas plusieurs.

Reponds par un unique objet JSON minimal et rien d'autre :
{"label": "<une categorie, recopiee exactement depuis la liste>"}

Aucune explication, aucun score de confiance, aucun bloc de code markdown, \
aucun texte avant ou apres l'objet JSON."""

# ---------------------------------------------------------------------------
# VARIANTES DE LA PHASE 3
# ---------------------------------------------------------------------------
# Chaque variante est une modification ISOLEE de v1 : sinon un gain ne pourrait
# pas etre attribue. `_INSTRUCTION_V1_EN` n'est JAMAIS modifiee -- v1 a deja ete
# mesuree, et changer son texte d'un octet invaliderait la comparaison.

# --- v2 : regle produit vs prejudice ---------------------------------------
# Vise le mecanisme d'erreur identifie en phase 2 : le texte decrit un prejudice
# ("ca abime mon credit") qui pointe vers une autre classe que le produit
# concerne. 2 des 7 erreurs de mistral-small en relevaient.
_REGLE_PRODUIT = """

Decision rule: classify by the FINANCIAL PRODUCT the complaint is about, not by the harm the consumer describes. Many complaints describe damage to a credit report, or a debt being pursued, while concerning a different underlying product such as a mortgage, an auto loan or a student loan. Identify the product first."""

# --- v3 : une definition d'une ligne par etiquette --------------------------
_DEFINITIONS = {
    "Credit reporting": "disputes about credit reports, credit scores, credit repair services or consumer reporting agencies",
    "Debt collection": "a collector attempting to collect a debt: contact practices, validation, disputed amounts",
    "Mortgage": "home loans: origination, servicing, escrow, modification, foreclosure",
    "Credit card or prepaid card": "credit cards, prepaid cards and gift cards: billing, interest, rewards, fees",
    "Bank account or service": "checking and savings accounts: deposits, withdrawals, overdrafts, account closure",
    "Student loan": "federal or private education loans: servicing, repayment plans, forgiveness",
    "Money transfer or virtual currency": "wire transfers, remittances, mobile payments, cryptocurrency",
    "Payday, title or personal loan": "short-term or installment consumer loans not secured by a home",
    "Vehicle loan or lease": "auto loans and leases: financing, repossession, title",
}

# --- styles few-shot : v6, v7, v8 -------------------------------------------
# Les exemples vivent dans le PREFIXE, apres la liste des libelles et avant le
# texte : ils sont constants d'un appel a l'autre, donc cacheables. Les y mettre
# apres le texte casserait le cache et fausserait la mesure de latence.
#
# POURQUOI UNE LISTE ET NON UN DICTIONNAIRE PAR CLASSE
# ----------------------------------------------------
# La version precedente indexait les exemples par `{classe: texte}`. Une seconde
# ligne portant la meme classe ECRASAIT SILENCIEUSEMENT la premiere : un jeu a
# deux exemples par classe aurait produit un prefixe amputes de moitie sans
# qu'aucun test ne le signale. La construction est passee en LISTE, ce qui rend
# plusieurs exemples par classe possibles sans toucher au reste.
#
# L'ORDRE FAIT PARTIE DU PREFIXE LITTERAL, DONC DU CACHE
# ------------------------------------------------------
# Il est totalement determine : rang de la classe dans `CLASS_ORDER`, puis
# `complaint_id` a l'interieur d'une classe. Aucune dependance a l'ordre des
# lignes du CSV ni au systeme de fichiers. Pour un jeu a UN exemple par classe
# -- le cas de v6 -- cet ordre est identique a celui de la version precedente,
# et le prefixe de v6 reste inchange A L'OCTET PRES.
#
# MEMOISATION PAR CHEMIN
# ----------------------
# Un cache global unique rendait les exemples de v6 a un appel demandant ceux de
# v7. Le cache est desormais indexe par chemin resolu.
_CACHE_EXEMPLES: dict[Path, tuple[tuple[str, str], ...]] = {}

# Fichier d'exemples de chaque style few-shot.
_FICHIER_EXEMPLES = {
    "v6_fewshot": cfg.LLM_FEWSHOT_FILE,
    "v7_fewshot_court": cfg.LLM_FEWSHOT_V7_FILE,
    "v8_fewshot_filtre": cfg.LLM_FEWSHOT_V8_FILE,
}

# Styles dont les exemples sont soumis au plafond de longueur.
# v6 en est ABSENT a dessein : ses exemples ont ete tires sans plafond, ils sont
# mesures, et son prefixe est GELE. Lui appliquer la garde le ferait echouer au
# chargement et invaliderait retroactivement une campagne publiee.
_STYLES_PLAFONNES = frozenset({"v7_fewshot_court", "v8_fewshot_filtre"})


def charge_exemples_fewshot(chemin=None, *,
                            plafond_mots: int | None = None
                            ) -> list[tuple[str, str]]:
    """Charge les exemples few-shot d'un CSV. Rend une liste (classe, texte).

    Ces exemples viennent du TRAIN et sont EXCLUS des jeux de selection : un
    exemple present dans les deux donnerait a la variante une reponse deja vue,
    et son gain mesure serait un artefact.

    Parameters
    ----------
    chemin : CSV a charger. Par defaut `cfg.LLM_FEWSHOT_FILE` (v6).
    plafond_mots : si renseigne, LEVE des qu'un exemple depasse ce nombre de
        mots. La garde est deliberement une exception et non une troncature :
        `tronque()` ne s'applique JAMAIS au prefixe, un exemple trop long
        gonflerait donc le prompt de TOUS les appels de la campagne. Mieux vaut
        un echec au chargement qu'une facture decouverte apres coup.

    Raises
    ------
    ValueError si un exemple depasse `plafond_mots`, ou si le CSV est vide.
    """
    import csv
    chemin = Path(chemin or cfg.LLM_FEWSHOT_FILE)
    cle = chemin.resolve()
    if cle in _CACHE_EXEMPLES:
        return list(_CACHE_EXEMPLES[cle])

    with open(chemin, encoding="utf-8") as f:
        lignes = list(csv.DictReader(f))
    if not lignes:
        raise ValueError(f"{chemin} ne contient aucun exemple.")

    inconnues = {l[cfg.LABEL_COL] for l in lignes} - set(cfg.CLASS_ORDER)
    if inconnues:
        raise ValueError(f"{chemin} : classe(s) hors CLASS_ORDER {sorted(inconnues)}.")

    if plafond_mots is not None:
        trop = [(l[cfg.ID_COL], len(l[cfg.TEXT_COL].split()))
                for l in lignes if len(l[cfg.TEXT_COL].split()) > plafond_mots]
        if trop:
            raise ValueError(
                f"{chemin.name} : {len(trop)} exemple(s) au-dessus du plafond de "
                f"{plafond_mots} mots -- {trop}. Le plafond porte sur le PREFIXE, "
                f"que `tronque()` ne raccourcit jamais : un exemple trop long "
                f"alourdirait chaque appel de la campagne."
            )

    rang = {c: i for i, c in enumerate(cfg.CLASS_ORDER)}
    lignes.sort(key=lambda l: (rang[l[cfg.LABEL_COL]], str(l[cfg.ID_COL])))
    exemples = tuple((l[cfg.LABEL_COL], l[cfg.TEXT_COL]) for l in lignes)
    _CACHE_EXEMPLES[cle] = exemples
    return list(exemples)


# --- correspondance style -> referentiel d'etiquettes -----------------------
# v4 interroge le modele avec les libelles CFPB OFFICIELS. Ses reponses doivent
# etre REMAPPEES vers les libelles courts avant toute metrique, sans quoi tout
# partirait en `label_inconnu` et la variante serait mesuree a zero.
_LABELS_OFFICIELS = tuple(cfg.LABELS_CFPB_OFFICIAL[c] for c in cfg.CLASS_ORDER)
_OFFICIEL_VERS_COURT = {cfg.LABELS_CFPB_OFFICIAL[c]: c for c in cfg.CLASS_ORDER}


def labels_du_style(style: str) -> tuple[str, ...]:
    """Referentiel presente au modele pour ce style."""
    return _LABELS_OFFICIELS if style == "v4_libelles_officiels" \
        else tuple(cfg.CLASS_ORDER)


def remappe_vers_court(label: str, style: str) -> str:
    """Ramene une etiquette au referentiel court fige de `CLASS_ORDER`.

    Sans identite pour tous les styles sauf v4. `PARSE_ERROR` traverse inchange.
    """
    if style != "v4_libelles_officiels":
        return label
    return _OFFICIEL_VERS_COURT.get(label, label)


# Styles connus. La cle est le nom passe partout ailleurs (runner, nom de
# fichier de sortie, colonne des rapports).
_INSTRUCTIONS = {
    "v1_zeroshot": _INSTRUCTION_V1_EN,
    "v2_regle_produit": _INSTRUCTION_V1_EN + _REGLE_PRODUIT,
    "v3_definitions": _INSTRUCTION_V1_EN,
    "v4_libelles_officiels": _INSTRUCTION_V1_EN,
    "v5_francais": _INSTRUCTION_V1_FR,
    "v6_fewshot": _INSTRUCTION_V1_EN,
    # v7 et v8 : MEME instruction que v1 et v6. Seuls les EXEMPLES changent --
    # c'est ce qui rend l'ecart entre les trois few-shot attribuable.
    "v7_fewshot_court": _INSTRUCTION_V1_EN,
    "v8_fewshot_filtre": _INSTRUCTION_V1_EN,
    # conserve : ancien nom de la variante francaise, utilise par les tests
    "v1_zeroshot_fr": _INSTRUCTION_V1_FR,
}

STYLES_CONNUS = frozenset(_INSTRUCTIONS)


# ===========================================================================
# Prefixe cacheable
# ===========================================================================
@lru_cache(maxsize=None)
def prefixe_fige(style: str = "v1_zeroshot",
                 labels: tuple[str, ...] = tuple(cfg.CLASS_ORDER)) -> str:
    """Blocs 1 et 2 du prompt : instruction + liste des libelles.

    Memoisee : deux appels avec les memes arguments rendent le MEME objet, donc
    une chaine identique a l'octet pres. C'est ce que le cache de prefixe du
    fournisseur exige.

    `labels` est un tuple (hachable) et non une liste, contrainte de `lru_cache`
    qui a l'avantage de rendre l'immuabilite explicite.
    """
    if style not in _INSTRUCTIONS:
        raise KeyError(
            f"style '{style}' inconnu. Styles disponibles : {sorted(STYLES_CONNUS)}"
        )
    if not labels:
        raise ValueError("la liste des libelles ne peut pas etre vide.")

    if style == "v3_definitions":
        lignes = "\n".join(f"- {lab} : {_DEFINITIONS[lab]}" for lab in labels)
    else:
        lignes = "\n".join(f"- {lab}" for lab in labels)

    prefixe = f"{_INSTRUCTIONS[style]}\n\nCategories:\n{lignes}"

    if style in _FICHIER_EXEMPLES:
        # Exemples APRES les libelles, AVANT le texte : constants, donc dans la
        # partie cacheable du prompt. Le gabarit d'un bloc est IDENTIQUE pour
        # v6, v7 et v8 : seul le contenu des exemples distingue les trois.
        exemples = charge_exemples_fewshot(
            _FICHIER_EXEMPLES[style],
            plafond_mots=(cfg.LLM_FEWSHOT2_MAX_WORDS
                          if style in _STYLES_PLAFONNES else None))
        blocs = "\n\n".join(
            f"Complaint: {t}\nAnswer: {{\"label\": \"{c}\"}}"
            for c, t in exemples)
        prefixe += f"\n\nExamples:\n\n{blocs}"
    return prefixe


def tronque(texte: str, max_mots: int = cfg.LLM_MAX_WORDS) -> tuple[str, bool]:
    """Tronque a `max_mots` mots. Retourne (texte, a_ete_tronque).

    La troncature est appliquee ICI, a la construction du prompt, et JAMAIS dans
    les donnees (arbitrage H.3) : `data/processed/` reste le corpus integral,
    utilisable tel quel par l'etape 3. Elle ne touche que 0,82 % des textes mais
    plafonne le pire cas a ~1 300 tokens au lieu de 8 200.
    """
    mots = texte.split()
    if len(mots) <= max_mots:
        return texte, False
    return " ".join(mots[:max_mots]), True


def construit_prompt(texte: str,
                     labels: list[str] | tuple[str, ...] | None = None,
                     style: str = "v1_zeroshot") -> tuple[list[dict], bool]:
    """Construit les messages d'un appel. Retourne (messages, a_ete_tronque).

    Le decoupage system / user n'est pas cosmetique : il place le prefixe
    constant (blocs 1 et 2) dans le message `system`, et le SEUL contenu
    variable (bloc 3, le texte) dans le message `user`, en dernier. C'est la
    disposition qui rend le prefixe cacheable.

    Le drapeau de troncature est remonte a l'appelant pour etre journalise :
    savoir quelles lignes ont ete raccourcies fait partie de l'audit.
    """
    if not isinstance(texte, str):
        raise TypeError(f"texte doit etre une chaine, recu {type(texte).__name__}.")

    labels = tuple(labels) if labels is not None else tuple(cfg.CLASS_ORDER)
    corps, tronque_ = tronque(texte)

    messages = [
        {"role": "system", "content": prefixe_fige(style, labels)},
        {"role": "user", "content": corps},
    ]
    return messages, tronque_


# ===========================================================================
# Parsing STRICT
# ===========================================================================
# Les quatre statuts possibles. Figes : le runner les compte, les rapports les
# ventilent, et un statut ajoute plus tard invaliderait les comparaisons.
STATUTS = ("ok", "json_invalide", "label_inconnu", "vide")


def parse_reponse(brut: str | None,
                  labels: list[str] | tuple[str, ...] | None = None
                  ) -> tuple[str, str]:
    """Extrait l'etiquette d'une reponse brute. Retourne (label, statut).

    Retourne `config.PARSE_ERROR` comme etiquette des que le statut n'est pas
    "ok" : l'appelant n'a donc jamais a verifier le statut pour eviter une
    etiquette invalide, mais il DOIT le journaliser pour que le taux d'echec de
    format soit mesurable.

    Regles, appliquees dans l'ordre :
      - vide            : reponse absente, vide ou uniquement des blancs
      - json_invalide   : ce n'est pas du JSON, ou pas un objet, ou pas de cle
                          "label", ou "label" n'est pas une chaine. Un bloc
                          ```json ... ``` tombe ICI : l'instruction interdit
                          explicitement les fences, les emettre est une
                          non-conformite au format.
      - label_inconnu   : JSON valide, mais la valeur n'est pas un des libelles.
                          Une casse differente tombe ICI : le referentiel est
                          fourni verbatim dans le prompt, ne pas le recopier a
                          l'identique est une non-conformite.
      - ok              : la valeur est un libelle du referentiel.

    SEULE tolerance : les blancs qui entourent la reponse et la valeur sont
    retires. Ce n'est pas une normalisation semantique, juste du bruit de
    serialisation, et l'accepter n'avantage aucune approche.

    Une cle supplementaire dans l'objet JSON n'invalide PAS la reponse : la
    classification, elle, a bien eu lieu. La verbosite est journalisee par le
    compte de tokens de sortie, pas par le statut de parsing.
    """
    labels = tuple(labels) if labels is not None else tuple(cfg.CLASS_ORDER)

    if brut is None or not brut.strip():
        return cfg.PARSE_ERROR, "vide"

    try:
        objet = json.loads(brut.strip())
    except (json.JSONDecodeError, ValueError):
        return cfg.PARSE_ERROR, "json_invalide"

    if not isinstance(objet, dict) or "label" not in objet:
        return cfg.PARSE_ERROR, "json_invalide"

    valeur = objet["label"]
    if not isinstance(valeur, str):
        return cfg.PARSE_ERROR, "json_invalide"

    valeur = valeur.strip()
    if valeur in labels:
        return valeur, "ok"
    return cfg.PARSE_ERROR, "label_inconnu"


# ===========================================================================
# Re-parsing TOLERANT -- analyse hors ligne uniquement
# ===========================================================================
_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL | re.IGNORECASE)


def reparse_tolerant(brut: str | None,
                     labels: list[str] | tuple[str, ...] | None = None
                     ) -> tuple[str, str]:
    """Re-parsing indulgent. NE JAMAIS appeler pendant une campagne.

    Existe pour une seule raison : chiffrer ce que le parsing strict a coute.
    Les reponses brutes etant toutes journalisees, cette fonction se rejoue sur
    un JSONL deja ecrit, sans reappel ni surcout, et repond a la question
    "combien d'erreurs etaient de simples ecarts de format ?".

    Si l'ecart mesure est significatif, c'est un ARGUMENT A DOCUMENTER, pas une
    autorisation a assouplir le parsing retroactivement : le score officiel
    reste celui du parsing strict, sans quoi le LLM serait mesure avec une
    indulgence dont le ML ne beneficie pas.

    Retourne (label, methode) ou methode dit ce qui a ete necessaire :
    "strict", "fence", "casse", "sous_chaine", ou "echec".
    """
    labels = tuple(labels) if labels is not None else tuple(cfg.CLASS_ORDER)

    label, statut = parse_reponse(brut, labels)
    if statut == "ok":
        return label, "strict"
    if brut is None or not brut.strip():
        return cfg.PARSE_ERROR, "echec"

    texte = brut.strip()

    # 1. bloc de code markdown : on retire les fences et on rejoue le strict
    m = _FENCE.match(texte)
    if m:
        label, statut = parse_reponse(m.group(1), labels)
        if statut == "ok":
            return label, "fence"
        texte = m.group(1).strip()

    # 2. casse ou blancs internes differents, sur du JSON valide
    try:
        objet = json.loads(texte)
        if isinstance(objet, dict) and isinstance(objet.get("label"), str):
            candidat = " ".join(objet["label"].split()).casefold()
            for lab in labels:
                if " ".join(lab.split()).casefold() == candidat:
                    return lab, "casse"
    except (json.JSONDecodeError, ValueError):
        pass

    # 3. libelle noye dans une phrase : le plus long libelle present gagne, pour
    #    qu'un libelle inclus dans un autre ne vole pas la correspondance.
    trouves = [lab for lab in labels if lab.casefold() in texte.casefold()]
    if trouves:
        return max(trouves, key=len), "sous_chaine"

    return cfg.PARSE_ERROR, "echec"
