"""Verifications du prompt et du parsing LLM (etape 2).

AUCUN APPEL API. Tout est verifie sur des cas construits a la main : ce fichier
doit tourner hors ligne, sans cle, et sans depenser un centime.

Ce qu'il prouve, dans l'ordre d'importance :

  1. Le parsing STRICT classe bien chaque forme de non-conformite dans le bon
     statut, et ne rend JAMAIS une etiquette hors referentiel. Un parsing qui
     rattraperait silencieusement une reponse mal formee gonflerait le score du
     LLM d'une indulgence dont le ML de l'etape 3 ne beneficie pas.
  2. `PARSE_ERROR` n'entre jamais dans `CLASS_ORDER`.
  3. Le prefixe du prompt est identique A L'OCTET PRES d'un appel a l'autre.
     Sans cela le cache de prefixe ne s'active pas, `cached_tokens` reste a
     zero, et la projection de -42,6 % du notebook devient invérifiable.
  4. Le texte de la reclamation vient EN DERNIER, apres l'instruction et les
     libelles.
  5. Le re-parsing tolerant recupere bien ce que le strict rejette -- il
     mesure le cout de la severite, il ne l'annule pas.

    python tests/test_llm_prompts.py

Sortie : une ligne par verification, code de retour non nul si l'une echoue.

IMPORTABLE SANS EFFET DE BORD : tout le travail est dans `main()`.
"""

import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parent
while not (RACINE / "src" / "config.py").exists() and RACINE != RACINE.parent:
    RACINE = RACINE.parent
sys.path.insert(0, str(RACINE))

from src import config as cfg          # noqa: E402
from src import llm_prompts as prompts  # noqa: E402

OK, KO = "  OK  ", " ECHEC"
_RESULTATS: list[bool] = []


def check(nom, cond, detail=""):
    """Enregistre et affiche le resultat d'une verification."""
    _RESULTATS.append(bool(cond))
    print(f"[{OK if cond else KO}] {nom}" + (f"   -> {detail}" if detail else ""))


def cas(nom, brut, label_attendu, statut_attendu):
    """Verifie parse_reponse() sur un cas construit a la main."""
    label, statut = prompts.parse_reponse(brut)
    check(nom, label == label_attendu and statut == statut_attendu,
          f"rendu ({label!r}, {statut!r}), attendu "
          f"({label_attendu!r}, {statut_attendu!r})")


def main() -> int:
    """Execute toutes les verifications. Retourne 0 si tout passe, 1 sinon."""
    _RESULTATS.clear()
    PE = cfg.PARSE_ERROR

    print("=" * 78)
    print("REFERENTIEL")
    print("=" * 78)
    check("PARSE_ERROR absent de CLASS_ORDER", PE not in cfg.CLASS_ORDER)
    check("CLASS_ORDER compte toujours 9 classes", len(cfg.CLASS_ORDER) == 9,
          f"n={len(cfg.CLASS_ORDER)}")
    check("les 4 statuts de parsing sont figes",
          prompts.STATUTS == ("ok", "json_invalide", "label_inconnu", "vide"))

    print("\n" + "=" * 78)
    print("PARSING STRICT -- CAS CONFORMES")
    print("=" * 78)
    cas("JSON valide, libelle exact",
        '{"label": "Mortgage"}', "Mortgage", "ok")
    cas("JSON valide, libelle a virgule",
        '{"label": "Payday, title or personal loan"}',
        "Payday, title or personal loan", "ok")
    cas("blancs parasites autour de la reponse",
        '   \n  {"label": "Debt collection"}  \n\t ', "Debt collection", "ok")
    cas("blancs parasites dans la valeur",
        '{"label": "  Student loan  "}', "Student loan", "ok")
    cas("cle supplementaire toleree (la classification a eu lieu)",
        '{"label": "Mortgage", "confidence": 0.9}', "Mortgage", "ok")

    print("\n" + "=" * 78)
    print("PARSING STRICT -- NON-CONFORMITES")
    print("=" * 78)
    cas("reponse vide", "", PE, "vide")
    cas("reponse blancs seuls", "   \n\t  ", PE, "vide")
    cas("reponse None", None, PE, "vide")
    cas("JSON malforme (accolade manquante)",
        '{"label": "Mortgage"', PE, "json_invalide")
    cas("JSON malforme (guillemets simples)",
        "{'label': 'Mortgage'}", PE, "json_invalide")
    cas("pas de JSON du tout", "Mortgage", PE, "json_invalide")
    cas("JSON valide mais pas un objet",
        '["Mortgage"]', PE, "json_invalide")
    cas("objet JSON sans cle 'label'",
        '{"categorie": "Mortgage"}', PE, "json_invalide")
    cas("valeur de 'label' non textuelle",
        '{"label": 3}', PE, "json_invalide")
    cas("label noye dans une phrase",
        'The complaint is about {"label": "Mortgage"} I think.',
        PE, "json_invalide")
    cas("reponse entouree de ```json (fences interdites par l'instruction)",
        '```json\n{"label": "Mortgage"}\n```', PE, "json_invalide")
    cas("label invente", '{"label": "Car insurance"}', PE, "label_inconnu")
    cas("casse differente (le referentiel est fourni verbatim)",
        '{"label": "mortgage"}', PE, "label_inconnu")
    cas("libelle CFPB officiel au lieu du libelle court",
        '{"label": "Checking or savings account"}', PE, "label_inconnu")
    cas("deux libelles concatenes",
        '{"label": "Mortgage / Debt collection"}', PE, "label_inconnu")

    print("\n" + "=" * 78)
    print("LE PARSING NE REND JAMAIS D'ETIQUETTE HORS REFERENTIEL")
    print("=" * 78)
    formes = ['{"label": "Mortgage"}', "", "   ", None, '{"label": "X"}',
              "```json\n{}\n```", "[]", '{"label": null}', "n'importe quoi",
              '{"label": "MORTGAGE"}']
    valides = set(cfg.CLASS_ORDER) | {PE}
    check("toute reponse rend soit une des 9 classes, soit PARSE_ERROR",
          all(prompts.parse_reponse(f)[0] in valides for f in formes))
    check("tout statut rendu appartient aux 4 statuts figes",
          all(prompts.parse_reponse(f)[1] in prompts.STATUTS for f in formes))

    print("\n" + "=" * 78)
    print("STRUCTURE DU PROMPT ET CACHE DE PREFIXE")
    print("=" * 78)
    m1, _ = prompts.construit_prompt("Texte A, une reclamation.")
    m2, _ = prompts.construit_prompt("Texte B, tout autre, plus long. XXXX.")

    check("deux messages : system puis user",
          [m["role"] for m in m1] == ["system", "user"])
    check("prefixe identique A L'OCTET PRES entre deux appels",
          m1[0]["content"] == m2[0]["content"])
    check("le prefixe est le MEME objet (memoisation)",
          prompts.prefixe_fige() is prompts.prefixe_fige())
    check("le texte de la reclamation est en DERNIER",
          m1[-1]["content"] == "Texte A, une reclamation.")
    check("le texte n'apparait pas dans le prefixe",
          "Texte A" not in m1[0]["content"])

    prefixe = m1[0]["content"]
    check("le prefixe contient les 9 libelles",
          all(lab in prefixe for lab in cfg.CLASS_ORDER))
    check("l'instruction mentionne le masquage XXXX",
          "XXXX" in prefixe)
    check("l'instruction impose le format JSON attendu",
          '{"label":' in prefixe.replace('{"label": ', '{"label":'))
    check("l'instruction interdit les fences markdown",
          "code fence" in prefixe.lower())
    check("aucune date ni horodatage interpole dans le prefixe",
          not any(j in prefixe for j in ("2026", "2025", "T00:", "UTC")))

    ordre_instruction = prefixe.index("classification system")
    ordre_labels = prefixe.index("Categories:")
    check("ordre respecte : instruction, puis libelles",
          ordre_instruction < ordre_labels)

    print("\n" + "=" * 78)
    print("TRONCATURE (a la construction du prompt, jamais dans les donnees)")
    print("=" * 78)
    court = "mot " * 10
    long_ = "mot " * (cfg.LLM_MAX_WORDS + 500)
    t_court, drapeau_court = prompts.tronque(court)
    t_long, drapeau_long = prompts.tronque(long_)
    check("texte court laisse strictement intact",
          t_court == court and drapeau_court is False)
    check(f"texte long ramene a {cfg.LLM_MAX_WORDS} mots",
          len(t_long.split()) == cfg.LLM_MAX_WORDS and drapeau_long is True,
          f"{len(t_long.split())} mots")
    _, drapeau = prompts.construit_prompt(long_)
    check("construit_prompt remonte le drapeau de troncature", drapeau is True)
    check("le seuil exact n'est pas tronque",
          prompts.tronque("mot " * cfg.LLM_MAX_WORDS)[1] is False)

    print("\n" + "=" * 78)
    print("STYLES")
    print("=" * 78)
    check("style inconnu leve KeyError",
          _leve(KeyError, prompts.construit_prompt, "t", None, "style_bidon"))
    check("la variante francaise existe et differe de l'anglaise",
          prompts.prefixe_fige("v1_zeroshot_fr") != prompts.prefixe_fige("v1_zeroshot"))
    check("v1_zeroshot est le seul style fige (phase 3 close)",
          cfg.LLM_STYLES_FIGES == {"v1_zeroshot"},
          "le runner n'acceptera l'echantillon d'evaluation que pour v1")
    check("aucune variante de la phase 3 n'est figee",
          not (cfg.LLM_STYLES_FIGES & {"v2_regle_produit", "v3_definitions",
                                       "v4_libelles_officiels", "v5_francais",
                                       "v6_fewshot"}))

    print("\n" + "=" * 78)
    print("VARIANTES DE LA PHASE 3")
    print("=" * 78)
    import hashlib
    # v1 a DEJA ETE MESUREE : son prefixe ne doit pas bouger d'un octet, sinon
    # la comparaison des variantes contre elle ne vaut plus rien.
    EMPREINTE_V1 = "16007502179e43f4"
    check("le prefixe v1 est inchange (empreinte figee)",
          hashlib.sha256(prompts.prefixe_fige("v1_zeroshot").encode())
          .hexdigest()[:16] == EMPREINTE_V1,
          "toute derive de v1 invaliderait les mesures de la phase 2")

    VARIANTES = ["v1_zeroshot", "v2_regle_produit", "v3_definitions",
                 "v4_libelles_officiels", "v5_francais", "v6_fewshot"]
    check("les 6 variantes sont declarees",
          all(v in prompts.STYLES_CONNUS for v in VARIANTES))

    prefixes = {v: prompts.prefixe_fige(v, prompts.labels_du_style(v))
                for v in VARIANTES}
    check("les 6 prefixes sont deux a deux distincts",
          len(set(prefixes.values())) == 6)
    for v in VARIANTES:
        check(f"{v} : prefixe stable a l'octet pres",
              prompts.prefixe_fige(v, prompts.labels_du_style(v)) is prefixes[v])

    v1p = prefixes["v1_zeroshot"]
    check("v2 = v1 + la regle produit, rien d'autre",
          "FINANCIAL PRODUCT" in prefixes["v2_regle_produit"]
          and prefixes["v2_regle_produit"].startswith(v1p.split("\n\nCategories:")[0]))
    check("v3 porte une definition par etiquette",
          all(f"- {c} :" in prefixes["v3_definitions"] for c in cfg.CLASS_ORDER))
    check("v4 presente les libelles CFPB officiels",
          all(cfg.LABELS_CFPB_OFFICIAL[c] in prefixes["v4_libelles_officiels"]
              for c in cfg.CLASS_ORDER))
    check("v4 n'utilise PAS les libelles courts pour les classes renommees",
          "- Bank account or service" not in prefixes["v4_libelles_officiels"])
    check("v5 est en francais", "categories de produit"
          in prefixes["v5_francais"])
    check("v6 porte 9 exemples, un par classe",
          prefixes["v6_fewshot"].count("Complaint:") == 9
          and prefixes["v6_fewshot"].count("Answer:") == 9)
    check("les exemples de v6 sont AVANT le texte, donc cacheables",
          "Examples:" in prefixes["v6_fewshot"])

    print("\n  -- remapping de v4 vers le referentiel court --")
    for court, officiel in cfg.LABELS_CFPB_OFFICIAL.items():
        lab, statut = prompts.parse_reponse(
            '{"label": "%s"}' % officiel, prompts.labels_du_style("v4_libelles_officiels"))
        remap = prompts.remappe_vers_court(lab, "v4_libelles_officiels")
        check(f"v4 · {officiel[:38]:38s} -> {court}",
              statut == "ok" and remap == court, f"{remap!r}")
    check("un libelle COURT est refuse par v4 (le referentiel a change)",
          prompts.parse_reponse('{"label": "Bank account or service"}',
                                prompts.labels_du_style("v4_libelles_officiels"))[1]
          == "label_inconnu")
    check("PARSE_ERROR traverse le remapping inchange",
          prompts.remappe_vers_court(cfg.PARSE_ERROR, "v4_libelles_officiels")
          == cfg.PARSE_ERROR)
    check("le remapping est l'identite pour les autres styles",
          all(prompts.remappe_vers_court(c, "v1_zeroshot") == c
              for c in cfg.CLASS_ORDER))

    print("\n  -- surcout de tokens des variantes (estimation par caracteres) --")
    for v in VARIANTES:
        n = len(prefixes[v])
        print(f"      {v:24s} {n:6d} caracteres  "
              f"(x{n/len(v1p):.1f} vs v1)")
    check("v6 porte un surcout de prefixe majeur, a chiffrer",
          len(prefixes["v6_fewshot"]) > 5 * len(v1p),
          f"x{len(prefixes['v6_fewshot'])/len(v1p):.1f} — pese dans la regle de decision")

    print("\n" + "=" * 78)
    print("RE-PARSING TOLERANT (analyse hors ligne, jamais en campagne)")
    print("=" * 78)
    for nom, brut, attendu, methode in [
        ("conforme -> 'strict'", '{"label": "Mortgage"}', "Mortgage", "strict"),
        ("fences retirees", '```json\n{"label": "Mortgage"}\n```', "Mortgage", "fence"),
        ("casse rattrapee", '{"label": "mortgage"}', "Mortgage", "casse"),
        ("libelle noye rattrape", 'I would say Mortgage.', "Mortgage", "sous_chaine"),
        ("vraiment irrecuperable", "aucune idee", cfg.PARSE_ERROR, "echec"),
    ]:
        label, meth = prompts.reparse_tolerant(brut)
        check(nom, label == attendu and meth == methode,
              f"rendu ({label!r}, {meth!r})")

    check("le plus long libelle gagne quand deux sont inclus l'un dans l'autre",
          prompts.reparse_tolerant("about a Credit card or prepaid card")[0]
          == "Credit card or prepaid card")
    check("le tolerant ne rend jamais d'etiquette hors referentiel",
          all(prompts.reparse_tolerant(f)[0] in valides for f in formes))

    print("\n" + "=" * 78)
    print(f"RESULTAT : {sum(_RESULTATS)}/{len(_RESULTATS)} verifications passees")
    print("=" * 78)
    return 0 if all(_RESULTATS) else 1


def _leve(exc, fn, *args) -> bool:
    """Vrai si `fn(*args)` leve bien `exc`."""
    try:
        fn(*args)
    except exc:
        return True
    except Exception:
        return False
    return False


if __name__ == "__main__":
    sys.exit(main())
