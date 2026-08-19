"""Coherence du referentiel de configuration de l'etape 2.

AUCUN APPEL API, AUCUN RESEAU. Ce fichier ne verifie pas que les tarifs sont
JUSTES -- seul un releve chez le fournisseur peut le dire, et il est date par
`PRICING_CHECKED_ON`. Il verifie que le referentiel est COHERENT AVEC LUI-MEME,
ce qui attrape la categorie d'erreur qui vient de se produire deux fois :

  - un champ documentaire faux (`cache_automatique` a False pour Mistral, dont
    le cache est pourtant automatique). Un champ documentaire faux est plus
    dangereux qu'un champ absent : il est cru sans etre reverifie.
  - un tarif recopie ailleurs, qui diverge (`.env.example`, huit jours durant).

Et il fige le constat central de l'etape 2 : les cles de `MODELS_PRICING` ne
sont PAS des identifiants d'API.

    python tests/test_llm_config.py
"""

import datetime as dt
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parent
while not (RACINE / "src" / "config.py").exists() and RACINE != RACINE.parent:
    RACINE = RACINE.parent
sys.path.insert(0, str(RACINE))

from src import config as cfg   # noqa: E402

OK, KO = "  OK  ", " ECHEC"
_RESULTATS: list[bool] = []


def check(nom, cond, detail=""):
    _RESULTATS.append(bool(cond))
    print(f"[{OK if cond else KO}] {nom}" + (f"   -> {detail}" if detail else ""))


def main() -> int:
    _RESULTATS.clear()

    print("=" * 78)
    print("MODELS_PRICING EST UNE GRILLE TARIFAIRE, PAS UN REGISTRE D'IDENTIFIANTS")
    print("=" * 78)
    cles_tarif = set(cfg.MODELS_PRICING)
    ids_api = {i["id_api_candidat"] for i in cfg.LLM_MODELES_A_COMPARER.values()}
    check("aucun identifiant d'API n'est aussi une cle de tarification",
          not (cles_tarif & ids_api),
          f"intersection : {sorted(cles_tarif & ids_api) or 'vide'}")
    check("chaque modele compare renvoie a une cle tarifaire existante",
          set(cfg.LLM_MODELES_A_COMPARER).issubset(cles_tarif),
          f"{sorted(set(cfg.LLM_MODELES_A_COMPARER) - cles_tarif) or 'toutes presentes'}")
    check("la note d'avertissement est presente en tete de MODELS_PRICING",
          "PAS UN REGISTRE" in (RACINE / "src" / "config.py").read_text())

    print("\n" + "=" * 78)
    print("IDENTIFIANTS VERIFIES CONTRE L'API")
    print("=" * 78)
    for cle, info in cfg.LLM_MODELES_A_COMPARER.items():
        check(f"{cle} : identifiant verifie et date",
              info.get("verifie") is True and info.get("verifie_le"),
              f"{info['id_api_candidat']} le {info.get('verifie_le')}")
        check(f"{cle} : version resolue consignee",
              bool(info.get("version_resolue"))
              and info["version_resolue"] != info["id_api_candidat"],
              f"{info.get('version_resolue')}")

    print("\n" + "=" * 78)
    print("COHERENCE DU CACHE POUR TOUS LES FOURNISSEURS")
    print("=" * 78)
    for nom, t in cfg.MODELS_PRICING.items():
        check(f"{nom} : cache_automatique est un booleen",
              isinstance(t.get("cache_automatique"), bool),
              f"{t.get('cache_automatique')!r}")
        r = t.get("remise_cache")
        check(f"{nom} : remise_cache est None ou dans [0, 1]",
              r is None or (isinstance(r, (int, float)) and 0.0 <= r <= 1.0),
              f"{r!r}")
        # Une remise non verifiee interdit d'affirmer un cache automatique :
        # on ne peut pas savoir qu'un fournisseur cache seul sans avoir
        # verifie qu'il applique une remise.
        check(f"{nom} : pas de cache automatique sans remise verifiee",
              not (r is None and t.get("cache_automatique") is True))

    print("\n  -- invariant par fournisseur --")
    par_fournisseur: dict[str, list] = {}
    for nom, t in cfg.MODELS_PRICING.items():
        par_fournisseur.setdefault(t["fournisseur"], []).append((nom, t))
    for f, modeles in sorted(par_fournisseur.items()):
        autos = {t["cache_automatique"] for _, t in modeles}
        remises = {t["remise_cache"] for _, t in modeles}
        # Le comportement de cache est une propriete du FOURNISSEUR, pas du
        # modele. Deux modeles d'un meme fournisseur qui divergent signalent
        # une saisie partielle -- exactement le defaut corrige le 2026-08-19,
        # ou seul un des trois modeles Mistral aurait pu etre mis a jour.
        check(f"{f} : cache_automatique identique sur ses {len(modeles)} modele(s)",
              len(autos) == 1, f"{autos}")
        check(f"{f} : remise_cache identique sur ses {len(modeles)} modele(s)",
              len(remises) == 1, f"{remises}")

    print("\n  -- valeurs verifiees chez le fournisseur --")
    check("Mistral : cache automatique (verifie 2026-08-19, docs Mistral)",
          all(t["cache_automatique"] is True for _, t in par_fournisseur["Mistral"]))
    check("Mistral : remise de cache a 90 %",
          all(t["remise_cache"] == 0.90 for _, t in par_fournisseur["Mistral"]))
    check("DeepSeek : cache automatique",
          cfg.MODELS_PRICING["deepseek-v4-flash"]["cache_automatique"] is True)
    check("Google : remise non verifiee, donc non modelisee",
          cfg.MODELS_PRICING["gemini-3.5-flash"]["remise_cache"] is None)

    print("\n" + "=" * 78)
    print("TARIFS DES DEUX MODELES DE L'ETAPE 2")
    print("=" * 78)
    small = cfg.MODELS_PRICING["mistral-small-4"]
    mini = cfg.MODELS_PRICING["ministral-3b"]
    check("Mistral Small 4 : 0,15 / 0,60 $ par M de tokens",
          (small["input_per_1m"], small["output_per_1m"]) == (0.15, 0.60),
          f"{small['input_per_1m']} / {small['output_per_1m']}")
    check("Ministral 3 3B : 0,10 / 0,10 $ par M de tokens",
          (mini["input_per_1m"], mini["output_per_1m"]) == (0.10, 0.10),
          f"{mini['input_per_1m']} / {mini['output_per_1m']}")
    check("aucun tarif promotionnel sur les deux modeles compares",
          small["tarif_provisoire_jusquau"] is None
          and mini["tarif_provisoire_jusquau"] is None)

    releve = dt.date.fromisoformat(cfg.PRICING_CHECKED_ON)
    check("la date de releve tarifaire est valide et pas dans le futur",
          releve <= dt.date.today(), f"{cfg.PRICING_CHECKED_ON}")

    print("\n" + "=" * 78)
    print("SOURCE UNIQUE DE VERITE TARIFAIRE")
    print("=" * 78)
    env = (RACINE / ".env.example").read_text()
    check(".env.example ne contient AUCUN chiffre tarifaire",
          not any(x in env for x in ("0,10 $", "0,30 $", "0.10 $", "$ / M",
                                     "$/M", "per_1m")),
          "un tarif recopie finit toujours par diverger")
    check(".env.example renvoie a MODELS_PRICING",
          "MODELS_PRICING" in env)

    print("\n" + "=" * 78)
    print("PARAMETRES D'APPEL")
    print("=" * 78)
    check("temperature a 0", cfg.LLM_TEMPERATURE == 0.0)
    check("max_tokens a 50", cfg.LLM_MAX_TOKENS == 50)
    check("PARSE_ERROR hors de CLASS_ORDER", cfg.PARSE_ERROR not in cfg.CLASS_ORDER)
    check("le prefixe cacheable depasse largement le bloc de 64 tokens",
          cfg.CACHED_PREFIX_TOKENS >= 64 * 2,
          f"{cfg.CACHED_PREFIX_TOKENS} tokens, soit "
          f"{cfg.CACHED_PREFIX_TOKENS / 64:.1f} blocs")
    check("la granularite de 64 tokens est documentee",
          "64 TOKENS" in (RACINE / "src" / "config.py").read_text())

    print("\n" + "=" * 78)
    print(f"RESULTAT : {sum(_RESULTATS)}/{len(_RESULTATS)} verifications passees")
    print("=" * 78)
    return 0 if all(_RESULTATS) else 1


if __name__ == "__main__":
    sys.exit(main())
