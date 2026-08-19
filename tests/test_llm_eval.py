"""Verifications du calcul des metriques LLM (etape 2).

AUCUN APPEL API : tous les journaux sont construits a la main, ligne par ligne,
avec des valeurs dont on connait le resultat attendu. C'est la seule facon de
prouver qu'une metrique est juste -- sur des donnees reelles on ne saurait pas
distinguer un bon score d'un bug.

Ce que ce fichier prouve :

  1. `llm_eval` DELEGUE a `metrics.py` : les garde-fous de ponderation restent
     actifs, et le F1 rendu est bien celui de `metrics.evaluate()`.
  2. Les identifiants manquants sont DETECTES et signales, jamais ignores.
  3. PARSE_ERROR et erreur HTTP sont comptes SEPAREMENT, et l'un ne contamine
     pas le taux de l'autre.
  4. Les couts et les tokens sont agreges depuis le journal, pas depuis les
     hypotheses de `config`.
  5. Une comparaison entre campagnes portant sur des lignes differentes est
     REFUSEE.

    python tests/test_llm_eval.py
"""

import ast
import json
import sys
import tempfile
from pathlib import Path

import pandas as pd

RACINE = Path(__file__).resolve().parent
while not (RACINE / "src" / "config.py").exists() and RACINE != RACINE.parent:
    RACINE = RACINE.parent
sys.path.insert(0, str(RACINE))

from src import config as cfg      # noqa: E402
from src import llm_eval as ev     # noqa: E402

OK, KO = "  OK  ", " ECHEC"
_RESULTATS: list[bool] = []


def check(nom, cond, detail=""):
    _RESULTATS.append(bool(cond))
    print(f"[{OK if cond else KO}] {nom}" + (f"   -> {detail}" if detail else ""))


def enr(cid, vrai, predit, statut="ok", brut='{"label": "x"}',
        tokens_in=300, tokens_out=10, tokens_caches=260, latence=0.5,
        attentes=1.5, erreur=None, cout=4.8e-05, execution=1,
        modele="mistral-small-latest", resolu="mistral-small-2603"):
    """Un enregistrement de journal, entierement controle."""
    return {
        cfg.ID_COL: cid, "label_vrai": vrai, "label_predit": predit,
        "statut_parsing": statut, "reponse_brute": brut, "style": "v1_zeroshot",
        "execution": execution, "texte_tronque": False, "cout_usd": cout,
        "tokens_in": tokens_in, "tokens_out": tokens_out,
        "tokens_total": (None if tokens_in is None or tokens_out is None
                         else tokens_in + tokens_out),
        "tokens_caches": tokens_caches,
        "service_tier": "free", "latence_s": latence,
        "latence_avec_attentes_s": attentes, "erreur_http": erreur,
        "tentatives": 1, "modele": modele, "modele_resolu": resolu,
        "horodatage": "2026-01-01T00:00:00+00:00",
    }


def ecrit(chemin, enregistrements):
    with Path(chemin).open("w", encoding="utf-8") as f:
        for e in enregistrements:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    return chemin


def source(n_par_classe=1, avec_poids=False, classes=None):
    classes = classes or cfg.CLASS_ORDER
    lignes = []
    for i, c in enumerate(classes):
        for k in range(n_par_classe):
            lignes.append({cfg.ID_COL: f"id{i}_{k}", cfg.TEXT_COL: "texte XXXX",
                           cfg.LABEL_COL: c})
    df = pd.DataFrame(lignes)
    if avec_poids:
        # Poids volontairement inegaux : sans eux le F1 serait different.
        df[cfg.WEIGHT_COL] = [1.0 if i % 2 == 0 else 50.0 for i in range(len(df))]
        df[cfg.STRATE_COL] = df[cfg.LABEL_COL]
    return df


def main() -> int:
    _RESULTATS.clear()
    tmp = Path(tempfile.mkdtemp(prefix="zen_eval_test_"))

    print("=" * 78)
    print("CAS PARFAIT : 9 CLASSES, 9 BONNES REPONSES")
    print("=" * 78)
    src = source()
    parfait = ecrit(tmp / "parfait.jsonl",
                    [enr(r[cfg.ID_COL], r[cfg.LABEL_COL], r[cfg.LABEL_COL])
                     for _, r in src.iterrows()])
    r = ev.evalue_campagne(parfait, src, with_ci=False, verbeux=False)
    check("F1-macro = 1.0 sur un sans-faute", r["f1_macro"] == 1.0,
          f"{r['f1_macro']}")
    check("n evalue = 9", r["n_evalue"] == 9, f"{r['n_evalue']}")
    check("aucun manquant", r["alignement"]["n_manquants"] == 0)
    check("taux PARSE_ERROR nul", r["taux_parse_error"] == 0.0)
    check("taux erreur HTTP nul", r["taux_erreur_http"] == 0.0)
    check("non pondere quand la source ne porte pas de poids",
          r["weighted"] is False)
    check("resultat exploitable par compare_results (cles attendues)",
          all(k in r for k in ("n", "accuracy", "f1_macro", "f1_weighted",
                               "classification_report", "weighted")))

    print("\n" + "=" * 78)
    print("ALIGNEMENT PAR IDENTIFIANT, PAS PAR POSITION")
    print("=" * 78)
    melange = ecrit(tmp / "melange.jsonl",
                    [enr(r[cfg.ID_COL], r[cfg.LABEL_COL], r[cfg.LABEL_COL])
                     for _, r in src.iloc[::-1].iterrows()])  # ordre inverse
    r_mel = ev.evalue_campagne(melange, src, with_ci=False, verbeux=False)
    check("journal en ordre inverse -> meme F1 (jointure sur l'id)",
          r_mel["f1_macro"] == 1.0, f"{r_mel['f1_macro']}")

    print("\n" + "=" * 78)
    print("IDENTIFIANTS MANQUANTS (appels perdus ou campagne incomplete)")
    print("=" * 78)
    partiel = ecrit(tmp / "partiel.jsonl",
                    [enr(r[cfg.ID_COL], r[cfg.LABEL_COL], r[cfg.LABEL_COL])
                     for _, r in src.iloc[:6].iterrows()])
    r_part = ev.evalue_campagne(partiel, src, with_ci=False, verbeux=False)
    diag = r_part["alignement"]
    check("3 identifiants manquants detectes", diag["n_manquants"] == 3,
          f"{diag['n_manquants']}")
    check("les manquants sont nommes", set(diag["ids_manquants"]) ==
          set(src[cfg.ID_COL].iloc[6:]), f"{diag['ids_manquants']}")
    check("n attendu conserve malgre l'incompletude",
          r_part["n_attendu"] == 9 and r_part["n_evalue"] == 6)

    hors = ecrit(tmp / "hors.jsonl",
                 [enr("inconnu_zz", "Mortgage", "Mortgage")]
                 + [enr(r[cfg.ID_COL], r[cfg.LABEL_COL], r[cfg.LABEL_COL])
                    for _, r in src.iterrows()])
    r_hors = ev.evalue_campagne(hors, src, with_ci=False, verbeux=False)
    check("identifiant hors source signale et ignore",
          r_hors["alignement"]["ids_hors_source"] == ["inconnu_zz"]
          and r_hors["n_evalue"] == 9)

    doublon = ecrit(tmp / "doublon.jsonl",
                    [enr("id0_0", "Credit reporting", "Credit reporting"),
                     enr("id0_0", "Credit reporting", "Mortgage")])
    leve = False
    try:
        ev.evalue_campagne(doublon, src, with_ci=False, verbeux=False)
    except ValueError as e:
        leve = "double" in str(e)
    check("doublon d'identifiant refuse", leve)

    print("\n" + "=" * 78)
    print("PARSE_ERROR ET ERREUR HTTP COMPTES SEPAREMENT")
    print("=" * 78)
    # 9 lignes : 6 ok, 2 PARSE_ERROR (le modele a repondu hors format),
    # 1 erreur HTTP (aucune reponse).
    melanges = []
    for i, (_, row) in enumerate(src.iterrows()):
        if i < 6:
            melanges.append(enr(row[cfg.ID_COL], row[cfg.LABEL_COL], row[cfg.LABEL_COL]))
        elif i < 8:
            melanges.append(enr(row[cfg.ID_COL], row[cfg.LABEL_COL],
                                cfg.PARSE_ERROR, statut="json_invalide",
                                brut="je dirais Mortgage"))
        else:
            melanges.append(enr(row[cfg.ID_COL], row[cfg.LABEL_COL],
                                cfg.PARSE_ERROR, statut="vide", brut=None,
                                erreur="HTTP 500 -- panne", latence=None,
                                cout=None, tokens_in=None, tokens_out=None,
                                tokens_caches=None))
    mix = ecrit(tmp / "mix.jsonl", melanges)
    r_mix = ev.evalue_campagne(mix, src, with_ci=False, verbeux=False)

    check("1 erreur HTTP comptee", r_mix["n_erreurs_http"] == 1,
          f"{r_mix['n_erreurs_http']}")
    check("taux HTTP = 1/9 sur le total des appels",
          abs(r_mix["taux_erreur_http"] - 1 / 9) < 1e-9,
          f"{r_mix['taux_erreur_http']:.4f}")
    check("2 PARSE_ERROR comptes", r_mix["n_parse_error"] == 2,
          f"{r_mix['n_parse_error']}")
    check("taux PARSE_ERROR = 2/8, hors lignes en panne HTTP",
          abs(r_mix["taux_parse_error"] - 2 / 8) < 1e-9,
          f"{r_mix['taux_parse_error']:.4f} (une panne reseau n'est pas un "
          f"defaut de format)")
    check("par defaut les erreurs HTTP sont exclues du F1",
          r_mix["n_evalue"] == 8, f"n_evalue={r_mix['n_evalue']}")
    check("detail des statuts ventile", r_mix["statuts_parsing"] ==
          {"ok": 6, "json_invalide": 2, "vide": 1}, f"{r_mix['statuts_parsing']}")

    # 8 evaluees, 6 justes, 2 PARSE_ERROR -> accuracy 6/8
    check("accuracy = 6/8 : PARSE_ERROR compte bien comme une erreur",
          abs(r_mix["accuracy"] - 6 / 8) < 1e-9, f"{r_mix['accuracy']:.4f}")
    check("PARSE_ERROR absent du rapport par classe",
          cfg.PARSE_ERROR not in r_mix["classification_report"])
    check("les 2 PARSE_ERROR sont hors matrice de confusion (limite sklearn)",
          r_mix["n_hors_matrice"] == 2, f"{r_mix['n_hors_matrice']}")

    r_cpt = ev.evalue_campagne(mix, src, with_ci=False, verbeux=False,
                               erreurs_http="compter_comme_erreur")
    check("erreurs_http='compter_comme_erreur' evalue les 9 lignes",
          r_cpt["n_evalue"] == 9 and abs(r_cpt["accuracy"] - 6 / 9) < 1e-9,
          f"n={r_cpt['n_evalue']}, accuracy={r_cpt['accuracy']:.4f}")
    check("valeur invalide pour erreurs_http refusee",
          _leve(ValueError, ev.evalue_campagne, mix, src,
                erreurs_http="n_importe_quoi"))

    print("\n" + "=" * 78)
    print("PONDERATION : LES GARDE-FOUS DE metrics.py RESTENT ACTIFS")
    print("=" * 78)
    src_p = source(avec_poids=True)
    imparfait = []
    for i, (_, row) in enumerate(src_p.iterrows()):
        predit = row[cfg.LABEL_COL] if i % 3 else "Mortgage"
        imparfait.append(enr(row[cfg.ID_COL], row[cfg.LABEL_COL], predit))
    pond = ecrit(tmp / "pondere.jsonl", imparfait)

    r_pond = ev.evalue_campagne(pond, src_p, with_ci=False, verbeux=False)
    check("source ponderee -> evaluation ponderee automatiquement",
          r_pond["weighted"] is True)

    r_nonp = ev.evalue_campagne(pond, source(), with_ci=False, verbeux=False)
    check("les poids changent effectivement le F1",
          abs(r_pond["f1_macro"] - r_nonp["f1_macro"]) > 1e-9,
          f"pondere {r_pond['f1_macro']:.4f} vs brut {r_nonp['f1_macro']:.4f}")
    check("le bootstrap est stratifie quand des poids sont fournis",
          r_pond["bootstrap_strata"].startswith("y_true"),
          r_pond["bootstrap_strata"])

    # Poids fournis separement, comme le fait load_eval_sample()
    src_sans = source()
    poids_ext = pd.Series([1.0 if i % 2 == 0 else 50.0 for i in range(9)])
    r_ext = ev.evalue_campagne(pond, src_sans, poids=poids_ext,
                               with_ci=False, verbeux=False)
    check("poids passes separement -> meme resultat que la colonne",
          abs(r_ext["f1_macro"] - r_pond["f1_macro"]) < 1e-9,
          f"{r_ext['f1_macro']:.4f} vs {r_pond['f1_macro']:.4f}")

    print("\n" + "=" * 78)
    print("AGREGATS DE TOKENS, DE COUT ET DE CACHE")
    print("=" * 78)
    check("tokens d'entree sommes depuis le journal",
          r["tokens_in_total"] == 9 * 300, f"{r['tokens_in_total']}")
    check("tokens d'entree moyens calcules, pas repris de config",
          r["tokens_in_moyen"] == 300.0
          and r["tokens_in_moyen"] != cfg.AVG_COMPLAINT_TOKENS,
          f"mesure {r['tokens_in_moyen']} vs hypothese {cfg.AVG_COMPLAINT_TOKENS}")
    check("taux de tokens servis par le cache",
          abs(r["taux_tokens_caches"] - 260 / 300) < 1e-9,
          f"{r['taux_tokens_caches']:.1%}")
    check("cout reel somme depuis le journal",
          abs(r["cout_reel_usd"] - 9 * 4.8e-05) < 1e-12,
          f"{r['cout_reel_usd']:.8f} $")
    check("cout pour 1000 predictions expose pour compare_results",
          abs(r["cout_pour_1000_predictions_usd"] - 4.8e-05 * 1000) < 1e-9)
    check("cout declare non calculable si un cout manque",
          r_mix["cout_calculable"] is False)

    print("\n" + "=" * 78)
    print("LATENCES : LES DEUX FORMES SONT DISTINGUEES")
    print("=" * 78)
    check("latence modele (inference seule) rapportee",
          r["latence_p50_s"] == 0.5, f"{r['latence_p50_s']}")
    check("latence avec attentes rapportee separement",
          r["attentes_latence_p50_s"] == 1.5, f"{r['attentes_latence_p50_s']}")
    check("les deux ne sont pas confondues",
          r["latence_p50_s"] != r["attentes_latence_p50_s"])
    check("p95 et max presents", "latence_p95_s" in r and "latence_max_s" in r)

    print("\n" + "=" * 78)
    print("COMPARAISON MULTI-CAMPAGNES")
    print("=" * 78)
    autre = ecrit(tmp / "autre.jsonl",
                  [enr(row[cfg.ID_COL], row[cfg.LABEL_COL],
                       row[cfg.LABEL_COL] if i % 2 else "Mortgage",
                       tokens_in=250, latence=0.2)
                   for i, (_, row) in enumerate(src.iterrows())])
    tab = ev.compare_campagnes({"v1_small": parfait, "v1_ministral": autre},
                               src, with_ci=False, verbeux=False)
    check("tableau a une ligne par campagne", list(tab.index) ==
          ["v1_small", "v1_ministral"], f"{list(tab.index)}")
    check("F1 par classe present dans le tableau",
          any(c.startswith("F1 ·") for c in tab.columns))
    check("colonnes LLM ajoutees",
          all(c in tab.columns for c in ("PARSE_ERROR", "erreur HTTP",
                                         "tokens entree moy.", "cout reel ($)")))
    check("les deux campagnes se distinguent bien",
          tab.loc["v1_small", "F1-macro"] > tab.loc["v1_ministral", "F1-macro"])

    refus = False
    try:
        ev.compare_campagnes({"complet": parfait, "partiel": partiel}, src,
                             with_ci=False, verbeux=False)
    except ValueError as e:
        refus = "memes lignes" in str(e)
    check("comparaison sur des lignes DIFFERENTES refusee", refus)

    print("\n" + "=" * 78)
    print("DETERMINISME")
    print("=" * 78)
    passes = []
    for p in range(1, 6):
        for i, (_, row) in enumerate(src.iterrows()):
            # id1_0 alterne entre deux etiquettes selon la passe
            predit = row[cfg.LABEL_COL]
            if row[cfg.ID_COL] == "id1_0" and p in (2, 4):
                predit = "Credit reporting"
            passes.append(enr(row[cfg.ID_COL], row[cfg.LABEL_COL], predit,
                              latence=0.5 + 0.1 * p, execution=p))
    det = ecrit(tmp / "det.jsonl", passes)
    d = ev.analyse_determinisme(det, verbeux=False)
    check("5 passes detectees", d["n_passes"] == 5, f"{d['n_passes']}")
    check("9 exemples suivis", d["n_exemples"] == 9)
    check("8 exemples sur 9 stables", d["n_stables"] == 8, f"{d['n_stables']}")
    check("taux de stabilite = 8/9", abs(d["taux_stabilite"] - 8 / 9) < 1e-9,
          f"{d['taux_stabilite']:.1%}")
    inst = d["instables"][0]
    check("l'exemple instable est nomme", inst[cfg.ID_COL] == "id1_0")
    check("les etiquettes concurrentes et leur frequence sont rapportees",
          inst["labels_concurrents"] == {"Credit reporting": 2, "Debt collection": 3},
          f"{inst['labels_concurrents']}")
    check("variabilite de latence sur un meme texte mesuree",
          d["latence_ecart_type_median_s"] > 0,
          f"{d['latence_ecart_type_median_s']:.4f} s")

    r_det = ev.evalue_campagne(det, src, with_ci=False, verbeux=False)
    check("evaluer un journal multi-passes n'evalue QU'UNE passe",
          r_det["n_evalue"] == 9, f"n_evalue={r_det['n_evalue']}")

    print("\n" + "=" * 78)
    print("DERIVE DE VERSION DERRIERE UN ALIAS")
    print("=" * 78)
    stable_a = ecrit(tmp / "va.jsonl", [enr(f"id{i}_0", "Mortgage", "Mortgage")
                                        for i in range(3)])
    stable_b = ecrit(tmp / "vb.jsonl", [enr(f"id{i}_0", "Mortgage", "Mortgage")
                                        for i in range(3)])
    rap = ev.verifie_derive_version({"a": stable_a, "b": stable_b})
    check("meme alias, meme version resolue -> stable", rap["stable"] is True)

    # Deux ALIAS differents servant deux versions differentes : cas NORMAL,
    # c'est toute la comparaison mistral-small / ministral.
    autre_modele = ecrit(tmp / "vmini.jsonl",
                         [enr(f"id{i}_0", "Mortgage", "Mortgage",
                              modele="ministral-3b-latest",
                              resolu="ministral-3b-2512") for i in range(3)])
    rap2 = ev.verifie_derive_version({"small": stable_a, "mini": autre_modele})
    check("deux alias distincts ne declenchent PAS de derive",
          rap2["stable"] is True, "comparer deux modeles est le but, pas un incident")

    # Meme alias, version differente : DERIVE.
    derive = ecrit(tmp / "vderive.jsonl",
                   [enr(f"id{i}_0", "Mortgage", "Mortgage",
                        resolu="mistral-small-2699") for i in range(3)])
    rap3 = ev.verifie_derive_version({"avant": stable_a, "apres": derive})
    check("meme alias, version differente -> DERIVE detectee",
          rap3["stable"] is False and len(rap3["derives"]) == 1,
          f"{rap3['derives']}")
    check("la derive nomme les deux versions",
          set(rap3["derives"][0]["versions"]) ==
          {"mistral-small-2603", "mistral-small-2699"})
    check("mode strict : la derive leve",
          _leve(RuntimeError, ev.verifie_derive_version,
                {"avant": stable_a, "apres": derive}, True))

    intra = ecrit(tmp / "vintra.jsonl",
                  [enr("id0_0", "Mortgage", "Mortgage"),
                   enr("id1_0", "Mortgage", "Mortgage", resolu="mistral-small-2699")])
    rap4 = ev.verifie_derive_version({"c": intra})
    check("repointage EN COURS de campagne detecte",
          len(rap4["derives_intra_campagne"]) == 1,
          f"{rap4['derives_intra_campagne']}")

    alias_brut = ecrit(tmp / "vnonresolu.jsonl",
                       [enr("id0_0", "Mortgage", "Mortgage",
                            resolu="mistral-small-latest")])
    rap5 = ev.verifie_derive_version({"d": alias_brut})
    check("alias journalise a la place de la version -> signale",
          len(rap5["versions_non_resolues"]) == 1,
          "sans ce controle, la detection de derive serait silencieusement "
          "inoperante")
    check("une version correctement resolue ne declenche pas l'alerte",
          ev.verifie_derive_version({"a": stable_a})["versions_non_resolues"] == [])

    print("\n" + "=" * 78)
    print("RECAPITULATIF DE FACTURATION")
    print("=" * 78)
    # 3 passes de 4 lignes + 1 echec HTTP : la facturation compte TOUT,
    # l'evaluation ne retient qu'une passe.
    factu = []
    for passe in (1, 2, 3):
        for i in range(4):
            factu.append(enr(f"id{i}_0", "Mortgage", "Mortgage", execution=passe))
    factu.append(enr("id9_0", "Mortgage", cfg.PARSE_ERROR, erreur="HTTP 500",
                     tokens_in=None, tokens_out=None, tokens_caches=None,
                     cout=None, latence=None, resolu=None))
    chemin_f = ecrit(tmp / "factu.jsonl", factu)
    rf = ev.recapitulatif_facturation(chemin_f, verbeux=False)
    t = rf["total"]
    check("les 12 appels aboutis sont comptes, l'echec HTTP exclu du facture",
          t["n_appels"] == 12, f"{t['n_appels']}")
    check("toutes les passes comptent dans la facturation",
          rf["campagnes"][0]["passes"] == [1, 2, 3])
    check("tokens d'entree cumules sur tous les appels",
          t["tokens_in"] == 12 * 300, f"{t['tokens_in']}")
    check("tokens de sortie cumules", t["tokens_out"] == 12 * 10)
    check("tokens en cache cumules", t["tokens_caches"] == 12 * 260)
    check("part non cachee deduite",
          t["tokens_in_non_caches"] == 12 * 40, f"{t['tokens_in_non_caches']}")
    check("taux de cache", abs(t["taux_cache"] - 260 / 300) < 1e-9)
    check("cout total cumule", abs(t["cout"] - 12 * 4.8e-05) < 1e-12)
    check("la version servie figure au recapitulatif",
          rf["campagnes"][0]["versions_servies"] == ["mistral-small-2603"])
    check("l'echec HTTP est signale a part",
          rf["campagnes"][0]["n_erreurs_http"] == 1
          and rf["campagnes"][0]["n_appels_emis"] == 13)

    # Perimetres volontairement differents : c'est le point a ne pas confondre.
    check("facturation (12) et evaluation (4/passe) ont des perimetres distincts",
          t["n_appels"] == 12 and rf["campagnes"][0]["passes"] == [1, 2, 3],
          "le fournisseur facture les 3 passes, aucune n'entre 3 fois dans un F1")

    print("\n" + "=" * 78)
    print("McNEMAR — COMPARAISON APPARIEE")
    print("=" * 78)
    # A juste partout ; B faux sur 3 lignes et juste nulle part ou A se trompe.
    a = ecrit(tmp / "mna.jsonl", [enr(r[cfg.ID_COL], r[cfg.LABEL_COL], r[cfg.LABEL_COL])
                                  for _, r in src.iterrows()])
    # Etiquette fausse GARANTIE sur les 3 premieres lignes : on decale d'un cran
    # dans CLASS_ORDER, ce qui ne peut jamais retomber sur la vraie classe.
    def _faux(vrai):
        return cfg.CLASS_ORDER[(cfg.CLASS_ORDER.index(vrai) + 1) % 9]

    b = ecrit(tmp / "mnb.jsonl",
              [enr(r[cfg.ID_COL], r[cfg.LABEL_COL],
                   _faux(r[cfg.LABEL_COL]) if i < 3 else r[cfg.LABEL_COL])
               for i, (_, r) in enumerate(src.iterrows())])
    ra = ev.evalue_campagne(a, src, with_ci=False, verbeux=False)
    rb = ev.evalue_campagne(b, src, with_ci=False, verbeux=False)
    mc = ev.mcnemar(ra, rb, "A", "B", verbeux=False)
    check("3 paires discordantes, toutes dans le meme sens",
          mc["b"] == 3 and mc["c"] == 0, f"b={mc['b']} c={mc['c']}")
    check("p exacte = 0,25 pour 3 discordances unilaterales",
          abs(mc["p_valeur"] - 0.25) < 1e-12, f"p={mc['p_valeur']}")
    check("NON significatif au seuil de 5 %",
          mc["significatif_5pct"] is False,
          "20 lignes ne peuvent pas demontrer une inclusion stricte")
    check("le sens de l'ecart est nomme", mc["sens"] == "A")

    idem = ev.mcnemar(ra, ra, verbeux=False)
    check("deux campagnes identiques -> p = 1", idem["p_valeur"] == 1.0
          and idem["n_discordantes"] == 0)
    check("McNemar refuse des lignes differentes",
          _leve(ValueError, ev.mcnemar, ra, r_part))

    # 10 discordances unilaterales : la meme structure devient significative.
    gros = ecrit(tmp / "mnc.jsonl",
                 [enr(f"g{i}", "Mortgage", "Mortgage") for i in range(20)])
    gros_b = ecrit(tmp / "mnd.jsonl",
                   [enr(f"g{i}", "Mortgage",
                        "Debt collection" if i < 10 else "Mortgage")
                    for i in range(20)])
    src_g = pd.DataFrame({cfg.ID_COL: [f"g{i}" for i in range(20)],
                          cfg.TEXT_COL: ["t"] * 20, cfg.LABEL_COL: ["Mortgage"] * 20})
    mg = ev.mcnemar(ev.evalue_campagne(gros, src_g, with_ci=False, verbeux=False),
                    ev.evalue_campagne(gros_b, src_g, with_ci=False, verbeux=False),
                    verbeux=False)
    check("10 discordances unilaterales -> significatif",
          mg["significatif_5pct"] is True and mg["p_valeur"] < 0.002,
          f"p={mg['p_valeur']:.5f} — la puissance vient de l'effectif discordant")

    print("\n" + "=" * 78)
    print("PROJECTION DE COUT : DEUX BORNES DE CACHE")
    print("=" * 78)
    theo = ev.projette_cout(1000, "mistral-small-4")
    obs = cfg.CACHE_MESURE["mistral-small-4"]
    reel = ev.projette_cout(1000, "mistral-small-4",
                            taux_activation_cache=obs["taux_activation"],
                            couverture_prefixe=obs["couverture_prefixe"])
    sans = ev.projette_cout(1000, "mistral-small-4", taux_activation_cache=0.0)
    check("defaut = borne theorique (activation 100 %)",
          theo["taux_activation_cache"] == 1.0 and theo["couverture_prefixe"] == 1.0,
          "la degradation doit etre un geste explicite")
    check("le cout mesure est SUPERIEUR au theorique",
          reel["cout_total_usd"] > theo["cout_total_usd"],
          f"{reel['cout_total_usd']:.4f} > {theo['cout_total_usd']:.4f} $")
    # TROIS regimes, dans un ordre qui ne doit jamais s'inverser :
    #   demarrage a froid  <  regime etabli  <  borne theorique
    froid = ev.projette_cout(
        1000, "mistral-small-4",
        taux_activation_cache=obs["taux_activation_demarrage"],
        couverture_prefixe=obs["couverture_prefixe"])
    eco = lambda r: 1 - r["cout_total_usd"] / sans["cout_total_usd"]
    check("ordre des trois regimes : froid < etabli < theorique",
          eco(froid) < eco(reel) < eco(theo),
          f"froid {eco(froid):.1%} < etabli {eco(reel):.1%} < "
          f"theorique {eco(theo):.1%}")
    # La couverture plafonnee (224/252) empeche STRUCTURELLEMENT d'atteindre la
    # borne, meme a 100 % d'activation : c'est ce qui distingue les deux.
    plein = ev.projette_cout(1000, "mistral-small-4", taux_activation_cache=1.0,
                             couverture_prefixe=obs["couverture_prefixe"])
    check("meme a 100 % d'activation, la couverture plafonnee borne l'economie",
          eco(plein) < eco(theo),
          f"activation totale mais couverture {obs['couverture_prefixe']:.0%} "
          f"-> {eco(plein):.1%} contre {eco(theo):.1%} en theorie")
    check("le demarrage a froid explique l'ecart initialement observe",
          0.18 < eco(froid) < 0.24,
          f"{eco(froid):.1%} — c'est la mesure sur 20 appels, pas le regime etabli")

    check("statut des tokens : mesure pour Mistral",
          theo["statut_tokens"] == "mesure" and theo["fournisseur_mesure"] is True)
    for autre in ("claude-opus-5", "gpt-5.6-sol", "gemini-3.5-flash",
                  "deepseek-v4-flash"):
        r_a = ev.projette_cout(1000, autre)
        check(f"statut des tokens : hypothese pour {autre}",
              r_a["statut_tokens"] == "hypothese",
              "tokenizer different, la mesure Mistral ne transfere pas")
    check("prefixe par modele : 252 pour small, 240 pour ministral",
          ev.projette_cout(1000, "mistral-small-4")["prefixe_tokens"] == 252
          and ev.projette_cout(1000, "ministral-3b")["prefixe_tokens"] == 240)
    check("prefixe non mesure pour les modeles non testes",
          ev.projette_cout(1000, "claude-opus-5")["prefixe_mesure"] is False)

    tab = ev.tableau_couts_deux_bornes(1000)
    check("le tableau porte les DEUX colonnes de cache",
          "$ cache THEORIQUE" in tab.columns and "$ cache MESURE" in tab.columns)
    check("la colonne mesuree n'est remplie que pour les modeles testes",
          tab["$ cache MESURE"].notna().sum() == len(cfg.CACHE_MESURE),
          f"{int(tab['$ cache MESURE'].notna().sum())} modele(s) sur {len(tab)}")
    check("aucune extrapolation vers les autres fournisseurs",
          tab.loc["claude-opus-5", "$ cache MESURE"] is None)
    check("la note du tableau precise le perimetre de la mesure",
          "Mistral" in tab.attrs["note"] and cfg.CACHE_MESURE_LE in tab.attrs["note"])

    print("\n" + "=" * 78)
    print("HOLM-BONFERRONI")
    print("=" * 78)
    h = ev.holm_bonferroni({"a": 0.001, "b": 0.02, "c": 0.30, "d": 0.40,
                            "e": 0.45}, verbeux=False)
    seuils = [r["seuil_holm"] for r in h["resultats"]]
    check("les seuils se relachent : 0,05/5 puis /4, /3, /2, /1",
          [round(x, 5) for x in seuils] == [0.01, 0.0125, round(0.05/3, 5),
                                            0.025, 0.05], f"{seuils}")
    check("les p sont tries par ordre croissant",
          [r["p_brut"] for r in h["resultats"]] ==
          sorted(r["p_brut"] for r in h["resultats"]))
    check("p=0,001 < 0,010 -> rejetee", h["resultats"][0]["rejetee"] is True)
    check("p=0,02 > 0,0125 -> non rejetee, procedure ARRETEE",
          h["resultats"][1]["rejetee"] is False
          and "arret" in h["resultats"][1]["motif"])
    check("une seule hypothese rejetee", h["rejetees"] == ["a"])

    # L'ARRET est ce qui distingue Holm d'une suite de seuils independants :
    # un p de rang 3 inferieur a son propre seuil ne doit PAS etre rejete si
    # le rang 2 a echoue.
    # Rangs : a=0,001 (seuil 0,0167, rejetee) ; b=0,030 (seuil 0,0250 -> ECHEC,
    # arret) ; c=0,040 qui passerait pourtant son propre seuil de 0,050.
    h2 = ev.holm_bonferroni({"a": 0.001, "b": 0.030, "c": 0.040}, verbeux=False)
    rang3 = [r for r in h2["resultats"] if r["nom"] == "c"][0]
    check("un p sous son seuil n'est PAS rejete si la procedure s'est arretee",
          rang3["p_brut"] < rang3["seuil_holm"] and rang3["rejetee"] is False,
          f"p={rang3['p_brut']} < seuil={rang3['seuil_holm']:.4f} mais non rejetee")

    # Holm DOMINE Bonferroni : il ne rejette jamais moins.
    ps = {"a": 0.008, "b": 0.011, "c": 0.02, "d": 0.30, "e": 0.40}
    holm = set(ev.holm_bonferroni(ps, verbeux=False)["rejetees"])
    bonf = {k for k, v in ps.items() if v < 0.05 / len(ps)}
    check("Holm domine Bonferroni (ne rejette jamais moins)",
          bonf.issubset(holm), f"Holm={sorted(holm)} Bonferroni={sorted(bonf)}")

    tout = ev.holm_bonferroni({"a": 0.0001, "b": 0.0002, "c": 0.0003},
                              verbeux=False)
    check("toutes rejetees si tous les p sont tres petits",
          len(tout["rejetees"]) == 3)
    aucune = ev.holm_bonferroni({"a": 0.2, "b": 0.5}, verbeux=False)
    check("aucune rejetee est une issue legitime", aucune["rejetees"] == [])
    check("famille vide geree", ev.holm_bonferroni({}, verbeux=False)["m"] == 0)

    print("\n  -- effet minimal detectable par rang --")
    emd = ev.effet_minimal_detectable_holm(verbeux=False)
    r1 = emd[(emd["rang"] == 1) & (emd["lignes_cassees"] == 0)].iloc[0]
    r5 = emd[(emd["rang"] == 5) & (emd["lignes_cassees"] == 0)].iloc[0]
    check("le rang 1 est le plus exigeant, le rang 5 le moins",
          r1["lignes_a_corriger"] > r5["lignes_a_corriger"],
          f"rang1 : {int(r1['lignes_a_corriger'])} corrections, "
          f"rang5 : {int(r5['lignes_a_corriger'])}")
    check("l'exigence croit avec le nombre de lignes cassees",
          emd[emd["rang"] == 1].sort_values("lignes_cassees")
             ["lignes_a_corriger"].is_monotonic_increasing)

    print("\n" + "=" * 78)
    print("ANALYSE D'ERREURS PAR PAIRE DE CLASSES")
    print("=" * 78)
    # Erreurs construites a la main sur des paires connues : 3 sur la paire
    # annoncee n1 du C.4, 1 sur une paire non annoncee.
    src_e = source()
    def _pred(vrai, i):
        if vrai == "Credit reporting" and i == 0: return "Debt collection"
        if vrai == "Debt collection": return "Credit reporting"
        if vrai == "Mortgage": return "Student loan"      # hors C.4
        return vrai
    jl = ecrit(tmp / "paires.jsonl",
               [enr(r[cfg.ID_COL], r[cfg.LABEL_COL], _pred(r[cfg.LABEL_COL], i))
                for i, (_, r) in enumerate(src_e.iterrows())])
    r_p = ev.evalue_campagne(jl, src_e, with_ci=False, verbeux=False)
    tab = ev.paires_erreurs(r_p, verbeux=False)

    check("les paires sont NON ORIENTEES (les deux sens fusionnent)",
          "Credit reporting ↔ Debt collection" in tab.index
          and int(tab.loc["Credit reporting ↔ Debt collection", "n"]) == 2,
          f"{dict(tab['n'])}")
    check("les paires du §C.4 sont marquees",
          bool(tab.loc["Credit reporting ↔ Debt collection", "annoncee_C4"]))
    check("une paire hors §C.4 n'est pas marquee",
          not bool(tab.loc["Mortgage ↔ Student loan", "annoncee_C4"]))
    check("les parts d'erreurs somment a 1",
          abs(tab["part_des_erreurs"].sum() - 1.0) < 1e-9,
          f"{tab['part_des_erreurs'].sum()}")
    check("les 5 paires annoncees au §C.4 sont declarees",
          len(ev.PAIRES_ANNONCEES_C4) == 5)
    check("toutes les paires du §C.4 portent des classes valides",
          all(a in cfg.CLASS_ORDER and b in cfg.CLASS_ORDER
              for a, b in ev.PAIRES_ANNONCEES_C4))
    check("aucune erreur -> tableau vide, pas d'exception",
          ev.paires_erreurs(r, verbeux=False).empty)

    print("\n  -- export pour l'analyse croisee de l'etape 3 --")
    ch_err = ev.exporte_erreurs(r_p, tmp / "erreurs.csv")
    exp = pd.read_csv(ch_err)
    n_err = int((pd.Series(r_p["_y_true"]) != pd.Series(r_p["_y_pred"])).sum())
    check("l'export contient exactement les lignes mal classees",
          len(exp) == n_err, f"{len(exp)} vs {n_err}")
    check("l'export porte l'id et les deux etiquettes",
          all(c in exp.columns for c in (cfg.ID_COL, "label_vrai",
                                         "label_predit")))
    check("aucune ligne correctement classee dans l'export",
          (exp["label_vrai"] != exp["label_predit"]).all())
    check("l'export ne contient QUE id, vrai, predit",
          list(exp.columns) == [cfg.ID_COL, "label_vrai", "label_predit"],
          f"{list(exp.columns)}")
    check("aucun texte de reclamation dans l'export",
          cfg.TEXT_COL not in exp.columns,
          "fichier relu plusieurs fois : le texte se rejoint par complaint_id")
    check("aucun poids dans l'export",
          cfg.WEIGHT_COL not in exp.columns,
          "les poids appartiennent a l'echantillon, pas aux erreurs")

    print("\n  -- confrontation aux paires du §C.4, references fixees avant mesure --")
    c4 = ev.confronte_paires_c4(r_p, verbeux=False)
    check("les 36 paires possibles sont comptees", c4["n_paires_possibles"] == 36)
    check("la reference uniforme vaut 5/36",
          abs(c4["attendu_uniforme"] - 5/36) < 1e-3, f"{c4['attendu_uniforme']}")
    check("a effectifs EGAUX, les deux references coincident (comportement juste)",
          abs(c4["attendu_effectifs"] - c4["attendu_uniforme"]) < 1e-9,
          "la source de test a 1 ligne par classe : n_i x n_j est constant")

    # Source DESEQUILIBREE : les deux references doivent alors diverger, sinon
    # la reference par effectifs n'apporterait rien.
    src_d = source(n_par_classe=1, classes=(["Credit reporting"] * 20
                                            + ["Debt collection"] * 15
                                            + ["Mortgage"] * 3
                                            + ["Student loan"] * 2))
    jd = ecrit(tmp / "desequilibre.jsonl",
               [enr(r[cfg.ID_COL], r[cfg.LABEL_COL],
                    "Debt collection" if r[cfg.LABEL_COL] == "Credit reporting"
                    else r[cfg.LABEL_COL])
                for _, r in src_d.iterrows()])
    r_d = ev.evalue_campagne(jd, src_d, with_ci=False, verbeux=False)
    c4d = ev.confronte_paires_c4(r_d, verbeux=False)
    check("a effectifs DESEQUILIBRES, la reference par effectifs diverge",
          abs(c4d["attendu_effectifs"] - c4d["attendu_uniforme"]) > 1e-3,
          f"uniforme {c4d['attendu_uniforme']} vs effectifs {c4d['attendu_effectifs']}")
    check("la reference par effectifs est la plus SEVERE ici",
          c4d["attendu_effectifs"] > c4d["attendu_uniforme"],
          "les grosses classes rendent leur paire mecaniquement plus probable")
    check("les deux lifts sont rendus",
          c4["lift_uniforme"] is not None and c4["lift_effectifs"] is not None)
    check("PARSE_ERROR est exclu du denombrement des paires",
          c4["n_erreurs_hors_parse_error"] <= c4["n_erreurs"])

    print("\n  -- verdict §C.4 : criteres pre-enregistres, verdict CALCULE --")
    # Cas construit pour CONFIRMER : erreurs concentrees sur la paire
    # principale du C.4, aucune paire lourde hors C.4.
    src_c = source(n_par_classe=1, classes=(["Credit reporting"] * 10
                                            + ["Debt collection"] * 10
                                            + ["Mortgage"] * 10))
    jc = ecrit(tmp / "c4_confirme.jsonl",
               [enr(r[cfg.ID_COL], r[cfg.LABEL_COL],
                    "Debt collection" if (r[cfg.LABEL_COL] == "Credit reporting"
                                          and i % 2 == 0) else r[cfg.LABEL_COL])
                for i, (_, r) in enumerate(src_c.iterrows())])
    v = ev.verdict_c4(ev.evalue_campagne(jc, src_c, with_ci=False, verbeux=False),
                      verbeux=False)
    check("cas concentre sur la paire principale -> B confirme",
          v["B_verdict"] == "confirme" and v["B_rang_paire_principale"] == 1,
          f"rang {v['B_rang_paire_principale']}")
    check("aucune paire lourde hors C.4 -> C confirme",
          v["C_verdict"] == "confirme" and v["C_lacunes"] == [])

    # Cas construit pour INFIRMER : toutes les erreurs sur une paire NON annoncee.
    ji = ecrit(tmp / "c4_infirme.jsonl",
               [enr(r[cfg.ID_COL], r[cfg.LABEL_COL],
                    "Student loan" if r[cfg.LABEL_COL] == "Mortgage"
                    else r[cfg.LABEL_COL])
                for _, r in src_c.iterrows()])
    vi = ev.verdict_c4(ev.evalue_campagne(ji, src_c, with_ci=False, verbeux=False),
                       verbeux=False)
    check("erreurs hors C.4 -> A infirme", vi["A_concentration"] == "infirme",
          f"lift severe {vi['A_lift_severe']}")
    check("paire principale absente -> B infirme",
          vi["B_verdict"] == "infirme" and vi["B_rang_paire_principale"] is None)
    check("paire lourde non annoncee -> C infirme (lacune detectee)",
          vi["C_verdict"] == "infirme" and len(vi["C_lacunes"]) == 1,
          f"{[l['paire'] for l in vi['C_lacunes']]}")

    check("le classement COMPLET est rendu, pas seulement les paires du C.4",
          len(vi["classement_complet"]) >= 1
          and not vi["classement_complet"]["annoncee_C4"].all())
    check("les seuils sont des constantes de module, pas des litteraux locaux",
          (ev.SEUIL_A_CONFIRME, ev.SEUIL_A_INFIRME, ev.RANG_B_CONFIRME,
           ev.RANG_B_INFIRME, ev.PART_C_LACUNE, ev.RANG_C_LACUNE)
          == (1.50, 1.00, 2, 5, 0.10, 3),
          "pre-enregistres avant la campagne d'evaluation")

    print("\n" + "=" * 78)
    print("AUCUNE METRIQUE N'EST REIMPLEMENTEE")
    print("=" * 78)
    source_mod = (RACINE / "src" / "llm_eval.py").read_text()
    arbre = ast.parse(source_mod)

    # Analyse SYNTAXIQUE, pas textuelle : une recherche de sous-chaine prend
    # `mt.plot_confusion_matrix(` pour un appel a `confusion_matrix(` de
    # sklearn et crie au loup. On distingue donc un appel direct d'une
    # delegation a metrics.py.
    INTERDITS = {"f1_score", "accuracy_score", "precision_score", "recall_score",
                 "confusion_matrix", "classification_report", "precision_recall_fscore_support"}
    directs = []
    for n in ast.walk(arbre):
        if not isinstance(n, ast.Call):
            continue
        f = n.func
        if isinstance(f, ast.Name) and f.id in INTERDITS:
            directs.append(f.id)
        elif (isinstance(f, ast.Attribute) and f.attr in INTERDITS
              and not (isinstance(f.value, ast.Name) and f.value.id == "mt")):
            directs.append(f.attr)
    check("llm_eval.py n'appelle aucune metrique sklearn en direct",
          not directs, f"appels directs : {directs}")

    importe_sklearn = [n for n in ast.walk(arbre)
                       if isinstance(n, ast.ImportFrom) and (n.module or "").startswith("sklearn")]
    importe_sklearn += [n for n in ast.walk(arbre) if isinstance(n, ast.Import)
                        and any(a.name.startswith("sklearn") for a in n.names)]
    check("llm_eval.py n'importe meme pas sklearn", not importe_sklearn)

    appels_mt = {n.func.attr for n in ast.walk(arbre)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                 and isinstance(n.func.value, ast.Name) and n.func.value.id == "mt"}
    check("llm_eval.py delegue bien a metrics.py",
          {"evaluate", "compare_results"}.issubset(appels_mt),
          f"appels a metrics : {sorted(appels_mt)}")

    print("\n" + "=" * 78)
    print("SOUS-ENSEMBLE EVALUE : LE CONTRAT AVEC L'ETAPE 3")
    print("=" * 78)
    ids = ev.ids_evalues(r_mix)
    check("les identifiants evalues excluent la ligne perdue en HTTP",
          len(ids) == 8 and "id8_0" not in ids, f"{len(ids)} ids")
    check("un resultat etranger est refuse",
          _leve(KeyError, ev.ids_evalues, {"f1_macro": 0.5}))

    chemin_ids = ev.exporte_ids_evalues(r_mix, tmp / "ids.json")
    relus = ev.charge_ids_evalues(chemin_ids)
    check("aller-retour export/relecture sans perte", relus == ids)
    charge = json.loads(Path(chemin_ids).read_text(encoding="utf-8"))
    check("l'export porte le contexte et l'avertissement",
          charge["n_attendu"] == 9 and charge["n_evalue"] == 8
          and "etape 3" in charge["avertissement"])
    check("l'export ne contient AUCUN texte de reclamation",
          cfg.TEXT_COL not in json.dumps(charge))

    restreint = ev.restreint_au_sous_ensemble(src, relus)
    check("la source se restreint au sous-ensemble exact",
          len(restreint) == 8 and set(restreint[cfg.ID_COL]) == set(relus))
    check("un identifiant absent du DataFrame leve",
          _leve(ValueError, ev.restreint_au_sous_ensemble, src.iloc[:3], relus))

    # Le scenario que la contrainte previent : ML sur 9, LLM sur 8.
    faux_ml = dict(r_part)
    faux_ml["n"] = 9
    refus_n = _leve(ValueError, ev.compare_campagnes,
                    {"llm": r_mix, "ml": faux_ml})
    check("comparer un ML sur 9 lignes a un LLM sur 8 est refuse", refus_n)

    print("\n" + "=" * 78)
    print("FIGURE ETENDUE : SEPAREE DU CALCUL DES METRIQUES")
    print("=" * 78)
    import matplotlib
    matplotlib.use("Agg")

    f1_avant = r_mix["f1_macro"]
    fig, ax = ev.figure_confusion_etendue(r_mix, save_path=tmp / "cm.png")
    check("le F1-macro est INCHANGE apres production de la figure",
          r_mix["f1_macro"] == f1_avant, f"{r_mix['f1_macro']:.6f}")
    check("le rapport par classe reste sur 9 classes, sans PARSE_ERROR",
          cfg.PARSE_ERROR not in r_mix["classification_report"])
    check("la figure porte bien 10 colonnes (9 classes + PARSE_ERROR)",
          len(ax.get_xticklabels()) == 10,
          f"{len(ax.get_xticklabels())}")
    check("PARSE_ERROR est la derniere colonne",
          ax.get_xticklabels()[-1].get_text() == cfg.PARSE_ERROR)
    check("la figure est enregistree", (tmp / "cm.png").exists())

    # Preuve chiffree que les deux chemins DOIVENT rester separes : inclure
    # PARSE_ERROR dans le calcul creerait une 10e classe de F1 nul.
    from src import metrics as mt
    etendu = mt.evaluate(r_mix["_y_true"], r_mix["_y_pred"],
                         labels=list(cfg.CLASS_ORDER) + [cfg.PARSE_ERROR],
                         sample_weight=None, with_ci=False)
    check("inclure PARSE_ERROR au CALCUL degraderait le F1 sans raison",
          etendu["f1_macro"] < r_mix["f1_macro"],
          f"9 classes {r_mix['f1_macro']:.4f} -> 10 classes "
          f"{etendu['f1_macro']:.4f} (dilution mecanique)")
    check("la 10e classe a bien un F1 nul (aucune etiquette vraie)",
          etendu["classification_report"][cfg.PARSE_ERROR]["f1-score"] == 0.0
          and etendu["classification_report"][cfg.PARSE_ERROR]["support"] == 0)

    ventil = ev.hors_format_par_classe(r_mix)
    check("ventilation des hors-format par classe vraie",
          int(ventil["hors_format"].sum()) == 2, f"{int(ventil['hors_format'].sum())}")
    check("la ventilation suit l'ordre fige des classes",
          list(ventil.index) == [c for c in cfg.CLASS_ORDER if c in ventil.index])

    print("\n" + "=" * 78)
    print(f"RESULTAT : {sum(_RESULTATS)}/{len(_RESULTATS)} verifications passees")
    print("=" * 78)
    print(f"(fichiers de travail : {tmp})")
    return 0 if all(_RESULTATS) else 1


def _leve(exc, fn, *args, **kw) -> bool:
    try:
        fn(*args, **kw)
    except exc:
        return True
    except Exception:
        return False
    return False


if __name__ == "__main__":
    sys.exit(main())
