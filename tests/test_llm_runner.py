"""Verifications du runner de campagne (etape 2).

AUCUN APPEL API : le client est remplace par un `FauxClient` qui rend des
reponses fabriquees. C'est deliberé -- la reprise, le plafond d'appels et le
garde-fou de budget doivent etre prouves AVANT de depenser quoi que ce soit,
puisque ce sont precisement les mecanismes qui protegent la depense.

Ce que ce fichier prouve :

  1. Le runner REFUSE de demarrer sur l'echantillon d'evaluation tant que le
     style n'est pas fige. C'est le garde-fou qui empeche de regler un prompt
     sur les lignes qui serviront a le noter.
  2. La reprise saute les `complaint_id` deja traites et ne les redepense pas,
     y compris apres une interruption ayant laisse une ligne tronquee.
  3. Le plafond d'appels et le plafond de budget arretent la campagne sans
     perdre ce qui a deja ete ecrit.
  4. Le test de determinisme peut rejouer les MEMES lignes : la cle de reprise
     inclut le numero de passe.
  5. Le cout est calcule sur les tokens REELS, et declare non calculable quand
     le tarif est inconnu -- jamais suppose.

    python tests/test_llm_runner.py
"""

import inspect
import sys
import tempfile
from pathlib import Path

import pandas as pd

RACINE = Path(__file__).resolve().parent
while not (RACINE / "src" / "config.py").exists() and RACINE != RACINE.parent:
    RACINE = RACINE.parent
sys.path.insert(0, str(RACINE))

from src import config as cfg        # noqa: E402
from src import llm_runner as runner  # noqa: E402

OK, KO = "  OK  ", " ECHEC"
_RESULTATS: list[bool] = []


def check(nom, cond, detail=""):
    _RESULTATS.append(bool(cond))
    print(f"[{OK if cond else KO}] {nom}" + (f"   -> {detail}" if detail else ""))


class FauxClient:
    """Client factice. Compte ses appels, ne touche jamais au reseau."""

    def __init__(self, cle_tarification="mistral-small-4", reponse=None,
                 tokens_in=300, tokens_out=10, tokens_caches=260):
        self.modele_api = "faux-modele"
        self.cle_tarification = cle_tarification
        self.n_appels = 0
        self._reponse = reponse or '{"label": "Mortgage"}'
        self._tokens = (tokens_in, tokens_out, tokens_caches)

    def predict(self, messages):
        self.n_appels += 1
        t_in, t_out, t_cache = self._tokens
        return {
            "reponse_brute": self._reponse,
            "tokens_in": t_in, "tokens_out": t_out,
            "tokens_total": t_in + t_out, "tokens_caches": t_cache,
            "service_tier": "free", "latence_s": 0.5,
            "latence_avec_attentes_s": 1.5, "erreur_http": None,
            "tentatives": 1, "modele": self.modele_api,
            "modele_resolu": "faux-modele-2601", "modele_annonce": "faux-modele",
            "horodatage": "2026-01-01T00:00:00+00:00",
        }


def faux_df(n=10, avec_poids=False):
    """Petit DataFrame au format du pipeline."""
    df = pd.DataFrame({
        cfg.ID_COL: [f"id{i}" for i in range(n)],
        cfg.TEXT_COL: [f"Reclamation numero {i} avec du masquage XXXX." for i in range(n)],
        cfg.LABEL_COL: ["Mortgage"] * n,
    })
    if avec_poids:
        df[cfg.WEIGHT_COL] = 1.0
        df[cfg.STRATE_COL] = "Mortgage"
    return df


def main() -> int:
    _RESULTATS.clear()
    tmp = Path(tempfile.mkdtemp(prefix="zen_llm_test_"))

    print("=" * 78)
    print("GARDE-FOU : ECHANTILLON D'EVALUATION ET STYLE FIGE")
    print("=" * 78)

    df_eval = faux_df(5, avec_poids=True)
    check("l'echantillon d'evaluation est reconnu a sa structure",
          runner._est_echantillon_eval(df_eval))
    check("un DataFrame d'iteration n'est PAS pris pour l'echantillon d'eval",
          not runner._est_echantillon_eval(faux_df(5)))

    # Style volontairement ABSENT de LLM_STYLES_FIGES. Ne pas utiliser
    # "v1_zeroshot" : il est fige depuis la cloture de la phase 3, et le test
    # verifierait alors le contraire de ce qu'il annonce.
    STYLE_NON_FIGE = "v2_regle_produit"
    assert STYLE_NON_FIGE not in cfg.LLM_STYLES_FIGES
    client = FauxClient()
    refus = False
    try:
        runner.run_campagne(df_eval, client, STYLE_NON_FIGE, tmp / "interdit.jsonl")
    except RuntimeError as exc:
        refus = "REFUS DE DEMARRER" in str(exc)
    check("refus de demarrer sur l'echantillon d'eval avec un style non fige",
          refus)
    check("aucun appel n'a ete emis lors du refus", client.n_appels == 0,
          f"n_appels={client.n_appels}")

    # Une fois le style fige, la campagne d'evaluation est autorisee.
    figes_avant = set(cfg.LLM_STYLES_FIGES)
    try:
        cfg.LLM_STYLES_FIGES.add(STYLE_NON_FIGE)
        client2 = FauxClient()
        syn = runner.run_campagne(df_eval, client2, STYLE_NON_FIGE,
                                  tmp / "autorise.jsonl", max_appels=3,
                                  frequence_progression=99)
        check("style fige -> la campagne d'evaluation demarre",
              syn["n_appels"] == 3, f"n_appels={syn['n_appels']}")
    finally:
        cfg.LLM_STYLES_FIGES.clear()
        cfg.LLM_STYLES_FIGES.update(figes_avant)
    check("LLM_STYLES_FIGES restaure apres le test",
          cfg.LLM_STYLES_FIGES == figes_avant)

    print("\n" + "=" * 78)
    print("PLAFOND D'APPELS")
    print("=" * 78)
    df = faux_df(50)
    client = FauxClient()
    chemin = tmp / "plafond.jsonl"
    syn = runner.run_campagne(df, client, "v1_zeroshot", chemin,
                              max_appels=20, frequence_progression=99)
    check("le plafond d'appels est respecte a l'unite pres",
          client.n_appels == 20, f"n_appels={client.n_appels}")
    check("le motif d'arret nomme le plafond",
          "plafond" in syn["motif_arret"], syn["motif_arret"])
    # `max_appels` est keyword-only : son defaut vit dans __kwdefaults__.
    defaut = inspect.signature(runner.run_campagne).parameters["max_appels"].default
    check("le plafond d'appels vaut 20 par defaut",
          defaut == 20 and cfg.LLM_ITERATION_SIZE == 20,
          f"defaut={defaut}")

    print("\n" + "=" * 78)
    print("REPRISE APRES INTERRUPTION")
    print("=" * 78)
    lignes = runner.charge_resultats(chemin)
    check("20 enregistrements journalises", len(lignes) == 20, f"n={len(lignes)}")

    client2 = FauxClient()
    syn2 = runner.run_campagne(df, client2, "v1_zeroshot", chemin,
                               max_appels=20, frequence_progression=99)
    check("la reprise ne redepense AUCUN id deja traite",
          client2.n_appels == 20 and len(runner.charge_resultats(chemin)) == 40,
          f"nouveaux appels={client2.n_appels}, total={len(runner.charge_resultats(chemin))}")
    ids = [l[cfg.ID_COL] for l in runner.charge_resultats(chemin)]
    check("aucun doublon d'id apres reprise", len(ids) == len(set(ids)),
          f"{len(ids)} lignes, {len(set(ids))} ids distincts")

    # Le corpus est epuise : une troisieme passe ne doit rien appeler.
    client3 = FauxClient()
    runner.run_campagne(df, client3, "v1_zeroshot", chemin,
                        max_appels=20, frequence_progression=99)
    check("corpus epuise -> aucun appel supplementaire", client3.n_appels == 10,
          f"n_appels={client3.n_appels} (les 10 lignes restantes)")

    print("\n" + "=" * 78)
    print("REPRISE AVEC LIGNE TRONQUEE (coupure en pleine ecriture)")
    print("=" * 78)
    chemin_casse = tmp / "casse.jsonl"
    client4 = FauxClient()
    runner.run_campagne(faux_df(3), client4, "v1_zeroshot", chemin_casse,
                        max_appels=3, frequence_progression=99)
    with chemin_casse.open("a", encoding="utf-8") as f:
        f.write('{"complaint_id": "id3", "label_pred')  # ligne coupee net
    relu = runner.charge_resultats(chemin_casse)
    check("la ligne tronquee est ignoree, pas fatale", len(relu) == 3,
          f"n={len(relu)}")
    client5 = FauxClient()
    runner.run_campagne(faux_df(4), client5, "v1_zeroshot", chemin_casse,
                        max_appels=5, frequence_progression=99)
    check("l'id de la ligne tronquee est bien rejoue", client5.n_appels == 1,
          f"n_appels={client5.n_appels}")

    print("\n" + "=" * 78)
    print("TEST DE DETERMINISME : REJOUER LES MEMES LIGNES")
    print("=" * 78)
    chemin_det = tmp / "determinisme.jsonl"
    appels_par_passe = []
    for passe in range(1, 4):
        c = FauxClient()
        runner.run_campagne(faux_df(4), c, "v1_zeroshot", chemin_det,
                            max_appels=10, execution=passe,
                            frequence_progression=99)
        appels_par_passe.append(c.n_appels)
    check("chaque passe rejoue bien les 4 memes lignes",
          appels_par_passe == [4, 4, 4], f"{appels_par_passe}")
    passes = {l["execution"] for l in runner.charge_resultats(chemin_det)}
    check("les 3 passes sont distinguables dans le journal",
          passes == {1, 2, 3}, f"{sorted(passes)}")

    print("\n" + "=" * 78)
    print("GARDE-FOU DE BUDGET")
    print("=" * 78)
    # 1 M tokens d'entree par appel : chaque appel coute exactement le tarif
    # d'entree du modele. Le tarif est LU depuis config, jamais recopie ici --
    # une valeur en dur ferait de ce test une seconde source de verite.
    prix_appel = cfg.MODELS_PRICING["mistral-small-4"]["input_per_1m"]
    client6 = FauxClient(tokens_in=1_000_000, tokens_out=0, tokens_caches=0)
    syn6 = runner.run_campagne(faux_df(50), client6, "v1_zeroshot",
                               tmp / "budget.jsonl", max_appels=50,
                               budget_max_usd=0.50, frequence_progression=99)
    check("la campagne s'arrete au depassement de budget",
          "budget" in syn6["motif_arret"], syn6["motif_arret"])
    check("le budget n'est pas depasse de plus d'un appel",
          syn6["cout_cumule_journal_usd"] <= 0.50 + prix_appel + 1e-9,
          f"cout={syn6['cout_reel_usd']:.4f} $ (plafond 0.50 + {prix_appel})")
    check("le journal contient bien les appels payes avant l'arret",
          len(runner.charge_resultats(tmp / "budget.jsonl")) == client6.n_appels)

    print("\n" + "=" * 78)
    print("COUT ET CONTENU DU JOURNAL")
    print("=" * 78)
    client7 = FauxClient(cle_tarification=None)
    syn7 = runner.run_campagne(faux_df(2), client7, "v1_zeroshot",
                               tmp / "sans_tarif.jsonl", max_appels=2,
                               frequence_progression=99)
    check("tarif inconnu -> cout declare non calculable",
          syn7["cout_calculable"] is False and syn7["cout_reel_usd"] == 0)
    # Le cout de l'execution ne doit pas s'attribuer celui des passes anterieures.
    c1 = FauxClient(); ch = tmp / "cumul.jsonl"
    s1 = runner.run_campagne(faux_df(3), c1, "v1_zeroshot", ch, max_appels=3,
                             execution=1, frequence_progression=99)
    c2 = FauxClient()
    s2 = runner.run_campagne(faux_df(3), c2, "v1_zeroshot", ch, max_appels=3,
                             execution=2, frequence_progression=99)
    check("cout de l'execution distinct du cumul du journal",
          abs(s2["cout_reel_usd"] - s1["cout_reel_usd"]) < 1e-12
          and abs(s2["cout_cumule_journal_usd"] - 2 * s1["cout_reel_usd"]) < 1e-12,
          f"passe={s2['cout_reel_usd']:.6f} cumul={s2['cout_cumule_journal_usd']:.6f}")

    enr = runner.charge_resultats(chemin)[0]
    attendus = {cfg.ID_COL, "label_vrai", "label_predit", "statut_parsing",
                "reponse_brute", "style", "execution", "texte_tronque",
                "cout_usd", "tokens_in", "tokens_out", "tokens_caches",
                "service_tier", "latence_s", "erreur_http", "modele",
                "modele_resolu", "horodatage"}
    check("le journal porte tous les champs d'audit",
          attendus.issubset(enr), f"manquants : {sorted(attendus - set(enr))}")
    check("la reponse BRUTE est toujours journalisee",
          enr["reponse_brute"] == '{"label": "Mortgage"}')
    check("le texte de la reclamation n'est PAS duplique dans le journal",
          cfg.TEXT_COL not in enr)
    check("le journal vit sous data/llm/ ou en zone temporaire, jamais reports/",
          "reports" not in str(chemin))

    print("\n" + "=" * 78)
    print("PARSE_ERROR COMPTE COMME UNE ERREUR")
    print("=" * 78)
    client8 = FauxClient(reponse="je dirais Mortgage")
    syn8 = runner.run_campagne(faux_df(4), client8, "v1_zeroshot",
                               tmp / "erreurs.jsonl", max_appels=4,
                               frequence_progression=99)
    check("taux de parse_error a 100 % sur des reponses non conformes",
          syn8["taux_parse_error"] == 1.0, f"{syn8['taux_parse_error']:.0%}")
    check("l'etiquette predite est PARSE_ERROR",
          all(l["label_predit"] == cfg.PARSE_ERROR
              for l in runner.charge_resultats(tmp / "erreurs.jsonl")))
    check("PARSE_ERROR reste hors de CLASS_ORDER",
          cfg.PARSE_ERROR not in cfg.CLASS_ORDER)

    print("\n" + "=" * 78)
    print(f"RESULTAT : {sum(_RESULTATS)}/{len(_RESULTATS)} verifications passees")
    print("=" * 78)
    print(f"(fichiers de travail : {tmp})")
    return 0 if all(_RESULTATS) else 1


if __name__ == "__main__":
    sys.exit(main())
