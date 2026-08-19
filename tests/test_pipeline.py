"""Verifications du pipeline ZenAssist et de ses garde-fous.

Ce fichier est le seul code qui prouve que les garde-fous ECHOUENT BIEN QUAND
ILS DOIVENT ECHOUER : `sample_weight` omis leve un TypeError, une source
ponderee sans poids leve une ValueError, `load_eval_sample()` refuse de servir
un echantillon de developpement, un modele de tarification inconnu leve une
KeyError. Un garde-fou qui n'est jamais mis en defaut n'est pas un garde-fou,
c'est une intention.

Il verifie aussi la coherence du pipeline lui-meme : aucun texte partage entre
train et test, plancher d'echantillonnage respecte, poids reconstituant la
population de reference.

Prerequis : le pipeline doit avoir tourne au moins une fois.

    python -m src.data_prep
    python tests/test_pipeline.py

Sortie : une ligne par verification, et un code de retour non nul si l'une
d'elles echoue (utilisable en integration continue).

IMPORTABLE SANS EFFET DE BORD
-----------------------------
Tout le travail est dans `main()`. Le nom de ce fichier correspond au motif de
collecte de pytest, qui l'importerait : si le code s'executait a l'import, la
collecte declencherait le pipeline complet puis mourrait sur le `sys.exit()`
final. `main()` n'est appele que sous `if __name__ == "__main__"`.
"""

import datetime as dt
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

# Racine du projet = le repertoire qui contient src/config.py. Aucun chemin en
# dur : ce fichier doit tourner depuis n'importe quel repertoire courant.
RACINE = Path(__file__).resolve().parent
while not (RACINE / "src" / "config.py").exists() and RACINE != RACINE.parent:
    RACINE = RACINE.parent
sys.path.insert(0, str(RACINE))

from src import config as cfg
from src import data_prep as dp
from src import metrics as mt

OK, KO = "  OK  ", " ECHEC"
_RESULTATS: list[bool] = []


def check(nom, cond, detail=""):
    """Enregistre et affiche le resultat d'une verification."""
    _RESULTATS.append(bool(cond))
    print(f"[{OK if cond else KO}] {nom}" + (f"   -> {detail}" if detail else ""))


@contextmanager
def fraction_temporaire(valeur):
    """Force `config.SAMPLE_FRACTION` le temps d'un bloc, puis la restaure.

    Le try/finally n'est pas cosmetique. Sans lui, une exception levee dans le
    bloc laisserait la constante a sa valeur de developpement pour tout le
    reste du script : `load_split()` lirait alors `train_dev5.csv` et les
    verifications de coherence porteraient silencieusement sur les mauvais
    fichiers.
    """
    ancienne = cfg.SAMPLE_FRACTION
    cfg.SAMPLE_FRACTION = valeur
    try:
        yield
    finally:
        cfg.SAMPLE_FRACTION = ancienne


def main() -> int:
    """Execute toutes les verifications. Retourne 0 si tout passe, 1 sinon."""
    _RESULTATS.clear()

    print("=" * 78)
    print("GARDE-FOUS")
    print("=" * 78)

    # --- load_eval_sample est le seul chargement supporte -------------------
    ech = dp.load_eval_sample()
    check("load_eval_sample() retourne un tuple",
          isinstance(ech, tuple) and len(ech) == 2)
    df_ech, poids = ech
    check("les poids sont une Serie alignee", len(poids) == len(df_ech))

    # Valeurs de NON-REGRESSION, relevees dans data/processed/split_metadata.json
    # pour le pipeline de reference (allocation B3) :
    #   Credit reporting      21 547 lignes de test pour 521 tirees -> w = 41,36
    #   Vehicle loan or lease  1 140 lignes de test pour  75 tirees -> w = 15,20
    # Le poids le plus FAIBLE revient a la classe la plus sur-echantillonnee.
    # A METTRE A JOUR si le nettoyage, le seuil de longueur, la fusion des
    # libelles ou l'allocation changent — sinon l'echec sera cryptique.
    check("poids = N_c / n_c coherents",
          np.isclose(poids.max(), 21547 / 521, atol=0.5)
          and np.isclose(poids.min(), 1140 / 75, atol=0.5),
          f"min={poids.min():.2f} (Vehicle loan) max={poids.max():.2f} "
          f"(Credit reporting)")

    # un pd.read_csv() nu doit rendre le fichier INEXPLOITABLE (KeyError au
    # premier acces). Il ne leve pas d'exception a la lecture : c'est le
    # comportement documente, pas une ParserError.
    brut = pd.read_csv(cfg.EVAL_SAMPLE_FILE)
    check("pd.read_csv() nu ne donne aucune colonne exploitable",
          cfg.LABEL_COL not in brut.columns and cfg.WEIGHT_COL not in brut.columns
          and brut.shape[1] == 1,
          f"shape={brut.shape}, en-tete = l'avertissement")
    try:
        brut[cfg.LABEL_COL]
        check("l'acces a 'label' leve un KeyError", False)
    except KeyError:
        check("l'acces a 'label' leve un KeyError", True)

    check("fichier README d'avertissement present",
          cfg.EVAL_SAMPLE_README_FILE.exists())

    # --- load_eval_sample refuse de servir un echantillon de developpement --
    with fraction_temporaire(0.05):
        try:
            dp.load_eval_sample()
            check("load_eval_sample() leve si SAMPLE_FRACTION < 1", False,
                  "aucune exception")
        except RuntimeError as e:
            check("load_eval_sample() leve si SAMPLE_FRACTION < 1", True,
                  "message : " + str(e).splitlines()[0][:44])
        except FileNotFoundError:
            check("load_eval_sample() leve si SAMPLE_FRACTION < 1", False,
                  "FileNotFoundError : le garde-fou n'est pas passe en premier")

        # `autoriser_dev=True` doit FRANCHIR le garde-fou. Deux issues sont
        # valides et dependent de l'etat du disque, pas du comportement teste :
        #   - le chargement reussit (un echantillon de developpement existe) ;
        #   - FileNotFoundError (il n'a jamais ete genere).
        # Seule une RuntimeError est un echec : c'est ce blocage-la qui doit
        # etre leve. Tester l'absence du fichier ferait echouer ce controle le
        # jour ou le pipeline tourne en mode developpement.
        try:
            dp.load_eval_sample(autoriser_dev=True)
            check("autoriser_dev=True franchit le garde-fou", True,
                  "chargement reussi : un echantillon de developpement existe")
        except FileNotFoundError:
            check("autoriser_dev=True franchit le garde-fou", True,
                  "garde-fou franchi ; le fichier de developpement est absent")
        except RuntimeError:
            check("autoriser_dev=True franchit le garde-fou", False,
                  "toujours bloque par le garde-fou")

    check("SAMPLE_FRACTION restauree apres le bloc", cfg.SAMPLE_FRACTION == 1.0,
          f"valeur = {cfg.SAMPLE_FRACTION}")

    # --- evaluate leve (et n'avertit pas) si les poids sont ignores ---------
    y_true = df_ech[cfg.LABEL_COL].values
    rng = np.random.default_rng(0)
    # predictions synthetiques : 70 % correctes, le reste tire au hasard
    y_pred = np.where(rng.random(len(y_true)) < 0.70, y_true,
                      rng.choice(cfg.CLASS_ORDER, len(y_true)))

    # sample_weight n'a PAS de valeur par defaut : l'omettre est un TypeError,
    # immediat et inconditionnel (il ne depend pas du passage de `source`).
    try:
        mt.evaluate(y_true, y_pred)
        check("evaluate() sans sample_weight -> TypeError", False,
              "aucune exception")
    except TypeError as e:
        check("evaluate() sans sample_weight -> TypeError", True, str(e)[-52:])
    try:
        mt.plot_confusion_matrix(y_true, y_pred)
        check("plot_confusion_matrix() sans sample_weight -> TypeError", False)
    except TypeError:
        check("plot_confusion_matrix() sans sample_weight -> TypeError", True)

    # second filet : sample_weight=None alors que la source porte des poids.
    try:
        mt.evaluate(y_true, y_pred, sample_weight=None, source=df_ech)
        check("evaluate(sample_weight=None, source=...) -> ValueError", False)
    except ValueError as e:
        check("evaluate(sample_weight=None, source=...) -> ValueError", True,
              str(e).splitlines()[0][:52])

    res_pond = mt.evaluate(y_true, y_pred, sample_weight=poids, source=df_ech,
                           n_boot=200)
    res_brut = mt.evaluate(y_true, y_pred, sample_weight=None, n_boot=200)
    check("flag weighted expose",
          res_pond["weighted"] is True and res_brut["weighted"] is False)

    ecart = res_brut["f1_macro"] - res_pond["f1_macro"]
    check("le biais non pondere va dans le sens prevu (positif)", ecart > 0,
          f"F1-macro brut {res_brut['f1_macro']:.4f} vs pondere "
          f"{res_pond['f1_macro']:.4f} (ecart {ecart:+.4f})")

    # --- compare_results refuse de melanger ---------------------------------
    try:
        mt.compare_results({"LLM": res_pond, "ML": res_brut})
        check("compare_results refuse le melange pondere/non pondere", False)
    except ValueError:
        check("compare_results refuse le melange pondere/non pondere", True)

    try:
        mt.compare_results({"a": res_pond, "b": dict(res_pond, n=999)})
        check("compare_results refuse des tailles differentes", False)
    except ValueError:
        check("compare_results refuse des tailles differentes", True)

    print("\n" + "=" * 78)
    print("METRICS")
    print("=" * 78)

    check("evaluate retourne toutes les cles attendues",
          all(k in res_pond for k in ["accuracy", "f1_macro", "f1_weighted",
                                      "precision_macro", "recall_macro",
                                      "classification_report", "confusion_matrix"]))
    ci = res_pond["f1_macro_ci"]
    check("bootstrap intra-strates encadre la valeur",
          ci["bas"] < ci["valeur"] < ci["haut"],
          f"{ci['valeur']:.4f} [{ci['bas']:.4f} ; {ci['haut']:.4f}]")

    # --- le plan de reechantillonnage est expose et journalise --------------
    check("strata='auto' -> y_true quand pondere",
          res_pond["bootstrap_strata"] == "y_true (auto)",
          res_pond["bootstrap_strata"])
    check("strata='auto' -> aucune quand non pondere",
          res_brut["bootstrap_strata"] == "aucune (auto)",
          res_brut["bootstrap_strata"])
    res_sans = mt.evaluate(y_true, y_pred, sample_weight=poids, source=df_ech,
                           strata=None, n_boot=200)
    largeur_unif = (res_sans["f1_macro_ci"]["haut"]
                    - res_sans["f1_macro_ci"]["bas"])
    check("strata=None force le tirage uniforme",
          res_sans["bootstrap_strata"] == "aucune"
          and largeur_unif > (ci["haut"] - ci["bas"]),
          f"largeur uniforme {largeur_unif:.4f} vs stratifiee "
          f"{ci['haut'] - ci['bas']:.4f}")
    res_expl = mt.evaluate(y_true, y_pred, sample_weight=poids, source=df_ech,
                           strata=y_true, n_boot=200)
    check("strata=<sequence> accepte", res_expl["bootstrap_strata"] == "explicite")
    try:
        mt.evaluate(y_true, y_pred, sample_weight=poids, strata="oui", n_boot=10)
        check("strata invalide -> ValueError", False)
    except ValueError:
        check("strata invalide -> ValueError", True)

    # le bootstrap intra-strates doit donner un IC plus etroit que l'uniforme
    _, b1, h1 = mt.bootstrap_ci(y_true, y_pred, sample_weight=poids.values,
                                strata=y_true, n_boot=300)
    _, b2, h2 = mt.bootstrap_ci(y_true, y_pred, sample_weight=poids.values,
                                strata=None, n_boot=300)
    check("bootstrap stratifie plus etroit que l'uniforme", (h1 - b1) < (h2 - b2),
          f"stratifie {h1 - b1:.4f} vs uniforme {h2 - b2:.4f}")

    # --- latence ------------------------------------------------------------
    chrono = mt.Timer()
    for _ in range(25):
        with chrono:
            sum(range(2000))
    check("Timer enregistre un appel par bloc with", chrono.n == 25,
          f"n={chrono.n}")
    stats = chrono.summary()
    check("Timer expose p50, p95 et max",
          all(k in stats for k in ["latence_p50_s", "latence_p95_s",
                                   "latence_max_s"]))

    res_lat = mt.evaluate(y_true, y_pred, latencies=chrono.latencies,
                          sample_weight=poids, source=df_ech, with_ci=False)
    check("evaluate integre les latences", "latence_p95_s" in res_lat)

    # --- cout ---------------------------------------------------------------
    # Le cout attendu est DERIVE de config, jamais recopie : une valeur en dur
    # ferait de ce test une seconde source de verite, qui se perimerait des la
    # prochaine mesure. Les tokens ont ete mis a jour le 2026-08-19 (257 -> 245,
    # prefixe 260 -> 252) et cette assertion a suivi sans etre retouchee.
    tarif = cfg.MODELS_PRICING[cfg.DEFAULT_MODEL]
    tok_in = (cfg.AVG_COMPLAINT_TOKENS + cfg.PROMPT_INSTRUCTION_TOKENS
              + cfg.PROMPT_LABELS_TOKENS)
    attendu_1000 = (tok_in * tarif["input_per_1m"]
                    + cfg.AVG_OUTPUT_TOKENS * tarif["output_per_1m"]) / 1e6 * 1000
    cout = mt.estimate_cost(1000)
    check("estimate_cost defaut = mistral-small-4 a 0,15/0,60",
          cout["model"] == "mistral-small-4" and cout["input_per_1m"] == 0.15
          and abs(cout["cout_total_usd"] - attendu_1000) < 1e-6,
          f"{cout['cout_total_usd']:.4f} $ / 1000 pred. (attendu "
          f"{attendu_1000:.4f}), {cout['cout_annuel_usd']:.1f} $ par an")
    cout_test = mt.estimate_cost(70863)
    check("cout du test complet coherent avec le cout unitaire",
          abs(cout_test["cout_total_usd"] - attendu_1000 / 1000 * 70863) < 0.01,
          f"{cout_test['cout_total_usd']:.2f} $")

    # le cache de prefixe est le levier principal
    cache = mt.estimate_cost(1000, cached_prefix_tokens=cfg.CACHED_PREFIX_TOKENS)
    economie = (1 - cache["cout_total_usd"] / cout["cout_total_usd"]) * 100
    check("le cache de prefixe reduit le cout de plus de 40 %", economie > 40,
          f"{cout['cout_total_usd']:.4f} -> {cache['cout_total_usd']:.4f} $ "
          f"({economie:.1f} %)")
    check("cache non modelise si la remise n'est pas relevee",
          mt.estimate_cost(1000, "gemini-3.5-flash",
                           cached_prefix_tokens=260)["cache_modelise"] is False,
          "gemini-3.5-flash : remise non relevee, cout non minore")
    check("DeepSeek cache automatiquement a 98 %",
          cfg.MODELS_PRICING["deepseek-v4-flash"]["cache_automatique"] is True
          and cfg.MODELS_PRICING["deepseek-v4-flash"]["remise_cache"] == 0.98)

    # tarif d'introduction claude-sonnet-5 : bascule au 01/09/2026
    avant = mt.estimate_cost(100, "claude-sonnet-5",
                             date_reference=dt.date(2026, 8, 31))
    with warnings.catch_warnings(record=True) as capte:
        warnings.simplefilter("always")
        apres = mt.estimate_cost(100, "claude-sonnet-5",
                                 date_reference=dt.date(2026, 9, 1))
    check("tarif d'introduction actif jusqu'au 31/08/2026",
          avant["input_per_1m"] == 2.00
          and not avant["avertissements"][0].startswith("'claude-sonnet-5' : le tarif"),
          "2,00/10,00 $ + mention du caractere provisoire")
    check("bascule au tarif standard le 01/09/2026 AVEC avertissement",
          apres["input_per_1m"] == 3.00 and apres["output_per_1m"] == 15.00
          and len(capte) == 1,
          f"3,00/15,00 $ (+50 %), cout {avant['cout_total_usd']:.4f} -> "
          f"{apres['cout_total_usd']:.4f} $")

    # tableau comparatif multi-modeles
    tab_cout = mt.compare_models_cost(2000)
    check("compare_models_cost couvre les 10 modeles",
          len(tab_cout) == len(cfg.MODELS_PRICING) == 10, f"{len(tab_cout)} lignes")
    check("le tableau est trie du moins cher au plus cher",
          tab_cout.index[0] == "ministral-3b" and tab_cout.index[-1] == "gpt-5.6-sol",
          f"{tab_cout.index[0]} -> {tab_cout.index[-1]}")
    # Bornes DERIVEES des tarifs et des volumes de config, pas recopiees : ce
    # sont les memes chiffres qui ont bouge le 2026-08-19 avec la mesure des
    # tokens. Ce qui est verifie ici est l'ECART entre le moins cher et le plus
    # cher du catalogue -- deux ordres de grandeur -- pas sa valeur absolue.
    def _annuel(cle):
        t = cfg.MODELS_PRICING[cle]
        tok = (cfg.AVG_COMPLAINT_TOKENS + cfg.PROMPT_INSTRUCTION_TOKENS
               + cfg.PROMPT_LABELS_TOKENS)
        return ((tok * t["input_per_1m"] + cfg.AVG_OUTPUT_TOKENS
                 * t["output_per_1m"]) / 1e6 * cfg.DAILY_COMPLAINTS * 365)

    moins_cher = min(cfg.MODELS_PRICING, key=_annuel)
    plus_cher = max(cfg.MODELS_PRICING, key=_annuel)
    check("fourchette annuelle sans cache : deux ordres de grandeur",
          abs(tab_cout["$/an sans cache"].iloc[0] - _annuel(moins_cher)) < 1
          and abs(tab_cout["$/an sans cache"].iloc[-1] - _annuel(plus_cher)) < 5
          and _annuel(plus_cher) / _annuel(moins_cher) > 40,
          f"{tab_cout['$/an sans cache'].iloc[0]:.0f} $ ({moins_cher}) -> "
          f"{tab_cout['$/an sans cache'].iloc[-1]:.0f} $ ({plus_cher}), "
          f"rapport x{_annuel(plus_cher) / _annuel(moins_cher):.0f}")

    try:
        mt.estimate_cost(1000, "modele-inexistant")
        check("modele inconnu -> KeyError", False)
    except KeyError:
        check("modele inconnu -> KeyError", True)
    try:
        mt.estimate_cost(1000, cached_prefix_tokens=99_999)
        check("prefixe cache > entree -> ValueError", False)
    except ValueError:
        check("prefixe cache > entree -> ValueError", True)

    # --- figure -------------------------------------------------------------
    chemin = cfg.FIGURES_DIR / "_test_confusion.png"
    mt.plot_confusion_matrix(y_true, y_pred, save_path=chemin,
                             sample_weight=poids, source=df_ech)
    check("plot_confusion_matrix ecrit un PNG",
          chemin.exists() and chemin.stat().st_size > 10_000,
          f"{chemin.stat().st_size / 1024:.0f} Ko")
    chemin.unlink()

    # --- comparaison --------------------------------------------------------
    res_b = mt.evaluate(y_true,
                        np.where(rng.random(len(y_true)) < 0.62, y_true,
                                 rng.choice(cfg.CLASS_ORDER, len(y_true))),
                        sample_weight=poids, source=df_ech, n_boot=200)
    tab = mt.compare_results({"LLM (simule)": res_pond, "ML (simule)": res_b})
    check("compare_results produit un tableau",
          isinstance(tab, pd.DataFrame) and len(tab) == 2)
    print("\n" + tab[["n", "accuracy", "F1-macro", "F1-macro IC 95 %",
                      "F1 · Vehicle loan or lease"]].to_string())

    print("\n" + "=" * 78)
    print("COHERENCE DU PIPELINE")
    print("=" * 78)

    train, test = dp.load_split()
    check("train et test disjoints sur complaint_id",
          len(set(train[cfg.ID_COL]) & set(test[cfg.ID_COL])) == 0)
    check("aucun texte partage entre train et test",
          len(set(train[cfg.TEXT_COL]) & set(test[cfg.TEXT_COL])) == 0,
          "c'est ce que garantit le dedoublonnage avant le split")
    check("aucun doublon de texte dans le corpus",
          train[cfg.TEXT_COL].duplicated().sum() == 0
          and test[cfg.TEXT_COL].duplicated().sum() == 0)
    check("tous les textes font au moins MIN_TEXT_LENGTH caracteres",
          train[cfg.TEXT_COL].str.len().min() >= cfg.MIN_TEXT_LENGTH,
          f"min={train[cfg.TEXT_COL].str.len().min()}")
    check("9 classes exactement", set(train[cfg.LABEL_COL]) == set(cfg.CLASS_ORDER))
    check("echantillon d'evaluation inclus dans le test",
          set(df_ech[cfg.ID_COL]).issubset(set(test[cfg.ID_COL])))
    check("plancher de 50 respecte",
          df_ech[cfg.LABEL_COL].value_counts().min() >= cfg.LLM_EVAL_MIN_PER_CLASS,
          f"min={df_ech[cfg.LABEL_COL].value_counts().min()}")
    check("les poids reconstituent la population du test",
          abs(poids.sum() - len(test)) / len(test) < 0.001,
          f"somme des poids {poids.sum():,.0f} vs test {len(test):,}")
    check("colonne strate presente", cfg.STRATE_COL in df_ech.columns)

    print("\n" + "=" * 78)
    print(f"RESULTAT : {sum(_RESULTATS)}/{len(_RESULTATS)} verifications passees")
    print("=" * 78)
    return 0 if all(_RESULTATS) else 1


if __name__ == "__main__":
    sys.exit(main())
