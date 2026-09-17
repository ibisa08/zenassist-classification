"""Verifie que `tools/export_modele.py` produit un artefact CONTROLABLE.

`models/` est exclu du depot au motif que son contenu est reproductible. Ce
motif n'a de valeur que s'il est verifie : sans ce fichier, « reproductible »
est une affirmation, pas une propriete.

CE QUI EST VERIFIE
------------------
  1. deux ajustements identiques donnent la meme empreinte de contenu, dans le
     meme processus ET dans deux processus distincts ;
  2. l'empreinte est SENSIBLE -- un coefficient modifie au 1e-6, un terme de
     vocabulaire change, et elle change ;
  3. l'empreinte est TOLERANTE au dernier bit -- une perturbation a 1e-14 ne la
     change pas, sinon une mise a jour de BLAS suffirait a la faire echouer ;
  4. l'empreinte IGNORE `_stop_words_id`, seul attribut du vectoriseur dont la
     valeur varie d'un processus a l'autre (c'est `id(self.stop_words)`, une
     adresse memoire servant de cle de cache a scikit-learn). C'est la raison
     pour laquelle `sha256_pickle` ne peut PAS servir de controle de
     determinisme, et pour laquelle il en existe deux ;
  5. les metadonnees portent les champs exiges et aucun chemin absolu -- un
     chemin de machine locale dans un fichier versionne serait une fuite
     d'environnement, exactement ce que l'integration continue doit exclure.

Aucun appel reseau, aucune ecriture dans `models/`.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

RACINE = Path(__file__).resolve().parent
while not (RACINE / "src" / "config.py").exists() and RACINE != RACINE.parent:
    RACINE = RACINE.parent
sys.path.insert(0, str(RACINE))
sys.path.insert(0, str(RACINE / "tools"))

from src import config as cfg      # noqa: E402
import export_modele as ex         # noqa: E402

OK, KO = "  OK  ", " ECHEC"
_RESULTATS: list[bool] = []
N_LIGNES = 1200          # assez pour un vocabulaire non trivial, assez peu pour un test


def check(nom, cond, detail=""):
    _RESULTATS.append(bool(cond))
    print(f"[{OK if cond else KO}] {nom}" + (f"   -> {detail}" if detail else ""))


def main() -> int:
    if not cfg.TRAIN_FILE.exists():
        print(f"IGNORE : {cfg.TRAIN_FILE.name} absent (data/ est exclu du depot).")
        print("Reconstruire avec `python -m src.data_prep` pour executer ce test.")
        return 0

    # --- 1. determinisme -----------------------------------------------------
    p1, n1 = ex.entraine("LinearSVC", N_LIGNES)
    p2, _ = ex.entraine("LinearSVC", N_LIGNES)
    e1, e2 = ex.empreinte_contenu(p1), ex.empreinte_contenu(p2)
    check("deux ajustements, meme processus : empreinte identique", e1 == e2, e1[:16])

    code = ("import sys; sys.path.insert(0, r'%s'); sys.path.insert(0, r'%s');"
            "import export_modele as ex;"
            "p,_ = ex.entraine('LinearSVC', %d); print(ex.empreinte_contenu(p))"
            % (RACINE, RACINE / "tools", N_LIGNES))
    e3 = subprocess.run([sys.executable, "-c", code], capture_output=True,
                        text=True, check=True).stdout.strip()
    check("deux processus distincts : empreinte identique", e1 == e3, e3[:16])

    # --- 2. sensibilite ------------------------------------------------------
    clf = p2.named_steps["clf"]
    clf.coef_[0, 0] += 1e-6
    check("un coefficient modifie a 1e-6 change l'empreinte",
          ex.empreinte_contenu(p2) != e1)
    clf.coef_[0, 0] -= 1e-6
    check("coefficient restaure : empreinte revenue a l'identique",
          ex.empreinte_contenu(p2) == e1)

    vec = p2.named_steps["tfidf"]
    terme = next(iter(vec.vocabulary_))
    vec.vocabulary_["\x00terme-inexistant"] = vec.vocabulary_.pop(terme)
    check("un terme de vocabulaire change modifie l'empreinte",
          ex.empreinte_contenu(p2) != e1)
    vec.vocabulary_[terme] = vec.vocabulary_.pop("\x00terme-inexistant")

    # --- 3. tolerance au dernier bit ----------------------------------------
    clf.coef_[0, 0] += 1e-14
    check("une perturbation a 1e-14 ne change PAS l'empreinte",
          ex.empreinte_contenu(p2) == e1,
          "sinon une mise a jour de BLAS ferait echouer la verification")
    clf.coef_[0, 0] -= 1e-14

    # --- 4. le champ instable est bien exclu --------------------------------
    avant = ex.empreinte_contenu(p1)
    check("`_stop_words_id` present sur le vectoriseur ajuste",
          hasattr(p1.named_steps["tfidf"], "_stop_words_id"),
          "si scikit-learn le retire, la reserve sur sha256_pickle est a relire")
    p1.named_steps["tfidf"]._stop_words_id = 123456789
    check("`_stop_words_id` modifie ne change PAS l'empreinte de contenu",
          ex.empreinte_contenu(p1) == avant)

    # --- 5. metadonnees ------------------------------------------------------
    meta = ex.metadonnees("LinearSVC", p1, n1, None, None)
    for champ in ("modele", "genere_le", "vectoriseur", "classifieur",
                  "vocabulaire", "donnees", "empreintes", "scores", "environnement"):
        check(f"metadonnees : champ `{champ}` present", champ in meta)
    check("metadonnees : taille du vocabulaire renseignee",
          meta["vocabulaire"]["taille"] > 0, f"{meta['vocabulaire']['taille']:,}")
    check("metadonnees : classes toutes issues du referentiel",
          set(meta["vocabulaire"]["classes"]) <= set(cfg.CLASS_ORDER))
    # Sur 1 200 lignes, `Vehicle loan or lease` (0,34 % du train) est absente.
    # C'est voulu : le test verifie ici que l'incompletude est SIGNALEE, et non
    # qu'elle n'arrive pas. Le refus d'export, lui, ne s'applique qu'en l'absence
    # de `--echantillon` -- sinon aucun test rapide ne serait possible.
    check("metadonnees : incompletude du referentiel signalee",
          meta["vocabulaire"]["referentiel_complet"] is False,
          f"{meta['vocabulaire']['n_classes']}/9 classes sur {N_LIGNES} lignes")
    check("metadonnees : version de scikit-learn renseignee",
          bool(meta["environnement"]["scikit_learn"]))
    check("metadonnees : rebalancement documente",
          meta["classifieur"]["rebalancement"] == "class_weight='balanced'")

    texte = json.dumps(meta, ensure_ascii=False)
    check("metadonnees : aucun chemin absolu",
          str(cfg.PROJECT_ROOT) not in texte and "/Users/" not in texte
          and "/home/" not in texte,
          "un chemin de machine locale dans un fichier versionne est une fuite")
    check("metadonnees : chemin des donnees relatif a la racine",
          meta["donnees"]["fichier"] == "data/processed/train.csv",
          meta["donnees"]["fichier"])
    check("metadonnees : serialisables en JSON sans repli",
          json.loads(json.dumps(meta, ensure_ascii=False))["modele"] == "LinearSVC")

    # --- 6. garde-fou sur les scores ----------------------------------------
    check("`SCORES_FILE` pointe dans reports/, pas dans models/",
          ex.SCORES_FILE.parent == cfg.REPORTS_DIR, ex.SCORES_FILE.name)
    check("le modele par defaut est celui retenu en phase A",
          ex.MODELE_PAR_DEFAUT in ex.CLASSIFIEURS)
    check("configuration du vectoriseur conforme a la phase A",
          ex.VECTORISEUR == dict(lowercase=True, sublinear_tf=True,
                                 ngram_range=(1, 2), min_df=2))

    # --- 7. controle de la version de scikit-learn ---------------------------
    # L'export du 2026-08-19 est parti sous scikit-learn 1.6.1 alors que
    # requirements.txt epingle 1.9.0, sans que rien ne le signale. Le garde-fou
    # n'a de valeur que si son ECHEC est verifie, pas seulement son succes.
    import sklearn

    check("verifie_version_sklearn passe sous l'environnement courant",
          ex.verifie_version_sklearn() == sklearn.__version__,
          sklearn.__version__)
    check("version requise lue depuis requirements.txt",
          ex.version_sklearn_requise() == sklearn.__version__,
          ex.version_sklearn_requise())

    lecture_reelle = ex.version_sklearn_requise
    try:
        ex.version_sklearn_requise = lambda *a, **k: "0.0.0-inexistante"
        try:
            ex.verifie_version_sklearn()
            leve, message = False, ""
        except SystemExit as e:
            leve, message = True, str(e)
    finally:
        ex.version_sklearn_requise = lecture_reelle

    check("version attendue differente -> SystemExit", leve)
    check("le message nomme l'attendu, l'installe et l'interpreteur",
          leve and "0.0.0-inexistante" in message
          and sklearn.__version__ in message and sys.executable in message)
    check("lecture reelle restauree apres le test",
          ex.version_sklearn_requise is lecture_reelle
          and ex.version_sklearn_requise() == sklearn.__version__)

    # Le controle doit preceder l'entrainement, y compris sous `--verifie` :
    # on simule une version non conforme et on verifie que `main()` sort AVANT
    # d'avoir ajuste quoi que ce soit (`entraine` remplace par un piege).
    entraine_reel, appels = ex.entraine, []
    try:
        ex.version_sklearn_requise = lambda *a, **k: "0.0.0-inexistante"
        ex.entraine = lambda *a, **k: appels.append(1)
        try:
            ex.main(["--modele", "LinearSVC", "--verifie"])
            leve2 = False
        except SystemExit:
            leve2 = True
    finally:
        ex.version_sklearn_requise = lecture_reelle
        ex.entraine = entraine_reel

    check("`--verifie` est soumis au controle de version", leve2)
    check("aucun entrainement n'a ete lance avant le controle",
          appels == [], f"{len(appels)} appel(s) a entraine()")

    print("\n" + "=" * 78)
    print(f"RESULTAT : {sum(_RESULTATS)}/{len(_RESULTATS)} verifications passees")
    if not all(_RESULTATS):
        print("\n  L'export n'est plus verifiable. `models/` est exclu du depot au")
        print("  motif qu'il est reproductible : ce motif tombe si ce test echoue.")
    print("=" * 78)
    return 0 if all(_RESULTATS) else 1


if __name__ == "__main__":
    sys.exit(main())
