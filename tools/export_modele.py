"""Export deterministe et verifiable du modele ML de l'etape 3.

    python tools/export_modele.py                    # exporte le modele retenu
    python tools/export_modele.py --modele MultinomialNB
    python tools/export_modele.py --verifie          # reentraine et compare, n'ecrit rien

CE QUE CE SCRIPT GARANTIT, ET COMMENT LE VERIFIER
-------------------------------------------------
Le pickle n'est pas versionne : `models/` est exclu du depot, au meme titre que
`data/`, parce qu'il est reproductible. « Reproductible » n'est pas une
affirmation mais une propriete controlable :

  * la generation est ENTIEREMENT scriptee ici -- aucun reglage n'est saisi a la
    main, tout vient de constantes de ce fichier ou de `src/config.py` ;
  * elle est DETERMINISTE -- `random_state` fixe partout, `TfidfVectorizer` trie
    son vocabulaire par ordre alphabetique avant d'assigner les indices (donc
    independant de `PYTHONHASHSEED`), `MultinomialNB` ne fait que compter ;
  * elle produit une EMPREINTE controlable, et `--verifie` reentraine puis
    compare a l'empreinte enregistree, sans rien ecrire.

DEUX EMPREINTES, ET POURQUOI
----------------------------
`sha256_pickle` est le hachage des octets du fichier livre : c'est ce qu'une
release attache, et ce qu'un telechargement verifie.

`empreinte_contenu` hache le CONTENU du modele -- vocabulaire dans l'ordre des
indices, `idf_`, coefficients -- arrondi a 12 decimales. C'est elle qui porte la
verification de determinisme ; `sha256_pickle` porte l'integrite du transfert.
Les deux sont enregistrees.

  DEUX EXPORTS DU MEME MODELE ONT DES `sha256_pickle` DIFFERENTS.
  CE N'EST PAS UN DEFAUT DE REPRODUCTIBILITE. Lire ci-dessous avant de
  conclure quoi que ce soit d'un tel ecart.

Constat mesure le 2026-08-19 : deux executions de la commande identique, meme
machine, memes versions, memes donnees, produisent la meme `empreinte_contenu`
et deux `sha256_pickle` distincts. `PYTHONHASHSEED=0` n'y change rien.

La cause a ete localisee en hachant SEPAREMENT chaque attribut du vectoriseur
ajuste. Sur les 25 attributs, un seul varie d'un processus a l'autre :

    _stop_words_id      int, 19 octets       <-- VARIE
    vocabulary_         dict, 893 692 o      identique
    idf_                                     identique
    _tfidf              TfidfTransformer     identique
    coef_ (octets bruts, sans arrondi)       identique
    ... les 21 autres                        identiques

`_stop_words_id` vaut `id(self.stop_words)` : une ADRESSE MEMOIRE, que
scikit-learn conserve comme cle de cache pour eviter de revalider la liste de
mots vides. Elle change a chaque processus par construction, et n'a aucun effet
sur une prediction. Le modele est donc bit-a-bit identique ; seule cette
adresse ne l'est pas.

D'ou la separation des roles, qui est une conclusion de mesure et non une
precaution de principe :

  * `empreinte_contenu` ne hache que ce qui determine une prediction. Elle est
    stable entre processus (verifie par `tests/test_export_modele.py`), sensible
    a une modification de coefficient de 1e-6, et tolerante a 1e-14 pour qu'une
    mise a jour de BLAS ne la fasse pas echouer.
  * `sha256_pickle` identifie UN fichier livre. Il repond a « l'octet que je
    telecharge est-il celui qui a ete publie ? », et pas a « ce modele est-il
    reproductible ? ». Le comparer entre deux constructions n'a pas de sens.

Si une version future de scikit-learn retire `_stop_words_id`, le test le
signale et cette reserve est a relire.

METADONNEES VERSIONNEES
-----------------------
`models/<modele>.metadata.json` EST versionne, lui. Il permet de savoir ce qu'un
pickle contient sans le charger : configuration du vectoriseur, classifieur et
ses parametres, taille du vocabulaire, empreintes, scores, versions des
bibliotheques, date.

INTEGRATION CONTINUE
--------------------
Aucun chemin absolu, aucune variable d'environnement, aucune dependance a une
session : la racine est retrouvee depuis `__file__`, les donnees sont lues dans
`data/processed/` via `src.config`.

RESERVE A LEVER AVANT D'ECRIRE L'ACTION : `data/` est exclu du depot. Un runner
d'integration continue n'a donc PAS `data/processed/train.csv` apres un
`checkout` seul. Il devra d'abord recuperer le dataset brut (639 Mo) et lancer
`python -m src.data_prep`, ou restaurer `data/processed/` depuis un cache ou un
artefact. Le script echoue avec un message explicite si le fichier manque.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import platform
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

RACINE = Path(__file__).resolve().parent
while not (RACINE / "src" / "config.py").exists() and RACINE != RACINE.parent:
    RACINE = RACINE.parent
sys.path.insert(0, str(RACINE))

from src import config as cfg            # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer   # noqa: E402
from sklearn.linear_model import LogisticRegression           # noqa: E402
from sklearn.naive_bayes import MultinomialNB                 # noqa: E402
from sklearn.pipeline import Pipeline                         # noqa: E402
from sklearn.svm import LinearSVC                             # noqa: E402
from sklearn.utils.class_weight import compute_sample_weight  # noqa: E402

import scipy                                                  # noqa: E402
import sklearn                                                # noqa: E402

# ===========================================================================
# Configuration figee — issue de la phase A (reports/resultats_phaseA.md)
# ===========================================================================
VECTORISEUR = dict(lowercase=True, sublinear_tf=True, ngram_range=(1, 2), min_df=2)

# `solver="liblinear"` choisi sur sonde de temps mesuree, pas par defaut : c'est
# aussi l'analogue le plus proche de LinearSVC (meme bibliotheque, meme schema
# un-contre-tous), ce qui isole la fonction de perte comme seule difference.
CLASSIFIEURS = {
    "LinearSVC": lambda: LinearSVC(
        class_weight="balanced", random_state=cfg.RANDOM_SEED),
    "LogisticRegression": lambda: LogisticRegression(
        solver="liblinear", max_iter=200,
        class_weight="balanced", random_state=cfg.RANDOM_SEED),
    # MultinomialNB n'accepte pas `class_weight` : le rebalancement passe par
    # `sample_weight`. Voir reports/resultats_phaseB.md — cette voie n'agit PAS
    # comme un rebalancement de l'a priori, et son effet principal transite par
    # le lissage `alpha`.
    "MultinomialNB": lambda: MultinomialNB(),
}
MODELE_PAR_DEFAUT = "LinearSVC"
SCORES_FILE = cfg.REPORTS_DIR / "etape3_scores.json"
MODELS_DIR = cfg.PROJECT_ROOT / "models"

# Attributs decisifs par classifieur, dans un ordre FIXE : l'empreinte doit etre
# reproductible, donc l'ordre de hachage ne peut pas dependre d'un parcours de
# dictionnaire.
ATTRIBUTS_MODELE = ("classes_", "coef_", "intercept_",
                    "feature_log_prob_", "class_log_prior_")


# ===========================================================================
# Empreintes
# ===========================================================================
def _hache_tableau(h: "hashlib._Hash", a) -> None:
    """Ajoute un tableau au hachage, arrondi a 12 decimales.

    L'arrondi absorbe le dernier bit de la representation flottante, qui peut
    varier avec la version de BLAS sans que le modele soit different. `+ 0.0`
    normalise `-0.0` en `+0.0`, qui n'ont pas la meme representation binaire.
    """
    a = np.asarray(a)
    if a.dtype.kind in "OUS":                       # `classes_` : etiquettes texte
        h.update("\x1f".join(map(str, a.ravel())).encode("utf-8"))
        return
    a = np.ascontiguousarray(np.round(a.astype(np.float64), 12) + 0.0)
    h.update(str(a.shape).encode("utf-8"))
    h.update(a.tobytes())


def empreinte_contenu(pipe: Pipeline) -> str:
    """Hachage du contenu du modele, independant de la serialisation."""
    h = hashlib.sha256()
    vec, clf = pipe.named_steps["tfidf"], pipe.named_steps["clf"]
    h.update(json.dumps(vec.get_params(), sort_keys=True, default=str).encode("utf-8"))
    h.update(json.dumps(clf.get_params(), sort_keys=True, default=str).encode("utf-8"))
    # Vocabulaire dans l'ordre des indices : deux ajustements identiques donnent
    # la meme suite, un vocabulaire different la change.
    termes = sorted(vec.vocabulary_, key=vec.vocabulary_.get)
    h.update("\n".join(termes).encode("utf-8"))
    _hache_tableau(h, vec.idf_)
    for nom in ATTRIBUTS_MODELE:
        if hasattr(clf, nom):
            h.update(nom.encode("utf-8"))
            _hache_tableau(h, getattr(clf, nom))
    return h.hexdigest()


def sha256_fichier(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for bloc in iter(lambda: f.read(1 << 20), b""):
            h.update(bloc)
    return h.hexdigest()


# ===========================================================================
# Entrainement
# ===========================================================================
def charge_train(echantillon: int | None = None) -> tuple:
    if not cfg.TRAIN_FILE.exists():
        raise SystemExit(
            f"Fichier absent : {cfg.TRAIN_FILE.relative_to(cfg.PROJECT_ROOT)}\n"
            "`data/` est exclu du depot. Reconstruire avec :\n"
            "    python -m src.data_prep\n"
            "ce qui suppose `data/raw/dataset.csv` present (639 Mo)."
        )
    df = pd.read_csv(cfg.TRAIN_FILE, encoding="utf-8", nrows=echantillon)
    return df[cfg.TEXT_COL].values, df[cfg.LABEL_COL].values, len(df)


def entraine(nom: str, echantillon: int | None = None) -> tuple[Pipeline, int]:
    if nom not in CLASSIFIEURS:
        raise SystemExit(f"Modele inconnu : {nom}. Connus : {', '.join(CLASSIFIEURS)}")
    X, y, n = charge_train(echantillon)
    pipe = Pipeline([("tfidf", TfidfVectorizer(**VECTORISEUR)),
                     ("clf", CLASSIFIEURS[nom]())])
    if nom == "MultinomialNB":
        Xt = pipe.named_steps["tfidf"].fit_transform(X)
        pipe.named_steps["clf"].fit(
            Xt, y, sample_weight=compute_sample_weight("balanced", y))
    else:
        pipe.fit(X, y)

    # Un modele qui ne connait pas les 9 classes ne peut RIEN predire pour
    # celles qu'il ignore : son F1-macro serait plafonne sans que rien ne le
    # signale. `Vehicle loan or lease` ne pese que 0,34 % du train et disparait
    # du premier millier de lignes ; l'export sur donnees tronquees est donc un
    # risque reel, pas theorique.
    manquantes = sorted(set(cfg.CLASS_ORDER) - set(map(str, pipe.named_steps["clf"].classes_)))
    if manquantes and echantillon is None:
        raise SystemExit(
            "Referentiel incomplet : classe(s) absente(s) du train -> "
            + ", ".join(manquantes) + "\nExport refuse."
        )
    if manquantes:
        # sur stderr : stdout porte les empreintes, qu'un appelant peut lire.
        print(f"  AVERTISSEMENT : referentiel incomplet ({len(manquantes)} classe(s) "
              f"absente(s)) — echantillon reduit, export de test uniquement.",
              file=sys.stderr)
    return pipe, n


def metadonnees(nom: str, pipe: Pipeline, n_lignes: int,
                sha_pickle: str | None, scores: dict | None) -> dict:
    vec, clf = pipe.named_steps["tfidf"], pipe.named_steps["clf"]
    return {
        "modele": nom,
        "genere_le": date.today().isoformat(),
        "genere_par": "tools/export_modele.py",
        "avertissement": (
            "Les scores ci-dessous proviennent d'une validation croisee sur le "
            "TRAIN. Ils ne sont comparables a aucun chiffre de l'etape 2 : "
            "donnees, ponderation et role differents."),
        "vectoriseur": {k: (list(v) if isinstance(v, tuple) else v)
                        for k, v in VECTORISEUR.items()},
        "classifieur": {"classe": type(clf).__name__,
                        "parametres": {k: (str(v) if not isinstance(
                            v, (int, float, str, bool, type(None))) else v)
                            for k, v in sorted(clf.get_params().items())},
                        "rebalancement": ("sample_weight='balanced'"
                                          if nom == "MultinomialNB"
                                          else "class_weight='balanced'")},
        "vocabulaire": {"taille": len(vec.vocabulary_),
                        "classes": list(map(str, clf.classes_)),
                        "n_classes": len(clf.classes_),
                        "referentiel_complet": sorted(map(str, clf.classes_))
                                               == sorted(cfg.CLASS_ORDER)},
        "donnees": {"fichier": str(cfg.TRAIN_FILE.relative_to(cfg.PROJECT_ROOT)),
                    "lignes_utilisees": n_lignes,
                    "sha256": sha256_fichier(cfg.TRAIN_FILE)},
        "empreintes": {"contenu": empreinte_contenu(pipe),
                       "sha256_pickle": sha_pickle},
        "scores": scores,
        "environnement": {"python": platform.python_version(),
                          "scikit_learn": sklearn.__version__,
                          "numpy": np.__version__,
                          "scipy": scipy.__version__,
                          "pandas": pd.__version__,
                          "plateforme": platform.system()},
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--modele", default=MODELE_PAR_DEFAUT, choices=list(CLASSIFIEURS))
    p.add_argument("--echantillon", type=int, default=None,
                   help="limite le nombre de lignes (tests seulement)")
    p.add_argument("--verifie", action="store_true",
                   help="reentraine et compare a l'empreinte enregistree, n'ecrit rien")
    p.add_argument("--sans-scores", action="store_true",
                   help="autorise l'absence de reports/etape3_scores.json")
    a = p.parse_args(argv)

    pipe, n = entraine(a.modele, a.echantillon)
    emp = empreinte_contenu(pipe)

    if a.verifie:
        meta_path = MODELS_DIR / f"{a.modele}.metadata.json"
        if not meta_path.exists():
            print(f"Aucune metadonnee a comparer : {meta_path}"); return 1
        attendu = json.loads(meta_path.read_text(encoding="utf-8"))["empreintes"]["contenu"]
        ok = emp == attendu
        print(f"empreinte attendue : {attendu}\nempreinte obtenue  : {emp}\n"
              f"{'IDENTIQUE — export deterministe' if ok else 'DIFFERENTE — export NON deterministe'}")
        return 0 if ok else 1

    scores = None
    if SCORES_FILE.exists():
        scores = json.loads(SCORES_FILE.read_text(encoding="utf-8")).get(a.modele)
    elif not a.sans_scores:
        raise SystemExit(
            f"Fichier de scores absent : {SCORES_FILE.relative_to(cfg.PROJECT_ROOT)}\n"
            "Les metadonnees ne doivent pas etre ecrites sans scores. "
            "Utiliser --sans-scores pour passer outre (tests).")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    pkl = MODELS_DIR / f"{a.modele}.pkl"
    pkl.write_bytes(pickle.dumps(pipe, protocol=5))
    meta = metadonnees(a.modele, pipe, n, sha256_fichier(pkl), scores)
    (MODELS_DIR / f"{a.modele}.metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"  {pkl.relative_to(cfg.PROJECT_ROOT)}  {pkl.stat().st_size/1e6:.1f} Mo")
    print(f"  vocabulaire      {meta['vocabulaire']['taille']:,}")
    print(f"  empreinte contenu {emp}")
    print(f"  sha256 pickle     {meta['empreintes']['sha256_pickle']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
