# tools/ — scripts de génération des livrables

Ces scripts **produisent** des fichiers du dépôt. Ils ne font partie d'aucun pipeline
d'exécution : `src/` ne les importe pas et ils ne tournent pas en production. Ils sont
versionnés parce qu'ils sont la **source de vérité** de livrables qui, sans eux, ne
seraient pas régénérables sur un clone.

Tous se lancent depuis n'importe quel répertoire (résolution de la racine par
`__file__`) et écrivent leurs résultats intermédiaires dans `scratch/`, qui n'est pas
versionné.

| Script | Produit | Durée |
|---|---|---|
| `build_notebook.py` | `notebooks/01_exploration.ipynb`, écrit **et exécuté** | ~2 min |
| `limites_analyse.py` | les mesures chiffrées de `reports/limites.md` | ~1 min |
| `h1_simulation.py` | simulation Monte-Carlo du plan d'échantillonnage → `reports/h1_sampling_simulation.md` | ~30 s |
| `h1_apparie.py` | complément : précision de l'écart apparié LLM/ML | ~1 min |
| `h1_decomposition.py` | complément : décomposition du bruit sur la classe la plus faible | ~20 s |
| `h1_figures.py` | les trois figures `reports/figures/h1_*.png` | ~10 s |

```bash
python tools/build_notebook.py       # régénère le notebook
python tools/h1_simulation.py        # rejoue l'arbitrage du plan d'échantillonnage
python tools/h1_figures.py           # doit suivre h1_simulation.py et h1_decomposition.py
```

## Le notebook est un artefact généré

`notebooks/01_exploration.ipynb` **ne doit pas être édité à la main** : toute correction
se fait dans `build_notebook.py`, puis on régénère. Éditer le `.ipynb` directement fait
diverger les deux et la correction sera perdue à la prochaine génération.

Le notebook étant remis au client, ses **sorties** ne doivent contenir ni chemin absolu
ni nom d'utilisateur — un `.ipynb` conserve ce qu'il affiche. `build_notebook.py`
n'imprime que des chemins relatifs à la racine du projet.

## Dépendance entre les scripts H.1

`h1_apparie.py` et `h1_decomposition.py` importent `h1_simulation` (matrice de
confusion, allocations, fonction de métriques) : lancer `h1_simulation.py` en premier.
`h1_figures.py` lit les `.csv` et `.npz` que les trois précédents ont déposés dans
`scratch/`.
