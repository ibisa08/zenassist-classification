# Contrainte transmise à la mission suivante — export du modèle en intégration continue

**Destinataire : la mission « GitHub Action attachant le pickle aux assets d'une
release ».** Ce document n'arbitre pas ; il transmet un obstacle identifié pendant
l'étape 3, les options possibles, et ce que chacune coûte. La décision se prend
là-bas.

## L'obstacle

`tools/export_modele.py` a été vérifié comme indépendant de tout environnement
local : racine retrouvée depuis `__file__`, aucun chemin absolu, aucune variable
de session, données lues via `src.config`. Une exécution depuis `/` avec
`env -i` réussit.

**Mais `data/` est exclu du dépôt.** Un runner qui fait un `checkout` seul n'a
donc pas `data/processed/train.csv`, et ne peut pas entraîner. Le script échoue
avec un message explicite plutôt que de partir sur des données absentes.

Grandeurs mesurées le 2026-08-19 :

| fichier | taille | rôle |
|---|---|---|
| `data/raw/dataset.csv` | 639 Mo (609 Mio) | source, non versionnée |
| `data/processed/train.csv` | 325 Mo (310 Mio) | **seul fichier nécessaire à l'export** |
| `data/processed/test.csv` | 82 Mo (78 Mio) | non requis par l'export |
| `train.csv` compressé `gzip -9` | ≈ 98 Mo | extrapolé de 20 000 lignes, ratio 3,3 |
| pickle produit | ≈ 90 Mo | sous la limite de 2 Go par asset de release |

## Option A — rejouer `src.data_prep` depuis le dataset brut

Le runner télécharge `dataset.csv` (639 Mo) depuis la source CFPB, puis lance
`python -m src.data_prep`.

**Ce que ça implique.** Stockage sur le runner : 639 Mo de brut + 325 Mo de train
+ 82 Mo de test, soit **environ 1,05 Go**. Temps : celui du téléchargement, plus
celui de `data_prep` — **non mesuré, délibérément** (voir la réserve plus bas).

**Avantage.** La chaîne part de la source ; rien à conserver hors du dépôt, rien
qui puisse périmer.

**Inconvénient, et il est sérieux.** Le résultat dépend de la stabilité d'une
source externe. Le CFPB met son jeu à jour : si le brut change, le split change,
et le modèle change **silencieusement**. Le split est stratifié aléatoire à
graine fixe, ce qui garantit la reproductibilité *à données identiques*, pas la
stabilité *dans le temps*.

## Option B — restaurer `data/processed/` depuis un cache ou un artefact

**Ce que ça implique.** Environ 98 Mo compressés pour `train.csv` seul.
Restauration de l'ordre de quelques dizaines de secondes. Limites GitHub à
vérifier là-bas : 10 Go de cache par dépôt, et **éviction après 7 jours
d'inactivité** — une release peu fréquente retrouvera un cache vide. L'option B
a donc besoin d'un repli sur A.

**Avantage.** Rapide, et indépendant d'une source externe.

**Inconvénient.** Un cache périmé produit silencieusement un modèle différent.
C'est le même risque qu'en A, mais plus facile à déclencher.

## Le garde-fou commun, et il est obligatoire dans les deux cas

`models/<modele>.metadata.json`, qui **est versionné**, porte le `sha256` de
`data/processed/train.csv` tel qu'il était à l'export de référence :

```
eeaf757019362752104c97be5b55286a90f2ef48608070c2112803b1e15ba458   train.csv, 283 449 lignes
```

**L'Action doit comparer cette empreinte AVANT d'entraîner, et échouer si elle
diffère.** Sans ce contrôle, les deux options peuvent produire un modèle
différent sans que rien ne le signale — c'est le mode de défaillance principal,
et il est silencieux dans les deux branches.

Point à connaître : les empreintes de référence vivent aujourd'hui dans
`data/processed/split_metadata.json`, **qui n'est pas versionné** puisque `data/`
est exclu. C'est la métadonnée d'export, versionnée elle, qui les rend
disponibles au dépôt. La boucle ne se ferme donc qu'**une fois le premier export
committé**.

## Recommandation transmise, non imposée

**B avec repli sur A**, et contrôle d'empreinte obligatoire dans les deux
branches. B couvre le cas courant en quelques dizaines de secondes ; A couvre
l'éviction de cache et la reconstruction depuis zéro ; le contrôle d'empreinte
rend les deux vérifiables au lieu de simplement rapides.

## Réserve — ce que je n'ai pas mesuré, et pourquoi

**Le temps d'exécution de `python -m src.data_prep` n'a pas été mesuré.** Le
relancer réécrirait `data/processed/`. Si la chaîne n'est pas parfaitement
idempotente, `train.csv` changerait — et avec lui le split sur lequel reposent
les campagnes LLM de l'étape 2 et l'intégralité de l'étape 3. Le risque était
sans commune mesure avec la valeur d'un chiffre de durée. À mesurer sur une
copie, dans la mission suivante.
