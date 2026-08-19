# Coût de maintenance d'une solution LLM — démonstration chiffrée

**Destinataire : recommandation finale (étape 4).** Élément à présenter à
ZenAssist, qui ne dispose d'aucune compétence IA en interne.

Ce document ne contient aucune considération générale sur la « fragilité des
prompts ». Il rapporte **un incident réel, mesuré, avec son mécanisme exact**,
survenu pendant la phase 3 de l'étape 2.

---

## Le fait

| | conformité du format | F1-macro |
|---|---|---|
| prompt v1 | **99,5 %** | 0,781 |
| prompt v3 — v1 + une définition par étiquette | **75,5 %** | 0,690 |

Une modification d'apparence anodine — ajouter une définition d'une ligne à
chacune des 9 étiquettes, sans toucher à la consigne, au format de sortie ni au
modèle — a fait **chuter la conformité de 24 points** sur 200 réclamations.

Sur les 2 000 réclamations quotidiennes envisagées, cela représenterait environ
**490 réponses inexploitables par jour**, à réintroduire manuellement dans le
circuit de traitement.

## Le mécanisme exact

Les 9 étiquettes étaient présentées ainsi :

```
- Credit reporting : disputes about credit reports, credit scores, ...
```

Le modèle a recopié **la ligne entière**, définition comprise :

```json
{"label": "Credit reporting : disputes about credit reports, credit scores, credit repair services or consumer reporting agencies"}
```

**47 des 49 réponses non conformes sont exactement cette erreur**, toutes sur la
même classe. Le séparateur ` : ` rendait la ligne lisible comme un seul libellé.
Rien dans la formulation ne signalait ce risque, et la consigne de format
(« recopie l'étiquette mot pour mot ») était même respectée — au sens où le
modèle l'a comprise.

## Pourquoi c'est le point qui compte pour ZenAssist

**1. Le défaut est invisible à la relecture.** Ajouter des définitions est une
amélioration de bon sens, que n'importe quel opérateur tenterait. Aucune revue
de prompt ne l'aurait rejetée. Il n'existe pas de règle de style qui aurait
prévenu ce cas — seule la mesure l'a révélé.

**2. Le symptôme est silencieux.** L'API répond en 0,51 s, sans erreur HTTP,
avec un JSON syntaxiquement valide. Un système qui ne validerait pas l'étiquette
contre le référentiel enregistrerait 49 classifications erronées **sans qu'aucun
signal d'alerte ne se déclenche**. C'est le parsing strict — décidé avant toute
mesure — qui a rendu l'incident visible.

**3. Il est concentré, donc statistiquement discret.** 47 des 49 échecs portent
sur une seule classe. Un contrôle de qualité par échantillonnage aléatoire aurait
de bonnes chances de ne rien voir : sur 20 réclamations tirées au hasard, la
probabilité de ne rencontrer aucun cas est loin d'être négligeable.

**4. Le prompt n'est pas un paramètre de configuration.** C'est un composant dont
chaque modification exige une **campagne de mesure sur un jeu étiqueté**, un test
statistique apparié et une correction de multiplicité. Le protocole appliqué ici
— 200 lignes, McNemar exact, Holm-Bonferroni — a coûté 0,08 $ d'API mais
suppose un jeu de référence, un cadre de mesure et la compétence pour le lire.
**C'est cette compétence, et non le coût d'API, qui constitue la charge réelle.**

## Ce que le fait ne dit pas

L'idée d'ajouter des définitions **n'est pas invalidée**. Après re-parsing
tolérant hors ligne, qui récupère les 49 réponses, v3 obtient un F1-macro de
0,791 contre 0,781 pour v1 — et McNemar donne b=2, c=4, **p = 0,688**. La
variante ne se distingue pas de v1, défaut de format mis à part.

Autrement dit : **le défaut de format ne masquait aucun gain**. Mais un
opérateur qui aurait déployé v3 en l'état, sans mesure préalable, aurait dégradé
son service de 24 points de conformité en croyant l'améliorer.

## Contrepoint mesuré, à présenter conjointement

L'argument ne doit pas être présenté seul : trois mesures de l'étape 2 vont dans
l'autre sens et doivent l'accompagner, sans quoi la restitution serait
tendancieuse.

- **Déterminisme : 100 %.** 20 réclamations rejouées 5 fois sur deux modèles,
  200 appels, aucune réponse divergente. Une crainte fréquente sur les LLM ne
  s'est pas matérialisée ici.
- **Robustesse du format en régime nominal : 99,5 %** sur v1, et **0 erreur
  HTTP** sur plus de 1 400 appels.
- **Le prompt figé est stable.** Le risque décrit ci-dessus se matérialise à la
  **modification**, pas à l'exploitation. Un prompt gelé et versionné ne dérive
  pas de lui-même.

Le coût de maintenance est donc un **coût de changement**, pas un coût de
fonctionnement. La question à poser à ZenAssist n'est pas « le LLM est-il
fiable ? » mais **« qui, en interne, mesurera le prochain changement de
prompt — et avec quel jeu de référence ? »**

---

# Le taux de conformité brut est trompeur — la nature des échecs le renverse

**Élément distinct du précédent, même destinataire.** Mesuré sur la campagne
d'évaluation, 2 000 réclamations par modèle.

## Le chiffre qui tromperait

| | conformité du format |
|---|---|
| ministral-3b | **99,75 %** (5 non conformes) |
| mistral-small-4 | **99,05 %** (19 non conformes) |

Lu seul, ce tableau désigne ministral comme le plus fiable des deux. **La lecture
s'inverse dès qu'on regarde de quoi ces échecs sont faits.**

## mistral-small : 19 échecs, tous de forme

Les 19 non-conformités sont **la même erreur, répétée** : le modèle omet la clé
`label` et rend l'étiquette comme clé nue.

```
attendu : {"label": "Credit reporting"}
rendu   : {"Credit reporting"}
```

La catégorie choisie est **valide et présente** — elle appartient au référentiel,
elle est simplement mal enveloppée. **12 des 19 étaient la bonne réponse.**

Un défaut systématique de cette nature **se corrige en production par une couche
de parsing** : une dizaine de lignes de code, écrites une fois, qui acceptent la
clé nue en plus de la forme canonique. Le coût est celui d'un correctif, pas
d'un risque permanent.

## ministral-3b : 5 échecs, dont 3 qui inventent une catégorie

| n | réponse rendue | nature |
|---|---|---|
| 3 | `{"label": "Paypal, title or personal loan"}`<br>`{"label": "Paypal credit or prepaid card"}`<br>`{"label": "Paypal Credit"}` | **catégorie inexistante** |
| 2 | réponse entourée de ` ```json ` | forme |

Les trois premières sont d'une autre nature. Le modèle a **contaminé le libellé
avec le nom d'une entité citée dans la réclamation** — « Paypal » n'appartient à
aucune des 9 catégories, et « Paypal, title or personal loan » est une
construction hybride entre le texte lu et le référentiel fourni.

**Ces trois cas sont irrécupérables.** Aucune couche de parsing ne peut deviner
la catégorie visée : la valeur produite n'existe pas dans le référentiel de
routage. Une réclamation étiquetée `Paypal Credit` ne peut être dirigée nulle
part.

## Pourquoi cette distinction pèse plus que 0,7 point

Pour ZenAssist, sans compétence IA en interne, les deux défauts n'ont ni le même
coût ni la même durée de vie :

| | mistral-small | ministral-3b |
|---|---|---|
| échecs récupérables hors ligne | **19 / 19** | 2 / 5 |
| dont la bonne réponse était produite | **12** | 2 |
| gain maximal d'une couche de parsing | +0,60 pt d'accuracy | +0,10 pt |
| échecs touchant au **fond** | **0** | **3** |

Un défaut de forme est **systématique, prévisible et corrigeable une fois pour
toutes**. Une hallucination de catégorie est **imprévisible dans son
déclenchement** — elle dépend du contenu de la réclamation — et ne se corrige que
par un contrôle de validité en aval, qui transforme l'erreur en cas à traiter à
la main plutôt qu'en routage silencieusement faux.

**À retenir pour la restitution** : ne pas présenter les taux de conformité sans
leur décomposition. Un taux brut plus favorable peut recouvrir des échecs
strictement plus coûteux.

---

*Source : phase 3 de l'étape 2, 2026-08-19. Journaux bruts sous `data/llm/`
(`phase3_v3_definitions.jsonl`), reproductible par
`tools/build_selection_set.py` puis rejeu du style `v3_definitions`.*
