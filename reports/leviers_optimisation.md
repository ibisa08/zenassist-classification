# Leviers d'optimisation — à NE PAS utiliser pendant l'évaluation

**Statut : notes de travail de l'étape 2.** Destinataire : le plan
d'implémentation de l'étape 4.

Ce fichier existe pour qu'un levier écarté ne soit pas perdu. Chaque entrée a
été identifiée pendant l'étape 2, **écartée volontairement de l'évaluation**, et
reste pertinente pour un déploiement chez ZenAssist.

## Le principe qui les écarte tous

> L'évaluation doit mesurer **ce qu'obtient quelqu'un qui appelle l'API
> normalement**, sans réglage. C'est la situation réelle de ZenAssist, qui n'a
> aucune compétence IA en interne.

Optimiser l'appel pendant la mesure produirait un chiffre que le client
n'atteindrait qu'en reproduisant nos réglages — et masquerait le coût du
comportement par défaut, qui est précisément ce qu'il subira. Chaque levier
ci-dessous **améliorerait le résultat affiché tout en le rendant moins
représentatif**.

---

## 1. `prompt_cache_key` — routage du cache de préfixe

**Écarté de l'évaluation.** Décision explicite du 2026-08-19.

Le paramètre existe dans `chat.complete` (SDK v2, vérifié). Une clé stable
(identifiant de session, de workflow) améliore le **routage** : les requêtes
partageant un préfixe sont dirigées vers le même serveur, ce qui augmente le
taux de succès du cache.

- **Pourquoi écarté** : il modifie le protocole. On mesurerait un système
  optimisé pour le cache plutôt que le comportement par défaut. Le taux de
  cache mesuré à l'étape 2 doit être celui qu'obtient un appel normal.
- **Pourquoi le garder** : en production, ZenAssist enverrait un volume continu
  avec un préfixe identique. C'est exactement le cas d'usage du paramètre, et
  le gain porte sur le poste de coût le plus important du projet.
- **À faire à l'étape 4** : chiffrer le gain en rejouant un échantillon **avec**
  et **sans** la clé, et présenter les deux. Ne pas extrapoler le gain sans
  l'avoir mesuré.

## 2. `response_format` — sortie JSON contrainte

**Variante de phase 3, pas de la ligne de base.**

Forcerait le modèle à émettre du JSON valide, ce qui écraserait vraisemblablement
le taux de `PARSE_ERROR`.

- **Pourquoi écarté de v1** : on ne saurait plus **ce que coûte le format
  libre** — or c'est ce que ZenAssist devra gérer en production s'il retient le
  LLM. Un taux de non-conformité nul obtenu par contrainte ne dit rien de la
  docilité du modèle.
- **Intérêt réel** : si le taux de `PARSE_ERROR` mesuré en phase 2 est élevé,
  ce paramètre est la première réponse à proposer, et son effet doit être
  chiffré par différence avec la ligne de base.

## 3. `random_seed` — reproductibilité

**Variante de phase 3, pas de la ligne de base.**

- **Pourquoi écarté** : le test de déterminisme mesure si, à température 0, le
  modèle rend la même réponse d'une passe à l'autre. Fixer la graine
  répondrait à la question avant de l'avoir posée.
- **Intérêt réel** : si le test révèle une instabilité, `random_seed` est une
  atténuation à évaluer — et son efficacité est elle-même à vérifier, une
  graine fixe ne garantissant pas le déterminisme sur une infrastructure
  distribuée.

## 4. Palier tarifaire et limite de débit

Le tier gratuit plafonne à ~1 requête/seconde, ce qui donne ~35 minutes pour
2 000 appels. Cette attente est **isolée** dans `latence_avec_attentes_s` et
n'entre pas dans `latence_s`, qui mesure l'inférence seule.

- **À l'étape 4** : la latence à présenter au client dépend du palier retenu.
  Présenter `latence_s` comme la vitesse du modèle et l'attente de débit comme
  une contrainte contractuelle, jamais confondues.

---

## Détail technique à conserver — granularité du cache

Le cache de préfixe Mistral travaille par **blocs de 64 tokens**, et un prompt
de moins de 64 tokens n'obtient **jamais** de hit (docs Mistral, vérifié le
2026-08-19).

Conséquence contre-intuitive : **raccourcir le préfixe pour « économiser des
tokens » peut coûter plus cher**. Sous 64 tokens, on perd la remise de 90 % sur
la totalité de l'entrée. Notre préfixe (~260 tokens, 4 blocs pleins) est
confortablement au-dessus du seuil.

## Dérive d'alias

L'évaluation appelle un **alias** (`mistral-small-latest`), pas une version
figée, parce que c'est ce qu'un client utiliserait. Le fournisseur peut le
repointer à tout moment. La version réellement servie est donc consignée à
chaque appel (`modele_resolu`) et comparée entre campagnes par
`llm_eval.verifie_derive_version()`.

**Pour un déploiement**, l'arbitrage est inverse : épingler une version figée
(`mistral-small-2603`) donne un comportement stable et reproductible, au prix
d'une migration manuelle à chaque nouvelle version. À trancher avec ZenAssist.
