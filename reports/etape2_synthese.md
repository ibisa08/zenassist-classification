# Étape 2 — Évaluation de l'approche LLM · rapport de synthèse

**Date : 2026-08-19.** Aucune recommandation dans ce document : il rapporte des
mesures et les confronte aux critères fixés avant elles.

- **Modèles évalués** : `mistral-small-latest` → `mistral-small-2603` (Mistral Small 4)
  et `ministral-3b-latest` → `ministral-3b-2512` (Ministral 3 3B)
- **Prompt figé** : `v1_zeroshot`, sortie JSON minimale, température 0, parsing strict
- **Échantillon** : `test_sample_2000.csv`, stratifié à plancher (B3), pondéré Horvitz-Thompson
- **Volume** : 7 401 appels, **0,3752 $**, 0 erreur HTTP

> **Réserve de portée, à lire avant le reste.** Ce document rapporte des mesures
> et des défaillances *trouvées*. Il ne peut rien dire de celles qui ne se sont
> pas manifestées. L'erreur la plus coûteuse de l'étape — le modèle de cache,
> optimiste d'un facteur 2 — n'a été détectée que parce que l'API Mistral expose
> un champ `cached_tokens` qui a rendu la mesure possible. **Sans ce champ,
> l'hypothèse serait encore en place aujourd'hui.** La détection tenait à une
> propriété du fournisseur, pas à la qualité de la revue. Le §10 développe ce
> point ; il en limite la portée de tout ce qui précède.

---

## 1. Les deux modèles sur les 2 000 réclamations

> **Toutes les métriques de cette section sont PONDÉRÉES Horvitz-Thompson.**
> L'échantillon est stratifié à plancher, donc non représentatif : sans
> pondération, le F1-macro serait surestimé de +0,022 et la précision de
> `Vehicle loan or lease` de +0,29.

| | mistral-small-4 | ministral-3b |
|---|---|---|
| **F1-macro** | **0,7885** | **0,7144** |
| IC 95 % bootstrap | [0,7664 ; 0,8104] | [0,6932 ; 0,7362] |
| accuracy | 0,8149 | 0,7607 |
| précision macro | 0,8084 | 0,7275 |
| rappel macro | 0,7821 | 0,7464 |
| conformité du format | 99,05 % (19 non conformes, **toutes formelles**) | 99,75 % (5, dont **3 hallucinations**) |
| erreurs HTTP | **0** | **0** |
| coût pour 1 000 prédictions | 0,0516 $ | 0,0389 $ |

Les intervalles de confiance **ne se recouvrent pas**.

### F1 par classe

| classe | mistral-small | ministral-3b | écart |
|---|---|---|---|
| Mortgage | 0,9342 | 0,9136 | −0,021 |
| Student loan | 0,8532 | 0,8211 | −0,032 |
| Credit reporting | 0,8175 | 0,7940 | −0,024 |
| Bank account or service | 0,8112 | 0,7390 | −0,072 |
| Credit card or prepaid card | 0,7959 | 0,7710 | −0,025 |
| Debt collection | 0,7708 | 0,6246 | **−0,146** |
| Money transfer or virtual currency | 0,7315 | 0,7095 | −0,022 |
| Payday, title or personal loan | 0,6966 | 0,5360 | **−0,161** |
| Vehicle loan or lease | 0,6854 | 0,5204 | **−0,165** |

L'écart entre les deux modèles est très inégal selon les classes : négligeable
sur `Mortgage`, supérieur à 0,14 sur les trois classes que le §C.4 du diagnostic
désignait comme ayant la **signature lexicale la plus faible**.

### Nature des non-conformités — plus informative que leur nombre

Les comptes concordent entre les journaux JSONL et `evalue_campagne()` :
19 pour mistral-small (0,95 %), 5 pour ministral-3b (0,25 %), 0 erreur HTTP.
Mais leur **nature diffère qualitativement**, et c'est ce que l'étape 3 devra
comparer.

**mistral-small — 19 non-conformités, un seul mode d'échec, purement formel**

| statut | n | forme |
|---|---|---|
| `json_invalide` | 19 | `{"Credit reporting"}` |
| `label_inconnu` | 0 | — |

Le modèle **omet la clé `label`** et rend l'étiquette comme clé nue. La
classification est présente et le référentiel respecté : **12 des 19 auraient été
justes** après re-parsing tolérant, soit un impact maximal de +0,60 point
d'accuracy. C'est un défaut de sérialisation, pas de compréhension.

**ministral-3b — 5 non-conformités, dont 3 sémantiques**

| statut | n | forme |
|---|---|---|
| `label_inconnu` | 3 | `{"label": "Paypal, title or personal loan"}`, `{"label": "Paypal credit or prepaid card"}`, `{"label": "Paypal Credit"}` |
| `json_invalide` | 2 | réponse entourée de ` ```json ` |

Les trois `label_inconnu` sont des **catégories inventées** : le modèle a
contaminé le libellé avec le nom d'une entité citée dans la réclamation. Elles
sont **irrécupérables** — aucun re-parsing ne peut deviner la classe visée. Seules
les 2 réponses entourées de fences le sont, dont 2 justes.

**Lecture** : ministral produit quatre fois moins de non-conformités, mais 3 de
ses 5 échecs touchent au **fond** (hallucination de catégorie) quand les 19 de
mistral-small ne touchent qu'à la **forme**. Le taux brut favorise ministral ;
la nature des échecs favorise mistral-small.

Matrice de confusion étendue (9 classes + colonne `PARSE_ERROR`) :
[`figures/llm_confusion_mistral_small.png`](figures/llm_confusion_mistral_small.png).
La ligne `PARSE_ERROR` est vide par construction — aucune réclamation n'a cette
étiquette de référence. **Cette étiquette étendue sert uniquement à la figure ;
le F1-macro reste calculé sur les 9 classes.**

## 2. McNemar apparié — et le sort de l'hypothèse de la phase 2

| | |
|---|---|
| mistral-small juste / ministral faux (b) | **187** |
| mistral-small faux / ministral juste (c) | **87** |
| paires discordantes | 274 |
| accord | 1 726 |
| **p bilatérale exacte** | **1,47 × 10⁻⁹** |

Écart significatif en faveur de `mistral-small`.

### Le résultat de méthode le plus utile de l'étape

En phase 2, sur **20 exemples**, ministral se trompait sur un **sur-ensemble
strict** des erreurs de mistral-small : b=3, c=0, motif parfait, apparence d'une
hiérarchie nette. Le test donnait alors p = 0,250 — non concluant.

Sur **2 000 lignes** : **c = 87**. L'inclusion stricte est **infirmée**. Il existe
87 réclamations que le petit modèle classe correctement et que le grand manque.

**Le sens était bon, la forme forte était un artefact de petit échantillon.**
C'est exactement ce que le protocole devait attraper, et il l'a attrapé parce
que l'énoncé avait été formulé comme une **hypothèse à tester** et non comme un
constat. Un rapport qui aurait écrit « ministral commet toutes les erreurs de
mistral-small, plus d'autres » aurait été démenti par la mesure suivante.

## 3. Confrontation au §C.4 du diagnostic — critères pré-enregistrés

Les seuils ont été fixés dans le code **avant** la campagne
(`llm_eval.SEUIL_A_CONFIRME` et suivants) et le verdict est **calculé**, pas rédigé.

### Critère A — concentration des erreurs sur les 5 paires annoncées

> **La référence sévère est présentée en premier à dessein.** La référence
> uniforme est **favorable au §C.4 par construction** : elle suppose les erreurs
> réparties également entre les 36 paires, ce qui ignore que les grosses classes
> produisent mécaniquement plus de confusions. Elle ne peut pas valider seule la
> prédiction.

| | mistral-small | ministral-3b |
|---|---|---|
| part des erreurs sur les 5 paires | 50,5 % | 51,3 % |
| **référence sévère** (∝ effectifs) | 20,4 % → **lift ×2,48** | 20,9 % → **lift ×2,45** |
| référence uniforme (5/36 = 13,9 %) | lift ×3,64 | lift ×3,60 |
| **verdict** (confirmé si lift sévère ≥ 1,50) | **CONFIRMÉ** | **CONFIRMÉ** |

### Critère B — rang de la paire annoncée comme la plus coûteuse

`Credit reporting ↔ Debt collection` sort au **rang 1** pour les deux modèles
(28,6 % et 30,9 % des erreurs). Seuil : confirmé si rang ≤ 2. → **CONFIRMÉ**.

### Critère C — lacunes, paires lourdes non annoncées

**INFIRMÉ pour les deux modèles**, sur la même paire :
`Credit card or prepaid card ↔ Credit reporting`, **rang 2** dans les deux
classements (9,6 % et 8,0 % des erreurs). Le critère s'est déclenché sur la
règle de rang, pas sur celle de part (9,6 % < 10 %) : avoir fixé les deux
conditions a servi.

#### Ce que cette lacune est exactement

**Ce n'est pas la mesure qui a manqué.** Le §C.3 du diagnostic avait bien calculé
le recouvrement — **17,1 %** des textes `Credit card` contiennent le vocabulaire
du signalement de crédit — et le §C.4 le **cite explicitement**, dans la phrase
qui nomme trois classes par le même mécanisme : « 35,8 % des textes
`Debt collection`, 26,7 % des `Vehicle loan or lease` et **17,1 % des
`Credit card`** parlent explicitement du dossier de crédit ».

Sur ces trois classes citées ensemble, la table des paires en retient **deux** et
écarte la troisième. Or elle devance trois des cinq paires retenues :

| recouvrement lexical | paire | retenue au §C.4 |
|---|---|---|
| 35,8 % | Credit reporting ↔ Debt collection | oui |
| 26,7 % | Vehicle loan ↔ Credit reporting | oui |
| **17,1 %** | **Credit card ↔ Credit reporting** | **non** |
| 9,2 % / 4,7 % | Credit card ↔ Bank account | oui |
| 9,1 % | Money transfer ↔ Bank account | oui |
| 4,8 % | Payday ↔ Debt collection | oui |

**C'est le passage de la mesure à la liste qui a échoué.** Un tri par
recouvrement décroissant aurait retenu la paire ; une sélection à la main l'a
écartée. Le §C.4 est donc **juste sur son mécanisme et incomplet sur son
application**.

#### Robustesse de ce constat

Les deux modèles produisent les **mêmes trois verdicts** et signalent la **même
paire manquante au même rang**. Deux architectures différentes — 24 B et 3 B de
paramètres, entraînements distincts — convergeant ainsi rendent peu plausible
l'explication par un artefact de modèle. La lacune est une propriété du
**corpus**, pas d'un classifieur.

## 4. Stabilité sur un second tirage — critère pré-enregistré

Vérification prévue par `reports/limites.md` : second échantillon B3, **même
allocation et mêmes poids**, seed 2024, recouvrement de 2,2 % avec le tirage de
référence.

| | F1-macro | IC 95 % |
|---|---|---|
| référence (seed 42) | 0,7885 | [0,7664 ; 0,8104] |
| alternatif (seed 2024) | 0,7840 | [0,7624 ; 0,8064] |

**Écart mesuré : 0,0045.** Critère : stable si < 0,024. → **STABLE**.

L'écart vaut **0,37 écart-type** de la distribution simulée dans
`h1_sampling_simulation.md` (σ = 0,0123 entre tirages B3), et les intervalles se
recouvrent largement. Le tirage de référence n'était donc ni particulièrement
favorable ni défavorable.

## 5. Phase 3 — aucune variante retenue

Cinq modifications **indépendantes** de v1, mesurées sur 200 lignes du **train**
(jamais sur l'échantillon d'évaluation), gate = McNemar exact apparié sous
correction de **Holm-Bonferroni** (FWER 5 %).

| rang | variante | b | c | p brut | seuil Holm | décision |
|---|---|---|---|---|---|---|
| 1 | v3 définitions | 40 | 3 | < 10⁻⁵ | 0,01000 | rejetée — **en faveur de v1** |
| 2 | v4 libellés CFPB officiels | 8 | 5 | 0,581 | 0,01250 | non rejetée |
| 3 | v5 français | 4 | 6 | 0,754 | 0,01667 | non rejetée |
| 4 | v2 règle produit | 7 | 5 | 0,774 | 0,02500 | non rejetée |
| 5 | v6 few-shot | 10 | 9 | 1,000 | 0,05000 | non rejetée |

### Ce que ce résultat signifie exactement

**« Aucun gain distinguable à n = 200 »** — et non « les variantes sont
équivalentes à v1 ». La distinction est essentielle et n'est pas rhétorique.

À n = 200, le protocole exigeait, pour rejeter :

| lignes cassées | corrections nécessaires (rang 1) | gain net |
|---|---|---|
| 0 | 8 | +4,0 pts |
| 2 | 13 | +5,5 pts |
| 5 | 19 | +7,0 pts |

Une variante apportant un gain réel de 2 points n'aurait **aucune chance** d'être
détectée. Le non-rejet borne l'effet, il ne l'annule pas.

**v5 (français) illustre pourquoi le gate n'est pas le F1.** Elle affiche le
**meilleur F1 du tableau** — 0,7924 contre 0,7809 pour v1 — mais n'a corrigé que
6 lignes pour 4 cassées, là où ~15 corrections auraient été nécessaires. Avec un
critère fondé sur l'intervalle de confiance, la tentation de la retenir aurait
été forte. Le test apparié, qui ne regarde que les lignes où les deux variantes
diffèrent, montre qu'il n'y a rien à retenir. **C'est la justification empirique
du choix de McNemar comme critère de décision.**

**v3 se lit en deux temps, dans cet ordre :**

1. **L'implémentation était fautive.** Le format `- Étiquette : définition` a
   fait recopier la ligne entière au modèle : 47 des 49 non-conformités sont
   exactement la même réponse. La conformité tombe de 99,5 % à 75,5 %.
2. **Le défaut ne masquait aucun gain.** Après re-parsing tolérant hors ligne,
   qui récupère les 49 réponses, McNemar donne b=2, c=4, **p = 0,688**. Même sans
   le défaut, v3 ne se distingue pas de v1.

Le chiffre officiel reste le strict : appliquer le re-parsing tolérant à v3
seule l'avantagerait, aucune autre variante n'y ayant droit.

Cet incident est documenté séparément dans
[`cout_maintenance_prompt.md`](cout_maintenance_prompt.md) — c'est le résultat le
plus exploitable de la phase 3 pour la restitution client.

## 6. Déterminisme — 100 %, et sa portée exacte

**20 réclamations rejouées 5 fois sur les deux modèles, 200 appels : 20/20
réponses identiques pour chacun.** Aucune étiquette concurrente. Écart-type
médian de latence sur un même texte : 0,076 s (mistral-small), 0,068 s (ministral).

**Portée** : 5 passes sur **20 exemples**, à quelques minutes d'intervalle, un
seul jour. Ce n'est **pas** une mesure sur les 2 000, ni sur plusieurs jours, ni
à travers un changement de version du modèle. Le résultat dit que le
non-déterminisme ne s'est pas manifesté dans ces conditions ; il ne dit pas qu'il
est exclu.

## 7. Coût réel et les trois régimes de cache

### Une prédiction quantitative confirmée à 0,5 % près

Le modèle de cache construit en phase 2 **sur 200 appels** prédisait **220,3
tokens cachés par appel** en régime établi. La campagne de 2 000 en a mesuré
**221,4**, pour une activation de 98,85 % contre 98 % prédits.

**Écart : 0,5 %, sur dix fois plus de données.** C'est une validation de
**méthode** autant qu'un chiffre : une hypothèse structurelle a été décomposée en
deux paramètres mesurables — activation et couverture — puis chacun estimé sur un
petit échantillon, et la prédiction jointe a tenu à l'échelle. Cela vaut pour la
suite : la même décomposition est reproductible sur les projections de l'étape 3.

### Coût mesuré

| | mistral-small | ministral-3b |
|---|---|---|
| 2 000 appels | 0,1033 $ | 0,0778 $ |
| pour 1 000 prédictions | 0,0516 $ | 0,0389 $ |
| écart à la projection | +1,2 % | +2,3 % |

### Le cache : trois régimes, mesurés

C'est la découverte de l'étape. `estimate_cost()` modélisait implicitement une
activation à 100 % et une couverture totale du préfixe. Les deux sont fausses.

| régime | économie | ce que c'est |
|---|---|---|
| démarrage à froid (20 premiers appels) | 20,6 % | cache encore vide |
| **régime établi** | **36,8 %** | ce qu'obtient une campagne longue |
| borne théorique | 42,2 % | plafond, jamais atteint |

Deux effets, tous deux invisibles avant mesure :

- **la couverture est plafonnée et structurellement constante** : 224 tokens sur
  un préfixe de 252 chez mistral-small (89 %), 128 sur 240 chez ministral (53 %).
  Jamais une autre valeur sur 200 appels.
- **l'activation dépend du réchauffement** : 55 % sur les 20 premiers appels,
  ~98 % ensuite.

**La campagne de 2 000 confirme le modèle du régime établi** : activation mesurée
**98,85 %**, moyenne de **221,4 tokens** cachés par appel — contre 220,3 prédits
à partir des mesures de phase 2. La part de l'entrée servie par le cache est de
**43,7 %** (25,9 % chez ministral, dont le plafond de couverture est plus bas).

Le levier `prompt_cache_key`, qui rapprocherait le régime réel de la borne, a été
**volontairement écarté de l'évaluation** — mesurer un système optimisé n'aurait
pas décrit ce qu'obtient un appel par défaut. Il est consigné dans
[`leviers_optimisation.md`](leviers_optimisation.md) pour l'étape 4.

## 8. Latences — deux grandeurs à ne pas confondre

| | mistral-small | ministral-3b |
|---|---|---|
| **latence modèle** p50 / p95 | 0,470 / **0,740** s | 0,286 / **0,487** s |
| latence max | 16,83 s | 2,72 s |
| latence + attentes p50 / p95 | 1,011 / 1,416 s | 1,006 / 1,204 s |

- **Latence modèle** : durée de la seule tentative aboutie, hors limiteur de
  débit et hors backoff. C'est la vitesse du modèle, celle qui se compare à
  l'étape 3.
- **Latence avec attentes** : inclut le limiteur à 1 req/s du tier gratuit. Elle
  dépend du palier tarifaire, pas du modèle — les confondre ferait passer une
  contrainte contractuelle pour une lenteur du LLM.

Les latences des tentatives échouées sont retirées de la distribution : un 429
n'est pas une mesure de la vitesse du modèle. Il n'y en a eu aucun.

La latence maximale de 16,83 s chez mistral-small est une **valeur isolée** :
5 appels sur 2 000 dépassent 5 s, et le p95 reste à 0,740 s. C'est une queue de
distribution, pas une dérive.

## 9. Hypothèses de `config.py` : ce qui a été remplacé, ce qui reste

### Remplacé par des mesures — **tokenizer Mistral uniquement**

| constante | avant | après | vérification |
|---|---|---|---|
| `AVG_COMPLAINT_TOKENS` | 257 | **245** | 254,5 mesurés sur les 2 000 (+3,9 %) |
| `CACHED_PREFIX_TOKENS` | 260 | **252** / 240 par modèle | régression R² = 0,9962 |
| ratio tokens/mot | 1,3 (règle empirique) | **1,189** | 40 appels, 24 à 534 mots |
| `AVG_OUTPUT_TOKENS` | 8 | **10** | max observé 13, sur 50 autorisés |
| `cache_automatique` (Mistral) | `False` | **`True`** | docs Mistral, 2026-08-19 |
| `PRICING_CHECKED_ON` | 2026-08-11 | **2026-08-19** | deux sources concordantes |
| `CACHE_MESURE` | *(inexistant)* | **deux régimes** | 200 appels |

### Restent des hypothèses

- **Tous les volumes de tokens pour Anthropic, OpenAI, Google et DeepSeek.**
  Leurs tokenizers diffèrent ; une mesure faite chez Mistral ne transfère pas.
  `config.FOURNISSEURS_TOKENS_MESURES` porte cette distinction dans le code, et
  `llm_eval.projette_cout()` rend un `statut_tokens` **par fournisseur** — un flag
  global unique ferait passer 8 modèles sur 10 pour mesurés.
- **Le comportement de cache de tous les fournisseurs sauf Mistral.** La colonne
  « avec cache » du tableau des 10 modèles reste une **borne théorique**,
  optimiste d'environ 5 points d'après ce que la mesure Mistral révèle.
- **La remise de cache de Google**, jamais relevée — le cache n'est pas modélisé
  pour ce fournisseur plutôt que supposé.
- **`DAILY_COMPLAINTS = 1000`**, volume de référence du client, jamais vérifié
  auprès de ZenAssist.
- **Les tarifs des fournisseurs autres que Mistral**, relevés le 2026-08-11 et
  non reconfirmés le 19.

## 10. Leçons de méthode

Trois dispositifs de surveillance ont failli pendant l'étape 2. La question
posée était de savoir s'ils relèvent d'un même mécanisme ou de trois causes
distinctes.

### Les trois cas

| dispositif | ce qu'il devait garantir | pourquoi il n'a rien garanti |
|---|---|---|
| registre des risques (§4 de `limites.md`) | signaler les valeurs de coût fragiles | il surveillait des **constantes**, pas la forme du calcul |
| filtre de monitoring | alerter sur les anomalies de campagne | le motif `error` matchait `parse_error`, présent sur **chaque** ligne nominale |
| test du refus de démarrage | vérifier qu'un style non figé est rejeté | il utilisait `v1_zeroshot`, devenu figé — il testait l'inverse de son intitulé |

### Un mécanisme, pas trois

**Chacun des trois encodait une prémisse implicite sur son environnement,
vraie au moment de l'écriture, et qu'aucun ne vérifiait.**

- cache : « le cache s'active à 100 % et couvre tout le préfixe » — jamais vraie,
  jamais énoncée
- filtre : « la chaîne `error` n'apparaît que dans les anomalies » — vraie d'un
  log générique, fausse de notre format
- test : « `v1_zeroshot` n'est pas dans `LLM_STYLES_FIGES` » — vraie jusqu'à la
  clôture de la phase 3

Le remède a été **identique dans les trois cas** : promouvoir la prémisse au rang
d'expression explicite et vérifiée. Un paramètre nommé `taux_activation_cache` ;
un test négatif du filtre sur une ligne nominale ; une assertion
`assert STYLE_NON_FIGE not in cfg.LLM_STYLES_FIGES`.

### Deux régimes de détection, et une échelle de gravité

La différence utile n'est pas dans le mécanisme mais dans **l'objet** de la
prémisse, qui détermine comment on peut la prendre en défaut :

- **prémisse sur l'état interne du code** (filtre, test) → une assertion suffit,
  la vérification est gratuite et immédiate ;
- **prémisse sur le comportement du monde extérieur** (cache) → aucune relecture
  ne la trouve, **seule la mesure la révèle**.

La gravité suit exactement le degré d'implicite :

| dispositif | mode d'échec | coût s'il n'avait pas été vu |
|---|---|---|
| test | **bruyant** — passe au rouge | nul, détecté en quelques secondes |
| filtre | **bavard** — dégradation visible | accoutumance aux fausses alertes |
| cache | **silencieux** — chiffre plausible | un facteur 2 présenté au client |

Le cas du cache est le plus profond : sa prémisse n'était pas seulement tue,
elle était **inexprimable** — `estimate_cost()` n'offrait aucun paramètre où
l'écrire. Une hypothèse qu'on ne peut pas nommer dans le vocabulaire du code est
une hypothèse que personne ne peut contester.

### Règle transférable

Lors de la revue d'un contrôle, la question n'est pas « ce contrôle est-il
correct ? » mais **« qu'est-ce qui devrait devenir faux pour qu'il continue de
passer tout en ne protégeant plus rien ? »**

Et son corollaire, qui vaut pour l'étape 3 : une projection dont aucun paramètre
n'exprime une hypothèse structurelle doit être **mesurée tôt sur un petit
échantillon**, jamais seulement relue. C'est ce qu'a fait la phase 2 en lisant
`cached_tokens` sur 20 appels plutôt qu'en faisant confiance à
`CACHED_PREFIX_TOKENS` — pour 0,001 $.

### Réserve — ceci est un inventaire des cas résolus, pas un inventaire

Ces trois défaillances ont été **trouvées**. Rien ne dit combien de dispositifs
encodent encore une prémisse implicite non détectée : par construction, on ne
recense que ceux qui ont fini par se manifester.

Le cas du cache le montre. Il n'a été vu que parce qu'une mesure a été faite —
et cette mesure n'était possible que parce que l'API expose un champ
`cached_tokens`. **Sans ce champ, l'hypothèse d'une activation à 100 % serait
encore en place aujourd'hui**, et le facteur 2 serait parti dans la
recommandation sans que rien ne l'ait signalé. La détection tenait à une
propriété du fournisseur, pas à la qualité de la revue.

La règle énoncée plus haut vaut donc comme **méthode de revue**, pas comme
garantie d'exhaustivité. Elle augmente la probabilité de trouver ce genre de
prémisse ; elle ne borne pas ce qui reste.

---

## 11. Ce qui est transmis à l'étape 3

### La contrainte des mêmes lignes

La comparaison LLM / ML n'est valide que sur les **mêmes lignes exactement**.
`metrics.compare_results()` refuse d'ailleurs d'aligner deux résultats de tailles
différentes, et `llm_eval._verifie_memes_lignes()` compare les **jeux
d'identifiants**, pas seulement leurs cardinaux — deux campagnes de même taille
peuvent porter des lignes différentes si chacune a perdu un appel distinct.

**Cette contrainte n'a pas mordu ici** : **0 erreur HTTP sur 6 000 appels
d'évaluation**, donc les 2 000 lignes sont toutes évaluées et le ML peut l'être
sur les 2 000 également. Le mécanisme reste en place pour le cas contraire :
`exporte_ids_evalues()`, `charge_ids_evalues()`, `restreint_au_sous_ensemble()`.

### Les fichiers d'erreurs

| fichier | contenu |
|---|---|
| `data/llm/phase4_erreurs_mistral_small.csv` | 385 erreurs |
| `data/llm/phase4_erreurs_ministral_3b.csv` | 485 erreurs |

**Format arrêté avant mesure, volontairement minimal** :
`complaint_id, label_vrai, label_predit` — et rien d'autre.

Pas de texte de réclamation : ce fichier est destiné à être relu plusieurs fois,
et y embarquer le texte en ferait une copie diffuse du corpus, qu'une jointure
sur `complaint_id` retrouve à tout moment. Pas de poids non plus : ils
appartiennent à l'échantillon, pas aux erreurs, et les dupliquer créerait une
seconde source de vérité.

### Le réservoir de l'analyse croisée

**298 réclamations sont manquées par les deux modèles**, soit 52,1 % de l'union
de leurs erreurs.

Le croisement des erreurs du LLM et de celles du ML sera la **seule mesure
empirique du plafond de performance** annoncé au §C.4 : une réclamation manquée
par deux approches aux mécanismes indépendants relève probablement d'une
ambiguïté d'étiquetage plutôt que d'une faiblesse de modèle.

**Le protocole de ce croisement doit être arrêté avant que les erreurs du ML
soient connues.** Sinon on choisira après coup la définition de « même erreur »
qui arrange la conclusion — exactement le travers que la formulation de
l'hypothèse d'inclusion stricte comme *hypothèse* a permis d'éviter en phase 2.

### Ce que l'étape 3 doit reprendre tel quel

- `metrics.py` et `data_prep.py`, **gelés**. `tests/test_metrics_gel.py` rejoue un
  jeu figé et compare les sorties à des valeurs de référence : une modification
  de logique déguisée en correction de commentaire y échoue.
- La pondération Horvitz-Thompson, **obligatoire** — `sample_weight` est un
  argument nommé sans valeur par défaut, son omission lève.
- `PARSE_ERROR` hors de `CLASS_ORDER` : le ML n'en produira pas, mais l'ordre des
  9 classes doit rester identique pour que les figures se superposent.

---

## 12. Récapitulatif de facturation

À confronter au tableau de bord Mistral. Un écart signalerait une erreur de
comptage ou une tarification différente de `MODELS_PRICING`.

| campagne | appels | coût |
|---|---|---|
| phase 2 — appel unique | 1 | 0,00005 $ |
| phase 2 — mistral-small, 5 passes | 100 | 0,00568 $ |
| phase 2 — ministral-3b, 5 passes | 100 | 0,00324 $ |
| phase 3 — v1 zeroshot | 200 | 0,01106 $ |
| phase 3 — v2 règle produit | 200 | 0,01112 $ |
| phase 3 — v3 définitions | 200 | 0,01204 $ |
| phase 3 — v4 libellés officiels | 200 | 0,01118 $ |
| phase 3 — v5 français | 200 | 0,01123 $ |
| phase 3 — v6 few-shot | 200 | 0,02665 $ |
| **phase 4 — mistral-small, 2 000** | 2 000 | 0,10325 $ |
| **phase 4 — ministral-3b, 2 000** | 2 000 | 0,07775 $ |
| **phase 4 — stabilité seed 2024** | 2 000 | 0,10196 $ |
| **TOTAL** | **7 401** | **0,3752 $** |

Plafond autorisé 1,00 $ → **37,5 % consommé**.

**0 erreur HTTP sur 7 401 appels.** Versions servies stables sur toute l'étape :
`mistral-small-2603` et `ministral-3b-2512`, vérifié par
`llm_eval.verifie_derive_version()` — qui compare les versions **résolues via les
métadonnées**, la réponse de `chat.complete` ne renvoyant que l'alias demandé.

---

## Annexe — reproductibilité

```
python tools/build_iteration_set.py      # 20 lignes, seed 42
python tools/build_selection_set.py      # 200 lignes, seed 1337 + few-shot
python tests/test_pipeline.py            # 50 vérifications
python tests/test_metrics_gel.py         # 44 — gel de metrics.py
python tests/test_llm_config.py          # 62 — cohérence du référentiel
python tests/test_llm_prompts.py         # 80 — prompts et parsing
python tests/test_llm_runner.py          # 29 — campagne et garde-fous
python tests/test_llm_eval.py            # 152 — métriques LLM
```

**417 vérifications, aucune n'effectue d'appel API.**

Journaux bruts sous `data/llm/` (couvert par `.gitignore`) : chaque appel y
figure avec sa réponse brute, ses tokens `usage`, sa latence, sa version résolue
et son coût. Un re-parsing tolérant hors ligne reste possible sans réappel.
