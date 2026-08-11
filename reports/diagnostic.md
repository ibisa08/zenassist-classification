# Diagnostic exploratoire — ZenAssist

**Étape 1 / phase 2 — constat chiffré et recommandations**
Dataset : `data/raw/dataset.csv` (639 Mo, utf-8, séparateur `,`, quotechar `"`)
Colonne texte : `Consumer Claim` — Colonne étiquette : `Tag`
Chargement : `usecols=['Complaint ID','Date received','Consumer Claim','Tag']`
Date du diagnostic : 2026-08-11

---

## Périmètre

| Étape | Lignes |
|---|---:|
| Fichier brut | 1 282 355 |
| Ayant un `Consumer Claim` non nul | **383 564** (29,9 %) |

Les 898 791 lignes sans texte (70,1 %) sont inexploitables : ni le LLM ni le modèle ML
n'ont de signal d'entrée. Toutes les statistiques ci-dessous portent sur les 383 564
lignes textuelles, sauf mention contraire.

Point de contrôle : les lignes avec texte proviennent **toutes** du canal `Web`
(`Submitted via = Web`, 383 564 / 383 564). Les canaux Phone, Postal mail, Fax et Referral
ne produisent jamais de narratif. Le corpus est donc homogène en canal, mais il n'est
**pas représentatif de l'ensemble des réclamations** : c'est une limite à mentionner dans
la recommandation finale au client.

---

## A. Qualité des données

### A.1 Valeurs manquantes et invalides

| Contrôle | Nombre |
|---|---:|
| Étiquette `Tag` manquante ou vide | **0** |
| Texte vide ou composé uniquement d'espaces | **0** |

Aucune ligne à écarter à ce titre. Le filtre « texte non nul » suffit : il n'existe pas de
faux non-nuls (chaîne vide, `" "`, `"nan"`) dans cette colonne.

### A.2 Doublons

| Contrôle | Textes distincts | Lignes |
|---|---:|---:|
| Textes apparaissant ≥ 2 fois | 11 354 | 30 702 |
| Lignes à supprimer par dédoublonnage sur le texte seul | — | **19 348** |
| Lignes à supprimer par dédoublonnage sur (texte + étiquette) | — | 19 130 |
| **Cas critique : même texte, étiquettes différentes** | **201** | **1 905** |

L'écart de 218 lignes entre les deux dédoublonnages correspond aux doublons dont
l'étiquette diffère. Ces 201 textes sont majoritairement des **lettres types de credit
repair** diffusées en masse : le même courrier standard invoquant la FCRA est déposé par
des centaines de consommateurs différents, et les agents du CFPB l'ont classé tantôt en
`Credit reporting`, tantôt en `Debt collection`.

Exemples réels :

```
{'Checking or savings account': 1, 'Payday loan, title loan, or personal loan': 1}
  "A full payment was made but I only wanted the regular monthly payment! I called
   and said I could not afford that! They refused to reverse the debit pa..."

{'Credit reporting, credit repair services, or other personal consumer reports': 35,
 'Debt collection': 1}
  "According to the Fair Credit Reporting Act, Section 609 (a)(1)(A), you are
   required by federal law to verify - through the physical verificati..."

{'Credit reporting, credit repair services, or other personal consumer reports': 4,
 'Credit card or prepaid card': 1}
  "According to the Fair Credit Reporting Act, Section 609 (a)(1)(A), you are
   required by federal law to verify - through the physical verificati..."
```

3,8 % du corpus cite explicitement la FCRA / 15 USC / section 609-611 — 8,0 % au sein de
`Credit reporting`. C'est le marqueur de ces courriers templatisés.

**Conséquence méthodologique majeure** : sans dédoublonnage, ces textes identiques se
retrouveraient **simultanément dans le train et dans le test**. Le modèle TF-IDF
mémoriserait le courrier type et le score de test serait artificiellement gonflé. Le
dédoublonnage sur le texte seul, **avant** le split, est non négociable.

### A.3 Distribution des longueurs

| Percentile | Caractères | Mots |
|---|---:|---:|
| min | 5 | 1 |
| p5 | 132 | 24 |
| p25 | 392 | 71 |
| **médiane** | **742** | **136** |
| p75 | 1 359 | 249 |
| p95 | 3 216 | 587 |
| p99 | 4 996 | 899 |
| max | 31 634 | 6 314 |
| moyenne | 1 078 | 197 |

Distribution très asymétrique à droite (moyenne 1 078 car. contre médiane 742). Les
réclamations sont substantielles : le p5 est déjà à 132 caractères, donc **95 % des textes
ont assez de matière** pour être classés. Ce n'est pas un corpus de messages courts.

### A.4 Textes très courts

| Seuil | Lignes | % |
|---|---:|---:|
| < 20 caractères | **148** | 0,039 % |
| < 50 caractères | 2 305 | 0,60 % |
| < 100 caractères | 11 896 | 3,10 % |

Cinq exemples sous 20 caractères :

```
[Credit reporting]  "Needs to be removed"
[Credit reporting]  "Needs to be removed"
[Debt collection]   "FTC NOT MINE REMOVE"
[Credit reporting]  "Account is fraud"
[Credit reporting]  "Account is fraud"
```

Ces textes ne portent aucun signal de catégorie : « Account is fraud » pourrait relever de
n'importe laquelle des 9 classes. Ils sont non seulement inutiles mais **nuisibles** —
ils créent du bruit d'étiquetage identique à celui décrit en A.2. Leur suppression coûte
0,04 % du corpus.

### A.5 Textes anormalement longs

3 831 lignes dépassent le p99 (4 996 caractères). Le maximum atteint **31 634 caractères
(6 314 mots)**, soit environ 8 200 tokens pour une seule réclamation.

Extrait du plus long (classé `Debt collection`) :

```
"XXXX XXXX XXXX XXXX VIOLATED A COURT ORDER, TRESPASSED AND MISREPRESENTED IN COURT
 TO MOVE ON A WRONGFUL FORECLOSURE WITHOUT ANY EVIDENTIARY DOCUMENTS OR AUTHORITY.
 MANDATORY NOTICE AFFIDAVIT of LEGALITY I XXXX XXXX I am that I am a living spirit,
 flesh and blood natural man on the land, creation of..."
```

Il s'agit de dépôts pseudo-juridiques (mouvance *sovereign citizen*), pas de réclamations
ordinaires. Ils sont peu nombreux mais coûteux côté LLM : à eux seuls ils font exploser la
facture et la latence. Traitement recommandé en section D — **troncature, pas suppression**
(l'étiquette reste valide, seul l'excédent de texte est inutile).

### A.6 Bruit et artefacts — vue d'ensemble

| Artefact | % de lignes |
|---|---:|
| Masquage `XX+` | **84,97 %** |
| Date masquée `XX/XX` | 38,89 % |
| Montant masqué `{$...}` | 30,89 % |
| Majuscules > 60 % du texte | 1,73 % |
| Balises HTML / entités | 0,80 % |
| URL (`http`, `www.`) | 0,43 % |
| Caractères non-ASCII | 0,03 % |

Le HTML, les URL et les caractères non-latins sont **négligeables** (< 1 % chacun) : aucun
nettoyage spécifique ne se justifie, le rapport bénéfice/complexité est mauvais. Le corpus
est monolingue anglais (0,03 % de non-ASCII, dont 0,00 % au-delà de 20 caractères
non-ASCII : ce sont des signes typographiques isolés, pas des textes en langue étrangère).

Le seul artefact réellement structurant est le masquage, traité en section G.

---

## B. Distribution des étiquettes

### B.1 État initial : 18 modalités

Sur les 383 564 lignes textuelles, `Tag` prend 18 valeurs, avec un
**ratio majoritaire/minoritaire de 5 774** (92 378 contre 16).

| Tag | n | % |
|---|---:|---:|
| Credit reporting, credit repair services, or other personal consumer reports | 92 378 | 24,08 |
| Debt collection | 86 710 | 22,61 |
| Mortgage | 52 987 | 13,81 |
| Credit reporting | 31 588 | 8,24 |
| Student loan | 21 810 | 5,69 |
| Credit card or prepaid card | 21 379 | 5,57 |
| Credit card | 18 838 | 4,91 |
| Bank account or service | 14 885 | 3,88 |
| Checking or savings account | 12 881 | 3,36 |
| Consumer Loan | 9 474 | 2,47 |
| Vehicle loan or lease | 5 745 | 1,50 |
| Money transfer, virtual currency, or money service | 5 466 | 1,43 |
| Payday loan, title loan, or personal loan | 4 421 | 1,15 |
| Payday loan | 1 747 | 0,46 |
| Money transfers | 1 497 | 0,39 |
| Prepaid card | 1 450 | 0,38 |
| Other financial service | 292 | 0,08 |
| Virtual currency | **16** | 0,004 |

### B.2 Classes sous les seuils (avant fusion)

- **Sous 100 occurrences** : `Virtual currency` (16).
- **Sous 1 000 occurrences** : `Virtual currency` (16), `Other financial service` (292).
- **Sous 2 000** : s'y ajoutent `Payday loan` (1 747), `Money transfers` (1 497),
  `Prepaid card` (1 450).

### B.3 Libellés quasi-doublons

Oui, massivement — et ce n'est pas une affaire de casse ou d'espaces. Les 18 modalités
sont **deux référentiels successifs superposés**, entièrement confirmé par le croisement
Tag × année de la section F. Les groupes redondants :

| Groupe sémantique | Ancien référentiel | Nouveau référentiel |
|---|---|---|
| Signalement crédit | `Credit reporting` | `Credit reporting, credit repair services, or other personal consumer reports` |
| Cartes | `Credit card`, `Prepaid card` | `Credit card or prepaid card` |
| Compte bancaire | `Bank account or service` | `Checking or savings account` |
| Transfert d'argent | `Money transfers`, `Virtual currency` | `Money transfer, virtual currency, or money service` |
| Prêt court terme | `Payday loan`, *(`Consumer Loan`)* | `Payday loan, title loan, or personal loan` |
| Prêt auto | *(`Consumer Loan`)* | `Vehicle loan or lease` |
| Inchangés | `Debt collection`, `Mortgage`, `Student loan` | idem |

Distinguer `Credit card` de `Credit card or prepaid card` reviendrait à demander au modèle
de deviner **la date de dépôt** de la réclamation, pas son sujet. C'est une distinction
sans réalité sémantique : la fusion est justifiée.

---

## C. Séparabilité apparente

### C.1 Exemples réels — 3 classes les plus grosses (après fusion)

**`Credit reporting` (n = 107 753)**

> Equifax, XXXX, and XXXX is constantly fraudulently reporting accounts on my report
> inwhich I HAVE NO CONTRACT WITH, I have repeatedly disputed these accounts all to no
> prevail. The constant harassment and XXXX hardship only would lead me to suing this
> companies. Never once have these reporting agencies nor the companies they represent
> brought forth any physical documentation [...]

> Equifax, XXXX and XXXX have not changed my credit score in over 12 years. I have
> documentation to prove that my credit score remains the same and they never change it.
> I have had several items removed from my credit and they have never changed the score.

**`Debt collection` (n = 84 028)**

> COMPANY SENT DEMAND NOTICE ACCOUNT OUTDATED / CLOSED OVER 11YRS AGO AND FRAUD AFFIDAVIT
> FILED WITH ORIGINAL CREDITOR ON ABOT XXXX/2007 ... COMPANY CONTINUE VIOLATES FCRA, FCBA,
> FDCPA (1692g), STATE STATUE OF LIMITATION EXPIRED ; 5YRS TO COLLECT WITH IN THE STATE OF
> GEORGIA.. NOT MY ACCOUNT/ NEVER DONE BUSINESS WITH SAID CREDITOR

> starting in the beginning of XX/XX/XXXX I started receiving calls from CDR assc. about
> some kind of debt, i asked what kind of debt and they respond with they need my SSN # to
> proceed, i told them i was not going to give that information out over the phone and
> send me in writing a copy of the debt they were trying to collect [...]

**`Mortgage` (n = 52 927)**

> We were told to use a home equity line of credit to pay for the down payment on the
> purchase of our home. We then moved XXXX and lost our renters. We stopped payment on the
> home in XX/XX/XXXX and was foreclosed in XX/XX/XXXX. This line of credit was written off
> but still shows on our credit report. We'd like to have it removed [...]

> I signed a modification agreement (HFA Modification) with the federal government that
> supersedes any other agreement. It was for the entire loan balance at that time. My
> original mortgage note was fixed, I never had an adjustable rate loan [...]

### C.2 Exemples réels — 3 classes les plus petites (après fusion)

**`Money transfer or virtual currency` (n = 6 966)**

> I went into the XXXX Chase branch on XX/XX/18. I ordered a new atm card as I lost mine.
> The banker gave me a temporary card, which can not be used except at the atm machines.
> The banker never sent my new atm card until XX/XX/18 [...] This really is a problem as I
> can not pay my bills

> On the XXXX I recieved a payment in Bitcoin from XXXX I immediatly recieved the payment
> including an email from Coinbase saying I will be able to use the funds after 6 confirms.
> I've been waiting 1 month now [...] So I basically got scammed by Coinbase the company.

**`Payday, title or personal loan` (n = 6 146)**

> I was given a loan of {$650.00} and was told the finance charge was {$220.00}, i was not
> told that this was a charge i would be receiving every 2 weeks for over a year. I was not
> told that my finance rate was 800 % [...]

> I received an email from a law firm yesterday XXXX stating that I have an outstanding
> balance with ACE Cash Express for {$850.00} which is the settlement amount. It notes that
> I am being sued by the court. I called Ace Cash Express and I am not in their system at
> all and this is probably a SCAM [...]

**`Vehicle loan or lease` (n = 5 704)**

> On my credit report, it shows my XXXX XXXX Account as a "Charge Off" which significantly
> impacts credit report, and ability to get a new car. I have paid on time, everytime on
> this account [...] Ally refuses to change this, through direct contact or credit report
> disputes.

> I wrecked my car the weekend of XXXX XXXX and on my way back to XXXX, I ended up rear
> ending another car. My insurance company totaled it the next week and my insurance
> company sent XXXX XXXX a check for the difference of the loan [...]

### C.3 Recouvrement lexical mesuré

Pour objectiver le jugement, voici le **% de textes de chaque classe contenant le
vocabulaire signature des autres classes** (recherche insensible à la casse) :

| Classe réelle ↓ / mot-clé → | credit report | debt collect | mortgage | credit card | checking/savings | student loan | vehicle | payday | transfer |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Credit reporting | **62,7** | 5,6 | 6,0 | 9,5 | 0,9 | 3,6 | 4,4 | 0,7 | 0,1 |
| Debt collection | **35,8** | 27,8 | 3,2 | 6,9 | 1,3 | 2,0 | 2,1 | 1,6 | 0,3 |
| Mortgage | 9,8 | 2,4 | **80,4** | 2,0 | 2,5 | 2,2 | 0,9 | 0,2 | 0,5 |
| Credit card or prepaid card | 17,1 | 2,1 | 1,8 | **51,6** | 4,7 | 0,3 | 1,0 | 0,2 | 1,9 |
| Bank account or service | 3,5 | 0,9 | 3,6 | 9,2 | **42,9** | 0,5 | 1,5 | 0,7 | 3,1 |
| Student loan | 13,4 | 4,9 | 2,6 | 2,3 | 2,6 | **73,3** | 0,9 | 0,5 | 0,1 |
| Money transfer or virtual currency | 0,7 | 0,6 | 1,6 | 7,2 | 9,1 | 0,4 | 0,9 | 0,1 | **43,0** |
| Payday, title or personal loan | 11,2 | 4,8 | 2,0 | 3,8 | 8,8 | 0,8 | 4,7 | **24,3** | 0,9 |
| Vehicle loan or lease | **26,7** | 2,8 | 1,7 | 2,4 | 3,2 | 0,4 | **55,1** | 0,7 | 0,2 |

### C.4 Jugement

**Les frontières ne sont pas nettes, et un plafond de performance est à attendre quelle
que soit l'approche.** Trois raisons, toutes visibles dans les chiffres ci-dessus.

**1. `Credit reporting` est un attracteur transversal.** 35,8 % des textes `Debt
collection`, 26,7 % des `Vehicle loan or lease` et 17,1 % des `Credit card` parlent
explicitement du dossier de crédit. C'est logique : le consommateur se plaint d'un impact
sur son score, quel que soit le produit sous-jacent. L'étiquette CFPB traduit le **produit
financier concerné**, alors que le texte décrit souvent le **préjudice ressenti**, qui est
presque toujours « ça abîme mon crédit ». Le signal lexical pointe vers la mauvaise
étiquette.

**2. Paires de classes à confusion attendue** (par ordre de risque décroissant) :

| Paire | Mécanisme de confusion |
|---|---|
| `Credit reporting` ↔ `Debt collection` | Une dette recouvrée est *aussi* signalée aux bureaux de crédit. Les deux classes pèsent 54 % du corpus : c'est la confusion la plus coûteuse. |
| `Vehicle loan or lease` ↔ `Credit reporting` | 26,7 % des plaintes auto sont formulées comme un litige de signalement. |
| `Credit card or prepaid card` ↔ `Bank account or service` | Cartes de débit, découvert, carte prépayée adossée à un compte : le périmètre réel se chevauche. 9,2 % / 4,7 % de recouvrement croisé. |
| `Money transfer or virtual currency` ↔ `Bank account or service` | 9,1 % des plaintes « transfert » mentionnent un compte courant — le premier exemple C.2 (carte ATM Chase) est d'ailleurs, à mon sens, **mal étiqueté**. |
| `Payday, title or personal loan` ↔ `Debt collection` | Un prêt payday impayé finit en recouvrement. Signature lexicale propre faible (24,3 % seulement). |

**3. Deux classes ont une signature lexicale intrinsèquement faible** : `Payday, title or
personal loan` (24,3 %) et `Debt collection` (27,8 %). Ce sont les candidates au plus
faible F1 par classe, pour les deux approches.

**Conséquence pour la comparaison LLM vs ML** : une part de l'erreur est irréductible car
elle vient de l'ambiguïté de l'étiquetage humain, pas du modèle. Ni un F1-macro de 1,0 ni
même de 0,90 n'est atteignable. Il faut annoncer ce plafond **avant** de présenter les
résultats, sinon les scores se lisent comme un échec. Corollaire utile : si le LLM et
le ML plafonnent au même niveau sur les mêmes paires, cela **renforce** l'argument que la
limite est dans les données — c'est un résultat à mettre en avant, pas à masquer.

---

## D. Implications pour l'étape 2 (LLM)

### D.1 Volume de tokens par réclamation

Approximation retenue : **tokens ≈ mots × 1,3** (corpus anglais, tokenizer BPE).
Mesuré sur le corpus final nettoyé (354 395 lignes) :

| Statistique | Mots | Tokens estimés |
|---|---:|---:|
| Moyenne | 202 | **263** |
| Médiane | 141 | 183 |
| p95 | 595 | 774 |
| p99 | 929 | 1 208 |
| max | 6 314 | 8 208 |

### D.2 Effet d'une troncature

| Limite | % de textes tronqués | Tokens moyens |
|---|---:|---:|
| 500 mots | 7,48 % | 239 |
| **1 000 mots** | **0,82 %** | **257** |
| 1 500 mots | 0,28 % | 260 |
| 2 000 mots | 0,12 % | 261 |

**Recommandation : troncature à 1 000 mots.** Elle ne touche que 0,82 % des textes, ne
réduit le coût moyen que de 2 % — mais elle **plafonne le pire cas** à 1 300 tokens au lieu
de 8 200. C'est une protection contre la variance de coût et de latence, pas une
optimisation de coût moyen. La queue de distribution est le vrai risque opérationnel.

### D.3 Coût projeté pour 1 000 prédictions

**Hypothèses explicites** (à revalider sur la page tarifaire Mistral au moment de
l'étape 2) :

- Tarif `mistral-small-latest` : **0,10 $ / M tokens en entrée**, **0,30 $ / M tokens en
  sortie**.
- Prompt d'instruction (rôle, consigne, format de sortie attendu) : **~200 tokens**.
- Liste des 9 étiquettes : **~60 tokens**.
- Texte de la réclamation, tronqué à 1 000 mots : **257 tokens** (moyenne mesurée).
- Sortie : l'étiquette seule, **~8 tokens**.

| Scénario | Tokens in / appel | Coût 1 000 préd. | Coût / an à 1 000 réclamations par jour |
|---|---:|---:|---:|
| **Zero-shot** (instruction + 9 labels + texte) | 517 | **0,054 $** | **19,7 $** |
| **Few-shot** (+ 9 exemples ~270 tok chacun) | 2 947 | **0,297 $** | **108,4 $** |

Détail zero-shot : 517 × 1 000 = 517 000 tokens en entrée → 0,0517 $ ; 8 × 1 000 = 8 000
tokens en sortie → 0,0024 $ ; **total 0,0541 $**.

**Lecture** : le coût n'est **pas** un argument discriminant. Même en few-shot, classer
1 000 réclamations par jour pendant un an coûte environ 108 $ — moins qu'une journée
d'ingénieur. Évaluer le LLM sur l'intégralité du test set (70 879 lignes) coûterait ~3,8 $
en zero-shot.

**Le vrai facteur limitant est la latence, pas le prix.** À ~1 s par appel et sous
contrainte de rate limit, 70 879 appels séquentiels représentent une vingtaine d'heures.
C'est ce qui justifie l'échantillon d'évaluation de 1 000 lignes — et c'est un argument à
formuler ainsi dans la recommandation, pas comme une contrainte budgétaire.

### D.4 Le prompt peut-il lister toutes les étiquettes ?

**Oui, sans difficulté.** 9 étiquettes ≈ 60 tokens, soit ~12 % du prompt zero-shot. Même
en gardant les 18 libellés d'origine on resterait sous 150 tokens. La fusion n'était donc
**pas** motivée par une contrainte de taille de prompt — elle est motivée par la cohérence
sémantique (section F).

En revanche, avec 9 classes le LLM peut recevoir une **définition courte par étiquette**
(une ligne chacune, ~25 tokens) pour lever précisément les ambiguïtés identifiées en C.4 —
typiquement « une dette signalée au bureau de crédit relève de `Debt collection` si le
litige porte sur le recouvrement, de `Credit reporting` s'il porte sur l'exactitude du
signalement ». Coût : +225 tokens/appel, soit 0,082 $ pour 1 000 prédictions. **C'est le
levier de qualité le plus rentable de toute l'étape 2**, et il n'existe pas côté TF-IDF :
à noter pour la recommandation finale.

---

## E. Recommandations chiffrées

### E.1 `MIN_TEXT_LENGTH = 20` caractères ✅ (valeur arbitrée, confirmée)

**Argument** : coûte **148 lignes (0,039 %)** et supprime des textes structurellement
non classables (« Account is fraud », « Needs to be removed ») qui injectent du bruit
d'étiquetage. Le p5 étant à 132 caractères, le seuil est très loin de la masse de la
distribution : aucun risque de couper du signal utile.

Un seuil plus agressif serait défendable — 50 caractères coûterait 2 305 lignes (0,60 %) —
mais 20 est le bon arbitrage : il élimine l'inclassable sans jugement sur le contenu.
**Je confirme la valeur.**

### E.2 `MIN_SAMPLES_PER_CLASS = 1 000` ✅ (valeur arbitrée, confirmée — aucune classe sous le seuil)

**Vérification demandée, faite** : après fusion et nettoyage, la plus petite classe est
`Vehicle loan or lease` avec **5 704 lignes**, soit **5,7 × le seuil**.

- Classes sous 1 000 : **aucune**
- Classes sous 100 : **aucune**
- Classes sous 2 000 : **aucune**

Le seuil ne déclenche donc **rien** dans le pipeline. Il reste utile comme **garde-fou** :
si le dataset est rafraîchi (le CFPB publie en continu) ou si le mapping évolue, une classe
trop rare sera signalée au lieu de passer silencieusement dans le split — où elle
produirait un F1-macro instable sans que personne ne le voie. `handle_rare_classes()` doit
donc être écrite et journaliser explicitement « 0 classe sous le seuil » plutôt que d'être
omise.

### E.3 Classes rares : supprimer, fusionner ou garder ?

**Décision appliquée : fusionner puis exclure deux libellés**, conformément à ton arbitrage.

- **Fusionner** les 16 libellés vers 9 classes (le ratio maj/min passe de 5 774 à 18,9).
- **Exclure `Consumer Loan` (9 474 lignes)** : correspondance non injective avec le nouveau
  référentiel — vérifié en F.4, l'exclusion est justifiée.
- **Exclure `Other financial service` (292 lignes)** : fourre-tout sans définition
  sémantique, sous le seuil de 1 000.
- **Ne rien supprimer d'autre** : `Virtual currency` (16 lignes) est absorbé par la fusion
  et disparaît comme classe autonome. Aucune classe résiduelle n'est rare.

Après cela, **aucun ré-échantillonnage n'est nécessaire** : un ratio de 18,9 est un
déséquilibre modéré, gérable par `class_weight='balanced'` à l'étape 3 et par la métrique
F1-macro. Sur-échantillonner ou sous-échantillonner introduirait un biais sans bénéfice.

### E.4 Sous-échantillonnage du dataset ? **Non** ✅ (conforme à ton arbitrage)

**Aucun sous-échantillonnage du train.** 283 516 lignes d'entraînement pour un TF-IDF +
régression logistique ou SGD, c'est un volume que scikit-learn traite en quelques minutes
sur une machine ordinaire. Il n'y a pas de contrainte de temps de calcul à arbitrer.

C'est même un **argument central de la recommandation finale** : le ML classique exploite
283 516 exemples annotés, le LLM en exploite ~9 (few-shot). Brider le train reviendrait à
saboter le seul avantage structurel du ML avant même de mesurer.

`SAMPLE_FRACTION` est prévu dans `config.py` (valeur par défaut `1.0`) pour itérer vite en
développement, sans jamais toucher aux fichiers produits.

### E.5 Colonnes à conserver / à jeter

**À conserver** (4 colonnes chargées via `usecols`) :

| Colonne | Rôle |
|---|---|
| `Consumer Claim` → `text` | Feature unique |
| `Tag` → `label` (après mapping) | Cible |
| `Complaint ID` | Clé de traçabilité + **tri déterministe pour l'idempotence** (voir E.6) |
| `Date received` | **Métadonnée d'audit uniquement** — jamais une feature (voir avertissement ci-dessous) |

⚠️ **`Date received` ne doit jamais entrer dans le modèle.** Après fusion, `Vehicle loan or
lease` n'existe que depuis avril 2017 (voir F.4) : la date prédirait cette classe par pur
artefact de référentiel. Elle est conservée pour permettre l'audit et pour justifier le
choix d'un split aléatoire plutôt que temporel — rien de plus.

**À jeter** (11 colonnes, non chargées) : `Company public response`, `Company`, `State`,
`ZIP code`, `Tags`, `Consumer consent provided?`, `Submitted via`, `Date sent to company`,
`Company response to consumer`, `Timely response?`, `Consumer disputed?`.

Justification : toutes sont soit des métadonnées post-traitement (donc **fuite de données**
— `Company response to consumer` est connue *après* la classification), soit
non pertinentes pour classer un texte, soit constantes sur le périmètre
(`Submitted via = Web` à 100 %). Charger uniquement 4 colonnes fait par ailleurs passer
l'empreinte mémoire de ~5 Go à environ 500 Mo.

### E.6 Point d'implémentation : idempotence du dédoublonnage

`drop_duplicates(keep='first')` dépend de l'ordre des lignes. Pour que `main()` soit
réellement idempotent et reproductible, il faut **trier par `Complaint ID` avant de
dédoublonner**. Sans cela, la ligne conservée parmi 35 copies — et donc son étiquette pour
les 168 textes contradictoires — varierait d'une exécution à l'autre.

---

## F. Impact de la fusion

### F.1 Croisement `Tag` × année — vérification demandée

Sur les **lignes avec texte** (le corpus exploitable) :

| Tag | 2015 | 2016 | 2017 | 2018 | 2019 |
|---|---:|---:|---:|---:|---:|
| Credit reporting, credit repair services, or other... | 0 | 0 | 36 153 | 48 995 | 7 230 |
| Debt collection | 14 573 | 18 716 | 23 592 | 26 176 | 3 653 |
| Mortgage | 12 303 | 15 770 | 13 141 | 10 257 | 1 516 |
| Credit reporting | 10 258 | 15 081 | 6 249 | **0** | **0** |
| Student loan | 1 805 | 4 179 | 9 642 | 5 291 | 893 |
| Credit card or prepaid card | 0 | 0 | 7 791 | 11 854 | 1 734 |
| Credit card | 6 243 | 9 434 | 3 161 | **0** | **0** |
| Bank account or service | 4 559 | 7 757 | 2 569 | **0** | **0** |
| Checking or savings account | 0 | 0 | 4 743 | 6 997 | 1 141 |
| Consumer Loan | 2 974 | 4 672 | 1 828 | **0** | **0** |
| Vehicle loan or lease | 0 | 0 | 2 047 | 3 268 | 430 |
| Money transfer, virtual currency, or money service | 0 | 0 | 1 868 | 3 243 | 355 |
| Payday loan, title loan, or personal loan | 0 | 0 | 1 643 | 2 398 | 380 |
| Payday loan | 615 | 866 | 266 | **0** | **0** |
| Money transfers | 562 | 711 | 224 | **0** | **0** |
| Prepaid card | 772 | 492 | 186 | **0** | **0** |
| Other financial service | 87 | 139 | 66 | **0** | **0** |
| Virtual currency | 7 | 6 | 3 | **0** | **0** |

*(Les narratifs ne sont collectés que depuis 2015 ; le tableau sur fichier complet couvre
2011-2019 et donne exactement la même structure.)*

### F.2 Date exacte de bascule

Premières et dernières apparitions de chaque libellé, sur le **fichier complet** :

| Libellé | Première | Dernière |
|---|---|---|
| **Ancien référentiel** | | |
| Credit reporting | 2012-10-22 | **2017-04-22** |
| Credit card | 2011-12-01 | **2017-04-22** |
| Bank account or service | 2012-03-01 | **2017-04-22** |
| Consumer Loan | 2012-03-01 | **2017-04-21** |
| Payday loan | 2013-11-06 | **2017-04-21** |
| Money transfers | 2013-04-04 | **2017-04-21** |
| Prepaid card | 2014-07-20 | **2017-04-21** |
| Other financial service | 2014-07-19 | **2017-04-21** |
| Virtual currency | 2014-08-15 | 2017-04-03 |
| **Nouveau référentiel** | | |
| Checking or savings account | **2017-04-21** | 2019-05-10 |
| Credit reporting, credit repair services, or other... | **2017-04-24** | 2019-05-10 |
| Credit card or prepaid card | **2017-04-24** | 2019-05-09 |
| Vehicle loan or lease | **2017-04-24** | 2019-05-10 |
| Money transfer, virtual currency, or money service | **2017-04-24** | 2019-05-07 |
| Payday loan, title loan, or personal loan | **2017-04-24** | 2019-05-09 |
| **Permanents** | | |
| Mortgage | 2011-12-01 | 2019-05-10 |
| Debt collection | 2013-07-10 | 2019-05-10 |
| Student loan | 2012-03-01 | 2019-05-10 |

### F.3 Conclusion de la vérification : **le croisement confirme intégralement tes hypothèses**

La bascule est **nette et datée : week-end du 21-24 avril 2017** (le 22 avril 2017 était un
samedi, le 24 un lundi). Aucun ancien libellé n'apparaît après le 22/04/2017 ; aucun
nouveau libellé n'apparaît avant le 21/04/2017. Il n'y a **pas de zone de recouvrement**
au-delà de ces trois jours calendaires, donc pas de période où les deux référentiels
auraient coexisté.

Les trois libellés `Debt collection`, `Mortgage` et `Student loan` traversent la bascule
sans interruption — ce qui confirme qu'ils sont bien inchangés d'un référentiel à l'autre
et valide leur maintien tel quel dans le mapping.

**Aucune hypothèse de mapping n'est contredite. Le mapping est appliqué tel que tu l'as
défini.**

### F.4 Le cas `Consumer Loan` : ton exclusion est confirmée, avec une conséquence

`Consumer Loan` (2012-03-01 → 2017-04-21, 9 474 lignes textuelles) disparaît exactement à
la bascule, au moment où apparaissent **simultanément** `Vehicle loan or lease` et `Payday
loan, title loan, or personal loan`. Ces deux classes n'ont **aucun** prédécesseur dans
l'ancien référentiel — ce qui confirme que `Consumer Loan` s'est bien éclaté en deux, et
qu'aucun mapping injectif n'existe. **Exclusion justifiée.**

Conséquence à connaître : `Vehicle loan or lease` ne dispose d'**aucune donnée antérieure à
avril 2017**. C'est la seule des 9 classes dans ce cas. Deux implications :

1. **Le split doit rester aléatoire stratifié, pas temporel.** Un split temporel
   (train = avant 2018, test = après) priverait cette classe d'entraînement ou de test. Le
   choix de `train_test_split(stratify=y)` prescrit en phase 3 est donc le bon.
2. **`Date received` est une feature interdite** (déjà signalé en E.5) : elle prédirait
   `Vehicle loan or lease` à 100 % de rappel par simple artefact administratif.

### F.5 Distribution avant / après fusion

**Avant fusion**, après exclusion de `Consumer Loan` et `Other financial service`
(373 798 lignes, 16 classes) — ratio maj/min = **5 774**.

**Après fusion** (373 798 lignes, **9 classes**) — ratio maj/min = **21,6** :

| Classe fusionnée | n | % |
|---|---:|---:|
| Credit reporting | 123 966 | 33,16 |
| Debt collection | 86 710 | 23,20 |
| Mortgage | 52 987 | 14,18 |
| Credit card or prepaid card | 41 667 | 11,15 |
| Bank account or service | 27 766 | 7,43 |
| Student loan | 21 810 | 5,83 |
| Money transfer or virtual currency | 6 979 | 1,87 |
| Payday, title or personal loan | 6 168 | 1,65 |
| Vehicle loan or lease | 5 745 | 1,54 |

Le ratio de déséquilibre est **divisé par 267**. C'est le gain principal de la fusion : on
passe d'un problème à classe quasi-vide (16 exemples) à un déséquilibre modéré et
parfaitement traitable.

### F.6 Lignes conservées / exclues à chaque étape

| # | Étape | Delta | Restant |
|---|---|---:|---:|
| 0 | Fichier brut | — | 1 282 355 |
| 1 | Filtre `Consumer Claim` non nul | −898 791 | 383 564 |
| 2 | Exclusion `Consumer Loan` | −9 474 | 374 090 |
| 3 | Exclusion `Other financial service` | −292 | 373 798 |
| 4 | Application du mapping (16 → 9 classes) | 0 | 373 798 |
| 5 | Normalisation des espaces + texte vide | −0 | 373 798 |
| 6 | `MIN_TEXT_LENGTH = 20` | −147 | 373 651 |
| 7 | Doublons exacts sur le texte | −19 256 | **354 395** |
| 8 | `MIN_SAMPLES_PER_CLASS = 1 000` | −0 | **354 395** |

**Corpus final : 354 395 lignes, 9 classes** — soit 92,4 % des lignes textuelles et 27,6 %
du fichier brut.

*(Note : 147 lignes supprimées par le seuil de longueur au lieu des 148 mesurées en A.4 —
une des lignes courtes appartenait à une classe exclue à l'étape 2.)*

### F.7 Distribution finale et volumétrie du split

**354 395 lignes, 9 classes, ratio maj/min = 18,9** :

| Classe | n | % | ≈ train (80 %) | ≈ test (20 %) | ≈ éch. LLM 1 000 |
|---|---:|---:|---:|---:|---:|
| Credit reporting | 107 753 | 30,40 | 86 202 | 21 551 | 304 |
| Debt collection | 84 028 | 23,71 | 67 222 | 16 806 | 237 |
| Mortgage | 52 927 | 14,93 | 42 342 | 10 585 | 149 |
| Credit card or prepaid card | 41 399 | 11,68 | 33 119 | 8 280 | 117 |
| Bank account or service | 27 703 | 7,82 | 22 162 | 5 541 | 78 |
| Student loan | 21 769 | 6,14 | 17 415 | 4 354 | 61 |
| Money transfer or virtual currency | 6 966 | 1,97 | 5 573 | 1 393 | 20 |
| Payday, title or personal loan | 6 146 | 1,73 | 4 917 | 1 229 | 17 |
| Vehicle loan or lease | 5 704 | 1,61 | 4 563 | 1 141 | **16** |
| **Total** | **354 395** | 100 | **283 516** | **70 879** | **1 000** |

⚠️ **Point de vigilance** : dans un échantillon LLM de 1 000 lignes stratifié
proportionnellement, la plus petite classe n'est représentée que par **16 exemples**. Le F1
mesuré sur cette classe aura un intervalle de confiance très large (de l'ordre de ±0,15).
Comme le F1-**macro** pondère les 9 classes à égalité, cette imprécision se propage à la
métrique principale de comparaison. C'est la première décision à trancher en section H.

### F.8 Contradictions d'étiquetage : effet de la fusion

| | Avant fusion | Après fusion |
|---|---:|---:|
| Textes portant ≥ 2 étiquettes différentes | 201 | **168** |
| Lignes concernées | 1 905 | 1 757 |

La fusion résorbe 33 contradictions (16 %) — celles qui n'étaient qu'un artefact de
référentiel, typiquement le même courrier classé `Credit reporting` avant 2017 et
`Credit reporting, credit repair services...` après. Les 168 restantes sont de **vraies
divergences d'appréciation humaine**, et elles confirment quantitativement le jugement de
la section C : même un annotateur du CFPB hésite entre ces classes.

Après dédoublonnage sur le texte, ces 168 textes ne subsistent qu'en un seul exemplaire,
avec une étiquette retenue arbitrairement (la première dans l'ordre de tri). L'alternative
propre serait de les supprimer entièrement — 168 textes, soit 0,05 % du corpus. Question
ouverte en section H.

### F.9 Reste-t-il des paires confondables **sur le fond** ?

Oui — et c'est important de le distinguer du problème de référentiel, qui lui est résolu.

La fusion a supprimé toutes les confusions **administratives** (`Credit card` vs `Credit
card or prepaid card` : même chose à une date près). Elle ne touche pas aux confusions
**sémantiques**, qui restent entières et sont détaillées en C.4. Les trois qui subsistent
après fusion :

1. **`Credit reporting` ↔ `Debt collection`** — la plus grave : 54 % du corpus, 35,8 % de
   recouvrement lexical, et une frontière conceptuelle réellement floue (contester une
   dette *et* son signalement est le même acte pour le consommateur).
2. **`Credit card or prepaid card` ↔ `Bank account or service`** — la fusion a même
   *aggravé* cette paire : en absorbant `Prepaid card` dans les cartes et `Checking or
   savings account` dans les comptes, elle rapproche deux classes dont les périmètres
   réels se chevauchent (carte de débit, découvert, carte prépayée adossée à un compte).
3. **`Money transfer or virtual currency` ↔ `Bank account or service`** — 9,1 % de
   recouvrement, et l'exemple ATM Chase cité en C.2 illustre un cas où l'étiquette CFPB
   me paraît elle-même discutable.

**Mon avis** : la fusion était nécessaire et bien calibrée, mais elle ne fait pas
disparaître le plafond de performance. Elle le rend simplement *mesurable* — avec 9 classes
sémantiquement définies, la matrice de confusion de l'étape 3 sera interprétable, ce
qu'elle n'aurait jamais été avec 18 libellés dont la moitié sont des synonymes historiques.
C'est un résultat exploitable en soi.

---

## G. Masquage `XXXX`

### G.1 Quantification

| Mesure | Valeur |
|---|---:|
| Lignes contenant au moins un `XX+` | **325 921 / 383 564 → 84,97 %** |
| Occurrences par ligne (moyenne, tout le corpus) | 13,1 |
| Occurrences par ligne (moyenne, si présent) | **15,5** |
| Occurrences par ligne (médiane, si présent) | 8 |
| Occurrences maximum sur une ligne | **2 313** |
| Part moyenne de caractères `X` dans le texte | 4,53 % |
| Lignes dont > 10 % des caractères sont des `X` | 9,53 % |
| Lignes dont > 25 % des caractères sont des `X` | 1,45 % |
| Lignes contenant une date masquée `XX/XX` | 38,89 % |
| Lignes contenant un montant masqué `{$...}` | 30,89 % |

Le masquage est **la caractéristique dominante du corpus** : 85 % des textes en contiennent.
Il est appliqué par le CFPB avant publication et remplace indistinctement noms de personnes,
noms d'entreprises, dates, adresses, numéros de compte et montants.

### G.2 Trois exemples de textes fortement caviardés

**1. `Credit reporting` — 76 % de caractères `X`**
```
Victim of credit report inquiry fraud XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX
XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX
XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX XX...
```

**2. `Credit reporting` — 76 % de caractères `X`**
```
XXXX XXXX XXXX XXXX XXXX XXXX XXXXXXXX XXXX Clarity services XXXX XXXX XXXX XXXX
XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX. XXXX XXXX XXXX.
XXXX XXXX XXXX XXXX. XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX X...
```

**3. `Credit reporting` — 76 % de caractères `X`**
```
THE FOLLOWING INQUIRIES ARE ALL A RESULT OF FRAUD XXXX XXXX XXXX/XXXX/XXXX XXXX
XXXX XXXX XXXX XXXX/XXXX/XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX
XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX...
```

Ces trois exemples partagent la même structure : une phrase d'introduction porteuse de sens
(« Victim of credit report inquiry fraud »), suivie d'une **liste caviardée** de noms de
créanciers ou de dates. Le signal utile est concentré dans les premiers mots ; le reste est
du remplissage. À noter, les trois appartiennent à `Credit reporting` — c'est cohérent,
c'est la classe où l'on énumère des lignes de rapport de crédit.

### G.3 Impact pour TF-IDF

**Impact réel mais gérable, à traiter dans le pipeline de l'étape 3 — pas ici.**

- Le token `xxxx` sera présent dans 85 % des documents. Son **IDF sera donc quasi nul**
  (log(N/df) ≈ 0,16), et TF-IDF le neutralisera **automatiquement**. Le mécanisme même de
  l'IDF est conçu pour ça.
- Filet de sécurité : `max_df=0.9` dans le `TfidfVectorizer` l'élimine explicitement, ou
  son ajout à la liste de stop-words. C'est une ligne de configuration.
- Le risque résiduel est ailleurs : `XXXX` **détruit les entités nommées**. Un TF-IDF ne
  peut pas apprendre que « Navient » signale un prêt étudiant ou « Equifax » un problème de
  signalement, puisque la plupart des noms d'entreprises sont masqués. Le modèle doit
  s'appuyer sur le vocabulaire générique (`forbearance`, `escrow`, `repossess`), ce qui
  explique en partie les taux de signature lexicale mesurés en C.3.
- Effet secondaire à surveiller : les 1,45 % de textes à plus de 25 % de `X` ont une
  longueur effective très inférieure à leur longueur apparente. Avec une normalisation L2,
  leur vecteur sera dominé par les quelques tokens réels — ce qui est le comportement
  souhaité.

**Conclusion : aucun retrait du masquage en amont.** Le nettoyage se limite aux espaces,
conformément à la spécification de la phase 3. Neutraliser `xxxx` est le travail du
`TfidfVectorizer`, pas de `data_prep.py` — et le texte brut doit rester intact pour le LLM.

### G.4 Impact pour le prompt LLM

**Plus problématique que pour TF-IDF, et pour une raison différente.**

- **Coût direct** : les `XXXX` représentent 4,53 % des caractères en moyenne et sont
  tokenisés inefficacement (`XXXX` répété se découpe mal en BPE). On paie donc pour du
  vide. Sur 1 000 prédictions à 0,054 $, l'enjeu reste marginal — ce n'est pas un argument
  de coût.
- **Coût indirect, lui réel** : les textes à 76 % de `X` consomment leur budget de tokens
  en bruit pur. La troncature à 1 000 mots (D.2) peut alors **couper le signal utile** si
  la liste caviardée précède l'exposé du problème. Sur les exemples G.2 le signal est en
  tête, donc une troncature *par la fin* le préserve — mais ce n'est pas garanti.
- **Risque de confusion du modèle** : un LLM confronté à « XXXX XXXX XXXX XXXX » sur 300
  mots peut halluciner un contenu, ou répondre qu'il manque d'information. Ce n'est pas
  hypothétique.

**Recommandation pour l'étape 2** — à décider au moment d'écrire le prompt, pas maintenant :
mentionner explicitement le masquage dans l'instruction système, par exemple *« Le texte
a été anonymisé : les séquences XXXX remplacent des noms, dates ou montants. Ignore-les et
classe sur le contenu restant. »* Coût : ~25 tokens. Cela évite au modèle de traiter le
masquage comme une anomalie et lui indique quoi faire des textes très caviardés.

**Point d'égalité entre les deux approches** : le masquage handicape le ML et le LLM pour
la *même* raison — la perte des entités nommées. Ce n'est donc pas un facteur
discriminant dans la recommandation finale, et il faut le dire, car c'est un biais
d'analyse tentant.

---

## H. Décisions à trancher

Sept points où j'ai besoin de ton arbitrage. Pour chacun je donne ma recommandation ; les
points 1 et 2 sont les seuls réellement structurants.

### H.1 🔴 Composition de l'échantillon d'évaluation LLM (1 000 lignes)

**Le problème** : une stratification proportionnelle donne **16 exemples** pour
`Vehicle loan or lease`, 17 pour `Payday, title or personal loan`, 20 pour `Money
transfer`. Le F1-macro, métrique principale de comparaison, pondère ces classes à égalité
avec `Credit reporting` (304 exemples). Une seule erreur sur les 16 fait bouger le F1 de
cette classe de plusieurs points, et le F1-macro d'environ un demi-point.

| Option | Description | Conséquence |
|---|---|---|
| **A** — Stratification proportionnelle | Conforme à la spec ; 16 exemples au minimum | Reflète la réalité de production, mais F1-macro bruité sur 3 classes |
| **B** — Plancher par classe *(ma recommandation)* | Minimum 50 par classe, le reste proportionnel | F1-macro nettement plus stable ; l'échantillon ne reflète plus la distribution réelle, mais l'accuracy pondérée reste calculable via re-pondération |
| **C** — Taille portée à 2 000 | Double l'échantillon | Coût LLM ~0,11 $, latence ~40 min ; la plus petite classe passe à 32 — mieux, mais toujours peu |
| **D** — Échantillon équilibré (111/classe) | Chaque classe à égalité | F1-macro optimal statistiquement ; l'accuracy globale devient ininterprétable |

**Ma recommandation : B**, avec un plancher à 50. Argument : le F1-macro est la métrique de
décision du projet ; on ne peut pas la laisser reposer sur 16 observations. Et quelle que
soit l'option, **le modèle ML devra être évalué sur exactement le même échantillon** pour
que la comparaison soit valide — plus, séparément, sur le test set complet pour montrer ce
que le volume lui apporte.

### H.2 🔴 Que faire des 168 textes à étiquettes contradictoires ?

Après dédoublonnage, ils survivent avec une étiquette retenue arbitrairement. Ce sont, par
construction, des exemples dont on **sait** qu'ils sont mal étiquetés dans au moins un cas.

- **Option A** : les supprimer intégralement (−168 lignes, 0,05 %). Propre : on n'entraîne
  pas sur du bruit connu.
- **Option B** *(ma recommandation)* : garder, en retenant l'étiquette **majoritaire**
  parmi les doublons (pour le template FCRA : `Credit reporting` à 35 contre 1 pour
  `Debt collection`) plutôt que la première rencontrée. Coût nul en volume, gain en
  qualité d'étiquetage, et déterministe.
- **Option C** : garder tel quel avec `keep='first'` après tri par `Complaint ID`.

**Ma recommandation : B.** Le vote majoritaire est aussi simple à implémenter que
`keep='first'` et strictement meilleur. Note : ces textes sont des courriers types diffusés
en masse, donc leur suppression pure (option A) reste tout à fait défendable.

### H.3 🟡 Troncature des textes : où l'appliquer ?

La troncature à 1 000 mots (D.2) concerne 0,82 % des textes mais divise le pire cas par 6.

- **Option A** *(ma recommandation)* : **ne pas tronquer dans `data_prep.py`**. Les fichiers
  `train.csv` / `test.csv` contiennent le texte intégral ; la troncature est appliquée à
  l'étape 2 au moment de construire le prompt, et jamais côté TF-IDF (qui n'en a aucun
  besoin).
- **Option B** : tronquer dans `clean()`, donc pour les deux approches.

**Ma recommandation : A.** Tronquer en amont handicaperait le ML sans raison et rendrait les
deux étapes non indépendantes. La spec de la phase 3 va déjà dans ce sens (« le texte brut
est nécessaire au LLM »). Je le signale pour que ce soit un choix explicite et non un oubli.

### H.4 🟡 Nom des classes fusionnées dans les fichiers produits

J'ai utilisé les libellés de ton mapping. Deux d'entre eux sont des reformulations et non
des libellés CFPB existants :

- `Money transfer or virtual currency` (au lieu de `Money transfer, virtual currency, or money service`)
- `Payday, title or personal loan` (au lieu de `Payday loan, title loan, or personal loan`)

**Ma recommandation** : les garder. Ils sont plus courts (économie de tokens dans le prompt,
lisibilité des axes de la matrice de confusion) et sans ambiguïté. Je confirme juste que tu
assumes l'écart avec la nomenclature CFPB officielle, au cas où les équipes ZenAssist
travailleraient déjà avec elle.

### H.5 🟡 `SAMPLE_FRACTION` — où l'appliquer ?

Prévu dans `config.py` avec `1.0` par défaut. Mais s'applique-t-il **avant** le split
(échantillon global, permet d'itérer sur tout le pipeline) ou **après** (train réduit, test
complet) ?

**Ma recommandation** : avant le split, avec stratification, et **jamais écrit dans
`data/processed/`** — un `SAMPLE_FRACTION < 1.0` doit produire des fichiers suffixés
(`train_sample.csv`) pour qu'il soit impossible de livrer par erreur un modèle entraîné sur
un corpus partiel.

### H.6 🟢 Faut-il conserver `Date received` dans `train.csv` / `test.csv` ?

**Ma recommandation : oui**, en colonne de métadonnée. Elle ne coûte presque rien et permet
de vérifier après coup qu'aucune classe n'est concentrée sur une période — utile pour
justifier le choix du split a posteriori. Le risque de l'utiliser comme feature par
inadvertance est écarté par un commentaire explicite dans `config.py` et par le fait que
seule la colonne texte est passée au vectorizer.

### H.7 🟢 Confirmation du seuil `MIN_SAMPLES_PER_CLASS`

Tu demandais confirmation : **confirmée**. La plus petite classe finale compte 5 704 lignes,
soit 5,7 × le seuil de 1 000. `handle_rare_classes()` ne supprimera rien et journalisera
« 0 classe sous le seuil ». Je propose de garder la fonction malgré tout, comme garde-fou
documenté pour un rafraîchissement futur du dataset — dis-moi si tu préfères la retirer
pour alléger le code.

---

## Synthèse en une page

| | Valeur |
|---|---|
| **Corpus final** | 354 395 réclamations, 9 classes |
| **Train / test (80/20)** | 283 516 / 70 879 |
| **Échantillon LLM** | 1 000 (composition à arbitrer — H.1) |
| **Déséquilibre** | ratio 18,9 (contre 5 774 avant fusion) |
| **Métrique principale** | F1-macro — justifiée par le déséquilibre 18,9 et le fait que les classes rares comptent autant pour le client |
| **Longueur médiane** | 742 caractères / 136 mots / ~183 tokens |
| **Coût LLM (1 000 préd., zero-shot)** | ~0,054 $ — **non discriminant** |
| **Facteur limitant du LLM** | la latence (~20 h pour 70 879 appels), pas le prix |
| **Avantage structurel du ML** | 283 516 exemples annotés contre ~9 en few-shot |
| **Plafond de performance attendu** | réel, dû à l'ambiguïté d'étiquetage `Credit reporting` ↔ `Debt collection` (54 % du corpus) |

---

*Fin du diagnostic — phase 2. Aucun module ni notebook produit à ce stade.*
