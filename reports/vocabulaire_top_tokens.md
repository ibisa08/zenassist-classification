# Journal du vocabulaire — étape 3, phase B

Tokens de plus fort poids par classe, pour chaque modèle ajusté sur le **train
complet** (283 449 lignes). Agrégats uniquement : aucun texte de réclamation
n'apparaît ici.

Deux bases de classement, parce que les deux familles n'ont pas de grandeur
commune :

* modèles linéaires — **coefficient un-contre-tous**, signé ;
* `MultinomialNB` — **`log P(t|c)` recentré sur la moyenne des classes**. Sans ce
  recentrage, le classement serait dominé par les termes fréquents partout et ne
  dirait rien de la classe.

Les deux ne se comparent pas terme à terme ; ils se lisent en parallèle.

## Ce que la comparaison montre — l'indépendance conditionnelle prise sur le fait

Sur `Credit reporting`, `MultinomialNB` place dans ses dix premiers :

```
equifax | transunion | experian | trans union | equifax and |
to equifax | with equifax | equifax is | xxxx equifax | equifax has
```

Sept de ces dix termes portent **la même évidence** — la mention d'Equifax — sous
sept formes. NB les traite comme sept observations indépendantes et compte donc
sept fois ce qu'un modèle discriminatif compte une fois. C'est exactement le
mécanisme n° 1 du §7 du protocole de sélection, rendu visible sans qu'aucune
mesure ne soit nécessaire.

`LinearSVC`, sur la même classe, produit dix termes distincts :

```
equifax | experian | transunion | report | inquiries |
trans union | trans | tax lien | freeze | credit
```

Le poids se répartit entre variables corrélées au lieu de s'y répéter. La
différence de F1 entre les deux modèles (0,81448 contre 0,71800) a ici son
illustration la plus directe.

## Réserve de transférabilité

`xxxx` et ses composés apparaissent dans plusieurs classements. C'est le jeton
d'anonymisation du CFPB, présent dans 84,32 % des documents. Le modèle apprend
donc en partie **la chaîne d'anonymisation**, pas seulement le domaine. Sur des
réclamations non anonymisées de la même façon, ces variables ne se
retrouveraient pas. La décision de conserver `xxxx` est documentée dans
`reports/resultats_phaseA.md` §6.


---

## LinearSVC

Classement : coefficient (un-contre-tous).

**Bank account or service**

| # | terme | poids |
|---|---|---|
| 1 | `bank` | +6.7338 |
| 2 | `scottrade` | +4.2397 |
| 3 | `checking` | +4.0203 |
| 4 | `branch` | +3.9061 |
| 5 | `overdraft` | +3.9024 |
| 6 | `cd` | +3.8070 |
| 7 | `debit card` | +3.6677 |
| 8 | `usaa` | +3.5546 |
| 9 | `atm` | +3.5236 |
| 10 | `bill pay` | +3.4855 |
| 11 | `banking` | +3.2598 |
| 12 | `chase` | +3.2236 |
| 13 | `pnc` | +3.2155 |
| 14 | `deposit` | +3.1906 |
| 15 | `debit` | +3.1833 |
| 16 | `one 360` | +3.1325 |
| 17 | `savings` | +3.0328 |
| 18 | `360` | +3.0254 |
| 19 | `suntrust` | +2.7322 |
| 20 | `account` | +2.5805 |
| 21 | `citigold` | +2.5533 |
| 22 | `regions` | +2.5323 |
| 23 | `overdrawn` | +2.5102 |
| 24 | `citibank` | +2.4925 |
| 25 | `debt card` | +2.4736 |

**Credit card or prepaid card**

| # | terme | poids |
|---|---|---|
| 1 | `card` | +8.0110 |
| 2 | `rushcard` | +5.4404 |
| 3 | `citi` | +4.0547 |
| 4 | `balance transfer` | +3.9228 |
| 5 | `amex` | +3.9210 |
| 6 | `synchrony` | +3.8481 |
| 7 | `macy` | +3.7531 |
| 8 | `discover` | +3.6482 |
| 9 | `rewards` | +3.6355 |
| 10 | `barclay` | +3.6197 |
| 11 | `target` | +3.5123 |
| 12 | `barclaycard` | +3.5037 |
| 13 | `mastercard` | +3.3144 |
| 14 | `barclays` | +3.2981 |
| 15 | `visa` | +3.2978 |
| 16 | `citicard` | +3.2668 |
| 17 | `cards` | +3.2277 |
| 18 | `american express` | +3.2062 |
| 19 | `netspend` | +3.1312 |
| 20 | `capital one` | +2.9748 |
| 21 | `interest` | +2.9727 |
| 22 | `capitalone` | +2.9573 |
| 23 | `chase` | +2.9479 |
| 24 | `prepaid` | +2.8588 |
| 25 | `credit` | +2.7225 |

**Credit reporting**

| # | terme | poids |
|---|---|---|
| 1 | `equifax` | +11.4442 |
| 2 | `experian` | +10.7109 |
| 3 | `transunion` | +9.7400 |
| 4 | `report` | +4.6958 |
| 5 | `inquiries` | +4.5068 |
| 6 | `trans union` | +3.8736 |
| 7 | `trans` | +3.1409 |
| 8 | `tax lien` | +3.0810 |
| 9 | `freeze` | +3.0166 |
| 10 | `credit` | +2.9408 |
| 11 | `inquiry` | +2.8623 |
| 12 | `inquires` | +2.7697 |
| 13 | `reporting` | +2.7290 |
| 14 | `lexisnexis` | +2.6422 |
| 15 | `background` | +2.4897 |
| 16 | `incorrect` | +2.3952 |
| 17 | `investigation` | +2.2718 |
| 18 | `background check` | +2.2332 |
| 19 | `my report` | +2.0942 |
| 20 | `chexsystems` | +2.0654 |
| 21 | `information` | +2.0348 |
| 22 | `reappeared` | +1.9903 |
| 23 | `xxxx is` | +1.9607 |
| 24 | `score` | +1.9226 |
| 25 | `bureaus are` | +1.8424 |

**Debt collection**

| # | terme | poids |
|---|---|---|
| 1 | `debt` | +7.7613 |
| 2 | `collection` | +4.6172 |
| 3 | `midland` | +3.1428 |
| 4 | `collections` | +3.1101 |
| 5 | `erc` | +2.9782 |
| 6 | `owe` | +2.9138 |
| 7 | `convergent` | +2.8393 |
| 8 | `afni` | +2.6934 |
| 9 | `associates` | +2.6190 |
| 10 | `cmre` | +2.6051 |
| 11 | `recovery` | +2.4956 |
| 12 | `owed` | +2.4721 |
| 13 | `collect` | +2.4687 |
| 14 | `debts` | +2.3903 |
| 15 | `medical` | +2.3670 |
| 16 | `portfolio` | +2.3530 |
| 17 | `receivables` | +2.3524 |
| 18 | `report ftc` | +2.3453 |
| 19 | `calling` | +2.3250 |
| 20 | `pinnacle` | +2.3082 |
| 21 | `calls` | +2.2447 |
| 22 | `repeatedly` | +2.1543 |
| 23 | `this` | +2.1312 |
| 24 | `collector` | +2.1264 |
| 25 | `inc` | +2.1035 |

**Money transfer or virtual currency**

| # | terme | poids |
|---|---|---|
| 1 | `coinbase` | +8.6184 |
| 2 | `moneygram` | +5.6953 |
| 3 | `paypal` | +5.1283 |
| 4 | `western union` | +3.6207 |
| 5 | `wire` | +3.5372 |
| 6 | `google` | +3.5089 |
| 7 | `transfer` | +3.4498 |
| 8 | `money` | +3.3231 |
| 9 | `money transfer` | +3.2793 |
| 10 | `western` | +3.1559 |
| 11 | `venmo` | +2.8247 |
| 12 | `money gram` | +2.6341 |
| 13 | `gram` | +2.6073 |
| 14 | `bitcoin` | +2.5658 |
| 15 | `exchange` | +2.4675 |
| 16 | `xoom` | +2.3009 |
| 17 | `virtual` | +2.2900 |
| 18 | `support` | +2.2581 |
| 19 | `currency` | +2.1546 |
| 20 | `paypal had` | +2.1455 |
| 21 | `recipient` | +2.1297 |
| 22 | `scammed` | +2.0967 |
| 23 | `integrate` | +2.0863 |
| 24 | `received first` | +2.0525 |
| 25 | `wallet` | +2.0485 |

**Mortgage**

| # | terme | poids |
|---|---|---|
| 1 | `mortgage` | +11.5931 |
| 2 | `escrow` | +5.6227 |
| 3 | `ocwen` | +5.4549 |
| 4 | `modification` | +5.1380 |
| 5 | `ditech` | +4.7467 |
| 6 | `heloc` | +4.3310 |
| 7 | `appraisal` | +4.1247 |
| 8 | `home` | +4.1105 |
| 9 | `citimortgage` | +4.0228 |
| 10 | `loancare` | +3.9689 |
| 11 | `foreclosure` | +3.9688 |
| 12 | `seterus` | +3.9240 |
| 13 | `nationstar` | +3.8839 |
| 14 | `pmi` | +3.3482 |
| 15 | `caliber` | +3.2754 |
| 16 | `loan` | +3.1934 |
| 17 | `greentree` | +3.1263 |
| 18 | `my escrow` | +3.0213 |
| 19 | `quicken` | +3.0194 |
| 20 | `closing` | +2.8424 |
| 21 | `sls` | +2.8095 |
| 22 | `house` | +2.7524 |
| 23 | `property` | +2.6467 |
| 24 | `pennymac` | +2.6007 |
| 25 | `servicing` | +2.5977 |

**Payday, title or personal loan**

| # | terme | poids |
|---|---|---|
| 1 | `loan` | +6.0846 |
| 2 | `borrowed` | +4.0157 |
| 3 | `payday` | +3.6064 |
| 4 | `mobiloans` | +3.4234 |
| 5 | `title loan` | +3.3731 |
| 6 | `personal loan` | +3.0482 |
| 7 | `00 loan` | +2.9681 |
| 8 | `cashnetusa` | +2.7868 |
| 9 | `cashnet` | +2.7792 |
| 10 | `cash` | +2.7563 |
| 11 | `2014 60` | +2.6333 |
| 12 | `advance america` | +2.6146 |
| 13 | `ace` | +2.5179 |
| 14 | `line of` | +2.5014 |
| 15 | `titlemax` | +2.4636 |
| 16 | `internet page` | +2.4427 |
| 17 | `title` | +2.4136 |
| 18 | `rise` | +2.4127 |
| 19 | `personal line` | +2.3868 |
| 20 | `check go` | +2.3371 |
| 21 | `received loan` | +2.3258 |
| 22 | `lender` | +2.3179 |
| 23 | `avant` | +2.3000 |
| 24 | `interest` | +2.2700 |
| 25 | `lending` | +2.2448 |

**Student loan**

| # | terme | poids |
|---|---|---|
| 1 | `navient` | +9.6638 |
| 2 | `nelnet` | +4.8398 |
| 3 | `loans` | +4.7806 |
| 4 | `fedloan` | +4.3540 |
| 5 | `student` | +4.3325 |
| 6 | `aes` | +4.1917 |
| 7 | `acs` | +4.1902 |
| 8 | `forbearance` | +3.6636 |
| 9 | `loan` | +3.6126 |
| 10 | `discover student` | +3.5190 |
| 11 | `repayment` | +3.4785 |
| 12 | `education` | +3.4674 |
| 13 | `ibr` | +3.2287 |
| 14 | `school` | +3.1741 |
| 15 | `mohela` | +3.0587 |
| 16 | `private` | +2.8696 |
| 17 | `cosigner` | +2.8032 |
| 18 | `ecmc` | +2.7496 |
| 19 | `graduated` | +2.6571 |
| 20 | `sofi` | +2.6245 |
| 21 | `my loans` | +2.6007 |
| 22 | `ecsi` | +2.3929 |
| 23 | `forgiveness` | +2.3325 |
| 24 | `ameritech` | +2.3129 |
| 25 | `degree` | +2.3048 |

**Vehicle loan or lease**

| # | terme | poids |
|---|---|---|
| 1 | `car` | +5.1875 |
| 2 | `vehicle` | +4.6112 |
| 3 | `santander` | +3.9690 |
| 4 | `ally` | +3.3843 |
| 5 | `gm` | +3.3001 |
| 6 | `lease` | +3.1441 |
| 7 | `auto` | +3.0586 |
| 8 | `auto loan` | +2.8711 |
| 9 | `acceptance` | +2.8111 |
| 10 | `carmax` | +2.8018 |
| 11 | `toyota` | +2.7800 |
| 12 | `vehicles` | +2.5083 |
| 13 | `incorrect no` | +2.3669 |
| 14 | `repo` | +2.3326 |
| 15 | `honda` | +2.3278 |
| 16 | `repossessed` | +2.3206 |
| 17 | `carvana` | +2.2718 |
| 18 | `finance` | +2.2568 |
| 19 | `dealer` | +2.2159 |
| 20 | `bridgecrest` | +2.1912 |
| 21 | `gmfnancial` | +2.1694 |
| 22 | `posted past` | +2.1563 |
| 23 | `bmw` | +2.0759 |
| 24 | `new xxxx` | +2.0758 |
| 25 | `westlake` | +2.0683 |


---

## LogisticRegression

Classement : coefficient (un-contre-tous).

**Bank account or service**

| # | terme | poids |
|---|---|---|
| 1 | `bank` | +20.4972 |
| 2 | `checking` | +10.0782 |
| 3 | `branch` | +9.9623 |
| 4 | `overdraft` | +9.8247 |
| 5 | `deposit` | +9.0886 |
| 6 | `debit` | +9.0384 |
| 7 | `account` | +8.9258 |
| 8 | `debit card` | +8.7139 |
| 9 | `usaa` | +8.3859 |
| 10 | `banking` | +8.3659 |
| 11 | `chase` | +7.9081 |
| 12 | `atm` | +7.7535 |
| 13 | `scottrade` | +7.4722 |
| 14 | `cd` | +7.1600 |
| 15 | `savings` | +7.0159 |
| 16 | `citibank` | +6.9191 |
| 17 | `fees` | +6.9069 |
| 18 | `pnc` | +6.7735 |
| 19 | `funds` | +6.6380 |
| 20 | `my account` | +6.4224 |
| 21 | `transactions` | +6.3628 |
| 22 | `the bank` | +6.2777 |
| 23 | `checking account` | +6.0912 |
| 24 | `check` | +6.0352 |
| 25 | `checks` | +5.7787 |

**Credit card or prepaid card**

| # | terme | poids |
|---|---|---|
| 1 | `card` | +23.0560 |
| 2 | `citi` | +9.8244 |
| 3 | `discover` | +9.5581 |
| 4 | `amex` | +9.1992 |
| 5 | `cards` | +8.9981 |
| 6 | `synchrony` | +8.9855 |
| 7 | `rushcard` | +8.8610 |
| 8 | `chase` | +7.9051 |
| 9 | `american express` | +7.8367 |
| 10 | `capital one` | +7.5735 |
| 11 | `interest` | +7.5490 |
| 12 | `rewards` | +7.2846 |
| 13 | `macy` | +7.1824 |
| 14 | `visa` | +7.0626 |
| 15 | `citibank` | +7.0042 |
| 16 | `barclays` | +6.9466 |
| 17 | `credit card` | +6.7505 |
| 18 | `express` | +6.6331 |
| 19 | `barclay` | +6.6134 |
| 20 | `balance transfer` | +6.4867 |
| 21 | `target` | +6.2835 |
| 22 | `the card` | +6.0507 |
| 23 | `mastercard` | +6.0159 |
| 24 | `credit` | +5.7891 |
| 25 | `capital` | +5.7187 |

**Credit reporting**

| # | terme | poids |
|---|---|---|
| 1 | `equifax` | +25.5771 |
| 2 | `experian` | +22.5114 |
| 3 | `transunion` | +20.1074 |
| 4 | `report` | +9.7990 |
| 5 | `inquiries` | +9.4560 |
| 6 | `credit` | +7.6242 |
| 7 | `my report` | +7.0572 |
| 8 | `inquiry` | +6.7952 |
| 9 | `freeze` | +6.7740 |
| 10 | `reporting` | +6.7516 |
| 11 | `trans union` | +6.3495 |
| 12 | `trans` | +6.0776 |
| 13 | `xxxx xxxx` | +5.9612 |
| 14 | `score` | +5.9480 |
| 15 | `inquires` | +5.3839 |
| 16 | `information` | +5.3266 |
| 17 | `incorrect` | +5.3054 |
| 18 | `investigation` | +5.1195 |
| 19 | `free` | +4.7835 |
| 20 | `xxxx is` | +4.6322 |
| 21 | `dispute` | +4.5653 |
| 22 | `reported` | +4.5513 |
| 23 | `breach` | +4.5296 |
| 24 | `late` | +4.4645 |
| 25 | `xxxx` | +4.4009 |

**Debt collection**

| # | terme | poids |
|---|---|---|
| 1 | `debt` | +18.5082 |
| 2 | `collection` | +11.6321 |
| 3 | `collections` | +7.8653 |
| 4 | `owe` | +7.4064 |
| 5 | `recovery` | +6.8573 |
| 6 | `calls` | +6.5451 |
| 7 | `owed` | +6.2974 |
| 8 | `inc` | +5.9610 |
| 9 | `midland` | +5.8406 |
| 10 | `calling` | +5.6710 |
| 11 | `collect` | +5.5597 |
| 12 | `debts` | +5.3720 |
| 13 | `medical` | +5.3156 |
| 14 | `associates` | +5.1374 |
| 15 | `bill` | +5.1193 |
| 16 | `services` | +4.9421 |
| 17 | `to collect` | +4.6579 |
| 18 | `erc` | +4.5896 |
| 19 | `call` | +4.5237 |
| 20 | `this` | +4.4834 |
| 21 | `repeatedly` | +4.3720 |
| 22 | `portfolio` | +4.3664 |
| 23 | `systems` | +4.3494 |
| 24 | `llc` | +4.1383 |
| 25 | `convergent` | +4.1029 |

**Money transfer or virtual currency**

| # | terme | poids |
|---|---|---|
| 1 | `coinbase` | +22.6900 |
| 2 | `paypal` | +18.4922 |
| 3 | `money` | +13.5182 |
| 4 | `transfer` | +12.2118 |
| 5 | `moneygram` | +12.0755 |
| 6 | `wire` | +10.4154 |
| 7 | `western union` | +9.2793 |
| 8 | `western` | +9.0369 |
| 9 | `transaction` | +7.8541 |
| 10 | `support` | +7.7757 |
| 11 | `google` | +7.6633 |
| 12 | `funds` | +6.6745 |
| 13 | `scam` | +6.2384 |
| 14 | `exchange` | +6.0926 |
| 15 | `the money` | +6.0337 |
| 16 | `money transfer` | +5.9828 |
| 17 | `currency` | +5.9533 |
| 18 | `bitcoin` | +5.9439 |
| 19 | `money gram` | +5.7640 |
| 20 | `gram` | +5.7028 |
| 21 | `scammed` | +5.3746 |
| 22 | `wallet` | +5.2520 |
| 23 | `transfers` | +5.2484 |
| 24 | `usd` | +5.2440 |
| 25 | `app` | +5.2299 |

**Mortgage**

| # | terme | poids |
|---|---|---|
| 1 | `mortgage` | +30.9313 |
| 2 | `escrow` | +14.1824 |
| 3 | `modification` | +12.6274 |
| 4 | `home` | +12.0760 |
| 5 | `ocwen` | +11.8501 |
| 6 | `ditech` | +10.3913 |
| 7 | `foreclosure` | +9.8853 |
| 8 | `loan` | +9.8173 |
| 9 | `appraisal` | +8.8853 |
| 10 | `nationstar` | +8.5782 |
| 11 | `property` | +8.4330 |
| 12 | `my mortgage` | +8.1905 |
| 13 | `house` | +7.8103 |
| 14 | `closing` | +7.3277 |
| 15 | `heloc` | +7.1840 |
| 16 | `servicing` | +7.1200 |
| 17 | `refinance` | +6.7971 |
| 18 | `seterus` | +6.6722 |
| 19 | `lender` | +6.6025 |
| 20 | `pmi` | +6.3136 |
| 21 | `quicken` | +6.1548 |
| 22 | `loancare` | +5.9693 |
| 23 | `my escrow` | +5.8532 |
| 24 | `taxes` | +5.7259 |
| 25 | `citimortgage` | +5.6761 |

**Payday, title or personal loan**

| # | terme | poids |
|---|---|---|
| 1 | `loan` | +20.9833 |
| 2 | `payday` | +11.1923 |
| 3 | `borrowed` | +9.7760 |
| 4 | `cash` | +9.3261 |
| 5 | `personal loan` | +8.3032 |
| 6 | `line of` | +8.0818 |
| 7 | `title` | +7.7529 |
| 8 | `lending` | +7.5981 |
| 9 | `interest` | +7.0409 |
| 10 | `title loan` | +6.8014 |
| 11 | `of credit` | +6.6558 |
| 12 | `00 loan` | +6.5426 |
| 13 | `finance` | +6.2555 |
| 14 | `ach` | +6.0946 |
| 15 | `my bank` | +5.9168 |
| 16 | `line` | +5.8337 |
| 17 | `ace` | +5.7575 |
| 18 | `mobiloans` | +5.7553 |
| 19 | `payday loans` | +5.7224 |
| 20 | `speedy cash` | +5.5928 |
| 21 | `lender` | +5.5209 |
| 22 | `cashnet` | +5.4276 |
| 23 | `advance america` | +5.3674 |
| 24 | `rise` | +5.3609 |
| 25 | `avant` | +5.3528 |

**Student loan**

| # | terme | poids |
|---|---|---|
| 1 | `navient` | +26.4553 |
| 2 | `loans` | +14.6272 |
| 3 | `student` | +13.0176 |
| 4 | `loan` | +12.8583 |
| 5 | `nelnet` | +11.6303 |
| 6 | `school` | +10.0015 |
| 7 | `aes` | +9.7684 |
| 8 | `education` | +9.2085 |
| 9 | `forbearance` | +9.1548 |
| 10 | `fedloan` | +9.0348 |
| 11 | `repayment` | +8.8989 |
| 12 | `acs` | +8.3459 |
| 13 | `student loan` | +8.0849 |
| 14 | `payments` | +7.7603 |
| 15 | `my loans` | +7.5917 |
| 16 | `private` | +7.5707 |
| 17 | `interest` | +6.3613 |
| 18 | `forgiveness` | +6.0482 |
| 19 | `cosigner` | +5.9990 |
| 20 | `college` | +5.8649 |
| 21 | `payment` | +5.7723 |
| 22 | `my loan` | +5.7550 |
| 23 | `ibr` | +5.5138 |
| 24 | `income` | +5.4282 |
| 25 | `sallie` | +5.4158 |

**Vehicle loan or lease**

| # | terme | poids |
|---|---|---|
| 1 | `car` | +18.9033 |
| 2 | `vehicle` | +16.5338 |
| 3 | `auto` | +11.6480 |
| 4 | `santander` | +11.6205 |
| 5 | `lease` | +10.7278 |
| 6 | `ally` | +10.2382 |
| 7 | `auto loan` | +9.2754 |
| 8 | `acceptance` | +8.7222 |
| 9 | `gm` | +8.5404 |
| 10 | `finance` | +8.0431 |
| 11 | `dealer` | +7.6693 |
| 12 | `toyota` | +7.5536 |
| 13 | `loan` | +7.5243 |
| 14 | `payment` | +7.0366 |
| 15 | `financial` | +7.0260 |
| 16 | `payments` | +6.8082 |
| 17 | `repossessed` | +6.5473 |
| 18 | `honda` | +6.2493 |
| 19 | `title` | +6.1933 |
| 20 | `repossession` | +5.8897 |
| 21 | `credit acceptance` | +5.7983 |
| 22 | `my car` | +5.6140 |
| 23 | `payoff` | +5.5744 |
| 24 | `repo` | +5.5670 |
| 25 | `gm financial` | +5.5444 |


---

## MultinomialNB (vect. retenu)

Classement : log P(t|c) recentre sur la moyenne des classes.

**Bank account or service**

| # | terme | poids |
|---|---|---|
| 1 | `the bonus` | +3.7432 |
| 2 | `atm` | +3.6971 |
| 3 | `bonus` | +3.6080 |
| 4 | `00 bonus` | +3.5506 |
| 5 | `citigold` | +3.4667 |
| 6 | `the atm` | +3.3549 |
| 7 | `overdraft fees` | +3.1575 |
| 8 | `overdraft fee` | +3.1507 |
| 9 | `overdraft` | +3.1296 |
| 10 | `scottrade` | +3.0658 |
| 11 | `the promotion` | +3.0647 |
| 12 | `overdraft protection` | +3.0526 |
| 13 | `citigold checking` | +2.9936 |
| 14 | `promotion` | +2.9786 |
| 15 | `an atm` | +2.9709 |
| 16 | `xxxx overdraft` | +2.9092 |
| 17 | `deposits` | +2.8280 |
| 18 | `opened checking` | +2.8219 |
| 19 | `the overdraft` | +2.8216 |
| 20 | `new checking` | +2.8126 |
| 21 | `direct deposits` | +2.8104 |
| 22 | `savings account` | +2.7919 |
| 23 | `00 overdraft` | +2.7683 |
| 24 | `deposited check` | +2.7669 |
| 25 | `overdrafts` | +2.7663 |

**Credit card or prepaid card**

| # | terme | poids |
|---|---|---|
| 1 | `american express` | +3.6559 |
| 2 | `rewards` | +3.5579 |
| 3 | `amex` | +3.5034 |
| 4 | `macy` | +3.4000 |
| 5 | `annual fee` | +3.3376 |
| 6 | `the card` | +3.3364 |
| 7 | `this card` | +3.2879 |
| 8 | `balance transfer` | +3.1790 |
| 9 | `discover card` | +3.0603 |
| 10 | `new card` | +3.0017 |
| 11 | `best buy` | +3.0000 |
| 12 | `synchrony` | +2.9925 |
| 13 | `barclay` | +2.9295 |
| 14 | `synchrony bank` | +2.9176 |
| 15 | `barclaycard` | +2.9007 |
| 16 | `the merchant` | +2.8698 |
| 17 | `sears` | +2.8649 |
| 18 | `mastercard` | +2.8581 |
| 19 | `bonus` | +2.8428 |
| 20 | `card account` | +2.8127 |
| 21 | `card company` | +2.8047 |
| 22 | `rushcard` | +2.7997 |
| 23 | `barclays` | +2.7789 |
| 24 | `citi card` | +2.7736 |
| 25 | `platinum` | +2.7735 |

**Credit reporting**

| # | terme | poids |
|---|---|---|
| 1 | `equifax` | +4.6355 |
| 2 | `transunion` | +4.4462 |
| 3 | `experian` | +4.2609 |
| 4 | `trans union` | +3.7180 |
| 5 | `equifax and` | +3.6831 |
| 6 | `to equifax` | +3.6238 |
| 7 | `with equifax` | +3.6034 |
| 8 | `equifax is` | +3.5845 |
| 9 | `xxxx equifax` | +3.5648 |
| 10 | `equifax has` | +3.5602 |
| 11 | `equifax xxxx` | +3.4796 |
| 12 | `transunion and` | +3.4629 |
| 13 | `and equifax` | +3.4519 |
| 14 | `to experian` | +3.4260 |
| 15 | `experian and` | +3.4182 |
| 16 | `xxxx experian` | +3.4103 |
| 17 | `and experian` | +3.4081 |
| 18 | `from equifax` | +3.4042 |
| 19 | `my equifax` | +3.3970 |
| 20 | `inquiries that` | +3.3949 |
| 21 | `and transunion` | +3.3767 |
| 22 | `experian is` | +3.3614 |
| 23 | `the equifax` | +3.3606 |
| 24 | `experian xxxx` | +3.3332 |
| 25 | `equifax to` | +3.3189 |

**Debt collection**

| # | terme | poids |
|---|---|---|
| 1 | `portfolio recovery` | +3.5722 |
| 2 | `diversified` | +3.1581 |
| 3 | `this collection` | +3.1366 |
| 4 | `midland` | +3.1345 |
| 5 | `recovery associates` | +3.0620 |
| 6 | `erc` | +3.0611 |
| 7 | `diversified consultants` | +3.0340 |
| 8 | `afni` | +3.0105 |
| 9 | `about debt` | +2.9861 |
| 10 | `convergent` | +2.9789 |
| 11 | `consultants` | +2.9384 |
| 12 | `this debt` | +2.9361 |
| 13 | `credit systems` | +2.9060 |
| 14 | `enhanced recovery` | +2.8953 |
| 15 | `collect debt` | +2.8716 |
| 16 | `original creditor` | +2.8631 |
| 17 | `midland funding` | +2.8101 |
| 18 | `transworld` | +2.8095 |
| 19 | `debt that` | +2.7854 |
| 20 | `collection account` | +2.7834 |
| 21 | `debt is` | +2.7666 |
| 22 | `debt from` | +2.7243 |
| 23 | `portfolio` | +2.7060 |
| 24 | `medical bill` | +2.7050 |
| 25 | `alleged debt` | +2.6887 |

**Money transfer or virtual currency**

| # | terme | poids |
|---|---|---|
| 1 | `coinbase` | +5.8628 |
| 2 | `western union` | +4.8044 |
| 3 | `to coinbase` | +4.6354 |
| 4 | `my coinbase` | +4.6352 |
| 5 | `coinbase account` | +4.6344 |
| 6 | `moneygram` | +4.4466 |
| 7 | `bitcoin` | +4.3896 |
| 8 | `from coinbase` | +4.3881 |
| 9 | `western` | +4.2910 |
| 10 | `with coinbase` | +4.2198 |
| 11 | `coinbase com` | +4.1381 |
| 12 | `paypal` | +4.0192 |
| 13 | `on coinbase` | +3.9989 |
| 14 | `coinbase support` | +3.9815 |
| 15 | `money transfer` | +3.9466 |
| 16 | `coinbase on` | +3.9345 |
| 17 | `the coinbase` | +3.9047 |
| 18 | `money gram` | +3.8789 |
| 19 | `the wire` | +3.8454 |
| 20 | `coinbase and` | +3.8322 |
| 21 | `xxxx coinbase` | +3.8102 |
| 22 | `gram` | +3.7667 |
| 23 | `wire transfer` | +3.7473 |
| 24 | `wu` | +3.7377 |
| 25 | `btc` | +3.7283 |

**Mortgage**

| # | terme | poids |
|---|---|---|
| 1 | `ocwen` | +4.2059 |
| 2 | `nationstar` | +4.0883 |
| 3 | `escrow` | +4.0528 |
| 4 | `modification` | +4.0324 |
| 5 | `my escrow` | +3.9630 |
| 6 | `pmi` | +3.9549 |
| 7 | `loan modification` | +3.9351 |
| 8 | `foreclosure` | +3.8368 |
| 9 | `short sale` | +3.8346 |
| 10 | `appraisal` | +3.8080 |
| 11 | `escrow account` | +3.7883 |
| 12 | `ditech` | +3.6584 |
| 13 | `the modification` | +3.5836 |
| 14 | `sale date` | +3.5599 |
| 15 | `the foreclosure` | +3.5550 |
| 16 | `seterus` | +3.5394 |
| 17 | `for modification` | +3.5307 |
| 18 | `hamp` | +3.5300 |
| 19 | `nationstar mortgage` | +3.5287 |
| 20 | `the escrow` | +3.5133 |
| 21 | `our mortgage` | +3.5044 |
| 22 | `mortgage was` | +3.4797 |
| 23 | `foreclose` | +3.4654 |
| 24 | `sls` | +3.4568 |
| 25 | `the mortgage` | +3.4563 |

**Payday, title or personal loan**

| # | terme | poids |
|---|---|---|
| 1 | `payday loan` | +4.2151 |
| 2 | `big picture` | +4.0809 |
| 3 | `picture loans` | +4.0470 |
| 4 | `payday` | +3.9479 |
| 5 | `speedy cash` | +3.8148 |
| 6 | `payday loans` | +3.7215 |
| 7 | `for payday` | +3.6213 |
| 8 | `speedy` | +3.5416 |
| 9 | `title loan` | +3.4606 |
| 10 | `tribal` | +3.4259 |
| 11 | `cash express` | +3.3473 |
| 12 | `ace cash` | +3.3459 |
| 13 | `ace` | +3.3150 |
| 14 | `00 loan` | +3.2966 |
| 15 | `day loan` | +3.2665 |
| 16 | `cashnetusa` | +3.2414 |
| 17 | `cash central` | +3.2364 |
| 18 | `out payday` | +3.1920 |
| 19 | `cashnet` | +3.1870 |
| 20 | `pay day` | +3.1850 |
| 21 | `castle` | +3.1541 |
| 22 | `advance america` | +3.1522 |
| 23 | `castle payday` | +3.0785 |
| 24 | `borrowed xxxx` | +3.0552 |
| 25 | `from big` | +3.0464 |

**Student loan**

| # | terme | poids |
|---|---|---|
| 1 | `navient` | +5.1134 |
| 2 | `my loans` | +4.2164 |
| 3 | `income based` | +4.1995 |
| 4 | `with navient` | +4.1863 |
| 5 | `based repayment` | +4.1171 |
| 6 | `to navient` | +4.0884 |
| 7 | `fedloan` | +4.0700 |
| 8 | `sallie` | +4.0136 |
| 9 | `sallie mae` | +4.0082 |
| 10 | `private loans` | +4.0040 |
| 11 | `forbearance` | +3.9997 |
| 12 | `pslf` | +3.9954 |
| 13 | `loan forgiveness` | +3.9946 |
| 14 | `navient has` | +3.9741 |
| 15 | `navient and` | +3.9712 |
| 16 | `ibr` | +3.9697 |
| 17 | `public service` | +3.9471 |
| 18 | `nelnet` | +3.9440 |
| 19 | `private student` | +3.9087 |
| 20 | `from navient` | +3.8788 |
| 21 | `aes` | +3.8784 |
| 22 | `navient to` | +3.8700 |
| 23 | `called navient` | +3.8606 |
| 24 | `student loans` | +3.8509 |
| 25 | `mae` | +3.8419 |

**Vehicle loan or lease**

| # | terme | poids |
|---|---|---|
| 1 | `the dealership` | +3.9254 |
| 2 | `gm financial` | +3.9248 |
| 3 | `gm` | +3.9181 |
| 4 | `the vehicle` | +3.8665 |
| 5 | `dealership` | +3.8647 |
| 6 | `the car` | +3.7723 |
| 7 | `honda` | +3.7408 |
| 8 | `toyota` | +3.7250 |
| 9 | `dealer` | +3.7073 |
| 10 | `the dealer` | +3.6862 |
| 11 | `repossessed` | +3.6700 |
| 12 | `repossession` | +3.6192 |
| 13 | `auto finance` | +3.6085 |
| 14 | `gap insurance` | +3.6020 |
| 15 | `credit acceptance` | +3.5869 |
| 16 | `vehicle was` | +3.5719 |
| 17 | `car was` | +3.5436 |
| 18 | `vehicle` | +3.5189 |
| 19 | `santander consumer` | +3.5111 |
| 20 | `repo` | +3.4601 |
| 21 | `consumer usa` | +3.4523 |
| 22 | `this vehicle` | +3.4487 |
| 23 | `this car` | +3.4385 |
| 24 | `car back` | +3.4262 |
| 25 | `one auto` | +3.4023 |


---

## MultinomialNB (vect. propre)

Classement : log P(t|c) recentre sur la moyenne des classes.

**Bank account or service**

| # | terme | poids |
|---|---|---|
| 1 | `the bonus` | +3.7950 |
| 2 | `atm` | +3.7239 |
| 3 | `bonus` | +3.6130 |
| 4 | `00 bonus` | +3.6049 |
| 5 | `citigold` | +3.5228 |
| 6 | `the atm` | +3.4037 |
| 7 | `scottrade` | +3.1946 |
| 8 | `overdraft fee` | +3.1715 |
| 9 | `overdraft fees` | +3.1650 |
| 10 | `overdraft` | +3.1265 |
| 11 | `overdraft protection` | +3.0874 |
| 12 | `the promotion` | +3.0823 |
| 13 | `citigold checking` | +3.0485 |
| 14 | `an atm` | +3.0280 |
| 15 | `promotion` | +2.9713 |
| 16 | `xxxx overdraft` | +2.9642 |
| 17 | `direct deposits` | +2.8472 |
| 18 | `new checking` | +2.8453 |
| 19 | `opened checking` | +2.8448 |
| 20 | `the overdraft` | +2.8448 |
| 21 | `deposits` | +2.8285 |
| 22 | `overdrafts` | +2.8159 |
| 23 | `00 overdraft` | +2.8067 |
| 24 | `account opening` | +2.8004 |
| 25 | `deposited check` | +2.7961 |

**Credit card or prepaid card**

| # | terme | poids |
|---|---|---|
| 1 | `american express` | +3.6919 |
| 2 | `rewards` | +3.6086 |
| 3 | `amex` | +3.5540 |
| 4 | `macy` | +3.4563 |
| 5 | `annual fee` | +3.3839 |
| 6 | `the card` | +3.3488 |
| 7 | `this card` | +3.3249 |
| 8 | `balance transfer` | +3.2183 |
| 9 | `discover card` | +3.1146 |
| 10 | `best buy` | +3.0618 |
| 11 | `new card` | +3.0330 |
| 12 | `synchrony` | +3.0188 |
| 13 | `barclaycard` | +2.9866 |
| 14 | `barclay` | +2.9823 |
| 15 | `synchrony bank` | +2.9573 |
| 16 | `sears` | +2.9196 |
| 17 | `mastercard` | +2.9086 |
| 18 | `the merchant` | +2.9076 |
| 19 | `rushcard` | +2.8813 |
| 20 | `bonus` | +2.8751 |
| 21 | `balance transfers` | +2.8512 |
| 22 | `citi card` | +2.8450 |
| 23 | `card account` | +2.8296 |
| 24 | `barclays` | +2.8243 |
| 25 | `card company` | +2.8193 |

**Credit reporting**

| # | terme | poids |
|---|---|---|
| 1 | `equifax` | +4.7396 |
| 2 | `transunion` | +4.5522 |
| 3 | `experian` | +4.3565 |
| 4 | `trans union` | +3.8556 |
| 5 | `equifax and` | +3.8151 |
| 6 | `to equifax` | +3.7456 |
| 7 | `with equifax` | +3.7351 |
| 8 | `equifax is` | +3.7140 |
| 9 | `equifax has` | +3.6971 |
| 10 | `xxxx equifax` | +3.6966 |
| 11 | `equifax xxxx` | +3.6042 |
| 12 | `transunion and` | +3.5970 |
| 13 | `and equifax` | +3.5847 |
| 14 | `to experian` | +3.5561 |
| 15 | `experian and` | +3.5420 |
| 16 | `from equifax` | +3.5415 |
| 17 | `xxxx experian` | +3.5391 |
| 18 | `and experian` | +3.5372 |
| 19 | `the equifax` | +3.5109 |
| 20 | `and transunion` | +3.5102 |
| 21 | `my equifax` | +3.5099 |
| 22 | `inquiries that` | +3.4957 |
| 23 | `experian is` | +3.4875 |
| 24 | `experian xxxx` | +3.4638 |
| 25 | `equifax to` | +3.4553 |

**Debt collection**

| # | terme | poids |
|---|---|---|
| 1 | `portfolio recovery` | +3.6857 |
| 2 | `diversified` | +3.2607 |
| 3 | `midland` | +3.2400 |
| 4 | `this collection` | +3.2257 |
| 5 | `erc` | +3.1854 |
| 6 | `recovery associates` | +3.1756 |
| 7 | `diversified consultants` | +3.1495 |
| 8 | `afni` | +3.1344 |
| 9 | `convergent` | +3.1000 |
| 10 | `about debt` | +3.0951 |
| 11 | `consultants` | +3.0403 |
| 12 | `credit systems` | +3.0290 |
| 13 | `enhanced recovery` | +3.0056 |
| 14 | `this debt` | +2.9914 |
| 15 | `original creditor` | +2.9461 |
| 16 | `collect debt` | +2.9355 |
| 17 | `midland funding` | +2.9301 |
| 18 | `transworld` | +2.9261 |
| 19 | `collection account` | +2.8569 |
| 20 | `debt that` | +2.8427 |
| 21 | `debt is` | +2.8267 |
| 22 | `medical bill` | +2.8061 |
| 23 | `midland credit` | +2.7973 |
| 24 | `ic` | +2.7961 |
| 25 | `debt from` | +2.7939 |

**Money transfer or virtual currency**

| # | terme | poids |
|---|---|---|
| 1 | `coinbase` | +5.9631 |
| 2 | `western union` | +4.8872 |
| 3 | `to coinbase` | +4.7432 |
| 4 | `coinbase account` | +4.7423 |
| 5 | `my coinbase` | +4.7419 |
| 6 | `moneygram` | +4.5543 |
| 7 | `bitcoin` | +4.4999 |
| 8 | `from coinbase` | +4.4935 |
| 9 | `western` | +4.3379 |
| 10 | `with coinbase` | +4.3293 |
| 11 | `coinbase com` | +4.2517 |
| 12 | `on coinbase` | +4.1126 |
| 13 | `coinbase support` | +4.0993 |
| 14 | `paypal` | +4.0741 |
| 15 | `coinbase on` | +4.0336 |
| 16 | `the coinbase` | +4.0275 |
| 17 | `money transfer` | +4.0227 |
| 18 | `money gram` | +3.9616 |
| 19 | `coinbase and` | +3.9437 |
| 20 | `xxxx coinbase` | +3.9272 |
| 21 | `the wire` | +3.9054 |
| 22 | `btc` | +3.8654 |
| 23 | `wu` | +3.8511 |
| 24 | `gram` | +3.8429 |
| 25 | `by coinbase` | +3.8095 |

**Mortgage**

| # | terme | poids |
|---|---|---|
| 1 | `ocwen` | +4.2124 |
| 2 | `nationstar` | +4.0977 |
| 3 | `escrow` | +4.0243 |
| 4 | `modification` | +3.9944 |
| 5 | `pmi` | +3.9849 |
| 6 | `my escrow` | +3.9768 |
| 7 | `loan modification` | +3.9146 |
| 8 | `short sale` | +3.8425 |
| 9 | `appraisal` | +3.8077 |
| 10 | `foreclosure` | +3.7997 |
| 11 | `escrow account` | +3.7874 |
| 12 | `ditech` | +3.6546 |
| 13 | `the modification` | +3.5811 |
| 14 | `the foreclosure` | +3.5709 |
| 15 | `sale date` | +3.5708 |
| 16 | `hamp` | +3.5654 |
| 17 | `seterus` | +3.5623 |
| 18 | `for modification` | +3.5502 |
| 19 | `nationstar mortgage` | +3.5478 |
| 20 | `the escrow` | +3.5120 |
| 21 | `our mortgage` | +3.4882 |
| 22 | `sls` | +3.4865 |
| 23 | `foreclose` | +3.4733 |
| 24 | `mortgage was` | +3.4721 |
| 25 | `sps` | +3.4709 |

**Payday, title or personal loan**

| # | terme | poids |
|---|---|---|
| 1 | `payday loan` | +4.2807 |
| 2 | `big picture` | +4.1685 |
| 3 | `picture loans` | +4.1509 |
| 4 | `payday` | +3.9866 |
| 5 | `speedy cash` | +3.9087 |
| 6 | `payday loans` | +3.8119 |
| 7 | `for payday` | +3.7090 |
| 8 | `speedy` | +3.6089 |
| 9 | `title loan` | +3.5547 |
| 10 | `tribal` | +3.5513 |
| 11 | `cash express` | +3.4308 |
| 12 | `ace cash` | +3.4255 |
| 13 | `ace` | +3.3788 |
| 14 | `cashnetusa` | +3.3688 |
| 15 | `00 loan` | +3.3444 |
| 16 | `day loan` | +3.3316 |
| 17 | `cash central` | +3.3295 |
| 18 | `cashnet` | +3.2973 |
| 19 | `out payday` | +3.2870 |
| 20 | `castle` | +3.2497 |
| 21 | `advance america` | +3.2481 |
| 22 | `pay day` | +3.2392 |
| 23 | `castle payday` | +3.1849 |
| 24 | `mobiloans` | +3.1611 |
| 25 | `borrowed xxxx` | +3.1438 |

**Student loan**

| # | terme | poids |
|---|---|---|
| 1 | `navient` | +5.1450 |
| 2 | `income based` | +4.2416 |
| 3 | `with navient` | +4.2307 |
| 4 | `my loans` | +4.2280 |
| 5 | `based repayment` | +4.1757 |
| 6 | `fedloan` | +4.1397 |
| 7 | `to navient` | +4.1311 |
| 8 | `pslf` | +4.0723 |
| 9 | `private loans` | +4.0670 |
| 10 | `sallie` | +4.0656 |
| 11 | `sallie mae` | +4.0602 |
| 12 | `ibr` | +4.0365 |
| 13 | `navient has` | +4.0320 |
| 14 | `loan forgiveness` | +4.0303 |
| 15 | `navient and` | +4.0205 |
| 16 | `forbearance` | +4.0122 |
| 17 | `public service` | +4.0110 |
| 18 | `nelnet` | +4.0016 |
| 19 | `private student` | +3.9592 |
| 20 | `aes` | +3.9315 |
| 21 | `from navient` | +3.9249 |
| 22 | `navient to` | +3.9234 |
| 23 | `called navient` | +3.9115 |
| 24 | `income driven` | +3.8840 |
| 25 | `mae` | +3.8814 |

**Vehicle loan or lease**

| # | terme | poids |
|---|---|---|
| 1 | `gm financial` | +3.9468 |
| 2 | `the dealership` | +3.9380 |
| 3 | `gm` | +3.9278 |
| 4 | `dealership` | +3.8621 |
| 5 | `the vehicle` | +3.8544 |
| 6 | `honda` | +3.7810 |
| 7 | `the car` | +3.7493 |
| 8 | `toyota` | +3.7473 |
| 9 | `dealer` | +3.7003 |
| 10 | `the dealer` | +3.6996 |
| 11 | `repossessed` | +3.6687 |
| 12 | `auto finance` | +3.6340 |
| 13 | `repossession` | +3.6290 |
| 14 | `gap insurance` | +3.6254 |
| 15 | `credit acceptance` | +3.6159 |
| 16 | `vehicle was` | +3.5779 |
| 17 | `santander consumer` | +3.5356 |
| 18 | `car was` | +3.5319 |
| 19 | `vehicle` | +3.4880 |
| 20 | `this vehicle` | +3.4809 |
| 21 | `consumer usa` | +3.4767 |
| 22 | `repo` | +3.4717 |
| 23 | `this car` | +3.4596 |
| 24 | `car back` | +3.4442 |
| 25 | `one auto` | +3.4187 |

