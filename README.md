# Path2Win: Interpretable Pregame Win Prediction for Professional League of Legends

This repository contains the code, data, and output summaries for **Path2Win**, an interpretable pregame win-probability model for professional League of Legends matches from the 2024 LPL Summer Season.

The goal of this project is to predict, before a match begins, the probability that the **blue side wins**, using only information available before the match. In addition to predicting the outcome, Path2Win is designed to explain the predicted route to victory through interpretable win-condition indices.

---

## Project objective

Professional League of Legends matches depend on many interacting factors, including lane pressure, role-specific player strength, objective control, teamfighting, jungle resource control, and patch-specific game dynamics.

Rather than building only a black-box classifier, this project formulates pregame prediction as an interpretable pathway problem:

```text
Pregame information
→ predicted win-condition advantages
→ interpretable condition indices
→ blue-side win probability
````

The final model estimates:

```text
P(blue side wins before the match starts)
```

and explains the prediction through four indices:

```text
early-lane control
objective control
fight control
resource control
```

---

## Repository structure

```text
Path2Win-LPL-Pregame-Win-Prediction/
├── README.md
├── data/
│   └── LPL_SummerSeason_2024_Player.csv
├── scripts/
│   ├── model8_final_FULL_END_TO_END_m1_m2_rwts_indices_logistic_v2.py
│   ├── model9_patch_aware_win_condition_compare.py
│   └── model10_hardcoded_patch_win_conditions_compare.py
├── outputs/
│   ├── final_model/
│   │   ├── features_A_original_model2_rebuilt.csv
│   │   ├── full_end_to_end_cv_summary.csv
│   │   ├── full_end_to_end_fold_results.csv
│   │   ├── full_end_to_end_test_predictions.csv
│   │   ├── full_end_to_end_integrated_features.csv
│   │   ├── full_end_to_end_feature_lists.json
│   │   ├── full_end_to_end_condition_prediction_summary.csv
│   │   ├── full_end_to_end_condition_index_values.csv
│   │   ├── full_end_to_end_logistic_coefficient_summary.csv
│   │   └── full_end_to_end_rwts_player_ratings.csv
│   ├── patch_aware_extension/
│   │   └── model9_cv_summary.csv
│   └── hardcoded_condition_extension/
│       └── model10_cv_summary.csv
└── report/
    └── path2win_final_report.tex
```

The exact structure may vary slightly depending on which output files are included, but the main reproducibility files are the raw CSV, the final end-to-end script, and the final output summaries.

---

## Data

The primary input file is:

```text
data/LPL_SummerSeason_2024_Player.csv
```

The dataset contains player-level match records from the 2024 LPL Summer Season.

The raw unit of observation is a **player-match row**. A standard match contains ten rows:

```text
5 blue-side players
5 red-side players
```

The final prediction unit is the **match**. The pipeline converts player-level rows into match-level blue-minus-red features.

Main dataset summary:

```text
Player-level rows: 2820
Unique canonical matches: 283
Final integrated match rows: 282
Date range: 2024-06-01 to 2024-07-31
Roles: TOP, JUNGLE, MID, ADCARRY, SUPPORT
```

Patch distribution:

```text
v14.10: 107 matches
v14.11: 23 matches
v14.13: 103 matches
v14.14: 49 matches
```

---

## Model overview

Path2Win contains four major modeling components.

### 1. Expected-performance branch

This branch builds leakage-safe historical player features and predicts role-specific expected performance before the match.

Historical feature families include:

```text
PR  = player-role history
PCR = player-role-champion history
CR  = champion-role history
RG  = role-global history
```

For each role, four performance targets were selected using validation standardized mean absolute error (zMAE).

Selected targets:

| Role    | Selected targets          |
| ------- | ------------------------- |
| TOP     | CSM, DMG%, GPM, DPM       |
| JUNGLE  | CSM, KP%, GOLD%, Kills    |
| MID     | KP%, DMG%, Assists, GD@15 |
| ADCARRY | CSM, GOLD%, DMG%, DPM     |
| SUPPORT | VSPM, VS%, WPM, GOLD%     |

These targets are not claimed to be the only measures of player strength. They are domain-relevant targets that were most reliably predictable from pregame history.

This branch produces:

```text
60 expected-performance features
```

---

### 2. Role-weighted rating branch

This branch constructs chronological TrueSkill-style player-role ratings.

Each player-role pair has:

```text
mu: estimated strength
sigma: uncertainty
conservative rating: mu - 3 sigma
```

Ratings are recorded before the current match result is used for updating, so the features remain pregame-safe.

Role-weighted team strength is computed using fixed role weights:

| Role    | Weight |
| ------- | -----: |
| TOP     |   1.00 |
| JUNGLE  |   1.15 |
| MID     |   1.15 |
| ADCARRY |   1.10 |
| SUPPORT |   0.90 |

This branch produces:

```text
83 role-weighted rating features
```

---

### 3. Win-condition prediction layer

The two pregame branches are combined:

```text
60 expected-performance features
+ 83 role-weighted rating features
= 143 pregame features
```

These 143 pregame features are used to predict realized match-level win-condition advantages.

The primary 12 win-condition targets are:

```text
actual_delta_GD15_sum
actual_delta_KP15_mean
actual_delta_Tower15
actual_delta_FirstTower
actual_delta_FirstDragon
actual_delta_NumberDragon
actual_delta_NumberVB
actual_delta_Herald
actual_delta_TeamJGShare
actual_delta_MiniTF_count
actual_delta_MiniTF_WR
actual_delta_TeamFight_WR
```

Each condition is constructed as:

```text
blue value - red value
```

Important leakage-control point:

```text
Actual win conditions are used only as intermediate training labels.
The final prediction uses predicted win conditions, not actual realized win conditions.
```

---

### 4. Final four-index logistic model

The predicted win conditions are grouped into four interpretable indices:

```text
early-lane control
objective control
fight control
resource control
```

The final logistic model maps these four indices to blue-side win probability.

The final model is intentionally simple and interpretable:

```text
4 final features → P(blue win)
```

---

## Main results

The final four-index Path2Win model achieved:

| Metric                                 |   Mean |
| -------------------------------------- | -----: |
| Logloss                                | 0.6658 |
| Brier score                            | 0.2364 |
| AUC                                    | 0.6197 |
| Accuracy at validation-tuned threshold | 0.6012 |
| Accuracy at threshold 0.5              | 0.6250 |

Final logistic coefficient summary:

| Feature                 | Mean coefficient |
| ----------------------- | ---------------: |
| Intercept               |            0.174 |
| Resource-control index  |            0.127 |
| Early-lane index        |            0.122 |
| Objective-control index |            0.106 |
| Fight-control index     |            0.105 |

All four condition-index coefficients are positive, which is directionally consistent with the game: predicted advantages in early lane, objectives, fights, and resources all increase blue-side win probability.

---

## Alternative model comparisons

Several alternative and robustness models were tested.

Representative comparison:

| Model                                           | Final features | Logloss | Brier |   AUC | Acc@0.5 |
| ----------------------------------------------- | -------------: | ------: | ----: | ----: | ------: |
| Final four-index Path2Win                       |              4 |   0.666 | 0.236 | 0.620 |   0.625 |
| Direct CatBoost on 143 pregame features         |            143 |   0.663 | 0.235 | 0.649 |   0.583 |
| Hard-coded global top-20 + interactions         |            190 |   0.663 | 0.235 | 0.637 |   0.595 |
| Patch-aware pooled top-10                       |           42.5 |   0.687 | 0.246 | 0.584 |   0.583 |
| Patch-aware pooled top-20                       |           69.0 |   0.695 | 0.249 | 0.573 |   0.607 |
| Hard-coded separate patch top-20 + interactions |              — |   0.685 | 0.238 | 0.630 |   0.607 |

The more complex models sometimes slightly improved probability metrics, but they required substantially more features and reduced interpretability. The final four-index Path2Win model was selected because it provides the best balance between interpretability, leakage control, stability, and predictive performance.

---

## Validation design

The model is evaluated using chronological rolling train/validation/test splits.

This design reflects the real use case:

```text
past matches → train
later matches → validation
future matches → test
```

Validation data are used for model selection, including:

```text
target selection
ridge penalty selection
logistic regularization
classification threshold tuning
```

Test data are used only for out-of-sample evaluation.

This chronological setup reduces leakage from future matches and is more realistic than random cross-validation for time-ordered sports data.

---

## Leakage control

The project explicitly separates pregame predictors from realized match outcomes.

The model enforces the following rules:

```text
Historical features for match m use only matches before m.
Role-weighted ratings are recorded before the current match updates the ratings.
Actual win conditions are used only as training labels.
The final win model uses predicted win conditions, not actual win conditions.
Model selection uses validation folds, not test folds.
```

This distinguishes Path2Win from a postgame oracle model that directly uses in-game outcomes such as dragons, towers, kills, or teamfight win rate.

---

## Main scripts

### Final Path2Win pipeline

```text
scripts/model8_final_FULL_END_TO_END_m1_m2_rwts_indices_logistic_v2.py
```

Although the filename contains earlier internal development names, this is the final end-to-end Path2Win script.

It rebuilds the full pipeline from the raw CSV:

```text
raw data
→ expected-performance features
→ role-weighted rating features
→ predicted win conditions
→ four condition indices
→ final logistic evaluation
```

### Patch-aware extension

```text
scripts/model9_patch_aware_win_condition_compare.py
```

Compares patch-aware final layers and predicted-condition interaction models.

### Hard-coded condition extension

```text
scripts/model10_hardcoded_patch_win_conditions_compare.py
```

Tests hard-coded patch-specific and expanded-condition final layers.

---

## How to run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the final end-to-end model:

```bash
python scripts/model8_final_FULL_END_TO_END_m1_m2_rwts_indices_logistic_v2.py
```

Run the patch-aware extension:

```bash
python scripts/model9_patch_aware_win_condition_compare.py
```

Run the hard-coded condition extension:

```bash
python scripts/model10_hardcoded_patch_win_conditions_compare.py
```

Depending on the script paths, you may need to run scripts from the repository root or adjust file paths inside the scripts.

---

## Key output files

Final model outputs are stored in:

```text
outputs/final_model/
```

Important files:

```text
features_A_original_model2_rebuilt.csv
full_end_to_end_cv_summary.csv
full_end_to_end_fold_results.csv
full_end_to_end_test_predictions.csv
full_end_to_end_integrated_features.csv
full_end_to_end_feature_lists.json
full_end_to_end_condition_prediction_summary.csv
full_end_to_end_condition_index_values.csv
full_end_to_end_logistic_coefficient_summary.csv
full_end_to_end_rwts_player_ratings.csv
```

Extension summaries:

```text
outputs/patch_aware_extension/model9_cv_summary.csv
outputs/hardcoded_condition_extension/model10_cv_summary.csv
```

---

## Report

The final written report is located in:

```text
report/path2win_final_report.tex
```

The report describes:

```text
problem formulation
data source and cleaning
feature engineering
validation design
leakage control
model construction
results
alternative model comparisons
assumptions and limitations
reproducibility
```

---

## Requirements

The main Python packages are:

```text
numpy
pandas
scikit-learn
catboost
scipy
```

These are listed in:

```text
requirements.txt
```

---

## Reproducibility note

The final script does not require prebuilt intermediate feature files. It reconstructs the full Path2Win pipeline directly from the raw player-level CSV.

The output summaries included in this repository are the files used to report the final results.

---

## Project status

This repository was created as the final project submission for a data mining / applied modeling project. The main emphasis is on a well-formulated prediction problem, leakage-safe validation, interpretable feature construction, and reproducible results.

```
```
