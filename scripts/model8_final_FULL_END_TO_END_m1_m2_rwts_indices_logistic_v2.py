#!/usr/bin/env python3
"""
model8_final_FULL_END_TO_END_m1_m2_rwts_indices_logistic.py

FULL END-TO-END FINAL SCRIPT
Only required input:
    LPL_SummerSeason_2024_Player.csv

Final selected model:
    M1 + M2 + role-weighted TrueSkill-style ratings -> condition indices -> logistic win probability

This script is designed to replace the two-step workflow:
    1. Run Model 2 script to create model2_bayes_ts_four_variants_outputs/features_A_original_model2.csv
    2. Run final Model 8 script that reads features_A_original_model2.csv

Instead, this script builds the exact original Model 2 feature block internally:
    raw player CSV
    -> PR/PCR/CR/RG leakage-safe history features using shift(1)
    -> role-specific Layer 1 target selection
    -> Layer 1 ridge predictions
    -> Layer 2 blue-minus-red delta / uncertainty / z features

Then it builds:
    -> Model 1 clean win-condition targets
    -> role-weighted TrueSkill-style pregame features
    -> predicted Model 1 conditions
    -> four interpretable condition indices
    -> final regularized logistic regression

Strict pregame rule:
    - PR/PCR/CR/RG historical means use shift(1).
    - TrueSkill-style ratings are recorded before current-match updates.
    - Actual Model 1 conditions are only used as training targets, never final predictors.
    - Final win model uses only predicted condition indices.

Main outputs:
    model8_final_FULL_END_TO_END_m1_m2_rwts_indices_logistic_outputs/
        features_A_original_model2_rebuilt.csv
        full_end_to_end_integrated_features.csv
        full_end_to_end_feature_lists.json
        full_end_to_end_cv_summary.csv
        full_end_to_end_fold_results.csv
        full_end_to_end_test_predictions.csv
        full_end_to_end_condition_prediction_summary.csv
        full_end_to_end_condition_index_values.csv
        full_end_to_end_logistic_coefficient_summary.csv
        full_end_to_end_rwts_player_ratings.csv
"""

from __future__ import annotations

from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import json
import re
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    log_loss,
    brier_score_loss,
    roc_auc_score,
    accuracy_score,
)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# ============================================================
# 0. CONFIG
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "LPL_SummerSeason_2024_Player.csv"

OUT_DIR = PROJECT_DIR / "model8_final_FULL_END_TO_END_m1_m2_rwts_indices_logistic_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATE_COL = "Date"
DATE_FORMAT_PRIMARY = "%d.%m.%Y"
TEAM_COL = "Team"
OPP_TEAM_COL = "Opponent Team"
GAME_COL = "No Game"
SIDE_COL = "Side"
ROLE_COL = "Role"
PLAYER_COL = "Player"
CHAMP_COL = "Champion"
PATCH_COL = "Patch"
OUTCOME_COL = "Outcome"

# Original Model 2 Layer 1 split/tuning.
TRAIN_FRAC = 0.70
VALID_FRAC = 0.15
K_LABELS = 4
LABEL_SELECTION_ALPHA = 50.0
ALPHA_GRID = [1.0, 10.0, 50.0, 200.0, 1000.0]

ROLE_CANDIDATES = {
    "TOP":     ["GD@15", "CSM", "GPM", "DMG%", "DPM", "Total damage taken", "Deaths", "KP%", "GOLD%", "Assists", "Kills"],
    "JUNGLE":  ["KP%", "DPM", "GPM", "CSM", "Deaths", "Assists", "Kills", "GOLD%", "CS in Enemy Jungle", "CS in Team's Jungle"],
    "MID":     ["GD@15", "CSM", "KP%", "DPM", "DMG%", "GPM", "Deaths", "Assists", "GOLD%", "Kills"],
    "ADCARRY": ["DPM", "DMG%", "GPM", "CSM", "Deaths", "Kills", "Assists", "GOLD%", "KP%"],
    "SUPPORT": ["VSPM", "WPM", "VWPM", "WCPM", "VS%", "KP%", "Deaths", "Assists", "GOLD%", "GPM"],
}

CORE_STATS = sorted(list({
    "Kills", "Deaths", "Assists",
    "CS in Team's Jungle", "CS in Enemy Jungle",
    "CSM", "GPM", "GOLD%",
    "VSPM", "WPM", "VWPM", "WCPM", "VS%",
    "DPM", "DMG%", "KP%", "GD@15", "Total damage taken",
}))

ROLES = ["TOP", "JUNGLE", "MID", "ADCARRY", "SUPPORT"]

# Final Model 8 rolling split.
N_FOLDS = 4
VALID_SIZE = 42
TEST_SIZE = 42
RANDOM_SEED = 2026

# Final condition-prediction ridge models.
RIDGE_ALPHA_GRID = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]

# Final logistic layer.
LOGISTIC_C_GRID = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]

# Role-weighted TrueSkill-style rating constants.
TS_INIT_MU = 25.0
TS_INIT_SIGMA = 25.0 / 3.0
TS_SIGMA_MIN = 1.0
TS_K = 1.25
TS_SIGMA_DECAY = 0.985
TS_SCALE = 10.0

# These are role weights for the final role-weighted TS block.
# They are intentionally simple and fixed for interpretability.
ROLE_WEIGHTS = {
    "TOP": 1.00,
    "JUNGLE": 1.15,
    "MID": 1.15,
    "ADCARRY": 1.10,
    "SUPPORT": 0.90,
}

TUNE_THRESHOLD_FOR_ACCURACY = True


# ============================================================
# 1. BASIC HELPERS
# ============================================================

def parse_mixed_date(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    d = pd.to_datetime(s, format=DATE_FORMAT_PRIMARY, errors="coerce")
    mask = d.isna()
    if mask.any():
        d.loc[mask] = pd.to_datetime(s[mask], format="%Y-%m-%d", errors="coerce")
    mask = d.isna()
    if mask.any():
        d.loc[mask] = pd.to_datetime(s[mask], format="%Y/%m/%d", errors="coerce")
    mask = d.isna()
    if mask.any():
        d.loc[mask] = pd.to_datetime(s[mask], errors="coerce")
    return d


def normalize_side(x) -> str | float:
    if pd.isna(x):
        return np.nan
    s = str(x).strip().upper()
    if s in {"BLUE", "B", "1"}:
        return "BLUE"
    if s in {"RED", "R", "0"}:
        return "RED"
    return s


def outcome_to_binary(x) -> int | float:
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        if x == 1:
            return 1
        if x == 0:
            return 0
    s = str(x).strip().lower()
    if s in {"win", "won", "w", "1", "true"}:
        return 1
    if s in {"loss", "lost", "l", "0", "false"}:
        return 0
    return np.nan


def to_numeric_percent(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    s = series.astype(str).str.replace("%", "", regex=False)
    return pd.to_numeric(s, errors="coerce").astype(float)


def build_match_id(df: pd.DataFrame) -> pd.Series:
    team_min = np.minimum(df[TEAM_COL].astype(str), df[OPP_TEAM_COL].astype(str))
    team_max = np.maximum(df[TEAM_COL].astype(str), df[OPP_TEAM_COL].astype(str))
    nogame = pd.to_numeric(df[GAME_COL], errors="coerce").astype("Int64").astype(str)
    return (
        df[DATE_COL].dt.strftime("%Y-%m-%d")
        + "__" + team_min.astype(str)
        + "__" + team_max.astype(str)
        + "__G" + nogame
    )


def load_and_clean(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required input file: {path}")

    df = pd.read_csv(path)
    required = [DATE_COL, TEAM_COL, OPP_TEAM_COL, GAME_COL, SIDE_COL, ROLE_COL, PLAYER_COL, CHAMP_COL, PATCH_COL, OUTCOME_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df[DATE_COL] = parse_mixed_date(df[DATE_COL])
    for col in [TEAM_COL, OPP_TEAM_COL, SIDE_COL, ROLE_COL, PLAYER_COL, CHAMP_COL, PATCH_COL]:
        df[col] = df[col].astype(str).str.strip()

    role_map = {
        "TOP": "TOP",
        "JUNGLE": "JUNGLE", "JNG": "JUNGLE", "JG": "JUNGLE",
        "MID": "MID",
        "ADCARRY": "ADCARRY", "ADC": "ADCARRY", "BOT": "ADCARRY", "BOTTOM": "ADCARRY",
        "SUPPORT": "SUPPORT", "SUP": "SUPPORT", "SUPP": "SUPPORT",
    }
    df[ROLE_COL] = df[ROLE_COL].astype(str).str.upper().map(role_map).fillna(df[ROLE_COL].astype(str).str.upper())
    df[SIDE_COL] = df[SIDE_COL].apply(normalize_side)
    df["win_binary"] = df[OUTCOME_COL].map(outcome_to_binary)

    df["match_id"] = build_match_id(df)
    df["canonical_match_key"] = df["match_id"]

    for col in ["KP%", "DMG%", "GOLD%", "VS%"]:
        if col in df.columns:
            df[col] = to_numeric_percent(df[col])

    for col in CORE_STATS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[DATE_COL, "match_id", PLAYER_COL, ROLE_COL, CHAMP_COL, PATCH_COL, SIDE_COL, "win_binary"])
    df = df[df[ROLE_COL].isin(ROLE_CANDIDATES.keys())].copy()
    df = df.sort_values([DATE_COL, "match_id", SIDE_COL, ROLE_COL]).reset_index(drop=True)
    df["win_binary"] = df["win_binary"].astype(int)

    print(f"Loaded rows: {len(df)}")
    print(f"Unique matches: {df['match_id'].nunique()}")
    print(f"Date range: {df[DATE_COL].min()} to {df[DATE_COL].max()}")
    print(f"Blue rows win/loss counts:\n{df['win_binary'].value_counts(dropna=False)}")

    return df


def first_numeric(s):
    x = pd.to_numeric(s, errors="coerce").dropna()
    return float(x.iloc[0]) if len(x) else np.nan


def mean_numeric(s):
    x = pd.to_numeric(s, errors="coerce")
    return float(x.mean()) if x.notna().sum() else np.nan


def sum_numeric(s):
    x = pd.to_numeric(s, errors="coerce")
    return float(x.sum()) if x.notna().sum() else np.nan


def safe_auc(y_true, p):
    if len(np.unique(y_true)) < 2:
        return np.nan
    try:
        return roc_auc_score(y_true, p)
    except Exception:
        return np.nan


def evaluate_probs(y_true, p, threshold=0.5):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    pred = (p >= threshold).astype(int)
    return {
        "logloss": log_loss(y_true, p, labels=[0, 1]),
        "brier": brier_score_loss(y_true, p),
        "auc": safe_auc(y_true, p),
        "accuracy": accuracy_score(y_true, pred),
        "threshold": float(threshold),
    }


def tune_threshold(y_valid, p_valid):
    grid = np.linspace(0.30, 0.70, 81)
    best_acc = -np.inf
    best_thr = 0.5

    for thr in grid:
        acc = accuracy_score(y_valid, (p_valid >= thr).astype(int))
        if (acc > best_acc) or (acc == best_acc and abs(thr - 0.5) < abs(best_thr - 0.5)):
            best_acc = acc
            best_thr = thr

    return float(best_thr), float(best_acc)


def corr_safe(y, x):
    y = pd.Series(y).astype(float)
    x = pd.Series(x).astype(float)
    mask = y.notna() & x.notna()
    if mask.sum() < 5:
        return np.nan
    if y.loc[mask].std(ddof=0) < 1e-10 or x.loc[mask].std(ddof=0) < 1e-10:
        return np.nan
    return float(np.corrcoef(y.loc[mask], x.loc[mask])[0, 1])


def sigmoid(x: float) -> float:
    x = float(np.clip(x, -50, 50))
    return 1.0 / (1.0 + np.exp(-x))


def weighted_mean(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights)

    if mask.sum() == 0 or weights[mask].sum() == 0:
        return np.nan

    return float(np.sum(values[mask] * weights[mask]) / np.sum(weights[mask]))


# ============================================================
# 2. EXACT ORIGINAL MODEL 2: LAYER 1 HISTORY FEATURES
# ============================================================

def _count_prev(df: pd.DataFrame, group_cols: List[str]) -> pd.Series:
    return df.groupby(group_cols, sort=False).cumcount()


def _expanding_mean_shifted(df: pd.DataFrame, group_cols: List[str], col: str) -> pd.Series:
    g = df.groupby(group_cols, sort=False)[col]
    shifted = g.shift(1)
    out = shifted.groupby([df[c] for c in group_cols], sort=False).expanding(min_periods=1).mean()
    return out.reset_index(level=list(range(len(group_cols))), drop=True)


def build_layer1_history_features(df: pd.DataFrame, core_stats: List[str]) -> pd.DataFrame:
    df = df.copy()
    group_specs = {
        "PR": [PLAYER_COL, ROLE_COL],
        "PCR": [PLAYER_COL, ROLE_COL, CHAMP_COL],
        "CR": [ROLE_COL, CHAMP_COL],
        "RG": [ROLE_COL],
    }

    for prefix, group_cols in group_specs.items():
        df[f"{prefix}_n_prev"] = _count_prev(df, group_cols).astype(float)

    for stat in core_stats:
        if stat not in df.columns:
            continue
        for prefix, group_cols in group_specs.items():
            df[f"{prefix}_{stat}_exp_mean"] = _expanding_mean_shifted(df, group_cols, stat)

    df["date_ordinal"] = df[DATE_COL].map(pd.Timestamp.toordinal).astype(float)

    hist_cols = [c for c in df.columns if c.startswith(("PR_", "PCR_", "CR_", "RG_"))]
    df[hist_cols + ["date_ordinal"]] = df[hist_cols + ["date_ordinal"]].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df


def time_split_by_date(dfr: pd.DataFrame, train_frac=0.70, valid_frac=0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    q_train = dfr[DATE_COL].quantile(train_frac)
    q_valid = dfr[DATE_COL].quantile(train_frac + valid_frac)
    train = dfr[dfr[DATE_COL] <= q_train].copy()
    valid = dfr[(dfr[DATE_COL] > q_train) & (dfr[DATE_COL] <= q_valid)].copy()
    test = dfr[dfr[DATE_COL] > q_valid].copy()
    return train, valid, test


def make_X(dfr: pd.DataFrame, columns: List[str] | None = None) -> pd.DataFrame:
    engineered = [
        c for c in dfr.columns
        if c.startswith(("PR_", "PCR_", "CR_")) and not c.endswith("_n_prev")
    ]
    base = ["date_ordinal"]
    counts = ["PR_n_prev", "PCR_n_prev", "CR_n_prev"]
    cat_cols = [PLAYER_COL, PATCH_COL]

    X = dfr[engineered + base + counts + cat_cols].copy()
    for c in cat_cols:
        X[c] = X[c].astype(str).str.strip()
    X = pd.get_dummies(X, columns=cat_cols, prefix=cat_cols, prefix_sep="=", dummy_na=False)
    X = X.loc[:, ~X.columns.duplicated()]
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if columns is not None:
        X = X.reindex(columns=columns, fill_value=0.0)
    return X


def zscore_params(y_train: pd.Series) -> Tuple[float, float]:
    mu = float(y_train.mean())
    sd = float(y_train.std(ddof=0))
    if sd == 0.0 or np.isnan(sd):
        sd = 1.0
    return mu, sd


def zscore(y: pd.Series, mu: float, sd: float) -> np.ndarray:
    return ((y - mu) / sd).values


def val_zmae_for_alpha(X_train, y_train, X_valid, y_valid, alpha: float) -> float:
    mu, sd = zscore_params(y_train)
    y_tr_z = zscore(y_train, mu, sd)
    y_va_z = zscore(y_valid, mu, sd)
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_tr_z)
    pred = model.predict(X_valid)
    return float(mean_absolute_error(y_va_z, pred))


def greedy_select(candidates: List[str], scores: Dict[str, float], k: int) -> List[str]:
    remaining = sorted([c for c in candidates if c in scores], key=lambda c: scores[c])
    chosen = []
    while len(chosen) < k and remaining:
        best_lab, best_obj = None, None
        for lab in remaining:
            obj = float(np.mean([scores[x] for x in chosen + [lab]]))
            if best_obj is None or obj < best_obj:
                best_lab, best_obj = lab, obj
        chosen.append(best_lab)
        remaining.remove(best_lab)
    return chosen


def is_label_usable(y_train: pd.Series, y_valid: pd.Series) -> bool:
    if y_train.notna().sum() < 50 or y_valid.notna().sum() < 20:
        return False
    if float(y_train.std(ddof=0)) <= 1e-9:
        return False
    if float(y_valid.std(ddof=0)) <= 1e-9:
        return False
    return True


def select_layer1_targets(df_feat: pd.DataFrame) -> Tuple[Dict[str, List[str]], pd.DataFrame]:
    selected_labels: Dict[str, List[str]] = {}
    rows = []

    for role, candidates in ROLE_CANDIDATES.items():
        dfr_role = df_feat[df_feat[ROLE_COL] == role].copy()
        train_df, valid_df, _ = time_split_by_date(dfr_role, TRAIN_FRAC, VALID_FRAC)
        if len(train_df) == 0 or len(valid_df) == 0:
            continue

        X_train = make_X(train_df)
        X_valid = make_X(valid_df, columns=list(X_train.columns))

        scores: Dict[str, float] = {}
        for lab in candidates:
            if lab not in dfr_role.columns:
                continue
            tr_mask = train_df[lab].notna()
            va_mask = valid_df[lab].notna()
            if tr_mask.sum() < 50 or va_mask.sum() < 20:
                continue
            y_train = train_df.loc[tr_mask, lab].astype(float)
            y_valid = valid_df.loc[va_mask, lab].astype(float)
            if not is_label_usable(y_train, y_valid):
                continue
            s = val_zmae_for_alpha(
                X_train.loc[y_train.index], y_train,
                X_valid.loc[y_valid.index], y_valid,
                LABEL_SELECTION_ALPHA,
            )
            scores[lab] = s
            rows.append({
                "role": role,
                "label": lab,
                "valid_zMAE": s,
                "train_n": int(tr_mask.sum()),
                "valid_n": int(va_mask.sum()),
            })

        chosen = greedy_select(candidates, scores, K_LABELS)
        selected_labels[role] = chosen
        print(f"[{role}] selected labels: {chosen}")
        print("  Top candidates:", [(k, round(v, 3)) for k, v in sorted(scores.items(), key=lambda kv: kv[1])[:10]])

    return selected_labels, pd.DataFrame(rows)


def tune_alpha(X_train, y_train, X_valid, y_valid) -> Tuple[float, Dict[float, float]]:
    scores = {}
    best_alpha, best_score = None, None
    for alpha in ALPHA_GRID:
        s = val_zmae_for_alpha(X_train, y_train, X_valid, y_valid, alpha)
        scores[alpha] = s
        if best_score is None or s < best_score:
            best_score = s
            best_alpha = alpha
    return float(best_alpha), scores


def fit_layer1_ridge_predictions(df_feat: pd.DataFrame, selected_labels: Dict[str, List[str]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_pred = df_feat.copy()
    result_rows = []

    for role, targets in selected_labels.items():
        dfr_role = df_pred[df_pred[ROLE_COL] == role].copy()
        train_df, valid_df, test_df = time_split_by_date(dfr_role, TRAIN_FRAC, VALID_FRAC)
        if len(train_df) == 0 or len(valid_df) == 0:
            continue

        X_train = make_X(train_df)
        X_valid = make_X(valid_df, columns=list(X_train.columns))
        X_all = make_X(dfr_role, columns=list(X_train.columns))
        X_test = make_X(test_df, columns=list(X_train.columns))

        for target in targets:
            if target not in dfr_role.columns:
                continue
            tr_mask = train_df[target].notna()
            va_mask = valid_df[target].notna()
            te_mask = test_df[target].notna()
            if tr_mask.sum() < 50 or va_mask.sum() < 20:
                continue

            y_train = train_df.loc[tr_mask, target].astype(float)
            y_valid = valid_df.loc[va_mask, target].astype(float)
            best_alpha, _ = tune_alpha(
                X_train.loc[y_train.index], y_train,
                X_valid.loc[y_valid.index], y_valid,
            )

            mu, sd = zscore_params(y_train)
            trainval_df = pd.concat([train_df, valid_df]).sort_values(DATE_COL)
            tv_mask = trainval_df[target].notna()
            y_trainval = trainval_df.loc[tv_mask, target].astype(float)
            X_trainval = make_X(trainval_df, columns=list(X_train.columns)).loc[y_trainval.index]

            model = Ridge(alpha=best_alpha)
            model.fit(X_trainval, zscore(y_trainval, mu, sd))

            pred_all_z = model.predict(X_all)
            pred_all = pred_all_z * sd + mu
            df_pred.loc[dfr_role.index, f"L1_mu__{role}__{target}"] = pred_all

            pred_train_z = model.predict(X_train.loc[y_train.index])
            pred_train = pred_train_z * sd + mu
            resid = y_train.values - pred_train
            resid_sd = float(np.nanstd(resid, ddof=1)) if len(resid) > 1 else float(sd)
            df_pred.loc[dfr_role.index, f"L1_sigma__{role}__{target}"] = resid_sd

            if te_mask.sum() >= 10:
                y_test = test_df.loc[te_mask, target].astype(float)
                pred_test_z = model.predict(X_test.loc[y_test.index])
                pred_test = pred_test_z * sd + mu
                result_rows.append({
                    "role": role,
                    "target": target,
                    "best_alpha": best_alpha,
                    "test_mae": float(mean_absolute_error(y_test, pred_test)),
                    "test_rmse": float(np.sqrt(mean_squared_error(y_test, pred_test))),
                    "test_zmae": float(mean_absolute_error(zscore(y_test, mu, sd), pred_test_z)),
                    "train_n": int(tr_mask.sum()),
                    "valid_n": int(va_mask.sum()),
                    "test_n": int(te_mask.sum()),
                })

            print(f"[Layer1 Ridge] {role} {target}: alpha={best_alpha}, train residual SD={resid_sd:.4f}")

    return df_pred, pd.DataFrame(result_rows)


# ============================================================
# 3. EXACT ORIGINAL MODEL 2: LAYER 2 FEATURES
# ============================================================

def build_base_layer2_features(df_pred: pd.DataFrame, selected_labels: Dict[str, List[str]]) -> pd.DataFrame:
    """Build original Model 2 delta__/unc__/z__ features from L1 Ridge predictions."""
    rows = []
    for match_id, match in df_pred.groupby("match_id", sort=False):
        rec = {
            "match_id": match_id,
            "canonical_match_key": match_id,
            "Date": match[DATE_COL].iloc[0],
            "y_blue_win": int(match.loc[match[SIDE_COL] == "BLUE", "win_binary"].iloc[0]) if (match[SIDE_COL] == "BLUE").any() else np.nan,
        }
        for role, targets in selected_labels.items():
            b = match[(match[SIDE_COL] == "BLUE") & (match[ROLE_COL] == role)]
            r = match[(match[SIDE_COL] == "RED") & (match[ROLE_COL] == role)]
            if len(b) == 0 or len(r) == 0:
                continue
            b = b.iloc[0]
            r = r.iloc[0]
            for target in targets:
                mu_col = f"L1_mu__{role}__{target}"
                sg_col = f"L1_sigma__{role}__{target}"
                if mu_col in df_pred.columns:
                    delta = float(b.get(mu_col, np.nan)) - float(r.get(mu_col, np.nan))
                    rec[f"delta__{role}__{target}"] = delta
                if sg_col in df_pred.columns:
                    unc = np.sqrt(float(b.get(sg_col, np.nan)) ** 2 + float(r.get(sg_col, np.nan)) ** 2)
                    rec[f"unc__{role}__{target}"] = unc
                    if f"delta__{role}__{target}" in rec:
                        rec[f"z__{role}__{target}"] = rec[f"delta__{role}__{target}"] / (unc + 1e-6)
        rows.append(rec)

    match_base = pd.DataFrame(rows).dropna(subset=["Date", "y_blue_win"]).sort_values(["Date", "match_id"]).reset_index(drop=True)
    return match_base


# ============================================================
# 4. MODEL 1 CONDITION TARGETS
# ============================================================

def get_matching_col(df, candidates):
    if isinstance(candidates, str):
        candidates = [candidates]

    for c in candidates:
        if c in df.columns:
            return c

    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]

    def norm(s):
        return re.sub(r"[^a-z0-9]+", "", str(s).lower())

    norm_map = {norm(c): c for c in df.columns}
    for c in candidates:
        if norm(c) in norm_map:
            return norm_map[norm(c)]

    return None


def build_model1_conditions(player_df):
    df = player_df.copy()

    source_cols = {
        "GD15_sum": get_matching_col(df, ["GD@15", "GD15"]),
        "KP15_mean": get_matching_col(df, ["KP%@15", "KP@15", "KP15"]),
        "Tower15": get_matching_col(df, ["Tower@15", "Tower15"]),
        "FirstTower": get_matching_col(df, ["Frist tower", "First tower", "First Tower"]),
        "FirstDragon": get_matching_col(df, ["First Dragon", "First dragon"]),
        "NumberDragon": get_matching_col(df, ["Number of dragon", "Number of Dragon"]),
        "NumberVB": get_matching_col(df, ["Number of VB", "Voidgrubs", "Number of Voidgrubs"]),
        "Herald": get_matching_col(df, ["Herald", "Number of Herald"]),
        "TeamJGShare": get_matching_col(df, ["Team jg share", "Team JG Share"]),
        "MiniTF_count": get_matching_col(df, ["mini team fight", "Mini Team Fight"]),
        "MiniTF_win": get_matching_col(df, ["mini team fight win", "Mini Team Fight Win"]),
        "MiniTF_tie": get_matching_col(df, ["mini team fight tie", "Mini Team Fight Tie"]),
        "TF_count": get_matching_col(df, ["team fight", "Team Fight"]),
        "TF_win": get_matching_col(df, ["team fight win", "Team Fight Win"]),
        "TF_tie": get_matching_col(df, ["team fight tie", "Team Fight Tie"]),
    }

    print("\nModel 1 source columns:")
    for k, v in source_cols.items():
        print(f"  {k}: {v}")

    team_rows = []

    for (match_id, side), g in df.groupby(["match_id", SIDE_COL], dropna=True):
        if side not in {"BLUE", "RED"}:
            continue

        row = {
            "match_id": match_id,
            "canonical_match_key": match_id,
            "Date": g[DATE_COL].iloc[0],
            "Side": side,
            "Team": str(g[TEAM_COL].iloc[0]),
            "team_win": mean_numeric(g["win_binary"]),
        }

        if source_cols["GD15_sum"] is not None:
            row["GD15_sum"] = sum_numeric(g[source_cols["GD15_sum"]])

        if source_cols["KP15_mean"] is not None:
            row["KP15_mean"] = mean_numeric(g[source_cols["KP15_mean"]])

        for name in [
            "Tower15", "FirstTower", "FirstDragon", "NumberDragon", "NumberVB",
            "Herald", "TeamJGShare", "MiniTF_count", "MiniTF_win", "MiniTF_tie",
            "TF_count", "TF_win", "TF_tie",
        ]:
            col = source_cols.get(name)
            if col is not None:
                row[name] = first_numeric(g[col])

        team_rows.append(row)

    team_df = pd.DataFrame(team_rows)

    if {"MiniTF_count", "MiniTF_win", "MiniTF_tie"}.issubset(team_df.columns):
        team_df["MiniTF_WR"] = (
            team_df["MiniTF_win"] + 0.5 * team_df["MiniTF_tie"]
        ) / team_df["MiniTF_count"].replace(0, np.nan)

    if {"TF_count", "TF_win", "TF_tie"}.issubset(team_df.columns):
        team_df["TeamFight_WR"] = (
            team_df["TF_win"] + 0.5 * team_df["TF_tie"]
        ) / team_df["TF_count"].replace(0, np.nan)

    base_condition_cols = [
        c for c in [
            "GD15_sum", "KP15_mean", "Tower15", "FirstTower", "FirstDragon",
            "NumberDragon", "NumberVB", "Herald", "TeamJGShare",
            "MiniTF_count", "MiniTF_WR", "TeamFight_WR",
        ]
        if c in team_df.columns
    ]

    rows = []

    for match_id, g in team_df.groupby("match_id"):
        blue = g[g["Side"].eq("BLUE")]
        red = g[g["Side"].eq("RED")]

        if blue.empty or red.empty:
            continue

        b = blue.iloc[0]
        r = red.iloc[0]

        row = {
            "match_id": match_id,
            "canonical_match_key": match_id,
            "Date": b["Date"],
            "blue_team_raw": b["Team"],
            "red_team_raw": r["Team"],
            "y_blue_win_from_conditions": int(round(float(b["team_win"]))) if pd.notna(b["team_win"]) else np.nan,
        }

        for c in base_condition_cols:
            row[f"actual_delta_{c}"] = b.get(c, np.nan) - r.get(c, np.nan)

        rows.append(row)

    cond_df = pd.DataFrame(rows)

    condition_cols = []
    for c in [c for c in cond_df.columns if c.startswith("actual_delta_")]:
        x = pd.to_numeric(cond_df[c], errors="coerce")
        if x.notna().mean() >= 0.55 and x.dropna().nunique() >= 2:
            condition_cols.append(c)
        else:
            print(f"[Drop condition] {c}: nonmissing={x.notna().mean():.2f}, nunique={x.dropna().nunique()}")

    cond_df = cond_df[
        ["match_id", "canonical_match_key", "Date", "blue_team_raw", "red_team_raw", "y_blue_win_from_conditions"]
        + condition_cols
    ].copy()

    print(f"\nBuilt Model 1 condition table: {cond_df.shape}")
    print("Condition targets:")
    for c in condition_cols:
        print(f"  {c}")

    return cond_df, condition_cols


# ============================================================
# 5. ROLE-WEIGHTED TS FEATURES
# ============================================================

def role_weight(role):
    return ROLE_WEIGHTS.get(str(role), 1.0)


def player_role_key(player, role):
    return (str(player), str(role))


def build_rwts_features(player_df):
    df = player_df.copy()
    df = df.dropna(subset=["match_id", DATE_COL, SIDE_COL, PLAYER_COL, ROLE_COL])

    match_order = (
        df[["match_id", DATE_COL]]
        .drop_duplicates()
        .sort_values([DATE_COL, "match_id"])
        .reset_index(drop=True)
    )

    ts_mu = defaultdict(lambda: TS_INIT_MU)
    ts_sigma = defaultdict(lambda: TS_INIT_SIGMA)
    ts_games = defaultdict(int)

    rows = []

    for _, m in match_order.iterrows():
        match_id = m["match_id"]
        date = m[DATE_COL]

        match_rows = df[df["match_id"].eq(match_id)].copy()
        blue = match_rows[match_rows[SIDE_COL].eq("BLUE")]
        red = match_rows[match_rows[SIDE_COL].eq("RED")]

        if blue.empty or red.empty:
            continue

        y_blue = first_numeric(blue["win_binary"])
        if pd.isna(y_blue):
            continue
        y_blue = int(round(float(y_blue)))

        row = {
            "match_id": match_id,
            "canonical_match_key": match_id,
            "Date": date,
            "y_blue_win_ts_source": int(y_blue),
        }

        side_cache = {}

        for side_name, side_df in [("blue", blue), ("red", red)]:
            mus, sigmas, conservatives, games, weights = [], [], [], [], []

            for _, r in side_df.iterrows():
                pr_key = player_role_key(r[PLAYER_COL], r[ROLE_COL])
                role = str(r[ROLE_COL])
                w = role_weight(role)

                mu = ts_mu[pr_key]
                sigma = ts_sigma[pr_key]
                cons = mu - 3.0 * sigma
                n_games = ts_games[pr_key]

                mus.append(mu)
                sigmas.append(sigma)
                conservatives.append(cons)
                games.append(n_games)
                weights.append(w)

                row[f"{side_name}_rwts_{role}_mu"] = mu
                row[f"{side_name}_rwts_{role}_sigma"] = sigma
                row[f"{side_name}_rwts_{role}_conservative"] = cons
                row[f"{side_name}_rwts_{role}_games"] = n_games
                row[f"{side_name}_rwts_{role}_weight"] = w

            side_cache[side_name] = {
                "mu": weighted_mean(mus, weights),
                "sigma": weighted_mean(sigmas, weights),
                "conservative": weighted_mean(conservatives, weights),
                "games": weighted_mean(games, weights),
            }

            row[f"{side_name}_rwts_mu"] = side_cache[side_name]["mu"]
            row[f"{side_name}_rwts_sigma"] = side_cache[side_name]["sigma"]
            row[f"{side_name}_rwts_conservative"] = side_cache[side_name]["conservative"]
            row[f"{side_name}_rwts_games"] = side_cache[side_name]["games"]

        for base in ["rwts_mu", "rwts_sigma", "rwts_conservative", "rwts_games"]:
            row[f"delta_{base}"] = row.get(f"blue_{base}", np.nan) - row.get(f"red_{base}", np.nan)

        for role in ROLES:
            for base in ["mu", "sigma", "conservative", "games"]:
                bcol = f"blue_rwts_{role}_{base}"
                rcol = f"red_rwts_{role}_{base}"
                row[f"delta_rwts_{base}_{role}"] = row.get(bcol, np.nan) - row.get(rcol, np.nan)

        if np.isfinite(row.get("delta_rwts_mu", np.nan)):
            row["rwts_implied_p_blue"] = sigmoid(row["delta_rwts_mu"] / TS_SCALE)

        rows.append(row)

        # Update only after pre-match feature recording.
        team_mu_b = side_cache["blue"]["mu"]
        team_mu_r = side_cache["red"]["mu"]
        p_blue = sigmoid((team_mu_b - team_mu_r) / TS_SCALE)
        error = y_blue - p_blue

        for side_df, direction in [(blue, 1.0), (red, -1.0)]:
            for _, r in side_df.iterrows():
                pr_key = player_role_key(r[PLAYER_COL], r[ROLE_COL])
                role = str(r[ROLE_COL])
                w = role_weight(role)

                mu = ts_mu[pr_key]
                sigma = ts_sigma[pr_key]

                delta = TS_K * w * direction * error * (sigma / TS_INIT_SIGMA)
                new_mu = mu + delta

                surprise = abs(error)
                new_sigma = max(TS_SIGMA_MIN, sigma * (TS_SIGMA_DECAY + 0.01 * surprise))

                ts_mu[pr_key] = new_mu
                ts_sigma[pr_key] = new_sigma
                ts_games[pr_key] += 1

    rwts_df = pd.DataFrame(rows)

    rwts_cols = [
        c for c in rwts_df.columns
        if pd.api.types.is_numeric_dtype(rwts_df[c])
        and (
            c.startswith("blue_rwts")
            or c.startswith("red_rwts")
            or c.startswith("delta_rwts")
            or c == "rwts_implied_p_blue"
        )
    ]

    rating_rows = []
    for (player, role), mu in ts_mu.items():
        rating_rows.append({
            "Player": player,
            "Role": role,
            "rwts_final_mu": mu,
            "rwts_final_sigma": ts_sigma[(player, role)],
            "rwts_final_conservative": mu - 3.0 * ts_sigma[(player, role)],
            "rwts_games": ts_games[(player, role)],
            "role_weight": role_weight(role),
        })

    final_ratings = pd.DataFrame(rating_rows).sort_values(
        ["Role", "rwts_final_conservative"],
        ascending=[True, False],
    )

    print(f"\nBuilt role-weighted TS feature table: {rwts_df.shape}")
    print(f"Role-weighted TS feature count: {len(rwts_cols)}")

    print("\n=== RWTS FEATURE COLUMNS ===")
    for i, c in enumerate(rwts_cols, start=1):
        print(f"{i:03d}. {c}")

    return rwts_df, rwts_cols, final_ratings


# ============================================================
# 6. FINAL CONDITION-PREDICTION + INDEX LOGISTIC MODEL
# ============================================================

def make_rolling_splits(df):
    order = df.sort_values(["Date", "match_id"]).index.to_numpy()
    n = len(order)

    first_test_start = n - N_FOLDS * TEST_SIZE
    splits = []

    for fold in range(N_FOLDS):
        test_start = first_test_start + fold * TEST_SIZE
        test_end = test_start + TEST_SIZE

        valid_start = test_start - VALID_SIZE
        valid_end = test_start
        train_end = valid_start

        if train_end <= 0 or valid_start < 0 or test_end > n:
            print(f"[Fold {fold + 1}] skipped due to small split.")
            continue

        splits.append((
            fold + 1,
            order[:train_end],
            order[valid_start:valid_end],
            order[test_start:test_end],
        ))

    return splits


def fit_ridge_condition_model(X_train, y_train, X_valid, y_valid):
    best_model = None
    best_alpha = None
    best_mse = np.inf

    for alpha in RIDGE_ALPHA_GRID:
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ])

        model.fit(X_train, y_train)
        pred_valid = model.predict(X_valid)
        mse = mean_squared_error(y_valid, pred_valid)

        if mse < best_mse:
            best_model = model
            best_alpha = alpha
            best_mse = mse

    return best_model, {
        "alpha": best_alpha,
        "valid_mse": float(best_mse),
    }


def predict_conditions_for_fold(df, train_idx, valid_idx, test_idx, predictor_cols, condition_cols):
    predictor_cols = [
        c for c in dict.fromkeys(predictor_cols)
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
    ]

    X_train_all = df.loc[train_idx, predictor_cols].copy()
    X_valid_all = df.loc[valid_idx, predictor_cols].copy()
    X_test_all = df.loc[test_idx, predictor_cols].copy()

    pred_train = pd.DataFrame(index=train_idx)
    pred_valid = pd.DataFrame(index=valid_idx)
    pred_test = pd.DataFrame(index=test_idx)

    eval_rows = []

    for cond in condition_cols:
        y_train_all = pd.to_numeric(df.loc[train_idx, cond], errors="coerce")
        y_valid_all = pd.to_numeric(df.loc[valid_idx, cond], errors="coerce")
        y_test_all = pd.to_numeric(df.loc[test_idx, cond], errors="coerce")

        train_mask = y_train_all.notna()
        valid_mask = y_valid_all.notna()
        test_mask = y_test_all.notna()

        if train_mask.sum() < 30 or valid_mask.sum() < 10:
            continue

        model, params = fit_ridge_condition_model(
            X_train_all.loc[train_mask],
            y_train_all.loc[train_mask],
            X_valid_all.loc[valid_mask],
            y_valid_all.loc[valid_mask],
        )

        pred_train_cond = model.predict(X_train_all)
        pred_valid_cond = model.predict(X_valid_all)
        pred_test_cond = model.predict(X_test_all)

        pred_col = "pred_" + cond.replace("actual_", "")

        pred_train[pred_col] = pred_train_cond
        pred_valid[pred_col] = pred_valid_cond
        pred_test[pred_col] = pred_test_cond

        valid_rmse = np.sqrt(mean_squared_error(y_valid_all.loc[valid_mask], pred_valid_cond[valid_mask]))
        valid_mae = mean_absolute_error(y_valid_all.loc[valid_mask], pred_valid_cond[valid_mask])
        valid_corr = corr_safe(y_valid_all.loc[valid_mask], pred_valid_cond[valid_mask])

        if test_mask.sum() > 0:
            test_rmse = np.sqrt(mean_squared_error(y_test_all.loc[test_mask], pred_test_cond[test_mask]))
            test_mae = mean_absolute_error(y_test_all.loc[test_mask], pred_test_cond[test_mask])
            test_corr = corr_safe(y_test_all.loc[test_mask], pred_test_cond[test_mask])
        else:
            test_rmse = np.nan
            test_mae = np.nan
            test_corr = np.nan

        eval_rows.append({
            "condition": cond,
            "pred_col": pred_col,
            "alpha": params["alpha"],
            "valid_rmse": valid_rmse,
            "valid_mae": valid_mae,
            "valid_corr": valid_corr,
            "test_rmse": test_rmse,
            "test_mae": test_mae,
            "test_corr": test_corr,
            "n_train": int(train_mask.sum()),
            "n_valid": int(valid_mask.sum()),
            "n_test": int(test_mask.sum()),
        })

    return pred_train, pred_valid, pred_test, pd.DataFrame(eval_rows)


def zscore_with_train(train_s, valid_s, test_s):
    train_s = pd.to_numeric(train_s, errors="coerce")
    valid_s = pd.to_numeric(valid_s, errors="coerce")
    test_s = pd.to_numeric(test_s, errors="coerce")

    med = train_s.median()
    if pd.isna(med):
        med = 0.0

    train_f = train_s.fillna(med)
    valid_f = valid_s.fillna(med)
    test_f = test_s.fillna(med)

    mu = train_f.mean()
    sd = train_f.std(ddof=0)

    if not np.isfinite(sd) or sd < 1e-12:
        return train_f * 0.0, valid_f * 0.0, test_f * 0.0

    return (train_f - mu) / sd, (valid_f - mu) / sd, (test_f - mu) / sd


def build_condition_indices_train_valid_test(C_train, C_valid, C_test):
    def col_map(df):
        return {c.replace("pred_delta_", ""): c for c in df.columns if c.startswith("pred_delta_")}

    cmap = col_map(C_train)

    groups = {
        "idx_early_lane": ["GD15_sum", "KP15_mean", "Tower15", "FirstTower"],
        "idx_objective_control": ["FirstDragon", "NumberDragon", "NumberVB", "Herald"],
        "idx_fight_control": ["MiniTF_count", "MiniTF_WR", "TeamFight_WR"],
        "idx_resource_control": ["TeamJGShare"],
    }

    I_train = pd.DataFrame(index=C_train.index)
    I_valid = pd.DataFrame(index=C_valid.index)
    I_test = pd.DataFrame(index=C_test.index)

    for idx_name, keys in groups.items():
        train_terms, valid_terms, test_terms = [], [], []

        for key in keys:
            if key not in cmap:
                continue

            c = cmap[key]
            tr, va, te = zscore_with_train(C_train[c], C_valid[c], C_test[c])
            train_terms.append(tr.to_numpy())
            valid_terms.append(va.to_numpy())
            test_terms.append(te.to_numpy())

        if train_terms:
            I_train[idx_name] = np.mean(np.vstack(train_terms), axis=0)
            I_valid[idx_name] = np.mean(np.vstack(valid_terms), axis=0)
            I_test[idx_name] = np.mean(np.vstack(test_terms), axis=0)

    return I_train, I_valid, I_test


def fit_final_logistic(X_train, y_train, X_valid, y_valid, X_test):
    best_model = None
    best_loss = np.inf
    best_C = None

    for C in LOGISTIC_C_GRID:
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                penalty="l2",
                C=C,
                solver="lbfgs",
                max_iter=5000,
                random_state=RANDOM_SEED,
            )),
        ])

        model.fit(X_train, y_train)
        p_valid = model.predict_proba(X_valid)[:, 1]
        loss = log_loss(y_valid, np.clip(p_valid, 1e-6, 1 - 1e-6), labels=[0, 1])

        if loss < best_loss:
            best_model = model
            best_loss = loss
            best_C = C

    p_valid = best_model.predict_proba(X_valid)[:, 1]
    p_test = best_model.predict_proba(X_test)[:, 1]

    params = {
        "model": "LogisticRegression",
        "C": best_C,
        "valid_logloss": float(best_loss),
    }

    return p_valid, p_test, best_model, params


def evaluate_final_model(df, target_col, predictor_cols, condition_cols):
    y = df[target_col].astype(int).to_numpy()

    fold_rows = []
    prediction_rows = []
    condition_eval_rows = []
    condition_prediction_rows = []
    condition_index_rows = []
    coef_rows = []

    for fold, train_idx, valid_idx, test_idx in make_rolling_splits(df):
        y_train = y[train_idx]
        y_valid = y[valid_idx]
        y_test = y[test_idx]

        C_train, C_valid, C_test, cond_eval = predict_conditions_for_fold(
            df,
            train_idx,
            valid_idx,
            test_idx,
            predictor_cols,
            condition_cols,
        )

        if C_train.empty:
            print(f"[Fold {fold}] no condition predictions; skipped.")
            continue

        I_train, I_valid, I_test = build_condition_indices_train_valid_test(C_train, C_valid, C_test)

        p_valid, p_test, model, params = fit_final_logistic(
            I_train,
            y_train,
            I_valid,
            y_valid,
            I_test,
        )

        try:
            clf_tmp = model.named_steps["clf"]
            print(f"[Fold {fold}] final logistic beta0/intercept = {clf_tmp.intercept_[0]:.6f}")
        except Exception:
            pass

        if TUNE_THRESHOLD_FOR_ACCURACY:
            threshold, valid_acc = tune_threshold(y_valid, p_valid)
        else:
            threshold, valid_acc = 0.5, accuracy_score(y_valid, (p_valid >= 0.5).astype(int))

        metrics = evaluate_probs(y_test, p_test, threshold=threshold)
        metrics_05 = evaluate_probs(y_test, p_test, threshold=0.5)

        row = {
            "variant": "M1plusM2plusRWTS_indices_logistic",
            "model_type": "logistic",
            "fold": fold,
            "n_features": I_train.shape[1],
            **metrics,
            "accuracy_at_0_5": metrics_05["accuracy"],
            "valid_accuracy_at_threshold": valid_acc,
            "params": json.dumps(params),
        }
        fold_rows.append(row)

        print(
            f"[M1plusM2plusRWTS_indices_logistic | Fold {fold}] "
            f"logloss={row['logloss']:.4f} "
            f"brier={row['brier']:.4f} "
            f"auc={row['auc']:.4f} "
            f"acc@thr={row['accuracy']:.4f} "
            f"acc@0.5={row['accuracy_at_0_5']:.4f} "
            f"features={I_train.shape[1]} "
            f"pred_conditions={C_train.shape[1]}"
        )

        for idx, yy, pp in zip(test_idx, y_test, p_test):
            prediction_rows.append({
                "variant": "M1plusM2plusRWTS_indices_logistic",
                "fold": fold,
                "row_index": int(idx),
                "match_id": df.loc[idx, "match_id"],
                "Date": df.loc[idx, "Date"],
                "y_true": int(yy),
                "p_blue_win": float(pp),
                "threshold": threshold,
                "pred_at_threshold": int(pp >= threshold),
                "pred_at_0_5": int(pp >= 0.5),
            })

        cond_eval = cond_eval.copy()
        cond_eval["fold"] = fold
        cond_eval["variant"] = "M1plusM2plusRWTS_indices_logistic"
        condition_eval_rows.extend(cond_eval.to_dict("records"))

        for idx in test_idx:
            r = {
                "variant": "M1plusM2plusRWTS_indices_logistic",
                "fold": fold,
                "row_index": int(idx),
                "match_id": df.loc[idx, "match_id"],
                "Date": df.loc[idx, "Date"],
                "y_true": int(y[idx]),
            }

            for c in C_test.columns:
                r[c] = C_test.loc[idx, c]

            for c in condition_cols:
                r[c] = df.loc[idx, c] if c in df.columns else np.nan

            condition_prediction_rows.append(r)

            ix = {
                "variant": "M1plusM2plusRWTS_indices_logistic",
                "fold": fold,
                "row_index": int(idx),
                "match_id": df.loc[idx, "match_id"],
                "Date": df.loc[idx, "Date"],
                "y_true": int(y[idx]),
            }

            for c in I_test.columns:
                ix[c] = I_test.loc[idx, c]

            condition_index_rows.append(ix)

        try:
            clf = model.named_steps["clf"]

            # Save beta0 / intercept.
            # This is the intercept in the standardized-feature logistic model:
            # eta = beta0 + beta1 * I_early_scaled + ... + beta4 * I_resource_scaled
            coef_rows.append({
                "variant": "M1plusM2plusRWTS_indices_logistic",
                "fold": fold,
                "feature": "intercept_beta0",
                "coef_standardized": float(clf.intercept_[0]),
                "abs_coef": float(abs(clf.intercept_[0])),
                "C": params["C"],
            })

            # Save slope coefficients for the four final condition indices.
            for feat, coef in zip(I_train.columns, clf.coef_[0]):
                coef_rows.append({
                    "variant": "M1plusM2plusRWTS_indices_logistic",
                    "fold": fold,
                    "feature": feat,
                    "coef_standardized": float(coef),
                    "abs_coef": float(abs(coef)),
                    "C": params["C"],
                })

        except Exception:
            pass

    return (
        pd.DataFrame(fold_rows),
        pd.DataFrame(prediction_rows),
        pd.DataFrame(condition_eval_rows),
        pd.DataFrame(condition_prediction_rows),
        pd.DataFrame(condition_index_rows),
        pd.DataFrame(coef_rows),
    )


# ============================================================
# 7. SUMMARIES
# ============================================================

def summarize_results(fold_results):
    if fold_results.empty:
        return pd.DataFrame()

    return (
        fold_results
        .groupby(["variant", "model_type"], as_index=False)
        .agg(
            n_folds=("fold", "count"),
            n_features=("n_features", "mean"),
            test_logloss_mean=("logloss", "mean"),
            test_logloss_std=("logloss", "std"),
            test_brier_mean=("brier", "mean"),
            test_brier_std=("brier", "std"),
            test_auc_mean=("auc", "mean"),
            test_auc_std=("auc", "std"),
            test_accuracy_tuned_mean=("accuracy", "mean"),
            test_accuracy_tuned_std=("accuracy", "std"),
            test_accuracy_0_5_mean=("accuracy_at_0_5", "mean"),
            test_accuracy_0_5_std=("accuracy_at_0_5", "std"),
            threshold_mean=("threshold", "mean"),
        )
        .sort_values(["test_logloss_mean", "test_brier_mean"], ascending=True)
    )


def summarize_condition_predictions(condition_eval):
    if condition_eval.empty:
        return pd.DataFrame()

    return (
        condition_eval
        .groupby(["variant", "condition"], as_index=False)
        .agg(
            valid_rmse_mean=("valid_rmse", "mean"),
            valid_mae_mean=("valid_mae", "mean"),
            valid_corr_mean=("valid_corr", "mean"),
            test_rmse_mean=("test_rmse", "mean"),
            test_mae_mean=("test_mae", "mean"),
            test_corr_mean=("test_corr", "mean"),
            n_train_mean=("n_train", "mean"),
            n_test_mean=("n_test", "mean"),
        )
        .sort_values(["variant", "test_corr_mean"], ascending=[True, False])
    )


def summarize_coefficients(coefs):
    if coefs.empty:
        return pd.DataFrame()

    return (
        coefs
        .groupby(["variant", "feature"], as_index=False)
        .agg(
            mean_coef=("coef_standardized", "mean"),
            mean_abs_coef=("abs_coef", "mean"),
            selected_n_folds=("fold", "nunique"),
        )
        .sort_values(["variant", "mean_abs_coef"], ascending=[True, False])
    )


# ============================================================
# 8. MAIN
# ============================================================

def main():
    print("=" * 100)
    print("FULL END-TO-END FINAL MODEL: EXACT M2 + M1 + ROLE-WEIGHTED TS INDICES LOGISTIC")
    print("=" * 100)

    # 1. Load and clean raw player CSV.
    df = load_and_clean(DATA_PATH)

    # 2. Build exact original Model 2 feature block internally.
    print("\n" + "=" * 100)
    print("BUILDING EXACT ORIGINAL MODEL 2 FEATURES")
    print("=" * 100)

    df_feat = build_layer1_history_features(df, CORE_STATS)

    selected_labels, selection_scores = select_layer1_targets(df_feat)
    selection_scores.to_csv(OUT_DIR / "layer1_label_selection_scores.csv", index=False)
    (OUT_DIR / "selected_labels.json").write_text(json.dumps(selected_labels, indent=2))

    df_ridge, layer1_results = fit_layer1_ridge_predictions(df_feat, selected_labels)
    layer1_results.to_csv(OUT_DIR / "layer1_ridge_test_results.csv", index=False)

    model2_df = build_base_layer2_features(df_ridge, selected_labels)
    model2_df.to_csv(OUT_DIR / "features_A_original_model2_rebuilt.csv", index=False)

    m2_cols = [c for c in model2_df.columns if c.startswith(("delta__", "unc__", "z__"))]

    print(f"\nRebuilt Model 2 dataframe shape: {model2_df.shape}")
    print(f"Rebuilt Model 2 feature count: {len(m2_cols)}")

    # 3. Build Model 1 condition targets.
    condition_df, condition_cols = build_model1_conditions(df)

    # 4. Build role-weighted TS features.
    rwts_df, rwts_cols, final_ratings = build_rwts_features(df)

    # 5. Integrate all final inputs.
    integrated = model2_df.merge(
        condition_df.drop(columns=["Date", "canonical_match_key"], errors="ignore"),
        on="match_id",
        how="left",
    )

    integrated = integrated.merge(
        rwts_df.drop(columns=["Date", "canonical_match_key"], errors="ignore"),
        on="match_id",
        how="left",
    )

    m2_cols = [c for c in m2_cols if c in integrated.columns and pd.api.types.is_numeric_dtype(integrated[c])]
    rwts_cols = [c for c in rwts_cols if c in integrated.columns and pd.api.types.is_numeric_dtype(integrated[c])]
    predictor_cols = list(dict.fromkeys(m2_cols + rwts_cols))

    print("\n" + "=" * 100)
    print("FINAL INTEGRATED DATA")
    print("=" * 100)
    print(f"Integrated dataframe shape: {integrated.shape}")
    print(f"Rows with Model 1 condition targets: {integrated[condition_cols].notna().any(axis=1).sum()} / {len(integrated)}")
    print(f"Rows with RWTS features: {integrated[rwts_cols].notna().any(axis=1).sum()} / {len(integrated)}")
    print("\nFeature block counts:")
    print(f"  Exact rebuilt Model 2 features: {len(m2_cols)}")
    print(f"  Role-weighted TS features: {len(rwts_cols)}")
    print(f"  Condition prediction inputs: {len(predictor_cols)}")
    print(f"  Model 1 condition targets: {len(condition_cols)}")
    print("  Final logistic features: 4 condition indices")

    # 6. Evaluate final selected model.
    fold_results, test_predictions, condition_eval, condition_predictions, condition_indices, coefs = evaluate_final_model(
        integrated,
        "y_blue_win",
        predictor_cols,
        condition_cols,
    )

    cv_summary = summarize_results(fold_results)
    condition_summary = summarize_condition_predictions(condition_eval)
    coef_summary = summarize_coefficients(coefs)

    # 7. Save outputs.
    integrated.to_csv(OUT_DIR / "full_end_to_end_integrated_features.csv", index=False)
    final_ratings.to_csv(OUT_DIR / "full_end_to_end_rwts_player_ratings.csv", index=False)

    with open(OUT_DIR / "full_end_to_end_feature_lists.json", "w") as f:
        json.dump(
            {
                "required_input": str(DATA_PATH.name),
                "model2_selected_labels": selected_labels,
                "m2_cols": m2_cols,
                "rwts_cols": rwts_cols,
                "predictor_cols": predictor_cols,
                "condition_cols": condition_cols,
                "final_condition_indices": [
                    "idx_early_lane",
                    "idx_objective_control",
                    "idx_fight_control",
                    "idx_resource_control",
                ],
                "role_weights": ROLE_WEIGHTS,
                "ts_settings": {
                    "TS_INIT_MU": TS_INIT_MU,
                    "TS_INIT_SIGMA": TS_INIT_SIGMA,
                    "TS_SIGMA_MIN": TS_SIGMA_MIN,
                    "TS_K": TS_K,
                    "TS_SIGMA_DECAY": TS_SIGMA_DECAY,
                    "TS_SCALE": TS_SCALE,
                },
                "model2_settings": {
                    "TRAIN_FRAC": TRAIN_FRAC,
                    "VALID_FRAC": VALID_FRAC,
                    "K_LABELS": K_LABELS,
                    "LABEL_SELECTION_ALPHA": LABEL_SELECTION_ALPHA,
                    "ALPHA_GRID": ALPHA_GRID,
                    "ROLE_CANDIDATES": ROLE_CANDIDATES,
                    "CORE_STATS": CORE_STATS,
                },
                "final_split_settings": {
                    "N_FOLDS": N_FOLDS,
                    "VALID_SIZE": VALID_SIZE,
                    "TEST_SIZE": TEST_SIZE,
                },
            },
            f,
            indent=2,
        )

    fold_results.to_csv(OUT_DIR / "full_end_to_end_fold_results.csv", index=False)
    test_predictions.to_csv(OUT_DIR / "full_end_to_end_test_predictions.csv", index=False)
    cv_summary.to_csv(OUT_DIR / "full_end_to_end_cv_summary.csv", index=False)
    condition_eval.to_csv(OUT_DIR / "full_end_to_end_condition_prediction_results.csv", index=False)
    condition_summary.to_csv(OUT_DIR / "full_end_to_end_condition_prediction_summary.csv", index=False)
    condition_predictions.to_csv(OUT_DIR / "full_end_to_end_condition_predictions.csv", index=False)
    condition_indices.to_csv(OUT_DIR / "full_end_to_end_condition_index_values.csv", index=False)
    coefs.to_csv(OUT_DIR / "full_end_to_end_logistic_coefficients.csv", index=False)
    coef_summary.to_csv(OUT_DIR / "full_end_to_end_logistic_coefficient_summary.csv", index=False)

    print("\n" + "=" * 100)
    print("FULL END-TO-END FINAL MODEL CV SUMMARY")
    print(cv_summary.to_string(index=False))

    print("\n" + "=" * 100)
    print("FULL END-TO-END FINAL LOGISTIC COEFFICIENT SUMMARY")
    if not coef_summary.empty:
        print(coef_summary.to_string(index=False))
    else:
        print("[No coefficient summary produced.]")

    print("\nSaved outputs to:", OUT_DIR)
    print("Key files:")
    print("  features_A_original_model2_rebuilt.csv")
    print("  full_end_to_end_cv_summary.csv")
    print("  full_end_to_end_fold_results.csv")
    print("  full_end_to_end_test_predictions.csv")
    print("  full_end_to_end_integrated_features.csv")
    print("  full_end_to_end_feature_lists.json")
    print("  full_end_to_end_condition_prediction_summary.csv")
    print("  full_end_to_end_condition_index_values.csv")
    print("  full_end_to_end_logistic_coefficient_summary.csv")
    print("  full_end_to_end_rwts_player_ratings.csv")


if __name__ == "__main__":
    main()
