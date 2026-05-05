from __future__ import annotations

from pathlib import Path
import json
import re
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    log_loss,
    brier_score_loss,
    roc_auc_score,
    accuracy_score,
)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# ============================================================
# Import the existing final pipeline construction
# ============================================================

import model8_final_FULL_END_TO_END_m1_m2_rwts_indices_logistic_v2 as base


# ============================================================
# 0. CONFIG
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
OUT_DIR = PROJECT_DIR / "model10_hardcoded_patch_win_conditions_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# We will compare hard-coded top K lists.
TOP_K_LIST = [10, 20]

# Logistic regularization grid.
LOGISTIC_C_GRID = base.LOGISTIC_C_GRID

# Patch-specific model minimums.
MIN_PATCH_TRAIN_N = 20
MIN_PATCH_VALID_N = 8

TUNE_THRESHOLD_FOR_ACCURACY = True
RANDOM_SEED = base.RANDOM_SEED


# ============================================================
# 1. HARD-CODED PATCH CONDITION LISTS
# ============================================================
# These are ordered from stronger/more important to weaker/lower priority.
# Edit these lists after you decide the final patch-specific order.
#
# The script will:
#   1. Keep the first K for each patch.
#   2. Use only conditions that actually exist in the constructed condition table.
#   3. Use predicted versions of these conditions in the final model.

HARDCODED_PATCH_CONDITIONS = {
    "v14.10": [
        "TeamFight_WR",
        "GD15_sum",
        "TeamJGShare",
        "Tower15",
        "NumberVB",
        "MiniTF_WR",
        "NumberDragon",
        "FirstDragon",
        "FirstTower",
        "KP15_mean",
        "Herald",
        "MiniTF_count",
        "TotalTower",
        "NumberBaron",
        "FirstBaron",
        "NumberElderDragon",
        "Kills",
        "Deaths",
        "Assists",
        "TotalDamage",
    ],

    "v14.11": [
        "TeamFight_WR",
        "TeamJGShare",
        "FirstDragon",
        "GD15_sum",
        "FirstTower",
        "NumberVB",
        "Tower15",
        "KP15_mean",
        "NumberDragon",
        "MiniTF_WR",
        "Herald",
        "MiniTF_count",
        "TotalTower",
        "NumberBaron",
        "FirstBaron",
        "NumberElderDragon",
        "Kills",
        "Deaths",
        "Assists",
        "TotalDamage",
    ],

    "v14.13": [
        "TeamFight_WR",
        "TotalTower",
        "NumberBaron",
        "Assists",
        "NumberDragon",
        "Deaths",
        "Kills",
        "TeamJGShare",
        "GD15_sum",
        "KP15_mean",
        "Tower15",
        "FirstTower",
        "NumberVB",
        "FirstDragon",
        "Herald",
        "MiniTF_WR",
        "MiniTF_count",
        "FirstBaron",
        "NumberElderDragon",
        "TotalDamage",
    ],

    "v14.14": [
        "TotalTower",
        "TeamFight_WR",
        "Kills",
        "Deaths",
        "Assists",
        "NumberDragon",
        "NumberBaron",
        "TeamJGShare",
        "NumberVB",
        "GD15_sum",
        "KP15_mean",
        "Tower15",
        "MiniTF_WR",
        "FirstTower",
        "FirstDragon",
        "Herald",
        "MiniTF_count",
        "FirstBaron",
        "NumberElderDragon",
        "TotalDamage",
    ],
}

# If a patch is unseen or has too little training data, this global fallback is used.
GLOBAL_FALLBACK_CONDITIONS = [
    "TeamFight_WR",
    "GD15_sum",
    "TeamJGShare",
    "NumberDragon",
    "Tower15",
    "KP15_mean",
    "FirstTower",
    "FirstDragon",
    "NumberVB",
    "MiniTF_WR",
    "Herald",
    "MiniTF_count",
    "TotalTower",
    "NumberBaron",
    "FirstBaron",
    "NumberElderDragon",
    "Kills",
    "Deaths",
    "Assists",
    "TotalDamage",
]


# ============================================================
# 2. BASIC HELPERS
# ============================================================

def normalize_patch(x):
    if pd.isna(x):
        return "UNKNOWN"
    s = str(x).strip()
    s = s.replace("Patch", "").replace("patch", "").strip()
    if not s:
        return "UNKNOWN"
    if not s.startswith("v"):
        s = "v" + s
    return s


def build_match_patch_table(player_df):
    patch_df = (
        player_df[["match_id", base.PATCH_COL]]
        .dropna()
        .drop_duplicates()
        .copy()
    )
    patch_df[base.PATCH_COL] = patch_df[base.PATCH_COL].map(normalize_patch)

    patch_df = (
        patch_df
        .sort_values(["match_id", base.PATCH_COL])
        .groupby("match_id", as_index=False)
        .agg(Patch=(base.PATCH_COL, "first"))
    )
    return patch_df


def safe_auc(y_true, p):
    y_true = np.asarray(y_true)
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


def first_numeric(s):
    x = pd.to_numeric(s, errors="coerce").dropna()
    return float(x.iloc[0]) if len(x) else np.nan


def mean_numeric(s):
    x = pd.to_numeric(s, errors="coerce")
    return float(x.mean()) if x.notna().sum() else np.nan


def sum_numeric(s):
    x = pd.to_numeric(s, errors="coerce")
    return float(x.sum()) if x.notna().sum() else np.nan


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


def clean_to_actual_condition(clean_name):
    return f"actual_delta_{clean_name}"


def actual_to_pred_condition(actual_name):
    return "pred_" + actual_name.replace("actual_", "")


def clean_to_pred_condition(clean_name):
    return f"pred_delta_{clean_name}"


def available_clean_conditions(condition_cols):
    out = []
    for c in condition_cols:
        if c.startswith("actual_delta_"):
            out.append(c.replace("actual_delta_", ""))
    return out


def select_patch_conditions(patch, top_k, available_clean):
    raw = HARDCODED_PATCH_CONDITIONS.get(str(patch), GLOBAL_FALLBACK_CONDITIONS)
    selected = [c for c in raw if c in available_clean]

    if len(selected) < min(top_k, len(raw)):
        # Fill from global fallback if hard-coded patch list references missing columns.
        for c in GLOBAL_FALLBACK_CONDITIONS:
            if c in available_clean and c not in selected:
                selected.append(c)

    return selected[:min(top_k, len(selected))]


# ============================================================
# 3. EXTENDED MODEL 1 CONDITION TARGETS
# ============================================================

def build_extended_model1_conditions(player_df):
    """
    Extended version of base.build_model1_conditions.

    It includes the original 12 clean conditions and attempts to add
    later-game/manual-review conditions if the raw CSV has those columns.

    Actual conditions are still only used as training targets.
    Final prediction uses predicted versions of these conditions.
    """
    df = player_df.copy()

    source_cols = {
        # Original clean conditions
        "GD15_sum": get_matching_col(df, ["GD@15", "GD15"]),
        "KP15_mean": get_matching_col(df, ["KP%@15", "KP@15", "KP15"]),
        "Tower15": get_matching_col(df, ["Tower@15", "Tower15"]),
        "FirstTower": get_matching_col(df, ["Frist tower", "First tower", "First Tower"]),
        "FirstDragon": get_matching_col(df, ["First Dragon", "First dragon"]),
        "NumberDragon": get_matching_col(df, ["Number of dragon", "Number of Dragon"]),
        "NumberVB": get_matching_col(df, ["Number of VB", "Voidgrubs", "Number of Voidgrubs"]),
        "Herald": get_matching_col(df, ["Herald", "Number of Herald"]),
        "TeamJGShare": get_matching_col(df, ["Team jg share", "Team JG Share"]),

        # Fight variables
        "MiniTF_count": get_matching_col(df, ["mini team fight", "Mini Team Fight"]),
        "MiniTF_win": get_matching_col(df, ["mini team fight win", "Mini Team Fight Win"]),
        "MiniTF_tie": get_matching_col(df, ["mini team fight tie", "Mini Team Fight Tie"]),
        "TF_count": get_matching_col(df, ["team fight", "Team Fight"]),
        "TF_win": get_matching_col(df, ["team fight win", "Team Fight Win"]),
        "TF_tie": get_matching_col(df, ["team fight tie", "Team Fight Tie"]),

        # Later-game / extended condition candidates
        "TotalTower": get_matching_col(df, [
            "Total tower", "Total Tower", "Total towers", "Total Towers",
            "Towers", "Tower"
        ]),
        "FirstBaron": get_matching_col(df, [
            "First baron", "First Baron", "First Daron", "FirstDaron", "FirstBaron"
        ]),
        "NumberBaron": get_matching_col(df, [
            "Number of baron", "Number of Baron", "Number Baron", "NumberBaron"
        ]),
        "NumberElderDragon": get_matching_col(df, [
            "Number of Elder Dragon", "Number of elder dragon",
            "Number of Ult dragon", "Number of Ult Dragon",
            "Elder Dragon", "ElderDragon"
        ]),

        # Basic final/combat stats if present
        "Kills": get_matching_col(df, ["Kills"]),
        "Deaths": get_matching_col(df, ["Deaths"]),
        "Assists": get_matching_col(df, ["Assists"]),
        "TotalDamage": get_matching_col(df, [
            "Total damage", "Total Damage", "Damage", "DPM"
        ]),
    }

    print("\nExtended Model 1 source columns:")
    for k, v in source_cols.items():
        print(f"  {k}: {v}")

    team_rows = []

    for (match_id, side), g in df.groupby(["match_id", base.SIDE_COL], dropna=True):
        if side not in {"BLUE", "RED"}:
            continue

        row = {
            "match_id": match_id,
            "canonical_match_key": match_id,
            "Date": g[base.DATE_COL].iloc[0],
            "Side": side,
            "Team": str(g[base.TEAM_COL].iloc[0]),
            "team_win": mean_numeric(g["win_binary"]),
        }

        # Aggregated player-level variables
        if source_cols["GD15_sum"] is not None:
            row["GD15_sum"] = sum_numeric(g[source_cols["GD15_sum"]])

        if source_cols["KP15_mean"] is not None:
            row["KP15_mean"] = mean_numeric(g[source_cols["KP15_mean"]])

        # Player-level sum stats if present
        for name in ["Kills", "Deaths", "Assists", "TotalDamage"]:
            col = source_cols.get(name)
            if col is not None:
                row[name] = sum_numeric(g[col])

        # Team-level variables
        for name in [
            "Tower15", "FirstTower", "FirstDragon", "NumberDragon", "NumberVB",
            "Herald", "TeamJGShare", "MiniTF_count", "MiniTF_win", "MiniTF_tie",
            "TF_count", "TF_win", "TF_tie",
            "TotalTower", "FirstBaron", "NumberBaron", "NumberElderDragon",
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
            "TotalTower", "FirstBaron", "NumberBaron", "NumberElderDragon",
            "Kills", "Deaths", "Assists", "TotalDamage",
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

    print(f"\nBuilt extended Model 1 condition table: {cond_df.shape}")
    print("Condition targets:")
    for c in condition_cols:
        print(f"  {c}")

    return cond_df, condition_cols


# ============================================================
# 4. FINAL FEATURE CONSTRUCTION
# ============================================================

def zscore_pred_cols_with_train(C_train, C_valid, C_test, pred_cols):
    X_train = pd.DataFrame(index=C_train.index)
    X_valid = pd.DataFrame(index=C_valid.index)
    X_test = pd.DataFrame(index=C_test.index)

    for c in pred_cols:
        if c not in C_train.columns:
            continue

        tr = pd.to_numeric(C_train[c], errors="coerce")
        va = pd.to_numeric(C_valid[c], errors="coerce")
        te = pd.to_numeric(C_test[c], errors="coerce")

        med = tr.median()
        if pd.isna(med):
            med = 0.0

        tr_f = tr.fillna(med)
        va_f = va.fillna(med)
        te_f = te.fillna(med)

        mu = tr_f.mean()
        sd = tr_f.std(ddof=0)

        if not np.isfinite(sd) or sd < 1e-12:
            sd = 1.0

        X_train[c] = (tr_f - mu) / sd
        X_valid[c] = (va_f - mu) / sd
        X_test[c] = (te_f - mu) / sd

    return X_train, X_valid, X_test


def add_pairwise_interactions(X):
    X2 = X.copy()
    cols = list(X.columns)

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            c1 = cols[i]
            c2 = cols[j]
            X2[f"{c1}_x_{c2}"] = X[c1] * X[c2]

    return X2


def make_hardcoded_patch_union_features(
    df,
    train_idx,
    valid_idx,
    test_idx,
    C_train,
    C_valid,
    C_test,
    top_k,
    add_interactions=False,
):
    """
    Pooled hard-coded patch-aware model.

    Feature design:
      - Use union of hard-coded conditions across patches.
      - Add patch dummies.
      - Add patch × condition terms.
      - Optionally add condition × condition interactions inside each patch.

    This lets each patch have different condition effects while using one pooled model.
    """
    available_clean = available_clean_conditions([
        "actual_" + c.replace("pred_", "") for c in C_train.columns
        if c.startswith("pred_delta_")
    ])

    train_patches = df.loc[train_idx, "Patch"].astype(str)
    valid_patches = df.loc[valid_idx, "Patch"].astype(str)
    test_patches = df.loc[test_idx, "Patch"].astype(str)

    # Union of all hard-coded selected conditions across patches.
    selected_union = []
    patch_selected = {}

    for patch in sorted(set(train_patches.unique()).union(valid_patches.unique()).union(test_patches.unique())):
        selected = select_patch_conditions(patch, top_k, available_clean)
        patch_selected[patch] = selected
        selected_union.extend(selected)

    selected_union = list(dict.fromkeys(selected_union))

    pred_cols = [clean_to_pred_condition(c) for c in selected_union]
    pred_cols = [c for c in pred_cols if c in C_train.columns]

    X_train_base, X_valid_base, X_test_base = zscore_pred_cols_with_train(
        C_train, C_valid, C_test, pred_cols
    )

    # Patch dummies from training patches only.
    train_dum = pd.get_dummies(train_patches, prefix="patch")
    valid_dum = pd.get_dummies(valid_patches, prefix="patch")
    test_dum = pd.get_dummies(test_patches, prefix="patch")

    patch_dummy_cols = sorted(train_dum.columns.tolist())

    train_dum = train_dum.reindex(columns=patch_dummy_cols, fill_value=0)
    valid_dum = valid_dum.reindex(columns=patch_dummy_cols, fill_value=0)
    test_dum = test_dum.reindex(columns=patch_dummy_cols, fill_value=0)

    train_dum.index = X_train_base.index
    valid_dum.index = X_valid_base.index
    test_dum.index = X_test_base.index

    X_train = pd.concat([X_train_base, train_dum], axis=1)
    X_valid = pd.concat([X_valid_base, valid_dum], axis=1)
    X_test = pd.concat([X_test_base, test_dum], axis=1)

    # Patch × condition interactions.
    for pcol in patch_dummy_cols:
        patch_name = pcol.replace("patch_", "")
        selected_for_patch = patch_selected.get(patch_name, [])
        selected_pred_cols = [clean_to_pred_condition(c) for c in selected_for_patch]
        selected_pred_cols = [c for c in selected_pred_cols if c in X_train_base.columns]

        for c in selected_pred_cols:
            new_col = f"{pcol}_x_{c}"
            X_train[new_col] = X_train[pcol] * X_train[c]
            X_valid[new_col] = X_valid[pcol] * X_valid[c]
            X_test[new_col] = X_test[pcol] * X_test[c]

        if add_interactions:
            for i in range(len(selected_pred_cols)):
                for j in range(i + 1, len(selected_pred_cols)):
                    c1 = selected_pred_cols[i]
                    c2 = selected_pred_cols[j]
                    new_col = f"{pcol}_x_{c1}_x_{c2}"

                    X_train[new_col] = X_train[pcol] * X_train[c1] * X_train[c2]
                    X_valid[new_col] = X_valid[pcol] * X_valid[c1] * X_valid[c2]
                    X_test[new_col] = X_test[pcol] * X_test[c1] * X_test[c2]

    X_valid = X_valid.reindex(columns=X_train.columns, fill_value=0.0)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0.0)

    return X_train, X_valid, X_test, patch_selected, selected_union


def make_direct_condition_features(C_train, C_valid, C_test, clean_conditions, add_interactions=False):
    pred_cols = [clean_to_pred_condition(c) for c in clean_conditions]
    pred_cols = [c for c in pred_cols if c in C_train.columns]

    X_train, X_valid, X_test = zscore_pred_cols_with_train(
        C_train, C_valid, C_test, pred_cols
    )

    if add_interactions:
        X_train = add_pairwise_interactions(X_train)
        X_valid = add_pairwise_interactions(X_valid)
        X_test = add_pairwise_interactions(X_test)

    return X_train, X_valid, X_test


# ============================================================
# 5. LOGISTIC FITTING
# ============================================================

def fit_logistic_grid(X_train, y_train, X_valid, y_valid, X_test):
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

    return p_valid, p_test, best_model, {
        "C": best_C,
        "valid_logloss": float(best_loss),
    }


def fit_logistic_fixed_C(X_train, y_train, X_valid, y_valid, X_test, C=0.1):
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

    p_valid = model.predict_proba(X_valid)[:, 1] if len(X_valid) else np.array([])
    p_test = model.predict_proba(X_test)[:, 1] if len(X_test) else np.array([])

    return p_valid, p_test, model, {
        "C": C,
        "valid_logloss": np.nan,
        "fixed_C": True,
    }


def save_coef_rows(coef_rows, variant, fold, model, feature_cols, params, patch="ALL"):
    try:
        clf = model.named_steps["clf"]

        coef_rows.append({
            "variant": variant,
            "fold": fold,
            "patch": patch,
            "feature": "intercept_beta0",
            "coef_standardized": float(clf.intercept_[0]),
            "abs_coef": float(abs(clf.intercept_[0])),
            "params": json.dumps(params),
        })

        for feat, coef in zip(feature_cols, clf.coef_[0]):
            coef_rows.append({
                "variant": variant,
                "fold": fold,
                "patch": patch,
                "feature": feat,
                "coef_standardized": float(coef),
                "abs_coef": float(abs(coef)),
                "params": json.dumps(params),
            })

    except Exception:
        pass


# ============================================================
# 6. RESULT HELPERS
# ============================================================

def append_fold_result(
    fold_rows,
    prediction_rows,
    variant,
    model_type,
    fold,
    top_k,
    interaction_mode,
    n_features,
    y_valid,
    p_valid,
    y_test,
    p_test,
    test_idx,
    df,
    params,
):
    if TUNE_THRESHOLD_FOR_ACCURACY and len(y_valid) > 0:
        threshold, valid_acc = tune_threshold(y_valid, p_valid)
    else:
        threshold = 0.5
        valid_acc = accuracy_score(y_valid, (p_valid >= 0.5).astype(int)) if len(y_valid) else np.nan

    metrics = evaluate_probs(y_test, p_test, threshold=threshold)
    metrics_05 = evaluate_probs(y_test, p_test, threshold=0.5)

    row = {
        "variant": variant,
        "model_type": model_type,
        "fold": fold,
        "top_k": top_k,
        "interaction_mode": interaction_mode,
        "n_features": n_features,
        **metrics,
        "accuracy_at_0_5": metrics_05["accuracy"],
        "valid_accuracy_at_threshold": valid_acc,
        "params": json.dumps(params),
    }
    fold_rows.append(row)

    print(
        f"[{variant} | Fold {fold} | K={top_k} | {interaction_mode}] "
        f"logloss={row['logloss']:.4f} "
        f"brier={row['brier']:.4f} "
        f"auc={row['auc']:.4f} "
        f"acc@thr={row['accuracy']:.4f} "
        f"acc@0.5={row['accuracy_at_0_5']:.4f} "
        f"features={n_features}"
    )

    for idx, yy, pp in zip(test_idx, y_test, p_test):
        prediction_rows.append({
            "variant": variant,
            "model_type": model_type,
            "fold": fold,
            "top_k": top_k,
            "interaction_mode": interaction_mode,
            "row_index": int(idx),
            "match_id": df.loc[idx, "match_id"],
            "Date": df.loc[idx, "Date"],
            "Patch": df.loc[idx, "Patch"],
            "y_true": int(yy),
            "p_blue_win": float(pp),
            "threshold": threshold,
            "pred_at_threshold": int(pp >= threshold),
            "pred_at_0_5": int(pp >= 0.5),
        })


def summarize_results(fold_results):
    if fold_results.empty:
        return pd.DataFrame()

    return (
        fold_results
        .groupby(["variant", "model_type", "top_k", "interaction_mode"], as_index=False)
        .agg(
            n_folds=("fold", "count"),
            n_features_mean=("n_features", "mean"),
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


# ============================================================
# 7. EVALUATION
# ============================================================

def evaluate_hardcoded_models(df, target_col, predictor_cols, condition_cols):
    y = df[target_col].astype(int).to_numpy()

    fold_rows = []
    prediction_rows = []
    coef_rows = []
    condition_eval_rows = []
    selected_rows = []

    splits = base.make_rolling_splits(df)

    available_clean = available_clean_conditions(condition_cols)

    for fold, train_idx, valid_idx, test_idx in splits:
        print("\n" + "=" * 100)
        print(f"FOLD {fold}")
        print("=" * 100)

        y_train = y[train_idx]
        y_valid = y[valid_idx]
        y_test = y[test_idx]

        # Same condition-prediction layer as Model 8:
        # 143 pregame features -> predicted condition advantages.
        C_train, C_valid, C_test, cond_eval = base.predict_conditions_for_fold(
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

        cond_eval = cond_eval.copy()
        cond_eval["fold"] = fold
        condition_eval_rows.extend(cond_eval.to_dict("records"))

        # ------------------------------------------------------------
        # Baseline: original 4-index Model 8
        # ------------------------------------------------------------
        I_train, I_valid, I_test = base.build_condition_indices_train_valid_test(
            C_train,
            C_valid,
            C_test,
        )

        p_valid, p_test, model, params = fit_logistic_grid(
            I_train,
            y_train,
            I_valid,
            y_valid,
            I_test,
        )

        append_fold_result(
            fold_rows,
            prediction_rows,
            variant="Baseline_4Index_Model8",
            model_type="four_index_logistic",
            fold=fold,
            top_k=4,
            interaction_mode="none",
            n_features=I_train.shape[1],
            y_valid=y_valid,
            p_valid=p_valid,
            y_test=y_test,
            p_test=p_test,
            test_idx=test_idx,
            df=df,
            params=params,
        )

        save_coef_rows(
            coef_rows,
            "Baseline_4Index_Model8",
            fold,
            model,
            list(I_train.columns),
            params,
            patch="ALL",
        )

        # ------------------------------------------------------------
        # Hard-coded global fallback conditions
        # ------------------------------------------------------------
        for top_k in TOP_K_LIST:
            selected_global = [c for c in GLOBAL_FALLBACK_CONDITIONS if c in available_clean][:top_k]

            for add_interactions in [False, True]:
                interaction_mode = "pairwise_condition_interactions" if add_interactions else "main_effects_only"

                X_train, X_valid, X_test = make_direct_condition_features(
                    C_train,
                    C_valid,
                    C_test,
                    selected_global,
                    add_interactions=add_interactions,
                )

                if X_train.shape[1] == 0:
                    continue

                p_valid, p_test, model, params = fit_logistic_grid(
                    X_train,
                    y_train,
                    X_valid,
                    y_valid,
                    X_test,
                )

                params["selected_conditions"] = selected_global
                params["selection_type"] = "hardcoded_global"

                variant = "HardcodedGlobalConditions"

                append_fold_result(
                    fold_rows,
                    prediction_rows,
                    variant=variant,
                    model_type="global_condition_logistic",
                    fold=fold,
                    top_k=top_k,
                    interaction_mode=interaction_mode,
                    n_features=X_train.shape[1],
                    y_valid=y_valid,
                    p_valid=p_valid,
                    y_test=y_test,
                    p_test=p_test,
                    test_idx=test_idx,
                    df=df,
                    params=params,
                )

                save_coef_rows(
                    coef_rows,
                    variant,
                    fold,
                    model,
                    list(X_train.columns),
                    params,
                    patch="ALL",
                )

            # ------------------------------------------------------------
            # Pooled hard-coded patch-aware model
            # ------------------------------------------------------------
            for add_interactions in [False, True]:
                interaction_mode = (
                    "patch_x_condition_plus_condition_pairwise"
                    if add_interactions
                    else "patch_x_condition"
                )

                X_train, X_valid, X_test, patch_selected, selected_union = make_hardcoded_patch_union_features(
                    df=df,
                    train_idx=train_idx,
                    valid_idx=valid_idx,
                    test_idx=test_idx,
                    C_train=C_train,
                    C_valid=C_valid,
                    C_test=C_test,
                    top_k=top_k,
                    add_interactions=add_interactions,
                )

                if X_train.shape[1] == 0:
                    continue

                p_valid, p_test, model, params = fit_logistic_grid(
                    X_train,
                    y_train,
                    X_valid,
                    y_valid,
                    X_test,
                )

                params["patch_selected_conditions"] = patch_selected
                params["selected_union_conditions"] = selected_union
                params["selection_type"] = "hardcoded_patch"

                variant = "HardcodedPatchAware_Pooled"

                append_fold_result(
                    fold_rows,
                    prediction_rows,
                    variant=variant,
                    model_type="pooled_patch_hardcoded_logistic",
                    fold=fold,
                    top_k=top_k,
                    interaction_mode=interaction_mode,
                    n_features=X_train.shape[1],
                    y_valid=y_valid,
                    p_valid=p_valid,
                    y_test=y_test,
                    p_test=p_test,
                    test_idx=test_idx,
                    df=df,
                    params=params,
                )

                save_coef_rows(
                    coef_rows,
                    variant,
                    fold,
                    model,
                    list(X_train.columns),
                    params,
                    patch="ALL",
                )

                for patch, conds in patch_selected.items():
                    for rank_order, cond in enumerate(conds, start=1):
                        selected_rows.append({
                            "fold": fold,
                            "top_k": top_k,
                            "patch": patch,
                            "rank_order": rank_order,
                            "condition": cond,
                            "actual_col": clean_to_actual_condition(cond),
                            "pred_col": clean_to_pred_condition(cond),
                            "available_in_condition_table": cond in available_clean,
                        })

            # ------------------------------------------------------------
            # Fully separate hard-coded patch models
            # ------------------------------------------------------------
            for add_interactions in [False, True]:
                interaction_mode = "pairwise_condition_interactions" if add_interactions else "main_effects_only"
                variant = "HardcodedSeparatePatch"

                p_valid_all = pd.Series(index=valid_idx, dtype=float)
                p_test_all = pd.Series(index=test_idx, dtype=float)

                train_patches = df.loc[train_idx, "Patch"].astype(str)
                valid_patches = df.loc[valid_idx, "Patch"].astype(str)
                test_patches = df.loc[test_idx, "Patch"].astype(str)

                all_eval_patches = sorted(set(valid_patches.unique()).union(set(test_patches.unique())))

                for patch in all_eval_patches:
                    tr_sub = train_idx[train_patches.eq(patch).to_numpy()]
                    va_sub = valid_idx[valid_patches.eq(patch).to_numpy()]
                    te_sub = test_idx[test_patches.eq(patch).to_numpy()]

                    use_patch_specific = (
                        len(tr_sub) >= MIN_PATCH_TRAIN_N
                        and len(np.unique(y[tr_sub])) >= 2
                    )

                    if use_patch_specific:
                        selected = select_patch_conditions(patch, top_k, available_clean)
                        train_use = tr_sub
                    else:
                        selected = [c for c in GLOBAL_FALLBACK_CONDITIONS if c in available_clean][:top_k]
                        train_use = train_idx

                    if len(va_sub) >= MIN_PATCH_VALID_N and len(np.unique(y[va_sub])) >= 2:
                        valid_use = va_sub
                    else:
                        valid_use = valid_idx

                    if len(te_sub) == 0:
                        continue

                    C_train_use = C_train.loc[train_use]
                    C_valid_use = C_valid.loc[valid_use]
                    C_test_use = C_test.loc[te_sub]

                    X_train, X_valid, X_test = make_direct_condition_features(
                        C_train_use,
                        C_valid_use,
                        C_test_use,
                        selected,
                        add_interactions=add_interactions,
                    )

                    if X_train.shape[1] == 0:
                        continue

                    y_train_use = y[train_use]
                    y_valid_use = y[valid_use]

                    if len(np.unique(y_train_use)) < 2:
                        p_test_all.loc[te_sub] = float(np.mean(y_train))
                        continue

                    if len(np.unique(y_valid_use)) >= 2:
                        p_valid_patch, p_test_patch, model, params = fit_logistic_grid(
                            X_train,
                            y_train_use,
                            X_valid,
                            y_valid_use,
                            X_test,
                        )
                    else:
                        p_valid_patch, p_test_patch, model, params = fit_logistic_fixed_C(
                            X_train,
                            y_train_use,
                            X_valid,
                            y_valid_use,
                            X_test,
                            C=0.1,
                        )

                    params["selected_conditions"] = selected
                    params["patch"] = patch
                    params["used_patch_specific_train"] = bool(use_patch_specific)
                    params["n_train_patch"] = int(len(tr_sub))
                    params["n_valid_patch"] = int(len(va_sub))
                    params["n_test_patch"] = int(len(te_sub))

                    # Predict same-patch validation rows for threshold tuning.
                    if len(va_sub) > 0:
                        C_valid_patch = C_valid.loc[va_sub]
                        _, X_valid_patch, _ = make_direct_condition_features(
                            C_train_use,
                            C_valid_patch,
                            C_test_use.iloc[0:0],
                            selected,
                            add_interactions=add_interactions,
                        )
                        X_valid_patch = X_valid_patch.reindex(columns=X_train.columns, fill_value=0.0)
                        p_valid_all.loc[va_sub] = model.predict_proba(X_valid_patch)[:, 1]

                    p_test_all.loc[te_sub] = p_test_patch

                    save_coef_rows(
                        coef_rows,
                        variant,
                        fold,
                        model,
                        list(X_train.columns),
                        params,
                        patch=patch,
                    )

                prior = float(np.mean(y_train))
                p_valid_all = p_valid_all.fillna(prior)
                p_test_all = p_test_all.fillna(prior)

                append_fold_result(
                    fold_rows,
                    prediction_rows,
                    variant=variant,
                    model_type="separate_patch_hardcoded_logistic",
                    fold=fold,
                    top_k=top_k,
                    interaction_mode=interaction_mode,
                    n_features=np.nan,
                    y_valid=y_valid,
                    p_valid=p_valid_all.loc[valid_idx].to_numpy(),
                    y_test=y_test,
                    p_test=p_test_all.loc[test_idx].to_numpy(),
                    test_idx=test_idx,
                    df=df,
                    params={
                        "top_k": top_k,
                        "interaction_mode": interaction_mode,
                        "fallback_prior_for_missing": prior,
                    },
                )

    return (
        pd.DataFrame(fold_rows),
        pd.DataFrame(prediction_rows),
        pd.DataFrame(coef_rows),
        pd.DataFrame(condition_eval_rows),
        pd.DataFrame(selected_rows),
    )


# ============================================================
# 8. MAIN
# ============================================================

def main():
    print("=" * 100)
    print("MODEL 10: HARD-CODED PATCH-SPECIFIC WIN-CONDITION FINAL LAYERS")
    print("Keeps Model 8 construction unchanged, but tests hard-coded patch condition lists.")
    print("=" * 100)

    # 1. Load and clean raw player CSV.
    df = base.load_and_clean(base.DATA_PATH)

    # 2. Build exact original Model 2 feature block internally.
    print("\n" + "=" * 100)
    print("BUILDING EXACT ORIGINAL MODEL 2 FEATURES")
    print("=" * 100)

    df_feat = base.build_layer1_history_features(df, base.CORE_STATS)

    selected_labels, selection_scores = base.select_layer1_targets(df_feat)
    selection_scores.to_csv(OUT_DIR / "layer1_label_selection_scores.csv", index=False)
    (OUT_DIR / "selected_labels.json").write_text(json.dumps(selected_labels, indent=2))

    df_ridge, layer1_results = base.fit_layer1_ridge_predictions(df_feat, selected_labels)
    layer1_results.to_csv(OUT_DIR / "layer1_ridge_test_results.csv", index=False)

    model2_df = base.build_base_layer2_features(df_ridge, selected_labels)
    model2_df.to_csv(OUT_DIR / "features_A_original_model2_rebuilt.csv", index=False)

    m2_cols = [c for c in model2_df.columns if c.startswith(("delta__", "unc__", "z__"))]

    print(f"\nRebuilt Model 2 dataframe shape: {model2_df.shape}")
    print(f"Rebuilt Model 2 feature count: {len(m2_cols)}")

    # 3. Build extended Model 1 condition targets.
    condition_df, condition_cols = build_extended_model1_conditions(df)

    # 4. Build role-weighted TS features.
    rwts_df, rwts_cols, final_ratings = base.build_rwts_features(df)

    # 5. Patch table.
    patch_df = build_match_patch_table(df)

    # 6. Integrate all final inputs.
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

    integrated = integrated.merge(
        patch_df,
        on="match_id",
        how="left",
    )

    integrated["Patch"] = integrated["Patch"].fillna("UNKNOWN").map(normalize_patch)

    m2_cols = [c for c in m2_cols if c in integrated.columns and pd.api.types.is_numeric_dtype(integrated[c])]
    rwts_cols = [c for c in rwts_cols if c in integrated.columns and pd.api.types.is_numeric_dtype(integrated[c])]
    predictor_cols = list(dict.fromkeys(m2_cols + rwts_cols))

    available_clean = available_clean_conditions(condition_cols)

    print("\n" + "=" * 100)
    print("FINAL INTEGRATED DATA")
    print("=" * 100)
    print(f"Integrated dataframe shape: {integrated.shape}")
    print(f"Patch counts:\n{integrated['Patch'].value_counts(dropna=False).sort_index()}")
    print(f"Rows with condition targets: {integrated[condition_cols].notna().any(axis=1).sum()} / {len(integrated)}")
    print(f"Rows with RWTS features: {integrated[rwts_cols].notna().any(axis=1).sum()} / {len(integrated)}")
    print("\nFeature block counts:")
    print(f"  Exact rebuilt Model 2 features: {len(m2_cols)}")
    print(f"  Role-weighted TS features: {len(rwts_cols)}")
    print(f"  Condition prediction inputs: {len(predictor_cols)}")
    print(f"  Extended Model 1 condition targets: {len(condition_cols)}")
    print(f"  Available clean condition names: {available_clean}")
    print(f"  Hard-coded patch top K: {TOP_K_LIST}")

    print("\nHard-coded patch lists after filtering to available conditions:")
    for patch in sorted(HARDCODED_PATCH_CONDITIONS.keys()):
        for k in TOP_K_LIST:
            print(f"  {patch}, top {k}: {select_patch_conditions(patch, k, available_clean)}")

    # 7. Evaluate hard-coded patch-aware final models.
    fold_results, test_predictions, coefs, condition_eval, selected_rows = evaluate_hardcoded_models(
        integrated,
        "y_blue_win",
        predictor_cols,
        condition_cols,
    )

    cv_summary = summarize_results(fold_results)

    coef_summary = (
        coefs
        .groupby(["variant", "patch", "feature"], as_index=False)
        .agg(
            mean_coef=("coef_standardized", "mean"),
            mean_abs_coef=("abs_coef", "mean"),
            selected_n_folds=("fold", "nunique"),
        )
        .sort_values(["variant", "patch", "mean_abs_coef"], ascending=[True, True, False])
        if not coefs.empty else pd.DataFrame()
    )

    # Optional: compare to previous Model 9 summary if available.
    model9_path = PROJECT_DIR / "model9_patch_aware_win_condition_compare_outputs" / "model9_cv_summary.csv"
    if model9_path.exists():
        previous_model9 = pd.read_csv(model9_path)
        previous_model9["source"] = "model9_previous_ranked_patch"
        cv_summary_with_previous = pd.concat(
            [cv_summary.assign(source="model10_hardcoded_patch"), previous_model9],
            ignore_index=True,
            sort=False,
        )
    else:
        cv_summary_with_previous = cv_summary.assign(source="model10_hardcoded_patch")

    # 8. Save outputs.
    integrated.to_csv(OUT_DIR / "model10_integrated_features.csv", index=False)
    final_ratings.to_csv(OUT_DIR / "model10_rwts_player_ratings.csv", index=False)

    fold_results.to_csv(OUT_DIR / "model10_fold_results.csv", index=False)
    test_predictions.to_csv(OUT_DIR / "model10_test_predictions.csv", index=False)
    cv_summary.to_csv(OUT_DIR / "model10_cv_summary.csv", index=False)
    cv_summary_with_previous.to_csv(OUT_DIR / "model10_cv_summary_with_model9_previous.csv", index=False)

    coefs.to_csv(OUT_DIR / "model10_logistic_coefficients.csv", index=False)
    coef_summary.to_csv(OUT_DIR / "model10_logistic_coefficient_summary.csv", index=False)

    condition_eval.to_csv(OUT_DIR / "model10_condition_prediction_results.csv", index=False)
    selected_rows.to_csv(OUT_DIR / "model10_hardcoded_selected_conditions_by_fold.csv", index=False)

    with open(OUT_DIR / "model10_feature_lists_and_settings.json", "w") as f:
        json.dump(
            {
                "required_input": str(base.DATA_PATH.name),
                "model2_selected_labels": selected_labels,
                "m2_cols": m2_cols,
                "rwts_cols": rwts_cols,
                "predictor_cols": predictor_cols,
                "condition_cols": condition_cols,
                "available_clean_conditions": available_clean,
                "top_k_list": TOP_K_LIST,
                "hardcoded_patch_conditions": HARDCODED_PATCH_CONDITIONS,
                "global_fallback_conditions": GLOBAL_FALLBACK_CONDITIONS,
                "role_weights": base.ROLE_WEIGHTS,
                "ts_settings": {
                    "TS_INIT_MU": base.TS_INIT_MU,
                    "TS_INIT_SIGMA": base.TS_INIT_SIGMA,
                    "TS_SIGMA_MIN": base.TS_SIGMA_MIN,
                    "TS_K": base.TS_K,
                    "TS_SIGMA_DECAY": base.TS_SIGMA_DECAY,
                    "TS_SCALE": base.TS_SCALE,
                },
                "final_split_settings": {
                    "N_FOLDS": base.N_FOLDS,
                    "VALID_SIZE": base.VALID_SIZE,
                    "TEST_SIZE": base.TEST_SIZE,
                },
            },
            f,
            indent=2,
        )

    print("\n" + "=" * 100)
    print("MODEL 10 HARD-CODED PATCH-AWARE CV SUMMARY")
    print(cv_summary.to_string(index=False))

    if model9_path.exists():
        print("\n" + "=" * 100)
        print("COMBINED MODEL 10 + PREVIOUS MODEL 9 SUMMARY SAVED")
        print("  model10_cv_summary_with_model9_previous.csv")

    print("\nSaved outputs to:", OUT_DIR)
    print("Key files:")
    print("  model10_cv_summary.csv")
    print("  model10_cv_summary_with_model9_previous.csv")
    print("  model10_fold_results.csv")
    print("  model10_test_predictions.csv")
    print("  model10_hardcoded_selected_conditions_by_fold.csv")
    print("  model10_logistic_coefficients.csv")
    print("  model10_logistic_coefficient_summary.csv")
    print("  model10_integrated_features.csv")
    print("  model10_feature_lists_and_settings.json")


if __name__ == "__main__":
    main()