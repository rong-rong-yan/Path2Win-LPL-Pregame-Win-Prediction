from __future__ import annotations

from pathlib import Path
import json
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
# Import previous full construction pipeline
# ============================================================
# This script keeps your previous construction part the same:
# raw data -> exact Model 2 features -> Model 1 conditions
# -> role-weighted TS features -> predicted conditions.
#
# Only the final logistic stage is replaced / extended.

import model8_final_FULL_END_TO_END_m1_m2_rwts_indices_logistic_v2 as base


# ============================================================
# 0. CONFIG
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
OUT_DIR = PROJECT_DIR / "model9_patch_aware_win_condition_compare_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Compare different numbers of final win conditions.
TOP_K_LIST = [3, 5, 7, 10, 12]

# Final logistic C grid.
LOGISTIC_C_GRID = base.LOGISTIC_C_GRID

# If True, tune threshold using validation accuracy.
TUNE_THRESHOLD_FOR_ACCURACY = True

# Minimum data requirements for fitting separate patch model.
MIN_PATCH_TRAIN_N = 20
MIN_PATCH_VALID_N = 8

# Ranking method for patch-specific condition importance.
# "coef" = fit L1 logistic on actual condition values in training rows.
# fallback is abs correlation if L1 fails or all coefficients are zero.
PATCH_RANKING_METHOD = "coef"

RANDOM_SEED = base.RANDOM_SEED


# ============================================================
# 1. Basic metric helpers
# ============================================================

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
# 2. Patch extraction
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

    # If multiple patch values appear inside one match, take the first sorted value.
    patch_df = (
        patch_df
        .sort_values(["match_id", base.PATCH_COL])
        .groupby("match_id", as_index=False)
        .agg(Patch=(base.PATCH_COL, "first"))
    )
    return patch_df


# ============================================================
# 3. Condition ranking
# ============================================================

def actual_condition_to_pred_col(cond):
    # actual_delta_GD15_sum -> pred_delta_GD15_sum
    return "pred_" + cond.replace("actual_", "")


def pred_col_to_clean_name(pred_col):
    # pred_delta_GD15_sum -> GD15_sum
    return pred_col.replace("pred_delta_", "")


def abs_corr_rank(df_train, y_train, condition_cols):
    rows = []

    for c in condition_cols:
        x = pd.to_numeric(df_train[c], errors="coerce")
        y = pd.Series(y_train, index=df_train.index).astype(float)
        mask = x.notna() & y.notna()

        if mask.sum() < 5:
            score = 0.0
        elif x.loc[mask].std(ddof=0) < 1e-12 or y.loc[mask].std(ddof=0) < 1e-12:
            score = 0.0
        else:
            score = abs(float(np.corrcoef(x.loc[mask], y.loc[mask])[0, 1]))

        rows.append({"condition": c, "score": score, "ranking_method": "abs_corr"})

    return pd.DataFrame(rows).sort_values("score", ascending=False)


def coef_rank_conditions(df_train, y_train, condition_cols):
    usable = []
    X = pd.DataFrame(index=df_train.index)

    for c in condition_cols:
        x = pd.to_numeric(df_train[c], errors="coerce")
        if x.notna().sum() >= 10 and x.nunique(dropna=True) >= 2:
            usable.append(c)
            X[c] = x

    y_train = np.asarray(y_train).astype(int)

    if len(usable) < 2 or len(np.unique(y_train)) < 2:
        return abs_corr_rank(df_train, y_train, condition_cols)

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="l1",
            solver="liblinear",
            C=0.3,
            max_iter=5000,
            random_state=RANDOM_SEED,
        )),
    ])

    try:
        model.fit(X[usable], y_train)
        coefs = model.named_steps["clf"].coef_[0]
        rows = []
        for c, coef in zip(usable, coefs):
            rows.append({
                "condition": c,
                "score": abs(float(coef)),
                "coef": float(coef),
                "ranking_method": "l1_logistic_abs_coef",
            })

        rank_df = pd.DataFrame(rows)

        # If L1 shrinks everything to zero, fallback to correlation.
        if rank_df["score"].max() <= 1e-12:
            return abs_corr_rank(df_train, y_train, condition_cols)

        # Include any dropped conditions with zero score so top-K logic is stable.
        missing = [c for c in condition_cols if c not in set(rank_df["condition"])]
        if missing:
            rank_df = pd.concat([
                rank_df,
                pd.DataFrame({
                    "condition": missing,
                    "score": 0.0,
                    "coef": 0.0,
                    "ranking_method": "l1_logistic_abs_coef_missing",
                }),
            ], ignore_index=True)

        return rank_df.sort_values("score", ascending=False)

    except Exception:
        return abs_corr_rank(df_train, y_train, condition_cols)


def rank_conditions_by_patch(df, train_idx, target_col, condition_cols):
    """
    Rank actual condition columns using training rows only.

    This is an oracle-style ranking step, but it is restricted to training rows
    inside each fold. The final predictive model still uses predicted conditions,
    not actual postgame conditions.
    """
    train_df = df.loc[train_idx].copy()
    y_train = train_df[target_col].astype(int).to_numpy()

    global_rank = coef_rank_conditions(train_df, y_train, condition_cols)

    patch_rankings = {}
    patch_values = sorted(train_df["Patch"].dropna().unique().tolist())

    for patch in patch_values:
        sub = train_df[train_df["Patch"].eq(patch)].copy()

        if len(sub) < MIN_PATCH_TRAIN_N or sub[target_col].nunique() < 2:
            patch_rankings[patch] = global_rank.copy()
            patch_rankings[patch]["patch"] = patch
            patch_rankings[patch]["used_fallback_global"] = True
            continue

        y_sub = sub[target_col].astype(int).to_numpy()
        rnk = coef_rank_conditions(sub, y_sub, condition_cols)
        rnk["patch"] = patch
        rnk["used_fallback_global"] = False
        patch_rankings[patch] = rnk

    # Global fallback for unseen patches in valid/test.
    global_rank = global_rank.copy()
    global_rank["patch"] = "GLOBAL_FALLBACK"
    global_rank["used_fallback_global"] = False

    return patch_rankings, global_rank


def get_top_conditions_for_patch(patch_rankings, global_rank, patch, top_k):
    if patch in patch_rankings:
        rnk = patch_rankings[patch]
    else:
        rnk = global_rank

    conds = rnk["condition"].drop_duplicates().tolist()
    return conds[:min(top_k, len(conds))]


# ============================================================
# 4. Final feature construction from predicted conditions
# ============================================================

def zscore_pred_cols_with_train(C_train, C_valid, C_test, pred_cols):
    """
    Standardize predicted condition columns using training-fold mean/sd.
    """
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


def make_direct_condition_features(C_train, C_valid, C_test, selected_conditions, add_interactions=False):
    pred_cols = [actual_condition_to_pred_col(c) for c in selected_conditions]
    pred_cols = [c for c in pred_cols if c in C_train.columns]

    X_train, X_valid, X_test = zscore_pred_cols_with_train(C_train, C_valid, C_test, pred_cols)

    if add_interactions:
        X_train = add_pairwise_interactions(X_train)
        X_valid = add_pairwise_interactions(X_valid)
        X_test = add_pairwise_interactions(X_test)

    return X_train, X_valid, X_test


def build_four_index_features(C_train, C_valid, C_test):
    """
    Reuse Model 8's current four-index construction.
    """
    I_train, I_valid, I_test = base.build_condition_indices_train_valid_test(C_train, C_valid, C_test)
    return I_train, I_valid, I_test


def add_patch_dummies(X_train, X_valid, X_test, patch_train, patch_valid, patch_test):
    train_dum = pd.get_dummies(patch_train.astype(str), prefix="patch")
    valid_dum = pd.get_dummies(patch_valid.astype(str), prefix="patch")
    test_dum = pd.get_dummies(patch_test.astype(str), prefix="patch")

    cols = sorted(train_dum.columns.tolist())

    train_dum = train_dum.reindex(columns=cols, fill_value=0)
    valid_dum = valid_dum.reindex(columns=cols, fill_value=0)
    test_dum = test_dum.reindex(columns=cols, fill_value=0)

    train_dum.index = X_train.index
    valid_dum.index = X_valid.index
    test_dum.index = X_test.index

    return (
        pd.concat([X_train, train_dum], axis=1),
        pd.concat([X_valid, valid_dum], axis=1),
        pd.concat([X_test, test_dum], axis=1),
    )


def make_pooled_patch_interaction_features(
    df,
    train_idx,
    valid_idx,
    test_idx,
    C_train,
    C_valid,
    C_test,
    patch_rankings,
    global_rank,
    top_k,
    condition_cols,
    add_condition_interactions=False,
):
    """
    One pooled model with patch indicators and patch x condition terms.

    It keeps all data together but lets condition effects vary by patch.
    """
    train_patches = df.loc[train_idx, "Patch"].astype(str)
    valid_patches = df.loc[valid_idx, "Patch"].astype(str)
    test_patches = df.loc[test_idx, "Patch"].astype(str)

    # Union of selected conditions across training patches.
    selected_union = []
    for patch in sorted(train_patches.unique().tolist()):
        selected_union.extend(get_top_conditions_for_patch(patch_rankings, global_rank, patch, top_k))

    # Fallback if no patch-specific selection works.
    if not selected_union:
        selected_union = global_rank["condition"].tolist()[:top_k]

    selected_union = list(dict.fromkeys(selected_union))
    pred_cols = [actual_condition_to_pred_col(c) for c in selected_union]
    pred_cols = [c for c in pred_cols if c in C_train.columns]

    X_train_base, X_valid_base, X_test_base = zscore_pred_cols_with_train(
        C_train,
        C_valid,
        C_test,
        pred_cols,
    )

    # Optional global condition-condition interactions.
    if add_condition_interactions:
        X_train_base = add_pairwise_interactions(X_train_base)
        X_valid_base = add_pairwise_interactions(X_valid_base)
        X_test_base = add_pairwise_interactions(X_test_base)

    # Add patch dummies.
    X_train, X_valid, X_test = add_patch_dummies(
        X_train_base,
        X_valid_base,
        X_test_base,
        train_patches,
        valid_patches,
        test_patches,
    )

    # Add patch × condition interactions for selected condition columns.
    train_patch_dummies = [c for c in X_train.columns if c.startswith("patch_")]
    condition_feature_cols = [c for c in X_train_base.columns]

    for pcol in train_patch_dummies:
        p_valid = pcol if pcol in X_valid.columns else None
        p_test = pcol if pcol in X_test.columns else None

        for c in condition_feature_cols:
            new_col = f"{pcol}_x_{c}"

            X_train[new_col] = X_train[pcol] * X_train[c]

            if p_valid is not None:
                X_valid[new_col] = X_valid[pcol] * X_valid[c]
            else:
                X_valid[new_col] = 0.0

            if p_test is not None:
                X_test[new_col] = X_test[pcol] * X_test[c]
            else:
                X_test[new_col] = 0.0

    # Align columns.
    X_valid = X_valid.reindex(columns=X_train.columns, fill_value=0.0)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0.0)

    return X_train, X_valid, X_test, selected_union


# ============================================================
# 5. Logistic fitting
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
# 6. Evaluation variants
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


def evaluate_patch_aware_models(df, target_col, predictor_cols, condition_cols):
    y = df[target_col].astype(int).to_numpy()

    fold_rows = []
    prediction_rows = []
    coef_rows = []
    selection_rows = []
    condition_eval_rows = []

    splits = base.make_rolling_splits(df)

    for fold, train_idx, valid_idx, test_idx in splits:
        print("\n" + "=" * 100)
        print(f"FOLD {fold}")
        print("=" * 100)

        y_train = y[train_idx]
        y_valid = y[valid_idx]
        y_test = y[test_idx]

        # Same condition-prediction layer as Model 8.
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

        # Patch-specific rankings use actual conditions in training only.
        patch_rankings, global_rank = rank_conditions_by_patch(
            df,
            train_idx,
            target_col,
            condition_cols,
        )

        # Save ranking rows.
        for patch, rnk in patch_rankings.items():
            tmp = rnk.copy()
            tmp["fold"] = fold
            tmp["ranking_scope"] = "patch"
            selection_rows.extend(tmp.to_dict("records"))

        tmp = global_rank.copy()
        tmp["fold"] = fold
        tmp["ranking_scope"] = "global_fallback"
        selection_rows.extend(tmp.to_dict("records"))

        # ------------------------------------------------------------
        # Baseline: current Model 8 four-index logistic.
        # ------------------------------------------------------------
        I_train, I_valid, I_test = build_four_index_features(C_train, C_valid, C_test)

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
        # Global direct condition model:
        # top K from global training ranking; no patch-specific effect.
        # ------------------------------------------------------------
        for top_k in TOP_K_LIST:
            selected_global = global_rank["condition"].drop_duplicates().tolist()[:top_k]

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
                params["ranking"] = "global_train_only"

                variant = "GlobalTopK_PredConditions"

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
            # Pooled patch-aware model:
            # one model, patch dummies, patch x condition interactions.
            # ------------------------------------------------------------
            for add_condition_interactions in [False, True]:
                interaction_mode = (
                    "patch_x_condition_plus_condition_pairwise"
                    if add_condition_interactions
                    else "patch_x_condition"
                )

                X_train, X_valid, X_test, selected_union = make_pooled_patch_interaction_features(
                    df=df,
                    train_idx=train_idx,
                    valid_idx=valid_idx,
                    test_idx=test_idx,
                    C_train=C_train,
                    C_valid=C_valid,
                    C_test=C_test,
                    patch_rankings=patch_rankings,
                    global_rank=global_rank,
                    top_k=top_k,
                    condition_cols=condition_cols,
                    add_condition_interactions=add_condition_interactions,
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

                params["selected_union_conditions"] = selected_union
                params["ranking"] = "patch_specific_train_only"

                variant = "PooledPatchAware_TopK"

                append_fold_result(
                    fold_rows,
                    prediction_rows,
                    variant=variant,
                    model_type="pooled_patch_interaction_logistic",
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
            # Fully separate patch models:
            # one logistic model per patch.
            # ------------------------------------------------------------
            for add_interactions in [False, True]:
                interaction_mode = "pairwise_condition_interactions" if add_interactions else "main_effects_only"
                variant = "SeparatePatch_TopK"

                p_valid_all = pd.Series(index=valid_idx, dtype=float)
                p_test_all = pd.Series(index=test_idx, dtype=float)

                valid_patches = df.loc[valid_idx, "Patch"].astype(str)
                test_patches = df.loc[test_idx, "Patch"].astype(str)
                train_patches = df.loc[train_idx, "Patch"].astype(str)

                all_eval_patches = sorted(set(valid_patches.unique()).union(set(test_patches.unique())))

                for patch in all_eval_patches:
                    tr_sub = train_idx[train_patches.eq(patch).to_numpy()]
                    va_sub = valid_idx[valid_patches.eq(patch).to_numpy()]
                    te_sub = test_idx[test_patches.eq(patch).to_numpy()]

                    # If patch is unseen or too small, fallback to global selected conditions
                    # and global training rows. This keeps predictions available for all test rows.
                    use_patch_specific = (
                        len(tr_sub) >= MIN_PATCH_TRAIN_N
                        and len(np.unique(y[tr_sub])) >= 2
                    )

                    if use_patch_specific:
                        selected = get_top_conditions_for_patch(patch_rankings, global_rank, patch, top_k)
                        train_use = tr_sub
                    else:
                        selected = global_rank["condition"].drop_duplicates().tolist()[:top_k]
                        train_use = train_idx

                    # Validation for C tuning:
                    # If same-patch validation is too small or one-class, use all validation.
                    if len(va_sub) >= MIN_PATCH_VALID_N and len(np.unique(y[va_sub])) >= 2:
                        valid_use = va_sub
                    else:
                        valid_use = valid_idx

                    # Build feature matrices for selected rows.
                    C_train_use = C_train.loc[train_use]
                    C_valid_use = C_valid.loc[valid_use]
                    C_test_use = C_test.loc[te_sub]

                    if len(te_sub) == 0:
                        continue

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
                        # Cannot train logistic with one class.
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

                    # Store valid predictions only for validation rows in this patch.
                    # If valid_use was all validation, only fill same-patch valid rows when possible.
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

                # If some valid/test predictions are missing, fill with train prior.
                prior = float(np.mean(y_train))
                p_valid_all = p_valid_all.fillna(prior)
                p_test_all = p_test_all.fillna(prior)

                append_fold_result(
                    fold_rows,
                    prediction_rows,
                    variant=variant,
                    model_type="separate_patch_logistic",
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
        pd.DataFrame(selection_rows),
        pd.DataFrame(condition_eval_rows),
    )


# ============================================================
# 7. Main construction, same as previous script
# ============================================================

def main():
    print("=" * 100)
    print("MODEL 9: PATCH-AWARE WIN-CONDITION FINAL LAYERS")
    print("Keeps Model 8 construction unchanged; compares final logistic variants.")
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

    # 3. Build Model 1 condition targets.
    condition_df, condition_cols = base.build_model1_conditions(df)

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

    print("\n" + "=" * 100)
    print("FINAL INTEGRATED DATA")
    print("=" * 100)
    print(f"Integrated dataframe shape: {integrated.shape}")
    print(f"Patch counts:\n{integrated['Patch'].value_counts(dropna=False).sort_index()}")
    print(f"Rows with Model 1 condition targets: {integrated[condition_cols].notna().any(axis=1).sum()} / {len(integrated)}")
    print(f"Rows with RWTS features: {integrated[rwts_cols].notna().any(axis=1).sum()} / {len(integrated)}")
    print("\nFeature block counts:")
    print(f"  Exact rebuilt Model 2 features: {len(m2_cols)}")
    print(f"  Role-weighted TS features: {len(rwts_cols)}")
    print(f"  Condition prediction inputs: {len(predictor_cols)}")
    print(f"  Model 1 condition targets: {len(condition_cols)}")
    print(f"  Patch-aware final models compare top K: {TOP_K_LIST}")

    # 7. Evaluate patch-aware final models.
    fold_results, test_predictions, coefs, selection_rows, condition_eval = evaluate_patch_aware_models(
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

    # 8. Save outputs.
    integrated.to_csv(OUT_DIR / "model9_integrated_features.csv", index=False)
    final_ratings.to_csv(OUT_DIR / "model9_rwts_player_ratings.csv", index=False)

    fold_results.to_csv(OUT_DIR / "model9_fold_results.csv", index=False)
    test_predictions.to_csv(OUT_DIR / "model9_test_predictions.csv", index=False)
    cv_summary.to_csv(OUT_DIR / "model9_cv_summary.csv", index=False)
    coefs.to_csv(OUT_DIR / "model9_logistic_coefficients.csv", index=False)
    coef_summary.to_csv(OUT_DIR / "model9_logistic_coefficient_summary.csv", index=False)
    selection_rows.to_csv(OUT_DIR / "model9_patch_condition_rankings.csv", index=False)
    condition_eval.to_csv(OUT_DIR / "model9_condition_prediction_results.csv", index=False)

    with open(OUT_DIR / "model9_feature_lists_and_settings.json", "w") as f:
        json.dump(
            {
                "required_input": str(base.DATA_PATH.name),
                "model2_selected_labels": selected_labels,
                "m2_cols": m2_cols,
                "rwts_cols": rwts_cols,
                "predictor_cols": predictor_cols,
                "condition_cols": condition_cols,
                "top_k_list": TOP_K_LIST,
                "patch_ranking_method": PATCH_RANKING_METHOD,
                "min_patch_train_n": MIN_PATCH_TRAIN_N,
                "min_patch_valid_n": MIN_PATCH_VALID_N,
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
    print("MODEL 9 PATCH-AWARE CV SUMMARY")
    print(cv_summary.to_string(index=False))

    print("\nSaved outputs to:", OUT_DIR)
    print("Key files:")
    print("  model9_cv_summary.csv")
    print("  model9_fold_results.csv")
    print("  model9_test_predictions.csv")
    print("  model9_patch_condition_rankings.csv")
    print("  model9_logistic_coefficients.csv")
    print("  model9_logistic_coefficient_summary.csv")
    print("  model9_integrated_features.csv")
    print("  model9_feature_lists_and_settings.json")


if __name__ == "__main__":
    main()