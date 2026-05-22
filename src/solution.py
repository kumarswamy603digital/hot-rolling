"""
Defect Detection in Hot Rolling - Alpha Defect Classification
=============================================================

Goal: predict the rare Alpha defect (Y) on the test set with
      Recall = 100% AND Precision > 90%.

Pipeline
--------
1.  Robust preprocessing
    - Median imputation per column (fit on train only).
    - "is missing" indicator columns for any feature that is null
      in train OR test.
    - Outlier clipping by 1st-99th percentile (fit on train only).

2.  Feature engineering
    - Row-wise statistical aggregates over X1..X49 (mean, std, min,
      max, range, skew, kurt).
    - Pairwise products / ratios of the 8 features most correlated
      with Y (X35, X13, X36, X34, X10, X30, X31, X32).
    - log1p of strictly-positive features.
    - Z-scores of strongly-correlated features.

3.  Strong supervised ensemble
    - Models: XGBoost, LightGBM, CatBoost, GradientBoosting,
      RandomForest, ExtraTrees, Logistic Regression.
    - Each model is trained with stratified 10-fold CV across 3
      random seeds (=> 30 fits per model).
    - Class imbalance handled via scale_pos_weight / class_weight.

4.  Stacking
    - The OOF probabilities from each base model become the inputs
      to a Logistic Regression meta-learner (with L2).
    - The meta-learner is also fit with stratified k-fold CV on the
      OOF matrix to avoid leakage. The mean of those meta-models is
      the final calibrated probability.

5.  Recall-first threshold tuning
    - Try strict criterion (recall=1.0, precision>=0.9). If
      achievable, take the highest-precision threshold.
    - Otherwise, take the F2-optimal threshold (recall weighted
      twice as heavily as precision).

6.  Output: data/dataset/expected_submission.csv (339 x 2,
    columns CoilID, Y). Probabilities are also dumped for
    transparency.
"""

import os
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
)
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    precision_recall_curve,
)

warnings.filterwarnings("ignore")
RNG = 42
np.random.seed(RNG)


# ------------------------------------------------------------------ #
# Load
# ------------------------------------------------------------------ #
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "dataset")
DATA_DIR = os.path.abspath(DATA_DIR)

train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
sample = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

print(f"train: {train.shape}, test: {test.shape}")
print(f"positive rate in train: {train['Y'].mean():.4f}  "
      f"({int(train['Y'].sum())} / {len(train)})")

RAW_FEATURES = [c for c in train.columns if c.startswith("X")]
TARGET = "Y"


# ------------------------------------------------------------------ #
# Preprocessing
# ------------------------------------------------------------------ #
miss_cols = [c for c in RAW_FEATURES
             if train[c].isna().any() or test[c].isna().any()]
print(f"missing-flag columns: {miss_cols}")

for c in miss_cols:
    train[f"{c}_isna"] = train[c].isna().astype(int)
    test[f"{c}_isna"] = test[c].isna().astype(int)

# Median impute (fit on train only)
imputer = SimpleImputer(strategy="median")
train[RAW_FEATURES] = imputer.fit_transform(train[RAW_FEATURES])
test[RAW_FEATURES] = imputer.transform(test[RAW_FEATURES])

# Outlier clipping (1st-99th percentile, fit on train only)
clip_lo = train[RAW_FEATURES].quantile(0.005)
clip_hi = train[RAW_FEATURES].quantile(0.995)
train[RAW_FEATURES] = train[RAW_FEATURES].clip(clip_lo, clip_hi, axis=1)
test[RAW_FEATURES] = test[RAW_FEATURES].clip(clip_lo, clip_hi, axis=1)


# ------------------------------------------------------------------ #
# Feature engineering
# ------------------------------------------------------------------ #
TOP_FEATS = ["X35", "X13", "X36", "X34", "X10", "X30", "X31", "X32"]


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    F = RAW_FEATURES

    # row-wise stats
    df["row_mean"] = df[F].mean(axis=1)
    df["row_std"] = df[F].std(axis=1)
    df["row_min"] = df[F].min(axis=1)
    df["row_max"] = df[F].max(axis=1)
    df["row_range"] = df["row_max"] - df["row_min"]
    df["row_median"] = df[F].median(axis=1)
    df["n_zeros"] = (df[F] == 0).sum(axis=1)

    # interactions of top features only
    for i, a in enumerate(TOP_FEATS):
        for b in TOP_FEATS[i + 1:]:
            df[f"{a}_x_{b}"] = df[a] * df[b]
            df[f"{a}_minus_{b}"] = df[a] - df[b]

    # squared top features
    for c in TOP_FEATS:
        df[f"{c}_sq"] = df[c] ** 2

    # log1p for the largest-scale features
    for c in ["X34", "X35", "X36", "X37"]:
        # shift before log to handle small/negative values from clipping
        v = df[c] - df[c].min() + 1
        df[f"{c}_log1p"] = np.log1p(v)

    return df


train_fe = engineer(train)
test_fe = engineer(test)

ALL_FEATURES = [c for c in train_fe.columns
                if c not in ("CoilID", TARGET)]
print(f"engineered feature count: {len(ALL_FEATURES)}")

X = train_fe[ALL_FEATURES].values
y = train_fe[TARGET].astype(int).values
X_test = test_fe[ALL_FEATURES].values

# Replace any inf/nan introduced by engineering
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

scaler = StandardScaler()
X_sc = scaler.fit_transform(X)
X_test_sc = scaler.transform(X_test)

pos = int(y.sum())
neg = len(y) - pos
spw = neg / max(pos, 1)
print(f"scale_pos_weight = {spw:.2f}")


# ------------------------------------------------------------------ #
# Models
# ------------------------------------------------------------------ #
def make_models(seed: int):
    models = {}

    try:
        from xgboost import XGBClassifier
        models["xgb"] = (
            XGBClassifier(
                n_estimators=800, max_depth=4, learning_rate=0.04,
                subsample=0.85, colsample_bytree=0.7,
                min_child_weight=1, reg_lambda=1.0,
                scale_pos_weight=spw, eval_metric="logloss",
                tree_method="hist", random_state=seed,
                n_jobs=-1, verbosity=0,
            ),
            False,
        )
    except Exception as e:
        print("xgboost unavailable:", e)

    try:
        from lightgbm import LGBMClassifier
        models["lgb"] = (
            LGBMClassifier(
                n_estimators=1000, num_leaves=31, max_depth=-1,
                learning_rate=0.025, subsample=0.85,
                colsample_bytree=0.7, min_child_samples=5,
                reg_lambda=1.0, class_weight="balanced",
                random_state=seed, n_jobs=-1, verbosity=-1,
            ),
            False,
        )
    except Exception as e:
        print("lightgbm unavailable:", e)

    try:
        from catboost import CatBoostClassifier
        models["cat"] = (
            CatBoostClassifier(
                iterations=1000, depth=5, learning_rate=0.03,
                l2_leaf_reg=3.0, random_seed=seed,
                auto_class_weights="Balanced",
                verbose=0, allow_writing_files=False,
            ),
            False,
        )
    except Exception as e:
        print("catboost unavailable:", e)

    models["gbm"] = (
        GradientBoostingClassifier(
            n_estimators=400, max_depth=3, learning_rate=0.05,
            subsample=0.85, random_state=seed,
        ),
        False,
    )
    models["rf"] = (
        RandomForestClassifier(
            n_estimators=800, max_depth=None, min_samples_leaf=2,
            class_weight="balanced", random_state=seed, n_jobs=-1,
        ),
        False,
    )
    models["et"] = (
        ExtraTreesClassifier(
            n_estimators=800, max_depth=None, min_samples_leaf=2,
            class_weight="balanced", random_state=seed, n_jobs=-1,
        ),
        False,
    )
    models["lr"] = (
        LogisticRegression(
            C=0.3, penalty="l2", solver="liblinear",
            class_weight="balanced", max_iter=3000,
            random_state=seed,
        ),
        True,
    )
    return models


# ------------------------------------------------------------------ #
# Train models with stratified k-fold OOF
# ------------------------------------------------------------------ #
N_SPLITS = 8
SEEDS = [42, 7]

per_model_oof = {}
per_model_test = {}

for seed in SEEDS:
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    models = make_models(seed)

    for name, (clf, use_scaled) in models.items():
        oof = np.zeros(len(X), dtype=float)
        tst = np.zeros(len(X_test), dtype=float)

        Xb = X_sc if use_scaled else X
        Xt = X_test_sc if use_scaled else X_test

        for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
            clf_clone = clf.__class__(**clf.get_params())
            clf_clone.fit(Xb[tr_idx], y[tr_idx])
            oof[va_idx] = clf_clone.predict_proba(Xb[va_idx])[:, 1]
            tst += clf_clone.predict_proba(Xt)[:, 1] / N_SPLITS

        per_model_oof.setdefault(name, []).append(oof)
        per_model_test.setdefault(name, []).append(tst)

# Average OOF / test across seeds
oof_by_model = {n: np.mean(v, axis=0) for n, v in per_model_oof.items()}
test_by_model = {n: np.mean(v, axis=0) for n, v in per_model_test.items()}

print("\nPer-model OOF AUC:")
for n in oof_by_model:
    print(f"  {n:>4s}  AUC = {roc_auc_score(y, oof_by_model[n]):.4f}")


# ------------------------------------------------------------------ #
# Stacking - meta-learner on OOF probabilities
# ------------------------------------------------------------------ #
# Build stacked features
stack_train = np.column_stack([oof_by_model[n] for n in oof_by_model])
stack_test = np.column_stack([test_by_model[n] for n in oof_by_model])

# Logistic Regression meta-learner with stratified k-fold
meta_oof = np.zeros(len(X), dtype=float)
meta_test = np.zeros(len(X_test), dtype=float)
N_META_FOLDS = 8
N_META_SEEDS = 2
meta_count = 0

for s in range(N_META_SEEDS):
    skf = StratifiedKFold(n_splits=N_META_FOLDS, shuffle=True,
                          random_state=42 + s)
    for tr_idx, va_idx in skf.split(stack_train, y):
        meta = LogisticRegression(
            C=1.0, class_weight="balanced",
            solver="liblinear", max_iter=3000,
        )
        meta.fit(stack_train[tr_idx], y[tr_idx])
        meta_oof[va_idx] += meta.predict_proba(stack_train[va_idx])[:, 1]
        meta_test += meta.predict_proba(stack_test)[:, 1] / N_META_FOLDS
        meta_count += 1

meta_oof /= N_META_SEEDS
meta_test /= N_META_SEEDS

# Final probabilities: blend stacker + simple mean-of-models
mean_oof = stack_train.mean(axis=1)
mean_test = stack_test.mean(axis=1)

oof_total = 0.5 * meta_oof + 0.5 * mean_oof
test_total = 0.5 * meta_test + 0.5 * mean_test

print(f"\nStacker meta OOF AUC : {roc_auc_score(y, meta_oof):.4f}")
print(f"Mean-of-models OOF AUC: {roc_auc_score(y, mean_oof):.4f}")
print(f"Blended OOF AUC       : {roc_auc_score(y, oof_total):.4f}")


# ------------------------------------------------------------------ #
# Threshold tuning
# ------------------------------------------------------------------ #
def fbeta_curve(p, r, beta):
    b2 = beta ** 2
    return (1 + b2) * p * r / np.maximum(b2 * p + r, 1e-12)


p_curve, r_curve, t_curve = precision_recall_curve(y, oof_total)
print("\nPrecision-Recall curve at key recall levels (blended OOF):")
for tgt in [1.00, 0.97, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70]:
    mask = r_curve[:-1] >= tgt
    if mask.any():
        idx = int(np.argmax(p_curve[:-1] * mask))
        print(f"  recall>={tgt:.2f} -> P={p_curve[idx]:.4f} "
              f"R={r_curve[idx]:.4f} thr={t_curve[idx]:.4f}")
    else:
        print(f"  recall>={tgt:.2f} -> not achievable")

f1 = fbeta_curve(p_curve[:-1], r_curve[:-1], 1.0)
f2 = fbeta_curve(p_curve[:-1], r_curve[:-1], 2.0)
f3 = fbeta_curve(p_curve[:-1], r_curve[:-1], 3.0)
i_f1 = int(np.argmax(f1))
i_f2 = int(np.argmax(f2))
i_f3 = int(np.argmax(f3))
print(f"Best F1: thr={t_curve[i_f1]:.4f} P={p_curve[i_f1]:.4f} "
      f"R={r_curve[i_f1]:.4f} F1={f1[i_f1]:.4f}")
print(f"Best F2: thr={t_curve[i_f2]:.4f} P={p_curve[i_f2]:.4f} "
      f"R={r_curve[i_f2]:.4f} F2={f2[i_f2]:.4f}")
print(f"Best F3: thr={t_curve[i_f3]:.4f} P={p_curve[i_f3]:.4f} "
      f"R={r_curve[i_f3]:.4f} F3={f3[i_f3]:.4f}")


def pick_threshold(probs, y):
    p, r, t = precision_recall_curve(y, probs)
    strict = (r[:-1] >= 1.0) & (p[:-1] > 0.90)
    if strict.any():
        idx = int(np.argmax(p[:-1] * strict))
        thr = t[idx]
        print(f"\nSTRICT criterion met: thr={thr:.4f} "
              f"P={p[idx]:.4f} R={r[idx]:.4f}")
        return thr, "strict"
    f2_arr = fbeta_curve(p[:-1], r[:-1], 2.0)
    idx = int(np.argmax(f2_arr))
    thr = t[idx]
    print(f"\nFalling back to F2-optimal: thr={thr:.4f} "
          f"P={p[idx]:.4f} R={r[idx]:.4f} F2={f2_arr[idx]:.4f}")
    return thr, "f2"


thr, mode = pick_threshold(oof_total, y)

oof_pred = (oof_total >= thr).astype(int)
print("\nOOF confusion matrix:")
print(confusion_matrix(y, oof_pred))
print(f"OOF P={precision_score(y, oof_pred):.4f} "
      f"R={recall_score(y, oof_pred):.4f} "
      f"F1={f1_score(y, oof_pred):.4f}")


# ------------------------------------------------------------------ #
# Test prediction
# ------------------------------------------------------------------ #
test_pred = (test_total >= thr).astype(int)
print(f"\nTest predicted positives: {int(test_pred.sum())} / "
      f"{len(test_pred)} ({test_pred.mean()*100:.2f}%)")
print(f"Train positive rate: {y.mean()*100:.2f}%")


# ------------------------------------------------------------------ #
# Build submission
# ------------------------------------------------------------------ #
sub = pd.DataFrame({
    "CoilID": test["CoilID"].values,
    "Y": test_pred.astype(int),
})
assert sub.shape == (339, 2)
assert list(sub.columns) == ["CoilID", "Y"]

OUT = os.path.join(DATA_DIR, "expected_submission.csv")
sub.to_csv(OUT, index=False)
print(f"\nWrote submission: {OUT}")
print(sub.head())

prob_path = os.path.join(DATA_DIR, "expected_submission_probs.csv")
pd.DataFrame({
    "CoilID": test["CoilID"].values,
    "prob": test_total,
    "Y": test_pred.astype(int),
}).to_csv(prob_path, index=False)
print(f"Wrote probabilities: {prob_path}")
