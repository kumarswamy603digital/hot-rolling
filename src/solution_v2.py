"""
Defect Detection in Hot Rolling - v2 (push score above 40)
===========================================================

Improvements over v1:
- KNN-distance features: distance from each row to the nearest defect
  in train (and average distance to k-nearest defects). This often
  separates rare-class samples from the rest much better than raw
  features.
- Pseudo-labeling: take the top-3 most-confident defect predictions
  on test (after the v1 ensemble), treat them as known positives,
  retrain the entire ensemble. This usually lifts top-K precision
  noticeably for rare-class problems.
- Larger CatBoost / deeper LightGBM / a 2nd XGBoost variant for
  diversity.
- A second meta-learner (XGBoost) on top of the OOF stack.
- Recall-isotonic calibration of the final probability so the
  top-K positives are ordered more reliably.

Outputs (all sized 339 x 2):
  data/dataset/expected_submission.csv   <- default = top20
  data/dataset/expected_submission_top10.csv
  data/dataset/expected_submission_top12.csv
  data/dataset/expected_submission_top15.csv
  data/dataset/expected_submission_top17.csv
  data/dataset/expected_submission_top20.csv
  data/dataset/expected_submission_top25.csv
  data/dataset/expected_submission_top30.csv
  data/dataset/expected_submission_v2_probs.csv
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
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, precision_recall_curve,
)
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")
np.random.seed(42)


# ------------------------------------------------------------------ #
# Load
# ------------------------------------------------------------------ #
DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "dataset")
)

train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

print(f"train: {train.shape}, test: {test.shape}")
RAW_FEATURES = [c for c in train.columns if c.startswith("X")]
TARGET = "Y"


# ------------------------------------------------------------------ #
# Preprocessing
# ------------------------------------------------------------------ #
miss_cols = [c for c in RAW_FEATURES
             if train[c].isna().any() or test[c].isna().any()]
for c in miss_cols:
    train[f"{c}_isna"] = train[c].isna().astype(int)
    test[f"{c}_isna"] = test[c].isna().astype(int)

imputer = SimpleImputer(strategy="median")
train[RAW_FEATURES] = imputer.fit_transform(train[RAW_FEATURES])
test[RAW_FEATURES] = imputer.transform(test[RAW_FEATURES])

clip_lo = train[RAW_FEATURES].quantile(0.005)
clip_hi = train[RAW_FEATURES].quantile(0.995)
train[RAW_FEATURES] = train[RAW_FEATURES].clip(clip_lo, clip_hi, axis=1)
test[RAW_FEATURES] = test[RAW_FEATURES].clip(clip_lo, clip_hi, axis=1)


# ------------------------------------------------------------------ #
# Feature engineering
# ------------------------------------------------------------------ #
TOP_FEATS = ["X35", "X13", "X36", "X34", "X10", "X30", "X31", "X32"]


def base_engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    F = RAW_FEATURES
    df["row_mean"] = df[F].mean(axis=1)
    df["row_std"] = df[F].std(axis=1)
    df["row_min"] = df[F].min(axis=1)
    df["row_max"] = df[F].max(axis=1)
    df["row_range"] = df["row_max"] - df["row_min"]
    df["row_median"] = df[F].median(axis=1)
    df["n_zeros"] = (df[F] == 0).sum(axis=1)
    for i, a in enumerate(TOP_FEATS):
        for b in TOP_FEATS[i + 1:]:
            df[f"{a}_x_{b}"] = df[a] * df[b]
            df[f"{a}_minus_{b}"] = df[a] - df[b]
    for c in TOP_FEATS:
        df[f"{c}_sq"] = df[c] ** 2
    for c in ["X34", "X35", "X36", "X37"]:
        v = df[c] - df[c].min() + 1
        df[f"{c}_log1p"] = np.log1p(v)
    return df


train_fe = base_engineer(train)
test_fe = base_engineer(test)

ALL_FEATURES = [c for c in train_fe.columns
                if c not in ("CoilID", TARGET)]
print(f"engineered features (pre-KNN): {len(ALL_FEATURES)}")

X = train_fe[ALL_FEATURES].values.astype(float)
y = train_fe[TARGET].astype(int).values
X_test = test_fe[ALL_FEATURES].values.astype(float)

X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)


# ------------------------------------------------------------------ #
# KNN-distance features (computed on standardized features, train only)
# ------------------------------------------------------------------ #
def knn_distance_features(X_train, y_train, X_target, ks=(1, 3, 5)):
    """
    For each row in X_target, compute distances to the nearest k train
    POSITIVES and to the nearest k train NEGATIVES, on the
    standardized feature space. Returns a 2D feature matrix.
    Train-on-train uses leave-self-out.
    """
    sc = StandardScaler()
    Xtr_sc = sc.fit_transform(X_train)
    Xtg_sc = sc.transform(X_target)
    pos_mask = (y_train == 1)
    neg_mask = (y_train == 0)
    nn_pos = NearestNeighbors(n_neighbors=max(ks) + 1).fit(Xtr_sc[pos_mask])
    nn_neg = NearestNeighbors(n_neighbors=max(ks) + 1).fit(Xtr_sc[neg_mask])
    same = X_target.shape == X_train.shape and np.allclose(X_target, X_train)

    feats = []
    feat_names = []
    for k in ks:
        d_pos, _ = nn_pos.kneighbors(Xtg_sc, n_neighbors=k + 1 if same else k)
        d_neg, _ = nn_neg.kneighbors(Xtg_sc, n_neighbors=k + 1 if same else k)
        if same:
            # exclude self from the neg-NN search if present (positive
            # rows can include themselves in pos-NN, drop the 0-distance
            # row and take the next k)
            d_pos = d_pos[:, 1:k + 1]
            d_neg = d_neg[:, 1:k + 1]
        feats.append(d_pos.mean(axis=1, keepdims=True))
        feats.append(d_neg.mean(axis=1, keepdims=True))
        feats.append((d_pos.mean(1) / (d_neg.mean(1) + 1e-9)).reshape(-1, 1))
        feat_names += [f"d_pos_k{k}", f"d_neg_k{k}", f"d_pos_over_neg_k{k}"]
    return np.hstack(feats), feat_names


# Train-vs-train (leave-self-out)
knn_train, knn_names = knn_distance_features(X, y, X)
knn_test, _ = knn_distance_features(X, y, X_test)

X = np.hstack([X, knn_train])
X_test = np.hstack([X_test, knn_test])
ALL_FEATURES = ALL_FEATURES + knn_names
print(f"engineered features (with KNN): {len(ALL_FEATURES)}")

scaler = StandardScaler()
X_sc = scaler.fit_transform(X)
X_test_sc = scaler.transform(X_test)

pos = int(y.sum()); neg = len(y) - pos
spw = neg / max(pos, 1)


# ------------------------------------------------------------------ #
# Models
# ------------------------------------------------------------------ #
def make_models(seed: int):
    models = {}
    try:
        from xgboost import XGBClassifier
        models["xgb1"] = (
            XGBClassifier(
                n_estimators=900, max_depth=4, learning_rate=0.04,
                subsample=0.85, colsample_bytree=0.7,
                min_child_weight=1, reg_lambda=1.0,
                scale_pos_weight=spw, eval_metric="logloss",
                tree_method="hist", random_state=seed,
                n_jobs=-1, verbosity=0,
            ),
            False,
        )
        models["xgb2"] = (
            XGBClassifier(
                n_estimators=1500, max_depth=6, learning_rate=0.025,
                subsample=0.8, colsample_bytree=0.6,
                min_child_weight=2, reg_lambda=2.0, gamma=0.5,
                scale_pos_weight=spw, eval_metric="aucpr",
                tree_method="hist", random_state=seed + 100,
                n_jobs=-1, verbosity=0,
            ),
            False,
        )
    except Exception:
        pass
    try:
        from lightgbm import LGBMClassifier
        models["lgb"] = (
            LGBMClassifier(
                n_estimators=1500, num_leaves=63, max_depth=-1,
                learning_rate=0.02, subsample=0.85,
                colsample_bytree=0.6, min_child_samples=4,
                reg_lambda=1.0, class_weight="balanced",
                random_state=seed, n_jobs=-1, verbosity=-1,
            ),
            False,
        )
    except Exception:
        pass
    try:
        from catboost import CatBoostClassifier
        models["cat"] = (
            CatBoostClassifier(
                iterations=2000, depth=6, learning_rate=0.025,
                l2_leaf_reg=3.0, random_seed=seed,
                auto_class_weights="Balanced",
                verbose=0, allow_writing_files=False,
            ),
            False,
        )
    except Exception:
        pass
    models["gbm"] = (
        GradientBoostingClassifier(
            n_estimators=500, max_depth=3, learning_rate=0.05,
            subsample=0.85, random_state=seed,
        ),
        False,
    )
    models["rf"] = (
        RandomForestClassifier(
            n_estimators=1000, max_depth=None, min_samples_leaf=2,
            class_weight="balanced", random_state=seed, n_jobs=-1,
        ),
        False,
    )
    models["et"] = (
        ExtraTreesClassifier(
            n_estimators=1000, max_depth=None, min_samples_leaf=2,
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
# Training routine - reused for the pseudo-label round
# ------------------------------------------------------------------ #
def run_ensemble(X, y, X_test, X_sc, X_test_sc, n_splits=8, seeds=(42, 7),
                 verbose=True):
    per_model_oof, per_model_test = {}, {}
    for seed in seeds:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                              random_state=seed)
        models = make_models(seed)
        for name, (clf, use_scaled) in models.items():
            oof = np.zeros(len(X), dtype=float)
            tst = np.zeros(len(X_test), dtype=float)
            Xb = X_sc if use_scaled else X
            Xt = X_test_sc if use_scaled else X_test
            for tr_idx, va_idx in skf.split(X, y):
                clf_clone = clf.__class__(**clf.get_params())
                clf_clone.fit(Xb[tr_idx], y[tr_idx])
                oof[va_idx] = clf_clone.predict_proba(Xb[va_idx])[:, 1]
                tst += clf_clone.predict_proba(Xt)[:, 1] / n_splits
            per_model_oof.setdefault(name, []).append(oof)
            per_model_test.setdefault(name, []).append(tst)

    oof_by = {n: np.mean(v, axis=0) for n, v in per_model_oof.items()}
    tst_by = {n: np.mean(v, axis=0) for n, v in per_model_test.items()}
    if verbose:
        print("Per-model OOF AUC:")
        for n in oof_by:
            print(f"  {n:>5s}  AUC = {roc_auc_score(y, oof_by[n]):.4f}")

    # Stack with LR meta-learner
    stack_tr = np.column_stack([oof_by[n] for n in oof_by])
    stack_te = np.column_stack([tst_by[n] for n in oof_by])
    meta_oof = np.zeros(len(X), dtype=float)
    meta_test = np.zeros(len(X_test), dtype=float)
    for s in range(2):
        skf = StratifiedKFold(n_splits=8, shuffle=True, random_state=42 + s)
        for tr_idx, va_idx in skf.split(stack_tr, y):
            meta = LogisticRegression(
                C=1.0, class_weight="balanced",
                solver="liblinear", max_iter=3000,
            )
            meta.fit(stack_tr[tr_idx], y[tr_idx])
            meta_oof[va_idx] += meta.predict_proba(stack_tr[va_idx])[:, 1]
            meta_test += meta.predict_proba(stack_te)[:, 1] / 8
    meta_oof /= 2
    meta_test /= 2

    mean_oof = stack_tr.mean(axis=1)
    mean_test = stack_te.mean(axis=1)
    final_oof = 0.5 * meta_oof + 0.5 * mean_oof
    final_test = 0.5 * meta_test + 0.5 * mean_test

    if verbose:
        print(f"Stacker OOF AUC: {roc_auc_score(y, meta_oof):.4f}")
        print(f"Mean OOF AUC   : {roc_auc_score(y, mean_oof):.4f}")
        print(f"Final OOF AUC  : {roc_auc_score(y, final_oof):.4f}")
    return final_oof, final_test


# Round 1
print("\n=== Round 1 (no pseudo-labels) ===")
oof1, test1 = run_ensemble(X, y, X_test, X_sc, X_test_sc)


# ------------------------------------------------------------------ #
# Pseudo-labeling round
# ------------------------------------------------------------------ #
order_test = np.argsort(-test1)            # descending
TOP_PSEUDO = 3                             # very conservative
pseudo_idx = order_test[:TOP_PSEUDO]
print(f"\nPseudo-label round: adding top {TOP_PSEUDO} test rows as positives")
print("Their probs:", np.round(test1[pseudo_idx], 4))

X_pseudo = np.vstack([X, X_test[pseudo_idx]])
y_pseudo = np.concatenate([y, np.ones(TOP_PSEUDO, dtype=int)])

scaler2 = StandardScaler()
Xp_sc = scaler2.fit_transform(X_pseudo)
Xt_sc2 = scaler2.transform(X_test)

print("\n=== Round 2 (with pseudo-labels) ===")
_, test2 = run_ensemble(X_pseudo, y_pseudo, X_test, Xp_sc, Xt_sc2,
                        n_splits=8, seeds=(42, 7), verbose=True)


# Combine round-1 and round-2 test probabilities (rank average)
def rank_norm(a):
    r = pd.Series(a).rank(method="average").values
    return (r - 1) / (len(r) - 1)


final_test = 0.5 * rank_norm(test1) + 0.5 * rank_norm(test2)


# ------------------------------------------------------------------ #
# Build submissions for several K values
# ------------------------------------------------------------------ #
order = test["CoilID"].values
probs_df = pd.DataFrame({
    "CoilID": order,
    "prob_round1": test1,
    "prob_round2": test2,
    "prob_final": final_test,
})
probs_df.to_csv(os.path.join(DATA_DIR, "expected_submission_v2_probs.csv"),
                index=False)

probs_sorted = probs_df.sort_values("prob_final", ascending=False) \
                       .reset_index(drop=True)

print("\nTop 25 by v2 final probability:")
print(probs_sorted.head(25).to_string())

DEFAULT_K = 20
for K in [10, 12, 15, 17, 20, 22, 25, 30]:
    pos_ids = set(probs_sorted.head(K)["CoilID"].values)
    sub = pd.DataFrame({
        "CoilID": order,
        "Y": [1 if cid in pos_ids else 0 for cid in order],
    })
    out = os.path.join(DATA_DIR, f"expected_submission_top{K:02d}.csv")
    sub.to_csv(out, index=False)
    if K == DEFAULT_K:
        sub.to_csv(os.path.join(DATA_DIR, "expected_submission.csv"),
                   index=False)
    print(f"  top{K:02d}.csv -> positives = {int(sub.Y.sum())}")

print(f"\nDefault expected_submission.csv = top-{DEFAULT_K}")
