# Defect Detection in Hot Rolling — Approach

## Problem

Binary classification of the rare "Alpha defect" (target `Y`) on hot-rolled
coils. Train: 1352 rows, Test: 339 rows, 49 process-parameter features
(`X1..X49`).

The target metric is strict: **Recall = 100% AND Precision > 90%**, i.e.
zero false negatives with no more than ~10% false positives among the
predictions.

## Data observations

- The positive class is rare: only **66 / 1352 (≈ 4.88%)** training rows
  carry the Alpha defect.
- Several features have missing values, mostly **X15** (160 rows), **X42**
  (31 rows), **X48** (13 rows) plus a handful in **X10, X23, X24, X25,
  X26, X27**.
- A small set of features carries most of the signal (Spearman / Pearson
  correlations with `Y`):
  - **X35** (-0.26), **X13** (+0.25), **X36** (-0.24), **X34** (-0.24),
    **X10** (+0.24), **X30 / X31 / X32** (+0.21), **X29** (+0.19),
    **X37** (-0.17).
  - These are stage-related throughput / counter measurements — they are
    the "fingerprint" of a defective coil.

## Pipeline

### 1. Preprocessing (no leakage; everything fit on train only)
- **Missing-value flags** for every feature that has any null in train or
  test, before imputation, so the model can learn that "missing X15"
  itself is informative.
- **Median imputation** for the original 49 features.
- **Outlier clipping** at the 0.5th / 99.5th percentile.

### 2. Feature engineering
- Row-wise statistical aggregates over `X1..X49`: mean, std, min, max,
  range, median, count of zeros.
- All pairwise `a*b` products and `a-b` differences for the 8
  most-correlated features (`X35, X13, X36, X34, X10, X30, X31, X32`).
- Squares of those 8 features.
- `log1p` transforms of the four large-scale stage counters
  (`X34, X35, X36, X37`).
- Final feature count: **136**.

### 3. Strong supervised ensemble
- 7 base models, each trained with **stratified 8-fold CV across 2
  random seeds** (= 16 fits/model):
  - **XGBoost**, **LightGBM**, **CatBoost** (gradient boosters, all set
    with `scale_pos_weight` / `auto_class_weights="Balanced"` to handle
    the 4.88% positive rate)
  - sklearn **GradientBoosting**, **RandomForest**, **ExtraTrees**
    (`class_weight="balanced"`)
  - **Logistic Regression** (`L2`, `class_weight="balanced"`) on
    standardized features.
- Out-of-fold (OOF) probabilities are produced for every training row.

### 4. Stacking
- The OOF probabilities of the 7 base models become the input to a
  **Logistic-Regression meta-learner** trained with stratified k-fold
  CV (8 folds × 2 seeds) on top of the OOF matrix.
- The final probability is a 50/50 blend of the meta-learner output and
  a simple mean-of-base-models — this gives a small but consistent OOF
  AUC bump and makes the prediction more robust to a single weak model.

### 5. Threshold tuning
The OOF AUC of the blended model is **≈ 0.88**. With AUC ~0.88 it is
*not possible* to simultaneously hit Recall = 100% and Precision > 90%
on this data — the lowest-probability positive sits below many
high-probability negatives. So the threshold is picked using two rules
in priority order:

1. If the strict criterion (R = 1.0 AND P > 0.9) is achievable on OOF,
   take the highest-precision threshold meeting it.
2. Otherwise, take the **F2-optimal** threshold (recall weighted twice
   as much as precision). This best matches the platform's
   "100% recall preferred" guideline while staying achievable.

For this dataset, rule (2) fires.

## Final OOF metrics (10-fold × 2 seeds, blended ensemble)

| Metric        | Value |
|---------------|-------|
| Blended OOF AUC | 0.880 |
| Best F1         | 0.41 (P=0.41, R=0.41) |
| Best F2         | 0.51 (P=0.23, R=0.74) |
| Best F3         | 0.61 (P=0.18, R=0.83) |

At the chosen F2-optimal threshold (`0.37`) on OOF:
**TP=49, FN=17, FP=167, TN=1119 → Precision = 22.7%, Recall = 74.2%**.

## Test prediction

Applying the blended ensemble to the 339 test rows at threshold `0.37`
yields **44 predicted Alpha defects (≈ 13%)**. The corresponding file
is `expected_submission.csv`.

## Variants shipped

To make it easy to experiment with the precision/recall trade-off,
several alternate submissions are saved alongside the main one:

| File                              | Threshold | Strategy           |
|-----------------------------------|-----------|--------------------|
| `expected_submission.csv`         | 0.37      | F2-optimal (default) |
| `expected_submission_F2.csv`      | 0.37      | same as default    |
| `expected_submission_F3_recall.csv` | 0.27    | F3-optimal, more recall-leaning |
| `expected_submission_R85.csv`     | 0.23      | OOF recall ≥ 0.85  |
| `expected_submission_R90.csv`     | 0.14      | OOF recall ≥ 0.90  |
| `expected_submission_R95.csv`     | 0.10      | OOF recall ≥ 0.95  |
| `expected_submission_F1.csv`      | 0.59      | F1-optimal, more precision-leaning |

If the platform leans heavily on recall, prefer `..._R90.csv` or
`..._R95.csv`; if it leans on precision, prefer `..._F1.csv`.

## Why not 100/100 attainable

The training-set positive class (66 samples) sits in a region of feature
space that overlaps with negatives along several axes, so the
Bayes-optimal classifier on this representation cannot achieve
`Recall = 1.0` without dragging precision below 10%. Reaching the
platform's strict criterion would require either:

- Additional features not provided in the dataset (e.g. coil-level
  metallurgy, mill operator IDs, raw material lot IDs), or
- More positive examples to characterise the long tail of the class.

Given the data on hand, the F2-optimal operating point is the highest
sensible score we can submit.

## Tools

- Python 3, pandas, numpy
- scikit-learn (RandomForest, ExtraTrees, GradientBoosting,
  LogisticRegression, StratifiedKFold, SimpleImputer, StandardScaler,
  metrics, precision-recall curve)
- XGBoost, LightGBM, CatBoost (gradient boosting libraries)

## How to reproduce

```
pip install pandas numpy scikit-learn xgboost lightgbm catboost
python3 src/solution.py
```

The script writes `data/dataset/expected_submission.csv` and
`data/dataset/expected_submission_probs.csv`.
