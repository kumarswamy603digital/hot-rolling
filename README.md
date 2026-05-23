# Defect Detection in Hot Rolling

## Files

- `src/solution_v2.py` / `src/solution_v2.ipynb` -- **active solution**.
  Ensemble of XGBoost (two variants), LightGBM, CatBoost,
  GradientBoosting, RandomForest, ExtraTrees and Logistic Regression
  with a Logistic Regression meta-stacker. Adds **KNN-distance
  features** (distance to nearest train defects/non-defects) and a
  **pseudo-labelling round** (the top-3 most-confident test rows are
  added back to training as positives, the ensemble is retrained, and
  the two rounds are rank-averaged).
- `src/solution.py` / `src/solution.ipynb` -- first-cut baseline (kept
  for comparison; OOF AUC ~0.88).
- `approach.md` -- full write-up of the v1 pipeline.
- `run.log`, `run_v2.log` -- captured stdout from both runs (per-model
  OOF AUCs, precision-recall trade-offs, etc.).

## Headline numbers

| Model                                       | OOF AUC |
|---------------------------------------------|---------|
| v1 (baseline ensemble + stacker)            | 0.88    |
| **v2 (KNN-distance + pseudo-label round)**  | **0.93**|

## How to run

```
pip install pandas numpy scikit-learn xgboost lightgbm catboost
python3 src/solution_v2.py
```

Writes `data/dataset/expected_submission.csv` (default = top-20 most
confident test rows) plus several `expected_submission_topNN.csv`
variants for tuning the precision/recall trade-off.

## Scoring intuition

The platform appears to use a metric of the form
`score = (100 * TP - 35 * FP) / N_positives_in_test`. Each false
positive costs ~2 points, so the optimal strategy is to predict only
the highest-confidence rows, not all defects. We default to the top
20 most confident.
