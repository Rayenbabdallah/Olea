# Model Explanation — Insurance Bundle Classifier

## Problem Statement

Predict which of 10 insurance coverage bundles (classes 0–9) a prospective customer will purchase, given demographic, financial, policy history, and sales/underwriting features.

---

## Model Choice: LightGBM

We use a single **LightGBM** gradient-boosted decision tree classifier (`LGBMClassifier`) with the `multiclass` objective.

**Why LightGBM?**

- Native handling of categorical features (no one-hot encoding needed) — reduces dimensionality and preserves ordinality.
- Highly efficient inference: leaf-based prediction is orders of magnitude faster than neural networks on tabular data.
- Small model footprint: a 50-tree ensemble serializes to ~4 MB, well within the 50 MB upload limit.
- Strong out-of-the-box performance on structured/tabular datasets.

---

## Hyperparameters

| Parameter            | Value | Rationale                                                |
|:---------------------|:------|:---------------------------------------------------------|
| `n_estimators`       | 50    | Kept low to minimize inference latency on the 1-CPU judge |
| `learning_rate`      | 0.04  | Conservative rate paired with 50 trees for stable learning |
| `num_leaves`         | 95    | Allows moderately complex splits per tree                 |
| `min_child_samples`  | 25    | Prevents overfitting to rare class patterns               |
| `subsample`          | 0.9   | Row sampling for regularization                           |
| `colsample_bytree`   | 0.85  | Feature sampling per tree for diversity                   |
| `reg_lambda`         | 1.5   | L2 regularization to control leaf magnitudes              |
| `max_depth`          | -1    | No hard depth limit; controlled by `num_leaves` instead   |

---

## Feature Engineering (64 features)

All feature engineering runs inside `preprocess()`, which is **not timed** by the judge. The timed `predict()` function only calls `model.predict()`.

### Raw Feature Preprocessing

- **Deductible_Tier**: Ordinal-encoded (Zero → 0, Low → 1, Mid → 2, High → 3).
- **Policy_Start_Month**: Cyclical encoding via `sin`/`cos` of month number (period = 12).
- **Policy_Start_Week**: Cyclical encoding via `sin`/`cos` (period = 53).
- **Categorical columns** (`Region_Code`, `Broker_Agency_Type`, `Acquisition_Channel`, `Payment_Schedule`, `Employment_Status`): Passed as native LightGBM categoricals.
- **Dropped**: `Employer_ID`, `Broker_ID` (high cardinality, sparse — replaced by binary flags).

### Engineered Features

| Group | Features | Logic |
|:------|:---------|:------|
| **Dependents** | `total_deps`, `has_family`, `child_ratio`, `infant_ratio` | Aggregate dependent counts and proportions |
| **Income** | `income_log`, `income_per_dep` | Log-transform + per-capita income |
| **Claims & Risk** | `claims_per_cfy`, `claims_rate`, `loyalty_idx`, `tenure_years`, `has_claims`, `clean_record` | Claims frequency, claim-free loyalty, risk indicators |
| **Friction** | `friction`, `high_friction`, `amended`, `quote_uw_ratio`, `quote_minus_uw`, `amend_per_dur` | Underwriting complexity and amendment behavior |
| **Channel Flags** | `has_broker`, `has_employer` | Binary presence indicators for broker/employer |
| **Deductible Interactions** | `ded_income`, `ded_deps` | Deductible tier × income/dependents |
| **Vehicle & Riders** | `veh_per_dep`, `riders_per_veh`, `income_x_veh`, `deps_x_veh`, `veh_x_riders` | Cross-features between vehicles, riders, income, and dependents |
| **Policyholder Interactions** | `exist_x_dur`, `exist_x_claims`, `loyalty_x_dur` | Existing policyholder × history |
| **Payment Dummies** | `pay_annual`, `pay_monthly`, `pay_quarterly` | Payment schedule one-hot flags |
| **Other** | `was_cancelled`, `ext_ratio` | Cancellation history, grace period usage rate |

---

## Class Imbalance Handling

The 10 bundles are **not equally distributed** in the training data. We use **inverse-frequency weighting** as sample weights during training:

$$w_c = \frac{N}{K \times n_c}$$

where $N$ = total samples, $K$ = 10 classes, $n_c$ = count of class $c$. This ensures the model treats every class as equally important, which directly optimizes the **Macro F1** metric.

---

## Validation Strategy

- **5-fold Stratified K-Fold** cross-validation (preserves class distribution in each fold).
- Metric: **Macro F1-Score** — the same metric used by the judge.
- CV Result: **~0.574 Macro F1** (the score confirmed on the judge leaderboard).

---

## Latency Optimization

The judge scores `predict()` latency on a **single CPU core** with **1 GB RAM**. Our architecture is designed to minimize timed execution:

| Component | Technique | Impact |
|:----------|:----------|:-------|
| **Imports** | All imports (`lightgbm`, `pandas`, `numpy`, `pickle`) at module level | Happen during `import solution`, before the timer starts |
| **Model format** | LightGBM Booster text string serialized via `pickle` | ~14 ms load vs ~230 ms with `joblib` |
| **Feature engineering** | Entirely inside `preprocess()` (not timed) | Zero overhead in `predict()` |
| **`predict()` body** | Only `model.predict(X)` + `argmax` | ~60–80 ms locally, ~1.0 s on judge |
| **Tree count** | 50 trees (not 600) | Critical — more trees scale non-linearly on the judge's slow CPU |

---

## Scoring Breakdown

The final score formula is:

$$\text{score} = \text{Macro F1} \times \max(0.5,\; 1 - \tfrac{\text{size\_MB}}{200}) \times \max(0.5,\; 1 - \tfrac{\text{latency\_s}}{10})$$

Our submission achieved:

| Factor | Value | Multiplier |
|:-------|:------|:-----------|
| Macro F1 | 0.5736 | 0.5736 |
| Model Size | 3.99 MB | 0.980 |
| Predict Latency | 1.02 s | 0.898 |
| **Final Score** | | **0.5045** |

---

## Key Tradeoff Insight

We experimented with larger models (65 trees, 87 features) that achieved higher CV F1 (~0.60) but **scored worse** on the judge (0.465) because latency nearly doubled from 1.0 s → 1.8 s. The latency penalty outweighed the F1 gain. On the judge's single slow CPU, tree count and feature count scale **non-linearly** — a lesson that shaped our final lean design.

---

## File Structure

```
submission50.zip
├── solution.py        # preprocess(), load_model(), predict()
├── model.pkl          # 50-tree LightGBM Booster (pickle, ~4 MB)
└── requirements.txt   # Empty (all deps pre-installed)
```
