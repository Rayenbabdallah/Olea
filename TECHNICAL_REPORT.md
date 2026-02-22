# OLEA AI — Full Technical Report

**Version:** 2.0.0
**Team:** OLEA
**Competition:** DataQuest Hackathon
**Date:** February 2026

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Definition](#2-problem-definition)
3. [Phase I — Model Training & Submission](#3-phase-i--model-training--submission)
4. [Phase II — Intelligent Backend Architecture](#4-phase-ii--intelligent-backend-architecture)
5. [Component Deep Dives](#5-component-deep-dives)
6. [API Specification](#6-api-specification)
7. [System Flow Diagrams](#7-system-flow-diagrams)
8. [Design Decisions & Tradeoffs](#8-design-decisions--tradeoffs)
9. [Results & Validation](#9-results--validation)
10. [Future Work](#10-future-work)

---

## 1. Executive Summary

OLEA AI is a **cognitive insurance advisory system** that goes beyond classification. Given a prospective customer's demographic, financial, and behavioral profile, the system:

1. **Predicts** which of 10 insurance bundles the customer will purchase (LightGBM, Macro F1 = 0.574).
2. **Explains** the prediction through a 4-agent cognitive architecture — each domain specialist provides an independent probability vote, and a Meta Decision Engine synthesizes their reasoning into an executive narrative.
3. **Simulates** counterfactual scenarios — "what if the customer's income doubled?" — to support strategic decision-making.
4. **Recommends** upsell and cross-sell strategies grounded in rule-based business logic and an adjacency graph across all 10 bundles.

The system comprises a FastAPI backend with 4 endpoints, a 64-feature engineering pipeline, 4 domain-specialist LightGBM agents, a bundle embedding space, perturbation-based sensitivity analysis, entropy-based uncertainty quantification, and a boardroom-ready business insight generator.

---

## 2. Problem Definition

### Task
Multi-class classification: predict `Purchased_Coverage_Bundle` (integer 0–9) from 27 raw customer features.

### Target Classes

| ID | Bundle | Category |
|:---|:-------|:---------|
| 0 | Auto_Comprehensive | Auto |
| 1 | Auto_Liability_Basic | Auto |
| 2 | Basic_Health | Health |
| 3 | Family_Comprehensive | Family |
| 4 | Health_Dental_Vision | Health |
| 5 | Home_Premium | Home |
| 6 | Home_Standard | Home |
| 7 | Premium_Health_Life | Health/Life |
| 8 | Renter_Basic | Renter |
| 9 | Renter_Premium | Renter |

### Scoring Formula

$$\text{score} = \text{Macro F1} \times \max\!\big(0.5,\; 1 - \tfrac{\text{size\_MB}}{200}\big) \times \max\!\big(0.5,\; 1 - \tfrac{\text{latency\_s}}{10}\big)$$

This penalizes both model bloat and slow inference — forcing a balance between accuracy and efficiency.

### Constraints

| Constraint | Value |
|:-----------|:------|
| Max upload | 50 MB |
| Container RAM | 1 GB |
| CPU cores | 1 |
| Timeout | 120 s |

---

## 3. Phase I — Model Training & Submission

### 3.1 Model Selection

We chose **LightGBM** (`lightgbm.Booster`) for its:
- Native categorical feature support (no one-hot explosion)
- Sub-second inference on single CPU
- Small serialized footprint (~4 MB for 50 trees)
- Strong tabular-data performance without extensive tuning

### 3.2 Hyperparameters

| Parameter | Value | Rationale |
|:----------|:------|:----------|
| `n_estimators` | 50 | Minimizes latency on the judge's single-CPU container |
| `learning_rate` | 0.04 | Conservative rate paired with 50 trees |
| `num_leaves` | 95 | Moderately complex splits |
| `min_child_samples` | 25 | Prevents overfitting to rare patterns |
| `subsample` | 0.9 | Row sampling for regularization |
| `colsample_bytree` | 0.85 | Feature diversity per tree |
| `reg_lambda` | 1.5 | L2 regularization on leaf weights |
| `max_depth` | -1 | Controlled by `num_leaves` |

### 3.3 Feature Engineering Pipeline (64 Features)

All engineering runs inside `preprocess()`, which is **not timed** by the judge.

#### Raw Feature Preprocessing

| Transformation | Features | Method |
|:---------------|:---------|:-------|
| Ordinal encoding | `Deductible_Tier` | Zero → 0, Low → 1, Mid → 2, High → 3 |
| Cyclical encoding | `Policy_Start_Month` | sin/cos with period 12 |
| Cyclical encoding | `Policy_Start_Week` | sin/cos with period 53 |
| Native categorical | `Region_Code`, `Broker_Agency_Type`, `Acquisition_Channel`, `Payment_Schedule`, `Employment_Status` | LightGBM categorical dtype |
| Binary flags + Drop | `Broker_ID`, `Employer_ID` | `has_broker`, `has_employer` flags; IDs dropped |

#### Engineered Feature Groups

| Group | Count | Features | Signal |
|:------|:------|:---------|:-------|
| **Dependents** | 4 | `total_deps`, `has_family`, `child_ratio`, `infant_ratio` | Family structure & coverage needs |
| **Income** | 2 | `income_log`, `income_per_dep` | Financial capacity per household member |
| **Claims & Risk** | 6 | `claims_per_cfy`, `claims_rate`, `loyalty_idx`, `tenure_years`, `has_claims`, `clean_record` | Risk profile & loyalty indicators |
| **Friction** | 6 | `friction`, `high_friction`, `amended`, `quote_uw_ratio`, `quote_minus_uw`, `amend_per_dur` | Underwriting complexity & price sensitivity |
| **Channel Flags** | 2 | `has_broker`, `has_employer` | Distribution channel influence |
| **Deductible Interactions** | 2 | `ded_income`, `ded_deps` | Deductible affordability relative to income/family |
| **Vehicle & Riders** | 5 | `veh_per_dep`, `riders_per_veh`, `income_x_veh`, `deps_x_veh`, `veh_x_riders` | Auto/property coverage complexity |
| **Policyholder Interactions** | 3 | `exist_x_dur`, `exist_x_claims`, `loyalty_x_dur` | Tenure × history cross-signals |
| **Payment Dummies** | 3 | `pay_annual`, `pay_monthly`, `pay_quarterly` | Payment behavior indicators |
| **Other Flags** | 2 | `was_cancelled`, `ext_ratio` | Cancellation risk & grace usage |
| **Cyclical Encodings** | 5 | `month_num`, `month_sin`, `month_cos`, `week_sin`, `week_cos` | Temporal seasonality |
| **Raw Numeric** | 24 | Original numeric columns carried through | Direct signal contribution |
| **Total** | **64** | | |

### 3.4 Class Imbalance Handling

Inverse-frequency sample weights during training:

$$w_c = \frac{N}{K \times n_c}$$

where $N$ = total samples, $K$ = 10 classes, $n_c$ = count of class $c$. This ensures the model treats all 10 bundles as equally important, directly optimizing Macro F1.

### 3.5 Validation

- **5-fold Stratified K-Fold** cross-validation
- Metric: **Macro F1-Score** (matching the judge)
- CV Result: ~0.574 Macro F1

### 3.6 Latency Optimization

| Technique | Impact |
|:----------|:-------|
| Module-level imports | Pre-loaded before timer starts |
| Pickle serialization of Booster text string | ~14 ms load vs ~230 ms with joblib |
| Feature engineering in `preprocess()` (untimed) | Entire `predict()` is just `model.predict()` + `argmax` |
| 50 trees (not 600) | Tree count scales non-linearly on slow CPU |

### 3.7 Phase I Results

| Factor | Value | Multiplier |
|:-------|:------|:-----------|
| Macro F1 | 0.5736 | 0.5736 |
| Model Size | 3.99 MB | 0.980 |
| Predict Latency | 1.02 s | 0.898 |
| **Final Score** | | **0.5045** |

#### Key Tradeoff Insight

Larger models (65 trees, 87 features) achieved higher CV F1 (~0.60) but **scored worse** on the judge (0.465) because latency nearly doubled to ~1.8 s. The latency penalty outweighed the F1 gain. This shaped our final lean 50-tree design.

---

## 4. Phase II — Intelligent Backend Architecture

Phase II transforms the Phase I classifier into a full **cognitive advisory system** without modifying `model.pkl`.

### 4.1 Design Principles

1. **Zero impact on Phase I model** — `model.pkl` remains untouched. All intelligence layers are additive.
2. **Train at startup, not at request time** — Domain agents and bundle embeddings are built from `train.csv` during server startup.
3. **Deterministic explanations** — No randomness in narratives; the same input always produces the same output.
4. **Business-readable output** — Every feature gets a human-readable label; every number gets contextual language.

### 4.2 Component Inventory

| Component | File | Lines | Purpose |
|:----------|:-----|:------|:--------|
| FastAPI Application | `app.py` | ~85 | 4 endpoints, lifespan startup |
| Model Singleton | `core/model.py` | ~63 | Load & predict with Phase I Booster |
| Feature Engineering | `core/preprocess.py` | ~150 | 64-feature pipeline (mirrors Phase I) |
| Multi-Agent Panel | `core/agents.py` | ~209 | 4 domain agents, vote, consensus |
| Bundle Embedding Space | `core/bundle_space.py` | ~80 | Per-bundle centroids, cosine similarity |
| Intelligence Engine | `core/explain.py` | ~600 | Sensitivity, uncertainty, narrative, upsell, business insight |
| Simulation Engine | `core/simulate.py` | ~55 | Base vs modified comparison |
| Schemas | `core/schemas.py` | ~180 | Pydantic request/response models |

---

## 5. Component Deep Dives

### 5.1 Multi-Agent Prediction Layer (`core/agents.py`)

#### Architecture

We deploy **4 domain-specialist LightGBM agents**, each trained on a focused subset of the 64 engineered features. Each agent outputs a full 10-class probability distribution, providing a real "vote" grounded in its domain expertise.

| Agent | Feature Subset | Count | Domain Focus |
|:------|:---------------|:------|:-------------|
| **Life-Stage Agent** | `Adult_Dependents`, `Child_Dependents`, `Infant_Dependents`, `total_deps`, `has_family`, `child_ratio`, `infant_ratio`, `income_per_dep`, `veh_per_dep`, `deps_x_veh`, `ded_deps` | 11 | Family structure & coverage orientation |
| **Financial Agent** | `Estimated_Annual_Income`, `income_log`, `income_per_dep`, `Deductible_Tier`, `ded_income`, `pay_annual`, `pay_monthly`, `pay_quarterly`, `income_x_veh`, `ext_ratio`, `Grace_Period_Extensions` | 11 | Financial capacity & payment behavior |
| **Stability Agent** | `Previous_Claims_Filed`, `Years_Without_Claims`, `Previous_Policy_Duration_Months`, `Existing_Policyholder`, `claims_per_cfy`, `claims_rate`, `loyalty_idx`, `tenure_years`, `has_claims`, `clean_record`, `exist_x_dur`, `exist_x_claims`, `loyalty_x_dur`, `was_cancelled` | 14 | Risk profile, loyalty & retention |
| **Friction Agent** | `Policy_Amendments_Count`, `Underwriting_Processing_Days`, `Days_Since_Quote`, `friction`, `high_friction`, `amended`, `quote_uw_ratio`, `quote_minus_uw`, `amend_per_dur`, `has_broker`, `has_employer` | 11 | Sales friction & channel influence |

#### Agent Hyperparameters

Each agent uses deliberately smaller models to prevent overfitting on limited feature sets:

| Parameter | Value |
|:----------|:------|
| `n_estimators` | 25 |
| `learning_rate` | 0.05 |
| `num_leaves` | 15 |
| `min_child_samples` | 30 |
| `objective` | multiclass (10 classes) |
| `seed` | 42 |

#### Consensus Mechanism

Agent consensus is computed as the **equal-weight arithmetic mean** of all 4 agents' probability vectors:

$$P_{\text{consensus}}(c) = \frac{1}{4}\sum_{a=1}^{4} P_a(c) \quad \text{for each class } c \in \{0,...,9\}$$

The `agreement` metric counts how many agents independently chose the same top class as the consensus pick. A 4/4 agreement is a strong signal; a 1/4 agreement indicates divergent domain perspectives.

#### Relationship to Phase I Model

The Phase I model (50-tree full-feature Booster) remains the **sole prediction authority** for `/predict`. The 4 agents are an **explainability-only enrichment layer** — they provide independent domain perspectives that are surfaced in `/explain` as `agent_votes`, `agent_consensus`, and embedded in the narrative timeline.

### 5.2 Feature Influence Sensitivity Engine (`core/explain.py`)

#### Method: Perturbation-Based Sensitivity Analysis

For each numeric feature $f$:

1. Set perturbation $\delta = 0.05 \times (|x_f| + 1)$ (5% of absolute value, with +1 floor)
2. Create $X^+ = X$ with $x_f + \delta$ and $X^- = X$ with $x_f - \delta$
3. Compute $\Delta_f = \frac{P(c^*|X^+) - P(c^*|X)}{2}$ averaged over both directions

where $c^*$ is the predicted class. Features are ranked by $|\Delta_f|$ and the top 3 are returned.

#### Output

Each influence includes:
- **Feature name** (raw column or engineered)
- **Business label** (from 64-entry `FEATURE_LABELS` dictionary)
- **Impact magnitude** (signed float)
- **Direction** ("positive" if feature supports the prediction, "negative" otherwise)

### 5.3 Bundle Embedding Space (`core/bundle_space.py`)

#### Construction

At startup, for each bundle $k \in \{0,...,9\}$:

1. Compute the **centroid** (mean feature vector) across all training samples in class $k$
2. L2-normalize the centroid vector

#### Similarity Query

For a predicted bundle $k$:

$$\text{sim}(k, j) = \hat{\mu}_k \cdot \hat{\mu}_j \quad \text{(cosine similarity of normalized centroids)}$$

Results are ranked and the top 3 nearest bundles are returned with **qualitative similarity levels**:

| Cosine Similarity | Level |
|:-------------------|:------|
| ≥ 0.995 | Very Strong |
| ≥ 0.98 | Strong |
| ≥ 0.95 | Moderate |
| < 0.95 | Weak |

Each result includes a context sentence: *"Represents approximately 97% profile overlap."*

No raw numeric similarity is exposed in the API response — only qualitative levels and context.

### 5.4 Uncertainty Quantification (`core/explain.py`)

#### Method: Shannon Entropy

$$H = -\sum_{c=0}^{9} P(c) \ln P(c)$$

| Entropy Range | Level | Interpretation |
|:-------------|:------|:---------------|
| $H < 1.0$ | Low | Highly confident; strong recommendation |
| $1.0 \leq H < 1.7$ | Medium | Reasonable but consider additional data |
| $H \geq 1.7$ | High | Multiple bundles competitive; gather more info |

A contextual explanation is generated dynamically, including the top-2 probability gap.

### 5.5 Multi-Agent Decision Timeline (`core/explain.py`)

A 4-step narrative simulating a committee deliberation:

| Step | Agent | Analysis |
|:-----|:------|:---------|
| 1 | **Life-Stage Agent** | Classifies household structure (single, young family, growing family, multi-adult, couple). Recommends coverage orientation. Includes real agent vote. |
| 2 | **Financial Agent** | Segments by income tier (budget-conscious, mid-income, high-income). Notes payment behavior and deductible strategy. Includes real agent vote. |
| 3 | **Stability Agent** | Assesses risk profile (excellent → elevated risk). Reviews tenure, claims history, grace extensions. Includes real agent vote. |
| 4 | **Meta Decision Engine** | Synthesizes all signals. Reports final bundle recommendation with confidence, key drivers, consensus level (high/medium/low based on top-2 probability gap), and agent agreement count. |

#### Consensus Classification

$$\text{gap} = P(c_1) - P(c_2) \quad \text{(top-2 probability gap)}$$

| Gap | Consensus Level | Executive Language |
|:----|:----------------|:-------------------|
| ≥ 20% | High | "Executive consensus is strong" |
| ≥ 10% | Medium | "Executive consensus is balanced" |
| < 10% | Low | "Executive consensus is limited" |

### 5.6 Upsell Strategy Generator (`core/explain.py`)

#### Rule-Based Triggers

| Rule | Trigger | Recommendation |
|:-----|:--------|:---------------|
| Payment Optimization | Monthly EFT or grace extensions ≥ 1 | Annual prepay incentive |
| Premium Upgrade | Income per dependent > $40K and not bundle 7 | Position Premium_Health_Life |
| Family Consolidation | ≥ 3 dependents and not bundle 3 | Recommend Family_Comprehensive |
| Deductible Rebalance | Low deductible + income > $60K | Propose higher deductible tier |

#### Cross-Sell Adjacency Graph

A hand-crafted adjacency map defines natural bundle pairings for all 10 classes. If rule-based suggestions yield fewer than 2, the system fills from the cross-sell graph. Maximum 3 suggestions per customer.

### 5.7 Business Insight Block (`core/explain.py`)

Deterministic commercial summary with three axes:

#### Customer Segment
| Condition | Segment |
|:----------|:--------|
| Dependents ≥ 3 | Family Growth Portfolio |
| Income > $80K | Affluent Protection Portfolio |
| Existing policyholder | Loyalty Renewal Portfolio |
| Default | Value Optimization Portfolio |

#### Retention Risk
| Condition | Risk Level |
|:----------|:-----------|
| Grace ≥ 2, or (Monthly + conf < 0.35), or gap < 8% | Elevated Retention Exposure |
| Grace ≥ 1, or claims ≥ 2, or gap < 15% | Moderate Retention Exposure |
| Default | Stable Retention Outlook |

#### Upsell Potential
| Condition | Potential |
|:----------|:---------|
| Income > $70K and ≥ 2 suggestions | High Expansion Headroom |
| Income > $30K or ≥ 2 suggestions | Selective Expansion Headroom |
| Default | Targeted Expansion Headroom |

### 5.8 Counterfactual Simulation Engine (`core/simulate.py`)

Given a base customer and a dictionary of field changes:

1. Run Phase I model on **base** → `base_prediction`
2. Merge changes into base → modified customer
3. Run Phase I model on **modified** → `modified_prediction`
4. Compute delta: `bundle_changed`, class with largest probability shift
5. Run sensitivity analysis + upsell on the modified scenario

This enables "what-if" exploration: *"If this customer's income went from $24K to $120K and they added 2 child dependents, would the recommended bundle change?"*

### 5.9 Presentation Polish

| Feature | Implementation |
|:--------|:---------------|
| **Business-readable labels** | 64-entry `FEATURE_LABELS` dictionary maps every raw and engineered feature to plain English |
| **Zero-income masking** | `_format_income()` replaces $0 income with "income not provided or minimal" |
| **Income density** | `_format_income_and_density()` shows "$X (Y per dependent)" when income > 0 |
| **Qualitative similarity** | Raw cosine similarity replaced with "very strong / strong / moderate / weak" levels |
| **Context sentences** | "Represents approximately 97% profile overlap" — no raw numbers exposed |
| **Executive upsell language** | "Revenue Strategy — Payment Optimization", "Portfolio Expansion", "Margin Optimization" |
| **Boardroom business insight** | "Family Growth Portfolio", "Elevated Retention Exposure", "High Expansion Headroom" |

---

## 6. API Specification

### 6.1 `GET /health`

| Field | Type | Description |
|:------|:-----|:------------|
| `status` | string | Always `"ok"` |
| `model_loaded` | boolean | Whether the Phase I model is loaded |

### 6.2 `POST /predict`

**Request body:** `CustomerFields` (27 raw customer fields matching CSV columns)

**Response:**

| Field | Type | Description |
|:------|:-----|:------------|
| `predicted_bundle` | int (0-9) | Predicted bundle class |
| `bundle_name` | string | Human-readable bundle name |
| `confidence` | float (0-1) | Probability of the predicted class |
| `probabilities` | float[10] | Full probability distribution |

### 6.3 `POST /explain`

**Request body:** `CustomerFields`

**Response:**

| Block | Type | Description |
|:------|:-----|:------------|
| `predicted_bundle` | int | Same as /predict |
| `confidence` | float | Same as /predict |
| `probabilities` | float[10] | Same as /predict |
| `uncertainty` | `UncertaintyInfo` | entropy, level (low/medium/high), contextual explanation |
| `top_influences` | `InfluenceItem[3]` | feature, label, impact, direction |
| `nearest_bundles` | `NearestBundleItem[3]` | bundle, name, similarity_level, similarity_context |
| `agent_votes` | `AgentVote[4]` | Per-agent: agent name, top_bundle, confidence, probabilities[10] |
| `agent_consensus` | `AgentConsensus` | Consensus: top_bundle, confidence, agreement count, probabilities[10] |
| `narrative` | `NarrativeInfo` | 4-step decision_timeline with agent names and summaries |
| `upsell_suggestions` | string[≤3] | Strategic upsell/cross-sell recommendations |
| `business_insight` | `BusinessInsight` | customer_segment, retention_risk, upsell_potential |

### 6.4 `POST /simulate`

**Request body:**
```json
{
  "base": { /* CustomerFields */ },
  "changes": { "field": "new_value", ... }
}
```

**Response:**

| Block | Type | Description |
|:------|:-----|:------------|
| `base` | `SimulatePrediction` | Original prediction |
| `modified` | `SimulatePrediction` | Prediction after changes |
| `delta` | `SimulateDelta` | `bundle_changed` flag + largest probability shift |
| `upsell_suggestions` | string[≤3] | Upsell for the modified scenario |

---

## 7. System Flow Diagrams

### 7.1 Startup Flow

```
Server Start
  │
  ├── model.load()           Load Phase I LightGBM Booster from model.pkl
  ├── bundle_space.build()   Read train.csv → compute 10 class centroids → normalize
  └── agent_panel.train()    Read train.csv → train 4 domain LightGBM agents (25 trees each)
  │
  ▼
Server Ready (port 8000)
```

### 7.2 `/predict` Flow

```
Customer JSON
  │
  ├── preprocess()           64-feature engineering
  ├── reindex(feature_cols)  Align to model's expected columns
  └── booster.predict()      50-tree LightGBM inference
  │
  ▼
{ predicted_bundle, confidence, probabilities }
```

### 7.3 `/explain` Flow

```
Customer JSON
  │
  ├── model.predict()                     Core prediction (Phase I model)
  ├── sensitivity_analysis()              Perturb each feature ±5% → rank impact
  ├── compute_uncertainty()               Shannon entropy of probability vector
  ├── agent_panel.vote()                  4 agents × preprocess → predict → probabilities
  ├── agent_panel.consensus()             Equal-weight average → top bundle + agreement
  ├── nearest_bundles_enriched()          Cosine similarity to predicted bundle centroid
  ├── generate_narrative(..., votes)      4-step timeline with real agent vote tags
  ├── generate_upsell()                   Rule-based + cross-sell adjacency
  └── generate_business_insight()         Segment, retention risk, upsell potential
  │
  ▼
Full ExplainResponse (13 blocks)
```

### 7.4 `/simulate` Flow

```
{ base, changes }
  │
  ├── model.predict(base)        Base prediction
  ├── model.predict(base+changes) Modified prediction
  ├── compute_delta()            Bundle change + max probability shift
  ├── sensitivity_analysis()     On modified scenario
  └── generate_upsell()          On modified scenario
  │
  ▼
{ base, modified, delta, upsell_suggestions }
```

---

## 8. Design Decisions & Tradeoffs

### 8.1 Why 4 Agents Instead of 6?

The project description envisioned 6 agents (including Loyalty and Behavioral Cluster). We consolidated to 4 because:
- **Loyalty** signals (`loyalty_idx`, `tenure_years`, `clean_record`) are absorbed into the Stability Agent alongside claims and existing-policyholder features — they are conceptually inseparable.
- **Behavioral Cluster Agent** would require unsupervised clustering (K-Means/DBSCAN) as a preprocessing step, adding complexity without clear business value in the narrative.
- 4 agents map cleanly to the 4-step Decision Timeline narrative.

### 8.2 Why Not a Meta-Decision Brain (Stacker)?

The original architecture proposed training a meta-LightGBM on concatenated agent probability vectors. We omitted this because:
- It would create a **new prediction model** that replaces or competes with the Phase I `model.pkl`.
- The Phase I model was optimized for the judge's scoring formula (latency, size, F1). Replacing it would invalidate our submission score.
- The agent consensus (equal-weight average) provides a similar insight without requiring stacked retraining.

### 8.3 Why Entropy Instead of Multi-Seed Variance?

Multi-seed uncertainty (training 3 models with different seeds and measuring prediction variance) would require:
- Retraining the Phase I model 3 times with different seeds
- Storing 3× the model size (~12 MB)
- 3× inference latency

Entropy over the single model's probability distribution provides a strong uncertainty signal without these costs.

### 8.4 Why Perturbation Sensitivity Instead of SHAP?

- SHAP (TreeExplainer) provides theoretically grounded Shapley values but requires `shap` as an additional dependency.
- Our perturbation approach is dependency-free, model-agnostic, and produces directionally equivalent results for top-3 feature ranking.
- The ±5% perturbation with +1 floor handles zero-valued features gracefully.

### 8.5 Why Qualitative Similarity Levels?

Raw cosine similarity values (e.g., 0.9847) are meaningless to business users. Converting to "strong overlap" with context sentences ("Represents approximately 98% profile overlap") makes the bundle similarity display presentation-ready without requiring data literacy from the audience.

---

## 9. Results & Validation

### 9.1 Phase I Judge Score

**Final Score: 0.5045** (Macro F1 = 0.574, Size = 3.99 MB, Latency = 1.02 s)

### 9.2 Agent Vote Example

Sample customer: Self-employed, 1 adult + 2 child dependents, income $0, Tier 3 deductible, monthly EFT, 14 UW days, 2 amendments, existing policyholder, 1 claim, 3 claim-free years.

| Agent | Top Vote | Confidence |
|:------|:---------|:-----------|
| Life-Stage | Basic_Health | 58.7% |
| Financial | Premium_Health_Life | 100.0% |
| Stability | Family_Comprehensive | 59.6% |
| Friction | Basic_Health | 79.7% |
| **Consensus** | **Basic_Health** | **40.0%** |
| **Phase I Model** | **Basic_Health** | — |

Agreement: 2/4 agents independently concur with the consensus (and with the Phase I model).

### 9.3 System Characteristics

| Metric | Value |
|:-------|:------|
| Startup time | ~10-15 s (agent training from train.csv) |
| `/predict` latency | < 100 ms (after startup) |
| `/explain` latency | ~500 ms (includes 4 agent inferences + perturbation) |
| `/simulate` latency | ~200 ms (2× model inference + perturbation) |
| Total backend code | ~1,400 lines Python |
| Model file size | 3.99 MB (Phase I model.pkl only) |
| Dependencies | pandas, numpy, scikit-learn, lightgbm, fastapi, uvicorn, pydantic |

---

## 10. Future Work

| Area | Enhancement | Status |
|:-----|:------------|:-------|
| **Frontend Dashboard** | Interactive UI with prediction card, agent timeline animation, simulation sliders, upsell display | Not started |
| **SHAP Integration** | Replace perturbation sensitivity with TreeExplainer for Shapley values | Considered, dependency concern |
| **Batch Prediction** | `/predict/batch` endpoint for CSV upload | Not implemented |
| **A/B Testing Framework** | Compare agent consensus vs Phase I model in production | Not implemented |
| **Model Versioning** | Support multiple model.pkl versions with routing | Not implemented |

---

## Appendix A: File Listing

| File | Lines | Purpose |
|:-----|:------|:--------|
| `backend/app.py` | ~85 | FastAPI app, lifespan, 4 routes |
| `backend/core/model.py` | ~63 | Phase I model singleton |
| `backend/core/preprocess.py` | ~150 | 64-feature engineering pipeline |
| `backend/core/agents.py` | ~209 | 4-agent panel (train, vote, consensus) |
| `backend/core/bundle_space.py` | ~80 | Bundle centroid embeddings |
| `backend/core/explain.py` | ~600 | All intelligence (sensitivity, uncertainty, narrative, upsell, insight) |
| `backend/core/simulate.py` | ~55 | Counterfactual simulation |
| `backend/core/schemas.py` | ~180 | Pydantic models |
| `backend/model.pkl` | — | Phase I LightGBM Booster (~4 MB) |

## Appendix B: Full Feature List (64 Features)

| # | Feature | Business Label | Source |
|:--|:--------|:---------------|:-------|
| 1 | `Policy_Cancelled_Post_Purchase` | Prior Cancellation History | Raw |
| 2 | `Policy_Start_Year` | Policy Start Year | Raw |
| 3 | `Policy_Start_Week` | Policy Start Week | Raw |
| 4 | `Policy_Start_Day` | Policy Start Day | Raw |
| 5 | `Grace_Period_Extensions` | Grace Period Extensions | Raw |
| 6 | `Previous_Policy_Duration_Months` | Prior Coverage Duration (months) | Raw |
| 7 | `Adult_Dependents` | Adult Dependents | Raw |
| 8 | `Child_Dependents` | Child Dependents | Raw |
| 9 | `Infant_Dependents` | Infant Dependents | Raw |
| 10 | `Existing_Policyholder` | Existing Policyholder Flag | Raw |
| 11 | `Previous_Claims_Filed` | Previous Claims Filed | Raw |
| 12 | `Years_Without_Claims` | Claim-Free Years | Raw |
| 13 | `Policy_Amendments_Count` | Quote Amendments | Raw |
| 14 | `Underwriting_Processing_Days` | Underwriting Duration (days) | Raw |
| 15 | `Vehicles_on_Policy` | Vehicles on Policy | Raw |
| 16 | `Custom_Riders_Requested` | Custom Riders Requested | Raw |
| 17 | `Estimated_Annual_Income` | Estimated Annual Income | Raw |
| 18 | `Days_Since_Quote` | Days Since Quote | Raw |
| 19 | `Deductible_Tier` | Deductible Tier | Ordinal encoded |
| 20 | `Region_Code` | Region Code | Categorical |
| 21 | `Broker_Agency_Type` | Broker Agency Type | Categorical |
| 22 | `Acquisition_Channel` | Acquisition Channel | Categorical |
| 23 | `Payment_Schedule` | Payment Schedule | Categorical |
| 24 | `Employment_Status` | Employment Status | Categorical |
| 25 | `month_num` | Policy Start Month | Engineered |
| 26 | `month_sin` | Seasonal Timing (sin) | Engineered |
| 27 | `month_cos` | Seasonal Timing (cos) | Engineered |
| 28 | `week_sin` | Weekly Timing (sin) | Engineered |
| 29 | `week_cos` | Weekly Timing (cos) | Engineered |
| 30 | `total_deps` | Total Dependents | Engineered |
| 31 | `has_family` | Has Family Coverage | Engineered |
| 32 | `child_ratio` | Child-to-Dependent Ratio | Engineered |
| 33 | `infant_ratio` | Infant-to-Dependent Ratio | Engineered |
| 34 | `income_log` | Log-Scaled Income | Engineered |
| 35 | `income_per_dep` | Income per Dependent | Engineered |
| 36 | `claims_per_cfy` | Claims per Claim-Free Year | Engineered |
| 37 | `claims_rate` | Annualized Claims Rate | Engineered |
| 38 | `loyalty_idx` | Customer Loyalty Index | Engineered |
| 39 | `tenure_years` | Policy Tenure (years) | Engineered |
| 40 | `has_claims` | Has Prior Claims | Engineered |
| 41 | `clean_record` | Clean Claims Record | Engineered |
| 42 | `friction` | Underwriting Friction Score | Engineered |
| 43 | `high_friction` | High Underwriting Friction | Engineered |
| 44 | `amended` | Policy Was Amended | Engineered |
| 45 | `quote_uw_ratio` | Quote-to-Underwriting Ratio | Engineered |
| 46 | `quote_minus_uw` | Days Gap (Quote - Underwriting) | Engineered |
| 47 | `amend_per_dur` | Amendments per Coverage Month | Engineered |
| 48 | `has_broker` | Has Broker/Agent | Engineered |
| 49 | `has_employer` | Has Employer Sponsorship | Engineered |
| 50 | `ded_income` | Deductible × Income | Engineered |
| 51 | `ded_deps` | Deductible × Dependents | Engineered |
| 52 | `veh_per_dep` | Vehicles per Dependent | Engineered |
| 53 | `riders_per_veh` | Riders per Vehicle | Engineered |
| 54 | `income_x_veh` | Income × Vehicles | Engineered |
| 55 | `deps_x_veh` | Dependents × Vehicles | Engineered |
| 56 | `veh_x_riders` | Vehicles × Riders | Engineered |
| 57 | `exist_x_dur` | Existing Customer × Tenure | Engineered |
| 58 | `exist_x_claims` | Existing Customer × Claims | Engineered |
| 59 | `loyalty_x_dur` | Loyalty × Tenure | Engineered |
| 60 | `pay_annual` | Pays Annually | Engineered |
| 61 | `pay_monthly` | Pays Monthly | Engineered |
| 62 | `pay_quarterly` | Pays Quarterly | Engineered |
| 63 | `was_cancelled` | Previously Cancelled | Engineered |
| 64 | `ext_ratio` | Grace Extension Ratio | Engineered |
