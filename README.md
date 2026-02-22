# OLEA AI — Cognitive Insurance Advisory System

> A multi-agent AI system that predicts insurance bundle selection, explains its reasoning through a 4-agent cognitive architecture, and simulates counterfactual "what-if" scenarios — built for the DataQuest Hackathon.

---

## Quick Start

```bash
# 1. Clone & enter project
cd Olea-ai

# 2. Create virtual environment (Python 3.10+)
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/Mac

# 3. Install dependencies
pip install pandas numpy scikit-learn lightgbm fastapi uvicorn pydantic requests

# 4. Start the API server
cd backend
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

The server trains 4 domain-specialist agents on `train.csv` at startup (~10-15 s), then serves on `http://127.0.0.1:8000`.

---

## Project Structure

```
Olea-ai/
├── backend/
│   ├── app.py                  # FastAPI application (4 endpoints)
│   ├── model.pkl               # Phase I LightGBM Booster (~4 MB)
│   └── core/
│       ├── model.py            # Model singleton — load & predict
│       ├── preprocess.py       # 64-feature engineering pipeline
│       ├── agents.py           # 4 domain-specialist LightGBM agents
│       ├── bundle_space.py     # Bundle embedding space (cosine similarity)
│       ├── explain.py          # Intelligence: sensitivity, narrative, upsell, business insight
│       ├── simulate.py         # Counterfactual simulation engine
│       └── schemas.py          # Pydantic request/response models
├── train.csv                   # Training data (features + target)
├── test.csv                    # Test data (features only)
├── solution.py                 # Phase I submission template
├── submission50.zip            # Phase I submission archive
├── MODEL_EXPLANATION.md        # Phase I model training rationale
├── TECHNICAL_REPORT.md         # Full backend technical report
├── PHASE2_CHECKLIST.md         # Development progress tracker
├── projectDescription.md       # System architecture vision document
└── README.md                   # This file
```

---

## API Endpoints

### `GET /health`

Returns server status and model readiness.

```json
{ "status": "ok", "model_loaded": true }
```

### `POST /predict`

Predicts insurance bundle for a customer. Original Phase I model is the sole prediction authority.

**Request:** Customer fields (same schema as CSV columns)
**Response:**
```json
{
  "predicted_bundle": 7,
  "bundle_name": "Premium_Health_Life",
  "confidence": 0.4821,
  "probabilities": [0.02, 0.01, 0.08, 0.12, 0.09, 0.01, 0.02, 0.48, 0.0, 0.0]
}
```

### `POST /explain`

Full intelligence analysis — sensitivity, multi-agent votes, narrative, upsell, business insight.

**Response includes:**
| Block | Description |
|:------|:------------|
| `predicted_bundle` / `confidence` / `probabilities` | Core prediction from Phase I model |
| `uncertainty` | Entropy-based confidence with contextual explanation |
| `top_influences` | Top 3 features by perturbation sensitivity with business labels |
| `nearest_bundles` | Cosine-similarity nearest bundles with qualitative levels |
| `agent_votes` | 4 per-agent probability distributions (Life-Stage, Financial, Stability, Friction) |
| `agent_consensus` | Equal-weight average across agents with agreement count |
| `narrative` | 4-step AI Decision Timeline (Life-Stage → Financial → Stability → Meta Decision Engine) |
| `upsell_suggestions` | Up to 3 rule-based + cross-sell strategic recommendations |
| `business_insight` | Customer segment, retention risk, upsell potential (boardroom-ready) |

### `POST /simulate`

Counterfactual simulation — "what if this customer's income doubled?"

**Request:**
```json
{
  "base": { /* customer fields */ },
  "changes": { "Estimated_Annual_Income": 120000, "Deductible_Tier": "Tier_2_Mid_Ded" }
}
```

**Response:**
```json
{
  "base": { "predicted_bundle": 2, "confidence": 0.58, "probabilities": [...] },
  "modified": { "predicted_bundle": 7, "confidence": 0.42, "probabilities": [...] },
  "delta": { "bundle_changed": true, "top_probability_shift": { "class": 7, "from": 0.08, "to": 0.42, "delta": 0.34 } },
  "upsell_suggestions": ["..."]
}
```

---

## Architecture Overview

```
Customer JSON ──► /predict ──► preprocess(64 feats) ──► LightGBM Booster ──► Bundle 0-9
                  /explain ──► ┌─ sensitivity_analysis (perturbation)
                               ├─ compute_uncertainty (entropy)
                               ├─ agent_panel.vote() (4 domain agents)
                               ├─ nearest_bundles_enriched (cosine sim)
                               ├─ generate_narrative (4-agent timeline)
                               ├─ generate_upsell (rules + cross-sell)
                               └─ generate_business_insight (segmentation)
                  /simulate ──► base_predict + modified_predict ──► delta + upsell
```

---

## Bundle Classes

| ID | Bundle Name |
|:---|:---|
| 0 | Auto_Comprehensive |
| 1 | Auto_Liability_Basic |
| 2 | Basic_Health |
| 3 | Family_Comprehensive |
| 4 | Health_Dental_Vision |
| 5 | Home_Premium |
| 6 | Home_Standard |
| 7 | Premium_Health_Life |
| 8 | Renter_Basic |
| 9 | Renter_Premium |

---

## Phase I Score

| Factor | Value | Multiplier |
|:-------|:------|:-----------|
| Macro F1 | 0.5736 | 0.5736 |
| Model Size | 3.99 MB | 0.980 |
| Predict Latency | 1.02 s | 0.898 |
| **Final Score** | | **0.5045** |

---

## Tech Stack

| Component | Technology |
|:----------|:-----------|
| ML Model | LightGBM 4.6.0 (50-tree Booster, multiclass) |
| Domain Agents | 4× LightGBM (25 trees each, feature subsets) |
| API | FastAPI + Uvicorn |
| Schemas | Pydantic v2 |
| Data | pandas 2.1.4, numpy 1.26.4 |
| Python | 3.10+ |
