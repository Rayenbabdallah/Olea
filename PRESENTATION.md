# OLEA AI — 5-Minute Presentation Script

> **DataQuest Hackathon | Team OLEA | February 2026**

---

## SLIDE 1 — Opening (30 seconds)

### "What if your insurance system could think like a team of experts?"

Good morning everyone. We're Team OLEA, and we built **OLEA AI** — a cognitive insurance advisory system that doesn't just predict which insurance bundle a customer will buy, it **explains why**, **shows how confident it is**, and lets brokers ask **"what if?"** — all in under one second.

Today I'll walk you through our two-phase approach: a lean machine learning model, and an intelligent advisory layer on top of it.

---

## SLIDE 2 — The Problem (30 seconds)

### Multi-Class Insurance Bundle Prediction

- **Task:** Given a customer's profile — income, dependents, claims history, employment, vehicles — predict which of **10 insurance bundles** they'll purchase.
- **10 bundles** spanning Auto, Health, Home, Family, and Renter categories.
- **Scoring formula penalizes bloat:**

$$\text{Score} = \text{F1} \times \text{Size\ Penalty} \times \text{Latency\ Penalty}$$

- This means a larger, more accurate model can **score worse** if it's slow. We had to be **lean and fast**.

| Bundle ID | Name                  | Category     |
|:---------:|:----------------------|:-------------|
| 0         | Auto Comprehensive    | Auto         |
| 1         | Auto Liability Basic  | Auto         |
| 2         | Basic Health          | Health       |
| 3         | Family Comprehensive  | Family       |
| 4         | Health Dental Vision  | Health       |
| 5         | Home Premium          | Home         |
| 6         | Home Standard         | Home         |
| 7         | Premium Health Life   | Health/Life  |
| 8         | Renter Basic          | Renter       |
| 9         | Renter Premium        | Renter       |

---

## SLIDE 3 — Phase I: The Model (60 seconds)

### LightGBM — 50 Trees, 64 Features, Sub-Second Inference

We chose **LightGBM** for its speed, small footprint, and native categorical support.

**Key design decisions:**

| Choice | Value | Why |
|:-------|:------|:----|
| Trees | 50 | Fast on single-CPU judge container |
| Learning Rate | 0.04 | Conservative for 50 trees |
| Leaves | 95 | Enough complexity without overfitting |
| Model Size | **3.99 MB** | Tiny — almost no size penalty |

**Feature Engineering (27 raw → 64 engineered):**

From 27 raw CSV columns, we engineered 64 features across 11 groups:

- **Family structure:** total dependents, child ratio, infant ratio, family flag
- **Financial capacity:** log income, income per dependent, deductible × income
- **Claims & risk:** claims rate, loyalty index, clean record flag
- **Friction signals:** underwriting days, quote-to-processing ratio, amendment rate
- **Temporal:** cyclical month/week encoding (sin/cos)
- **Channel:** broker and employer binary flags

**Class imbalance handling:** Inverse-frequency sample weights ($w_c = \frac{N}{K \times n_c}$) so all 10 bundles matter equally to Macro F1.

**Key insight we learned the hard way:** A 65-tree model with 87 features scored **worse** (0.465) because latency nearly doubled. The lean 50-tree design scored **0.5045** — latency discipline wins.

| Metric | Value |
|:-------|:------|
| Macro F1 | 0.574 |
| Model Size | 3.99 MB |
| Latency | 1.02 s |
| **Final Score** | **0.5045** |

---

## SLIDE 4 — Phase II: The Intelligence Layer (90 seconds)

### From Classifier to Cognitive Advisory System

Phase II wraps the model in a **full-stack advisory system** — without modifying `model.pkl`. Everything is additive.

#### 4 API Endpoints

| Endpoint | What it does |
|:---------|:-------------|
| `GET /health` | System status check |
| `POST /predict` | Predict bundle + confidence + 10-class probabilities |
| `POST /explain` | Full cognitive analysis (agents, narrative, sensitivity, upsell) |
| `POST /simulate` | "What-if" counterfactual scenario comparison |

---

#### Multi-Agent Architecture (The Core Innovation)

We deploy **4 domain-specialist LightGBM agents**, each trained on a focused feature subset:

| Agent | Focus | Features |
|:------|:------|:---------|
| 🧬 **Life-Stage** | Family structure | dependents, child ratio, family flag (11 features) |
| 💰 **Financial** | Payment capacity | income, deductible, payment schedule (11 features) |
| 🛡️ **Stability** | Risk & loyalty | claims, tenure, policyholder history (14 features) |
| ⚡ **Friction** | Sales complexity | amendments, underwriting days, broker channel (11 features) |

Each agent independently produces a **10-class probability vote**. A Meta Decision Engine fuses them:

$$P_{\text{consensus}}(c) = \frac{1}{4}\sum_{a=1}^{4} P_a(c)$$

The `agreement` score (e.g., "3/4 agents agree") gives brokers instant confidence calibration.

---

#### Intelligence Modules

| Module | What it provides |
|:-------|:-----------------|
| **Sensitivity Engine** | Perturbs each feature ±20% to measure prediction impact |
| **Uncertainty Quantification** | Shannon entropy on the probability distribution (Low / Moderate / High) |
| **Bundle Similarity** | Cosine similarity in feature space — "this customer also resembles Home Premium" |
| **Decision Narrative** | 4-step timeline: each agent summarizes its reasoning in natural language |
| **Business Insight** | Customer segment, retention risk, upsell potential — boardroom-ready |
| **Upsell Engine** | Rule-based + cross-sell adjacency graph across all 10 bundles |

---

#### Simulation Engine — "What If?"

Brokers can ask: *"What if this customer's income doubled and they added 2 children?"*

```
POST /simulate
{
  "base": { ...current customer profile... },
  "changes": { "Estimated_Annual_Income": 120000, "Child_Dependents": 4 }
}
```

Response shows:
- **Base prediction** vs. **Modified prediction**
- Whether the **bundle changed**
- Exact **probability shift** (e.g., +12.3% toward Family Comprehensive)
- New **upsell recommendations**

This turns static prediction into an **interactive decision tool**.

---

## SLIDE 5 — The Frontend (30 seconds)

### React 19 + Tailwind — OLEA Branded UI

- **Left panel:** Full customer profile form — sliders for numerics, dropdowns for categoricals, toggles for booleans, collapsible "Advanced Fields"
- **Center stage:** Animated agent deliberation timeline → final prediction with confidence, uncertainty level, and consensus score → decision narrative
- **Right panel:** Sensitivity bars (which features matter most), bundle similarity cards, executive business insights
- **Simulation panel:** Quick-simulate buttons for instant "what-if" scenarios
- Built with **React 19, Vite, Tailwind CSS, Framer Motion** — professional OLEA branding with orange/terracotta/dark palette

The frontend calls the FastAPI backend through a Vite proxy — zero mock data, everything is real inference.

---

## SLIDE 6 — Architecture Overview (30 seconds)

```
┌─────────────────────────────────────────────────────────┐
│                    React Frontend                       │
│   Customer Form → API Call → Animated Results Display   │
└──────────────────────┬──────────────────────────────────┘
                       │ /api/* → Vite Proxy
                       ▼
┌─────────────────────────────────────────────────────────┐
│                  FastAPI Backend                        │
│                                                        │
│  /predict ──→ Preprocess (64 feat) → LightGBM Booster  │
│                                                        │
│  /explain ──→ 4 Domain Agents (vote + consensus)       │
│              → Sensitivity (perturbation analysis)     │
│              → Uncertainty (Shannon entropy)           │
│              → Narrative (4-step decision timeline)    │
│              → Bundle Similarity (cosine in feat space)│
│              → Business Insight (segment/risk/upsell)  │
│                                                        │
│  /simulate ─→ Base vs Modified prediction + delta      │
└─────────────────────────────────────────────────────────┘
```

**Tech stack:** Python 3.10 • LightGBM 4.6 • FastAPI • Pydantic v2 • React 19 • Vite 6 • Tailwind 4

---

## SLIDE 7 — Live Demo Talking Points (60 seconds)

### Demo Flow (if presenting live)

1. **Open the UI** at `http://127.0.0.1:5173`
2. **Show the default customer** — a self-employed South African with 1 adult + 2 child dependents, moderate income
3. **Click "RUN COGNITIVE ANALYSIS"** — watch the 4 agents animate one by one, then the Meta Decision Engine fuses their votes
4. **Point out the result card:** predicted bundle (e.g., Premium Health Life), confidence (44%), uncertainty level (Moderate), consensus (3/4 agents agree)
5. **Scroll to the narrative:** each agent explains in natural language why it voted the way it did
6. **Right panel:** show the sensitivity bars — "Income has the highest positive impact on this prediction"
7. **Quick simulate:** click "+50% Income" and show the simulation result card — does the bundle change? What's the probability shift?
8. **Business insight panel:** "Family Growth Portfolio, Moderate Retention Exposure, Selective Expansion Headroom"

---

## SLIDE 8 — Closing (30 seconds)

### What Makes OLEA AI Different

| Traditional ML | OLEA AI |
|:---------------|:--------|
| Single model → single number | 4 specialist agents → consensus with agreement score |
| Black-box prediction | Sensitivity analysis + decision narrative |
| Static output | Interactive "what-if" simulation |
| Engineer-readable JSON | Boardroom-ready business insights |
| Prediction only | Prediction + explanation + simulation + upsell |

**OLEA AI transforms a classification model into a decision-making partner** — giving brokers not just a prediction, but the confidence to act on it.

### Final Numbers

| Metric | Value |
|:-------|:------|
| Phase I Score | **0.5045** |
| Macro F1 | 0.574 |
| Model Size | 3.99 MB |
| Inference Latency | < 1 second |
| Specialist Agents | 4 |
| Engineered Features | 64 |
| API Endpoints | 4 |
| Total Codebase | ~1,400 lines Python + ~650 lines TypeScript |

> *"Don't just predict the bundle. Explain it. Simulate it. Sell it."*

**Thank you. Questions?**

---

## Appendix — Timing Guide

| Section | Slide | Duration | Cumulative |
|:--------|:------|:---------|:-----------|
| Opening hook | 1 | 0:30 | 0:30 |
| Problem definition | 2 | 0:30 | 1:00 |
| Phase I model | 3 | 1:00 | 2:00 |
| Phase II intelligence | 4 | 1:30 | 3:30 |
| Frontend | 5 | 0:30 | 4:00 |
| Architecture | 6 | 0:30 | 4:30 |
| Live demo / talking points | 7 | 0:30 | 5:00 |
| Closing | 8 | 0:30 | 5:00* |

*\*Demo and closing overlap — skip demo if short on time, or cut architecture slide and extend demo to 60s.*
