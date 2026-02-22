# OLEA AI — Phase 2 Checklist

---

## Backend

### API Endpoints

- [x] `GET /health` — status + model loaded flag
- [x] `POST /predict` — returns bundle, confidence, probabilities
- [x] `POST /explain` — sensitivity, uncertainty, multi-agent narrative, upsell, business insight
- [x] `POST /simulate` — base vs modified delta + upsell suggestions

### Intelligence Modules

- [x] **Feature Engineering Layer** — 64 engineered features across 5 behavioral dimensions (life-stage, financial, loyalty, friction, broker influence)
- [x] **Multi-Agent Prediction Layer** — 4 domain-specialist LightGBM agents (Life-Stage, Financial, Stability, Friction) trained on feature subsets at startup. Real per-agent probability votes and consensus included in `/explain` response.
- [x] **Feature Influence Sensitivity Engine** — perturbation-based sensitivity analysis with business-labeled features and directional impact
- [x] **Bundle Embedding Space** — cosine-similarity centroids from training data with qualitative similarity levels
- [x] **Uncertainty Quantification (entropy-based)** — entropy + contextual explanation narrative
- [x] **Upsell Strategy Generator** — rule-based + cross-sell adjacency map, enterprise-ready wording
- [x] **Business Insight Block** — `customer_segment`, `retention_risk`, `upsell_potential` with boardroom-ready language
- [x] **Agent Consensus Logic** — probability-gap-based executive consensus tiers in Meta Decision Engine narrative

### Presentation Polish

- [x] Business-readable feature labels (64 mappings)
- [x] Zero-income masking in narratives
- [x] Qualitative similarity levels (very strong / strong / moderate / weak)
- [x] Numeric similarity removed from output; context sentences only
- [x] Strategic enterprise-ready upsell language
- [x] Boardroom-ready business insight values
- [x] Update `README.md` for Phase 2 — API docs, architecture overview, setup instructions
- [x] Full technical report (`TECHNICAL_REPORT.md`) — model training, all components, design decisions, API spec, flow diagrams
- [x] Demo script (`demo.py`) — walkthrough of all 4 endpoints with formatted output

---

## Frontend

### Dashboard App

- [ ] **Frontend project scaffold** — React / Next.js / HTML+JS app under `frontend/`
- [ ] **API integration layer** — fetch wrapper for `/predict`, `/explain`, `/simulate`
- [ ] **Landing / input form** — customer fields form to submit to `/predict`

### Core Panels

- [ ] **Prediction result card** — bundle name, confidence gauge, probability bar chart
- [ ] **AI Decision Timeline animation** — step-through visualization of 4 agent deliberations (Life-Stage → Financial → Stability → Meta Decision Engine)
- [ ] **Sensitivity panel** — top feature influences with impact bars and direction indicators
- [ ] **Bundle similarity display** — nearest bundles with qualitative levels and context sentences
- [ ] **Uncertainty indicator** — visual confidence meter with contextual explanation text
- [ ] **Business insight card** — customer segment, retention risk, upsell potential in executive styling

### Simulation Interface

- [ ] **Counterfactual simulation panel** — sliders / inputs for income, dependents, deductible, payment schedule
- [ ] **Live re-prediction** — call `/simulate` on slider change, animate probability shifts
- [ ] **Delta visualization** — before/after comparison with bundle-change highlight
- [ ] **Updated upsell display** — show modified-scenario upsell suggestions

### Visual Polish

- [ ] **Responsive layout** — works on projector / large screen for demo day
- [ ] **Dark / light theme** — professional appearance
- [ ] **Loading states & transitions** — smooth UX during API calls
- [ ] **Export / screenshot** — allow exporting a customer analysis as PDF or image
