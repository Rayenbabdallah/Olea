🧠 OLEA AI – FULL SYSTEM ARCHITECTURE
What We Are Building

We are building:

A Multi-Agent Cognitive Insurance Decision System
with simulation, uncertainty estimation, behavioral embeddings, and strategic upsell intelligence.

This is NOT just a classifier.

It is:

A committee of AI agents

A decision fusion brain

A sensitivity analyzer

A bundle embedding engine

An uncertainty estimator

A strategy generator

A visual decision timeline

This makes our project:

Technically advanced

Business intelligent

Visually impressive

Enterprise-ready

🏗 SYSTEM OVERVIEW

The system has 7 core components:

Feature Engineering Layer

Multi-Agent Prediction Layer

Meta-Decision Brain

Feature Influence Sensitivity Engine

Bundle Embedding Space

Uncertainty Quantification

Upsell Strategy Generator

AI Decision Timeline (Visualization)

1️⃣ FEATURE ENGINEERING LAYER

We transform raw data into meaningful behavioral signals.

We build five behavioral dimensions:

A. Life Stage Dimension

Total dependents

Income per dependent

Household responsibility score

Purpose:
Captures family-driven insurance needs.

B. Financial Behavior Dimension

Risk tolerance index

Liquidity stress score

Payment stability score

Purpose:
Captures risk appetite and financial flexibility.

C. Loyalty & Risk Dimension

Loyalty index

Claim risk score

Churn probability signal

Purpose:
Identifies premium vs unstable customers.

D. Friction & Complexity Dimension

Negotiation intensity

Processing friction score

Purpose:
Detects price sensitivity and complexity patterns.

E. Broker Influence Dimension

Smoothed broker encoding

Employer bias

Channel distribution

Purpose:
Captures sales-driven influence patterns.

This alone gives us strong leaderboard performance.

2️⃣ MULTI-AGENT PREDICTION LAYER

We build specialized agents:

Each agent sees different features and outputs:

Probability distribution over 10 bundles

Confidence score

Agents:

Life-Stage Agent

Financial Agent

Loyalty Agent

Broker Agent

Friction Agent

Behavioral Cluster Agent

Each agent focuses on its domain.

3️⃣ META-DECISION BRAIN

All agent probabilities are concatenated.

Example:

Agent1 → 10 probs
Agent2 → 10 probs
Agent3 → 10 probs

Total ≈ 60 features.

We train a final LightGBM model that learns:

Which agent to trust

How to combine their votes

How to maximize Macro F1

This is stacked generalization.

4️⃣ FEATURE INFLUENCE SENSITIVITY ENGINE

This is advanced.

For each customer:

We slightly perturb each numeric feature (small +Δ and -Δ).

We measure:

How much does the prediction change?

We rank features by influence.

Example output:

Top Influencing Factors:

Income per dependent

Deductible tier

Years without claims

This gives causal-style reasoning.

It makes our AI look analytical, not black-box.

5️⃣ BUNDLE EMBEDDING SPACE

Instead of treating bundles as independent classes,
we create a similarity space.

Steps:

Compute centroid of customer features for each bundle.

Represent each bundle as a vector.

Compute cosine similarity between bundles.

Now we can say:

Predicted: Premium_Health_Life
Close alternative: Health_Dental_Vision (similarity 0.82)

This makes the system feel like a recommendation engine.

Huge product intelligence.

6️⃣ UNCERTAINTY QUANTIFICATION

We train multiple meta-models (3 small ones with different seeds).

At prediction time:

Average predictions

Compute variance across models

Low variance → High confidence
High variance → Low confidence

We output:

Prediction: Bundle 7
Confidence: 0.83
Uncertainty: Low

Few teams will do this.

This adds technical credibility.

7️⃣ UPSELL STRATEGY GENERATOR

After prediction + sensitivity analysis:

We simulate:

What minimal change increases probability of premium bundle?

Example:

To move customer from Basic_Health → Premium_Health_Life:

Increase deductible tier

Improve loyalty discount

Adjust payment schedule

This converts prediction into actionable business strategy.

This is a major pitch advantage.

8️⃣ AI DECISION TIMELINE (Visual Feature)

During demo:

We show:

Life-Stage Agent votes

Financial Agent votes

Loyalty Agent votes

Broker Agent votes

Meta-Brain decides

Confidence displayed

It looks like an AI deliberation process.

This creates drama during presentation.

Very impactful visually.

🧠 COUNTERFACTUAL SIMULATION INTERFACE

We build a slider-based simulation panel (not full chatbot).

User can adjust:

Income

Dependents

Deductible

Payment schedule

Each adjustment:

Re-runs model

Updates bundle probabilities

Updates agent votes

Updates upsell recommendations

This turns system into interactive decision simulator.

⚙️ TECHNICAL EXECUTION PLAN

PHASE I (7 HOURS)

Build baseline LightGBM

Add feature engineering

Build agents

Train meta-model

Add uncertainty layer

Save model.pkl

Optimize speed and size

PHASE II (3 HOURS)

Build API endpoints:

/predict

/explain

/simulate

Build frontend dashboard

Add decision timeline animation

Add sensitivity panel

Add bundle similarity display

🚀 WHY THIS IS INSANE (IN A GOOD WAY)

We are combining:

Multi-agent architecture

Stacked ensemble learning

Sensitivity analysis

Embedding space modeling

Uncertainty quantification

Strategic optimization

Interactive simulation

That is:

Academic-level + Enterprise-level + Product-level.

🎯 POSITIONING STATEMENT

We are not building:

“A bundle classifier.”

We are building:

“A cognitive insurance advisory system that models behavioral intelligence, quantifies uncertainty, simulates strategic interventions, and optimizes customer-bundle alignment.”