"""Multi-Agent Prediction Layer — domain-specialist LightGBM agents.

Each agent is trained on a focused subset of the 64 engineered features,
giving it a domain-specific "world view".  The original model.pkl remains the
sole authority for /predict; these agents only enrich /explain with real
per-agent probability votes and top-pick reasoning.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from .preprocess import preprocess, CAT_COLS

_DIR = os.path.dirname(os.path.abspath(__file__))
_TRAIN_PATH = os.path.join(_DIR, "..", "..", "train.csv")

TARGET = "Purchased_Coverage_Bundle"
ID_COL = "User_ID"

BUNDLE_NAMES: dict[int, str] = {
    0: "Auto_Comprehensive",
    1: "Auto_Liability_Basic",
    2: "Basic_Health",
    3: "Family_Comprehensive",
    4: "Health_Dental_Vision",
    5: "Home_Premium",
    6: "Home_Standard",
    7: "Premium_Health_Life",
    8: "Renter_Basic",
    9: "Renter_Premium",
}

# ═══════════════════════════════════════════════════════════════════════
#  Feature subsets — each agent sees only its domain
# ═══════════════════════════════════════════════════════════════════════
LIFE_STAGE_FEATURES = [
    "Adult_Dependents", "Child_Dependents", "Infant_Dependents",
    "total_deps", "has_family", "child_ratio", "infant_ratio",
    "income_per_dep", "veh_per_dep", "deps_x_veh", "ded_deps",
]

FINANCIAL_FEATURES = [
    "Estimated_Annual_Income", "income_log", "income_per_dep",
    "Deductible_Tier", "ded_income", "pay_annual", "pay_monthly",
    "pay_quarterly", "income_x_veh", "ext_ratio",
    "Grace_Period_Extensions",
]

STABILITY_FEATURES = [
    "Previous_Claims_Filed", "Years_Without_Claims",
    "Previous_Policy_Duration_Months", "Existing_Policyholder",
    "claims_per_cfy", "claims_rate", "loyalty_idx", "tenure_years",
    "has_claims", "clean_record", "exist_x_dur", "exist_x_claims",
    "loyalty_x_dur", "was_cancelled",
]

FRICTION_FEATURES = [
    "Policy_Amendments_Count", "Underwriting_Processing_Days",
    "Days_Since_Quote",
    "friction", "high_friction", "amended",
    "quote_uw_ratio", "quote_minus_uw", "amend_per_dur",
    "has_broker", "has_employer",
]

# ═══════════════════════════════════════════════════════════════════════
#  Agent descriptor
# ═══════════════════════════════════════════════════════════════════════
@dataclass
class _AgentSpec:
    name: str
    features: list[str]
    n_estimators: int = 25
    learning_rate: float = 0.05
    booster: lgb.Booster | None = field(default=None, repr=False)
    # cat cols that fall inside this agent's feature set
    cat_cols: list[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════
#  Agent Panel — trains & queries all agents
# ═══════════════════════════════════════════════════════════════════════
class AgentPanel:
    """Manages the 4 domain-expert agents."""

    def __init__(self) -> None:
        self.agents: list[_AgentSpec] = [
            _AgentSpec(name="Life-Stage Agent", features=LIFE_STAGE_FEATURES),
            _AgentSpec(name="Financial Agent", features=FINANCIAL_FEATURES),
            _AgentSpec(name="Stability Agent", features=STABILITY_FEATURES),
            _AgentSpec(name="Friction Agent", features=FRICTION_FEATURES),
        ]
        self._ready = False

    # ── Training ─────────────────────────────────────────────────────
    def train(self, path: str = _TRAIN_PATH) -> None:
        """Train each agent on its feature subset from train.csv."""
        df = pd.read_csv(path)
        labels = df[TARGET].astype(int)
        df_eng = preprocess(df)

        for agent in self.agents:
            # resolve available features (some may be cat)
            cols = [c for c in agent.features if c in df_eng.columns]
            agent.cat_cols = [c for c in cols if c in CAT_COLS]
            X = df_eng[cols].copy()

            for c in agent.cat_cols:
                X[c] = X[c].astype("category")

            ds = lgb.Dataset(
                X, label=labels,
                categorical_feature=agent.cat_cols if agent.cat_cols else "auto",
                free_raw_data=False,
            )

            params = {
                "objective": "multiclass",
                "num_class": 10,
                "metric": "multi_logloss",
                "learning_rate": agent.learning_rate,
                "num_leaves": 15,
                "min_child_samples": 30,
                "verbose": -1,
                "seed": 42,
            }

            agent.booster = lgb.train(
                params, ds,
                num_boost_round=agent.n_estimators,
            )
            # Store the column order used during training
            agent.features = cols  # overwrite with resolved list

        self._ready = True

    @property
    def is_ready(self) -> bool:
        return self._ready

    # ── Inference ────────────────────────────────────────────────────
    def vote(self, raw: dict[str, Any]) -> list[dict[str, Any]]:
        """Run every agent and return per-agent probability votes.

        Returns a list of dicts, one per agent:
        {
          "agent": "Life-Stage Agent",
          "top_bundle": 3,
          "top_bundle_name": "Family_Comprehensive",
          "confidence": 0.42,
          "probabilities": [0.01, 0.02, ...]   # length 10
        }
        """
        if not self._ready:
            return []

        df = pd.DataFrame([raw])
        df_eng = preprocess(df)

        results: list[dict[str, Any]] = []
        for agent in self.agents:
            X = df_eng[agent.features].copy()
            for c in agent.cat_cols:
                if c in X.columns:
                    X[c] = X[c].astype("category")

            probs = agent.booster.predict(X)[0]  # type: ignore[union-attr]
            probs_list = [round(float(p), 6) for p in probs]
            top = int(np.argmax(probs))

            results.append({
                "agent": agent.name,
                "top_bundle": top,
                "top_bundle_name": BUNDLE_NAMES[top],
                "confidence": round(float(probs[top]), 4),
                "probabilities": probs_list,
            })

        return results

    def consensus(self, votes: list[dict[str, Any]]) -> dict[str, Any]:
        """Compute meta-consensus from agent votes (equal-weight average)."""
        if not votes:
            return {"probabilities": [0.1] * 10, "top_bundle": 0, "agreement": 0}

        all_probs = np.array([v["probabilities"] for v in votes])
        avg = all_probs.mean(axis=0)
        top = int(np.argmax(avg))
        # Agreement = how many agents agree with the consensus top pick
        agreement = sum(1 for v in votes if v["top_bundle"] == top)

        return {
            "probabilities": [round(float(p), 6) for p in avg],
            "top_bundle": top,
            "top_bundle_name": BUNDLE_NAMES[top],
            "confidence": round(float(avg[top]), 4),
            "agreement": agreement,
            "total_agents": len(votes),
        }


# Module-level singleton
agent_panel = AgentPanel()
