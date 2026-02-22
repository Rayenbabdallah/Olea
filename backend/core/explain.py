"""Explainability — sensitivity, uncertainty, multi-agent narrative, upsell."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .preprocess import preprocess, DEDUCTIBLE_MAP
from .bundle_space import bundle_space
from .agents import agent_panel

# ═══════════════════════════════════════════════════════════════════════
#  Lookups
# ═══════════════════════════════════════════════════════════════════════
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

# ── (1) Business-readable feature labels ─────────────────────────────
FEATURE_LABELS: dict[str, str] = {
    # raw columns
    "Policy_Cancelled_Post_Purchase": "Prior Cancellation History",
    "Policy_Start_Year": "Policy Start Year",
    "Policy_Start_Week": "Policy Start Week",
    "Policy_Start_Day": "Policy Start Day",
    "Grace_Period_Extensions": "Grace Period Extensions",
    "Previous_Policy_Duration_Months": "Prior Coverage Duration (months)",
    "Adult_Dependents": "Adult Dependents",
    "Child_Dependents": "Child Dependents",
    "Infant_Dependents": "Infant Dependents",
    "Existing_Policyholder": "Existing Policyholder Flag",
    "Previous_Claims_Filed": "Previous Claims Filed",
    "Years_Without_Claims": "Claim-Free Years",
    "Policy_Amendments_Count": "Quote Amendments",
    "Underwriting_Processing_Days": "Underwriting Duration (days)",
    "Vehicles_on_Policy": "Vehicles on Policy",
    "Custom_Riders_Requested": "Custom Riders Requested",
    "Estimated_Annual_Income": "Estimated Annual Income",
    "Days_Since_Quote": "Days Since Quote",
    "Deductible_Tier": "Deductible Tier",
    # engineered features
    "month_num": "Policy Start Month",
    "month_sin": "Seasonal Timing (sin)",
    "month_cos": "Seasonal Timing (cos)",
    "week_sin": "Weekly Timing (sin)",
    "week_cos": "Weekly Timing (cos)",
    "total_deps": "Total Dependents",
    "has_family": "Has Family Coverage",
    "child_ratio": "Child-to-Dependent Ratio",
    "infant_ratio": "Infant-to-Dependent Ratio",
    "income_log": "Log-Scaled Income",
    "income_per_dep": "Income per Dependent",
    "claims_per_cfy": "Claims per Claim-Free Year",
    "claims_rate": "Annualized Claims Rate",
    "loyalty_idx": "Customer Loyalty Index",
    "tenure_years": "Policy Tenure (years)",
    "has_claims": "Has Prior Claims",
    "clean_record": "Clean Claims Record",
    "friction": "Underwriting Friction Score",
    "high_friction": "High Underwriting Friction",
    "amended": "Policy Was Amended",
    "quote_uw_ratio": "Quote-to-Underwriting Ratio",
    "quote_minus_uw": "Days Gap (Quote - Underwriting)",
    "amend_per_dur": "Amendments per Coverage Month",
    "has_broker": "Has Broker/Agent",
    "has_employer": "Has Employer Sponsorship",
    "ded_income": "Deductible x Income",
    "ded_deps": "Deductible x Dependents",
    "veh_per_dep": "Vehicles per Dependent",
    "riders_per_veh": "Riders per Vehicle",
    "income_x_veh": "Income x Vehicles",
    "deps_x_veh": "Dependents x Vehicles",
    "veh_x_riders": "Vehicles x Riders",
    "exist_x_dur": "Existing Customer x Tenure",
    "exist_x_claims": "Existing Customer x Claims",
    "loyalty_x_dur": "Loyalty x Tenure",
    "pay_annual": "Pays Annually",
    "pay_monthly": "Pays Monthly",
    "pay_quarterly": "Pays Quarterly",
    "was_cancelled": "Previously Cancelled",
    "ext_ratio": "Grace Extension Ratio",
}


def _label(feat: str) -> str:
    """Return a business-readable label for a feature name."""
    return FEATURE_LABELS.get(feat, feat.replace("_", " ").title())


def _format_income(income: float) -> str:
    """Hide zero/empty income in narratives."""
    if income <= 0:
        return "income not provided or minimal"
    return f"${income:,.0f}"


def _format_income_and_density(income: float, deps: float) -> str:
    """Return executive-friendly income text with conditional per-dependent context."""
    if income <= 0:
        return "Income not provided or minimal"
    income_per_dep = income / (deps + 1)
    return f"Income {_format_income(income)} ({income_per_dep:,.0f} per dependent)"


def _consensus_from_gap(gap: float) -> dict[str, str]:
    """Deterministic consensus tier derived from top-2 probability gap."""
    if gap >= 0.20:
        return {
            "level": "high",
            "summary": "Executive consensus is strong",
            "detail": f"The lead bundle is ahead by {gap:.1%}, indicating clear strategic alignment.",
        }
    if gap >= 0.10:
        return {
            "level": "medium",
            "summary": "Executive consensus is balanced",
            "detail": f"The lead bundle is ahead by {gap:.1%}; a secondary pathway remains commercially credible.",
        }
    return {
        "level": "low",
        "summary": "Executive consensus is limited",
        "detail": f"The lead bundle is ahead by only {gap:.1%}, so alternatives remain materially close.",
    }


# ═══════════════════════════════════════════════════════════════════════
#  (2) Uncertainty — with contextual explanation
# ═══════════════════════════════════════════════════════════════════════
def compute_uncertainty(probs: list[float]) -> dict[str, Any]:
    """Entropy-based uncertainty with contextual narrative."""
    p = np.array(probs, dtype=np.float64)
    p = np.clip(p, 1e-12, 1.0)
    entropy = float(-np.sum(p * np.log(p)))

    top2 = sorted(probs, reverse=True)[:2]
    gap = top2[0] - top2[1]

    if entropy < 1.0:
        level = "low"
        explanation = (
            f"The model is highly confident (entropy {entropy:.2f}). "
            f"The top prediction leads the runner-up by {gap:.1%} — "
            f"this recommendation is strong."
        )
    elif entropy < 1.7:
        level = "medium"
        explanation = (
            f"Moderate uncertainty (entropy {entropy:.2f}). "
            f"The gap between the top two bundles is {gap:.1%}. "
            f"The recommendation is reasonable but a second opinion or "
            f"additional customer data could sharpen it."
        )
    else:
        level = "high"
        explanation = (
            f"High uncertainty (entropy {entropy:.2f}). "
            f"Multiple bundles have similar probabilities (top-2 gap only {gap:.1%}). "
            f"Consider gathering more customer information before finalizing."
        )

    return {
        "entropy": round(entropy, 4),
        "level": level,
        "explanation": explanation,
    }


# ═══════════════════════════════════════════════════════════════════════
#  Sensitivity Analysis — returns business-labeled features
# ═══════════════════════════════════════════════════════════════════════
def sensitivity_analysis(
    raw: dict[str, Any],
    model: Any,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """Perturb each numeric feature by +/-5% and measure impact."""
    df_base = pd.DataFrame([raw])
    df_base = preprocess(df_base)
    X_base = df_base.reindex(columns=model.feature_cols)
    for c in model.cat_cols:
        if c in X_base.columns:
            X_base[c] = X_base[c].astype("category")
    probs_base = model.booster.predict(X_base)[0]
    pred_class = int(np.argmax(probs_base))
    base_prob = float(probs_base[pred_class])

    numeric_feats = [c for c in model.feature_cols if c not in model.cat_cols]
    impacts: list[tuple[str, float]] = []

    for feat in numeric_feats:
        val = float(X_base[feat].iloc[0])
        delta = 0.05 * (abs(val) + 1)
        total = 0.0
        for sign in (+1, -1):
            X_pert = X_base.copy()
            X_pert[feat] = val + sign * delta
            for c in model.cat_cols:
                if c in X_pert.columns:
                    X_pert[c] = X_pert[c].astype("category")
            probs_pert = model.booster.predict(X_pert)[0]
            total += float(probs_pert[pred_class]) - base_prob
        impacts.append((feat, total / 2.0))

    impacts.sort(key=lambda t: abs(t[1]), reverse=True)
    return [
        {
            "feature": f,
            "label": _label(f),
            "impact": round(imp, 6),
            "direction": "positive" if imp > 0 else "negative",
        }
        for f, imp in impacts[:top_k]
    ]


# ═══════════════════════════════════════════════════════════════════════
#  (3) Nearest Bundles — with qualitative similarity levels
# ═══════════════════════════════════════════════════════════════════════
def _similarity_level(sim: float) -> str:
    if sim >= 0.995:
        return "very strong"
    if sim >= 0.98:
        return "strong"
    if sim >= 0.95:
        return "moderate"
    return "weak"


def nearest_bundles_enriched(
    bundle_id: int, top_k: int = 3,
) -> list[dict[str, Any]]:
    """Wrap bundle_space.nearest_bundles with qualitative labels."""
    raw = bundle_space.nearest_bundles(bundle_id, top_k)
    enriched: list[dict[str, Any]] = []
    for b in raw:
        sim_raw = float(b["similarity"])
        overlap_pct = int(sim_raw * 100)
        enriched.append(
            {
                "bundle": b["bundle"],
                "bundle_name": BUNDLE_NAMES.get(b["bundle"], f"Bundle {b['bundle']}"),
                "similarity_level": _similarity_level(sim_raw),
                "similarity_context": f"Represents approximately {overlap_pct}% profile overlap.",
            }
        )
    return enriched


# ═══════════════════════════════════════════════════════════════════════
#  (4) Multi-Agent Decision Timeline
# ═══════════════════════════════════════════════════════════════════════
def generate_narrative(
    raw: dict[str, Any],
    predicted_bundle: int,
    confidence: float,
    probabilities: list[float],
    top_influences: list[dict[str, Any]],
    uncertainty: dict[str, Any],
    nearest: list[dict[str, Any]],
    agent_votes: list[dict[str, Any]] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Build a 4-agent decision timeline narrative with real agent votes."""
    bundle_name = BUNDLE_NAMES.get(predicted_bundle, f"Bundle {predicted_bundle}")
    income = raw.get("Estimated_Annual_Income") or 0
    adults = raw.get("Adult_Dependents") or 0
    children = raw.get("Child_Dependents") or 0
    infants = raw.get("Infant_Dependents") or 0
    deps = adults + children + infants
    claims = raw.get("Previous_Claims_Filed") or 0
    cf = raw.get("Years_Without_Claims") or 0
    dur = raw.get("Previous_Policy_Duration_Months") or 0
    existing = raw.get("Existing_Policyholder") or 0
    veh = raw.get("Vehicles_on_Policy") or 0
    payment = raw.get("Payment_Schedule") or ""
    ded = raw.get("Deductible_Tier") or ""
    grace = raw.get("Grace_Period_Extensions") or 0

    inf_labels = (
        ", ".join(i["label"] for i in top_influences[:3])
        if top_influences else "general profile"
    )
    top2 = sorted(probabilities, reverse=True)[:2]
    prob_gap = top2[0] - top2[1] if len(top2) == 2 else 0.0
    consensus = _consensus_from_gap(prob_gap)

    # ── Helper: format agent vote annotation ─────────────────────────
    def _vote_tag(agent_name: str) -> str:
        """Return a vote annotation string if real agent votes are available."""
        if not agent_votes:
            return ""
        for v in agent_votes:
            if v["agent"] == agent_name:
                pick = BUNDLE_NAMES.get(v["top_bundle"], f"Bundle {v['top_bundle']}")
                return (
                    f" Agent vote: {pick} ({v['confidence']:.0%} confidence)."
                )
        return ""

    # ── Agent 1: Life-Stage Agent ────────────────────────────────────
    if deps == 0:
        life_stage = "single / no dependents"
        life_needs = "individual-focused coverage with lean premiums"
    elif infants > 0:
        life_stage = "young family with infant(s)"
        life_needs = "comprehensive family coverage with pediatric benefits"
    elif children > 0:
        life_stage = "growing family with children"
        life_needs = "broad family protection including education-age riders"
    elif adults >= 3:
        life_stage = "multi-adult household"
        life_needs = "group or family plans with adult wellness options"
    else:
        life_stage = "couple / small household"
        life_needs = "balanced dual-coverage with flexible riders"

    life_summary = (
        f"[Life-Stage Agent] Profile: {life_stage} "
        f"({int(adults)} adult, {int(children)} child, {int(infants)} infant dependents). "
        f"Recommended coverage orientation: {life_needs}."
        f"{_vote_tag('Life-Stage Agent')}"
    )

    # ── Agent 2: Financial Agent ─────────────────────────────────────
    income_per_dep = income / (deps + 1)
    if income > 80_000:
        fin_tier = "high-income"
        fin_rec = "premium or comprehensive bundles are affordable and appropriate"
    elif income > 35_000:
        fin_tier = "mid-income"
        fin_rec = "standard bundles with optional riders offer best value"
    else:
        fin_tier = "budget-conscious"
        fin_rec = "basic or liability plans minimize out-of-pocket while maintaining coverage"

    payment_note = {
        "Annual_Upfront": "Annual upfront payment signals financial stability.",
        "Monthly_EFT": "Monthly EFT payments suggest cash-flow sensitivity.",
        "Quarterly_Invoice": "Quarterly invoicing indicates moderate payment flexibility.",
    }.get(payment, "")

    fin_summary = (
        f"[Financial Agent] {_format_income_and_density(income, deps)} — {fin_tier} segment. "
        f"{fin_rec.capitalize()}. "
        f"Deductible tier: {ded or 'unspecified'}. {payment_note}"
        f"{_vote_tag('Financial Agent')}"
    )

    # ── Agent 3: Stability Agent ─────────────────────────────────────
    if existing and cf >= 3 and claims == 0:
        stability = "excellent — loyal, claim-free existing customer"
    elif existing and dur > 24:
        stability = "good — long-tenured policyholder with track record"
    elif claims == 0:
        stability = "promising — no claims history, relatively new"
    elif claims <= 2:
        stability = "moderate — limited claims, manageable risk"
    else:
        stability = "elevated risk — multiple prior claims on record"

    veh_note = f" {veh} vehicle(s) on policy." if veh > 0 else ""
    grace_note = (
        f" {grace} grace extension(s) used — monitor payment reliability."
        if grace > 0 else ""
    )

    stab_summary = (
        f"[Stability Agent] Risk assessment: {stability}. "
        f"{dur} months prior coverage, {cf} claim-free years, "
        f"{claims} claim(s).{veh_note}{grace_note}"
        f"{_vote_tag('Stability Agent')}"
    )

    # ── Agent 4: Meta Decision Engine ────────────────────────────────
    runner_up = nearest[0]["bundle_name"] if nearest else "N/A"
    runner_sim = nearest[0].get("similarity_level", "") if nearest else ""
    runner_ctx = nearest[0].get("similarity_context", "") if nearest else ""

    if consensus["level"] == "high":
        conf_note = (
            f"{consensus['summary']}: {consensus['detail']} "
            f"The nearest alternative is {runner_up} ({runner_sim}; {runner_ctx})."
        )
    elif consensus["level"] == "medium":
        conf_note = (
            f"{consensus['summary']}: {consensus['detail']} "
            f"Consider presenting {runner_up} ({runner_sim}; {runner_ctx}) as a structured secondary option."
        )
    else:
        conf_note = (
            f"{consensus['summary']}: {consensus['detail']} "
            f"A dual-track recommendation between {bundle_name} and {runner_up} is advisable for executive review."
        )

    # ── Agent agreement annotation for Meta ───────────────────────
    if agent_votes:
        agree_count = sum(1 for v in agent_votes if v["top_bundle"] == predicted_bundle)
        agree_note = f" {agree_count}/{len(agent_votes)} specialist agents independently concur."
    else:
        agree_note = ""

    meta_summary = (
        f"[Meta Decision Engine] After synthesising life-stage, financial, and stability signals, "
        f"the engine recommends **{bundle_name}** at {confidence:.1%} confidence. "
        f"Key drivers: {inf_labels}. {conf_note}{agree_note}"
    )

    steps = [
        {"step": 1, "agent": "Life-Stage Agent",
         "title": "Life-Stage Analysis", "summary": life_summary},
        {"step": 2, "agent": "Financial Agent",
         "title": "Financial Fit Assessment", "summary": fin_summary},
        {"step": 3, "agent": "Stability Agent",
         "title": "Risk & Stability Review", "summary": stab_summary},
        {"step": 4, "agent": "Meta Decision Engine",
         "title": "Final Recommendation", "summary": meta_summary},
    ]
    return {"decision_timeline": steps}


# ═══════════════════════════════════════════════════════════════════════
#  (5) Upsell / Cross-Sell — always proactive
# ═══════════════════════════════════════════════════════════════════════

# Cross-sell adjacency map: for each bundle, which others complement it
_CROSS_SELL: dict[int, list[tuple[int, str]]] = {
    0: [(5, "Pair Auto_Comprehensive with Home_Premium for a full property shield."),
        (7, "Add Premium_Health_Life to secure personal health alongside vehicle coverage.")],
    1: [(0, "Upgrade from liability-only to Auto_Comprehensive for collision & theft protection."),
        (2, "Bundle basic health coverage alongside auto liability for holistic protection.")],
    2: [(4, "Enhance Basic_Health with dental & vision via Health_Dental_Vision."),
        (7, "Step up to Premium_Health_Life for life insurance and expanded medical.")],
    3: [(5, "Complement Family_Comprehensive with Home_Premium to cover the family home."),
        (4, "Add Health_Dental_Vision for preventive care across all family members.")],
    4: [(7, "Upgrade to Premium_Health_Life for life-insurance coverage on top of dental & vision."),
        (3, "Consider Family_Comprehensive if household size grows.")],
    5: [(6, "If budget tightens, Home_Standard offers solid coverage at lower cost."),
        (0, "Add Auto_Comprehensive to protect vehicles alongside the home.")],
    6: [(5, "Upgrade to Home_Premium for extended liability and higher rebuild limits."),
        (8, "Add Renter_Basic for a secondary rental property.")],
    7: [(3, "Add Family_Comprehensive if dependents increase."),
        (5, "Protect property with Home_Premium alongside life & health.")],
    8: [(9, "Upgrade to Renter_Premium for broader personal-property and liability coverage."),
        (2, "Add Basic_Health to ensure medical coverage alongside renter's insurance.")],
    9: [(5, "If transitioning to homeownership, consider Home_Premium."),
        (4, "Add Health_Dental_Vision for preventive-care savings.")],
}


def generate_upsell(
    raw: dict[str, Any],
    predicted_bundle: int,
    top_influences: list[dict[str, Any]],
) -> list[str]:
    """Always-proactive upsell / cross-sell suggestions."""
    suggestions: list[str] = []
    income = raw.get("Estimated_Annual_Income") or 0
    deps = (
        (raw.get("Adult_Dependents") or 0)
        + (raw.get("Child_Dependents") or 0)
        + (raw.get("Infant_Dependents") or 0)
    )
    grace = raw.get("Grace_Period_Extensions") or 0
    payment = raw.get("Payment_Schedule") or ""
    ded_tier = raw.get("Deductible_Tier") or ""
    income_per_dep = income / (deps + 1)

    # Rule 1 — price-sensitive → annual discount
    is_monthly = payment == "Monthly_EFT"
    if grace >= 1 or is_monthly:
        suggestions.append(
            "Revenue Strategy — Payment Optimization: Customer pays monthly"
            + (", with grace extensions" if grace >= 1 else "")
            + ". Introduce a structured annual-prepay incentive to improve renewal stickiness and reduce servicing cost."
        )

    # Rule 2 — high income per dependent → premium upgrade
    if income_per_dep > 40_000 and predicted_bundle not in (7,):
        suggestions.append(
            f"Portfolio Expansion — Income per dependent is ${income_per_dep:,.0f}. "
            "Position Premium_Health_Life (bundle 7) as a value-accretive upgrade with broader protection depth."
        )

    # Rule 3 — many dependents → family comprehensive
    if deps >= 3 and predicted_bundle != 3:
        suggestions.append(
            f"Household Strategy — Household has {deps} dependents. "
            "Recommend Family_Comprehensive (bundle 3) to consolidate coverage and improve household-level economics."
        )

    # Rule 4 — low deductible + high income → higher deductible
    low_ded = ded_tier in ("Tier_4_Zero_Ded", "Tier_3_Low_Ded")
    if low_ded and income > 60_000:
        suggestions.append(
            "Margin Optimization — Deductible is low relative to earning capacity. "
            "Propose a deductible rebalance to reduce premium drag while enabling premium-tier package eligibility."
        )

    # ── Always-proactive cross-sell fill ─────────────────────────────
    if len(suggestions) < 2:
        cross = _CROSS_SELL.get(predicted_bundle, [])
        for _, msg in cross:
            s = f"Cross-Sell Strategy — {msg}"
            if s not in suggestions:
                suggestions.append(s)
            if len(suggestions) >= 3:
                break

    return suggestions[:3]


def generate_business_insight(
    raw: dict[str, Any],
    confidence: float,
    probabilities: list[float],
    upsell_suggestions: list[str],
) -> dict[str, str]:
    """Deterministic commercial summary for downstream business users."""
    income = float(raw.get("Estimated_Annual_Income") or 0)
    deps = (
        float(raw.get("Adult_Dependents") or 0)
        + float(raw.get("Child_Dependents") or 0)
        + float(raw.get("Infant_Dependents") or 0)
    )
    existing = int(raw.get("Existing_Policyholder") or 0)
    claims = int(raw.get("Previous_Claims_Filed") or 0)
    grace = int(raw.get("Grace_Period_Extensions") or 0)
    payment = raw.get("Payment_Schedule") or ""

    if deps >= 3:
        customer_segment = "Family Growth Portfolio"
    elif income > 80_000:
        customer_segment = "Affluent Protection Portfolio"
    elif existing:
        customer_segment = "Loyalty Renewal Portfolio"
    else:
        customer_segment = "Value Optimization Portfolio"

    top2 = sorted(probabilities, reverse=True)[:2]
    prob_gap = top2[0] - top2[1] if len(top2) == 2 else 0.0
    if grace >= 2 or (payment == "Monthly_EFT" and confidence < 0.35) or prob_gap < 0.08:
        retention_risk = "Elevated Retention Exposure"
    elif grace >= 1 or claims >= 2 or prob_gap < 0.15:
        retention_risk = "Moderate Retention Exposure"
    else:
        retention_risk = "Stable Retention Outlook"

    if income > 70_000 and len(upsell_suggestions) >= 2:
        upsell_potential = "High Expansion Headroom"
    elif income > 30_000 or len(upsell_suggestions) >= 2:
        upsell_potential = "Selective Expansion Headroom"
    else:
        upsell_potential = "Targeted Expansion Headroom"

    return {
        "customer_segment": customer_segment,
        "retention_risk": retention_risk,
        "upsell_potential": upsell_potential,
    }


# ═══════════════════════════════════════════════════════════════════════
#  Full Explain — composes everything
# ═══════════════════════════════════════════════════════════════════════
def full_explain(raw: dict[str, Any], model: Any) -> dict[str, Any]:
    """Compose the complete /explain response payload."""
    result = model.predict(raw)
    pred = result["predicted_bundle"]
    conf = result["confidence"]
    probs = result["probabilities"]

    uncertainty = compute_uncertainty(probs)
    influences = sensitivity_analysis(raw, model, top_k=3)
    nearest = nearest_bundles_enriched(pred, top_k=3)

    # ── Multi-agent votes ────────────────────────────────────────────
    votes = agent_panel.vote(raw) if agent_panel.is_ready else []
    meta = agent_panel.consensus(votes) if votes else {
        "probabilities": [0.1] * 10,
        "top_bundle": pred,
        "top_bundle_name": BUNDLE_NAMES.get(pred, ""),
        "confidence": conf,
        "agreement": 0,
        "total_agents": 0,
    }

    narrative = generate_narrative(
        raw, pred, conf, probs, influences, uncertainty, nearest,
        agent_votes=votes,
    )
    upsell = generate_upsell(raw, pred, influences)
    business_insight = generate_business_insight(raw, conf, probs, upsell)

    return {
        "predicted_bundle": pred,
        "confidence": conf,
        "probabilities": probs,
        "uncertainty": uncertainty,
        "top_influences": influences,
        "nearest_bundles": nearest,
        "narrative": narrative,
        "agent_votes": votes,
        "agent_consensus": meta,
        "upsell_suggestions": upsell,
        "business_insight": business_insight,
    }
