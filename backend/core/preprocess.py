"""Feature engineering — mirrors the Phase I solution.py preprocess() exactly."""
from __future__ import annotations

import numpy as np
import pandas as pd

MONTH_MAP = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12,
}

DEDUCTIBLE_MAP = {
    "Tier_4_Zero_Ded": 0, "Tier_3_Low_Ded": 1,
    "Tier_2_Mid_Ded": 2, "Tier_1_High_Ded": 3,
}

CAT_COLS = [
    "Region_Code", "Broker_Agency_Type", "Acquisition_Channel",
    "Payment_Schedule", "Employment_Status",
]

DROP_COLS = ["Employer_ID", "Broker_ID"]


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same feature engineering used during training."""
    X = df.copy()

    # ── Drop high-cardinality IDs (after extracting flags) ───────────
    # Flags must be built *before* dropping the columns
    X["has_broker"] = X["Broker_ID"].notna().astype(int) if "Broker_ID" in X.columns else 0
    X["has_employer"] = X["Employer_ID"].notna().astype(int) if "Employer_ID" in X.columns else 0

    for c in DROP_COLS:
        if c in X.columns:
            X.drop(columns=[c], inplace=True)

    # ── Deductible ordinal encoding ──────────────────────────────────
    if "Deductible_Tier" in X.columns:
        X["Deductible_Tier"] = (
            X["Deductible_Tier"].map(DEDUCTIBLE_MAP).fillna(-1).astype(int)
        )

    # ── Month cyclical encoding ──────────────────────────────────────
    if "Policy_Start_Month" in X.columns:
        mn = X["Policy_Start_Month"].map(MONTH_MAP).fillna(0).astype(int)
        X["month_num"] = mn
        X["month_sin"] = np.sin(2 * np.pi * mn / 12)
        X["month_cos"] = np.cos(2 * np.pi * mn / 12)
        X.drop(columns=["Policy_Start_Month"], inplace=True)

    # ── Week cyclical encoding ───────────────────────────────────────
    if "Policy_Start_Week" in X.columns:
        w = X["Policy_Start_Week"].fillna(0)
        X["week_sin"] = np.sin(2 * np.pi * w / 53)
        X["week_cos"] = np.cos(2 * np.pi * w / 53)

    # ── Dependents ───────────────────────────────────────────────────
    adult = X.get("Adult_Dependents", 0)
    child = pd.to_numeric(X.get("Child_Dependents", 0), errors="coerce").fillna(0)
    infant = X.get("Infant_Dependents", 0)
    X["total_deps"] = adult + child + infant
    X["has_family"] = (X["total_deps"] > 0).astype(int)
    X["child_ratio"] = child / (X["total_deps"] + 1)
    X["infant_ratio"] = infant / (X["total_deps"] + 1)

    # ── Income ───────────────────────────────────────────────────────
    income = pd.to_numeric(
        X.get("Estimated_Annual_Income", 0), errors="coerce"
    ).fillna(0).clip(lower=0)
    X["income_log"] = np.log1p(income)
    X["income_per_dep"] = income / (X["total_deps"] + 1)

    # ── Claims & risk ────────────────────────────────────────────────
    claims = pd.to_numeric(X.get("Previous_Claims_Filed", 0), errors="coerce").fillna(0)
    cf = pd.to_numeric(X.get("Years_Without_Claims", 0), errors="coerce").fillna(0)
    dur = pd.to_numeric(
        X.get("Previous_Policy_Duration_Months", 0), errors="coerce"
    ).fillna(0)
    X["claims_per_cfy"] = claims / (cf + 1)
    X["claims_rate"] = claims / (dur / 12 + 1)
    X["loyalty_idx"] = cf - claims
    X["tenure_years"] = dur / 12
    X["has_claims"] = (claims > 0).astype(int)
    X["clean_record"] = ((claims == 0) & (cf > 0)).astype(int)

    # ── Friction / underwriting ──────────────────────────────────────
    amend = pd.to_numeric(X.get("Policy_Amendments_Count", 0), errors="coerce").fillna(0)
    uw = pd.to_numeric(
        X.get("Underwriting_Processing_Days", 0), errors="coerce"
    ).fillna(0)
    quote = pd.to_numeric(X.get("Days_Since_Quote", 0), errors="coerce").fillna(0)
    X["friction"] = uw + amend
    X["high_friction"] = (uw > 7).astype(int)
    X["amended"] = (amend > 0).astype(int)
    X["quote_uw_ratio"] = quote / (uw + 1)
    X["quote_minus_uw"] = quote - uw
    X["amend_per_dur"] = amend / (dur + 1)

    # ── Deductible interactions ──────────────────────────────────────
    ded = X.get("Deductible_Tier", 0)
    X["ded_income"] = ded * income
    X["ded_deps"] = ded * X["total_deps"]

    # ── Vehicle & riders ─────────────────────────────────────────────
    veh = pd.to_numeric(X.get("Vehicles_on_Policy", 0), errors="coerce").fillna(0)
    riders = pd.to_numeric(
        X.get("Custom_Riders_Requested", 0), errors="coerce"
    ).fillna(0)
    X["veh_per_dep"] = veh / (X["total_deps"] + 1)
    X["riders_per_veh"] = riders / (veh + 1)
    X["income_x_veh"] = income * veh
    X["deps_x_veh"] = X["total_deps"] * veh
    X["veh_x_riders"] = veh * riders

    # ── Policyholder interactions ────────────────────────────────────
    exist = pd.to_numeric(X.get("Existing_Policyholder", 0), errors="coerce").fillna(0)
    X["exist_x_dur"] = exist * dur
    X["exist_x_claims"] = exist * claims
    X["loyalty_x_dur"] = X["loyalty_idx"] * dur

    # ── Payment schedule dummies ─────────────────────────────────────
    if "Payment_Schedule" in X.columns:
        X["pay_annual"] = (X["Payment_Schedule"] == "Annual_Upfront").astype(int)
        X["pay_monthly"] = (X["Payment_Schedule"] == "Monthly_EFT").astype(int)
        X["pay_quarterly"] = (X["Payment_Schedule"] == "Quarterly_Invoice").astype(int)

    # ── Other flags ──────────────────────────────────────────────────
    if "Policy_Cancelled_Post_Purchase" in X.columns:
        X["was_cancelled"] = pd.to_numeric(X["Policy_Cancelled_Post_Purchase"], errors="coerce").fillna(0).astype(int)
    else:
        X["was_cancelled"] = 0
    if "Grace_Period_Extensions" in X.columns:
        grace = pd.to_numeric(X["Grace_Period_Extensions"], errors="coerce").fillna(0)
    else:
        grace = 0
    X["ext_ratio"] = grace / (dur + 1)

    # ── Categorical dtype ────────────────────────────────────────────
    for col in CAT_COLS:
        if col in X.columns:
            X[col] = X[col].astype(str).fillna("_missing_").astype("category")

    return X
