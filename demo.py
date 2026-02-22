"""OLEA AI — Demo Walkthrough Script.

Run this after starting the server:
    cd backend
    python -m uvicorn app:app --host 127.0.0.1 --port 8000

Then from the project root:
    python demo.py
"""

import json
import sys
import requests

API = "http://127.0.0.1:8000"

SAMPLE_CUSTOMER = {
    "User_ID": "DEMO_001",
    "Policy_Start_Year": 2023,
    "Policy_Start_Month": "February",
    "Policy_Start_Week": 8,
    "Policy_Start_Day": 3,
    "Grace_Period_Extensions": 1,
    "Previous_Policy_Duration_Months": 36,
    "Adult_Dependents": 1,
    "Child_Dependents": 2,
    "Infant_Dependents": 0,
    "Region_Code": "R1001",
    "Existing_Policyholder": 1,
    "Previous_Claims_Filed": 1,
    "Years_Without_Claims": 3,
    "Policy_Amendments_Count": 2,
    "Underwriting_Processing_Days": 14,
    "Vehicles_on_Policy": 2,
    "Custom_Riders_Requested": 1,
    "Broker_Agency_Type": "Retail_Agency",
    "Deductible_Tier": "Tier_3_Low_Ded",
    "Acquisition_Channel": "Agent_Referral",
    "Payment_Schedule": "Monthly_EFT",
    "Employment_Status": "Self_Employed",
    "Estimated_Annual_Income": 55000,
    "Days_Since_Quote": 5,
}

DIVIDER = "=" * 70


def section(title: str) -> None:
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)


def main() -> None:
    # ── 1. Health Check ──────────────────────────────────────────────
    section("1. HEALTH CHECK")
    try:
        r = requests.get(f"{API}/health", timeout=5)
        r.raise_for_status()
        print(json.dumps(r.json(), indent=2))
    except requests.ConnectionError:
        print("ERROR: Server not reachable. Start it first:")
        print("  cd backend && python -m uvicorn app:app --host 127.0.0.1 --port 8000")
        sys.exit(1)

    # ── 2. Predict ───────────────────────────────────────────────────
    section("2. PREDICTION")
    r = requests.post(f"{API}/predict", json=SAMPLE_CUSTOMER)
    pred = r.json()
    print(f"  Predicted Bundle : {pred['predicted_bundle']} — {pred['bundle_name']}")
    print(f"  Confidence       : {pred['confidence']:.1%}")
    print(f"  Probabilities    : {['%.3f' % p for p in pred['probabilities']]}")

    # ── 3. Explain ───────────────────────────────────────────────────
    section("3. FULL EXPLANATION")
    r = requests.post(f"{API}/explain", json=SAMPLE_CUSTOMER)
    exp = r.json()

    # 3a. Uncertainty
    u = exp["uncertainty"]
    print(f"\n  Uncertainty: {u['level']} (entropy {u['entropy']:.2f})")
    print(f"  {u['explanation']}")

    # 3b. Top influences
    print("\n  Top Feature Influences:")
    for inf in exp["top_influences"]:
        arrow = "+" if inf["direction"] == "positive" else "-"
        print(f"    [{arrow}] {inf['label']:40s} impact={inf['impact']:.6f}")

    # 3c. Nearest bundles
    print("\n  Nearest Bundles:")
    for nb in exp["nearest_bundles"]:
        print(f"    {nb['bundle_name']:25s} [{nb['similarity_level']}] {nb['similarity_context']}")

    # 3d. Agent votes
    print("\n  Multi-Agent Votes:")
    for v in exp["agent_votes"]:
        print(f"    {v['agent']:20s} -> {v['top_bundle_name']:25s} ({v['confidence']:.0%})")

    # 3e. Agent consensus
    c = exp["agent_consensus"]
    print(f"\n  Consensus: {c['top_bundle_name']} ({c['agreement']}/{c['total_agents']} agents agree, {c['confidence']:.0%})")

    # 3f. Narrative timeline
    print("\n  AI Decision Timeline:")
    for step in exp["narrative"]["decision_timeline"]:
        print(f"\n  Step {step['step']}: {step['title']} ({step['agent']})")
        # wrap long summary
        summary = step["summary"]
        while len(summary) > 90:
            cut = summary[:90].rfind(" ")
            if cut < 40:
                cut = 90
            print(f"    {summary[:cut]}")
            summary = summary[cut:].lstrip()
        if summary:
            print(f"    {summary}")

    # 3g. Upsell
    print("\n  Upsell Suggestions:")
    for i, s in enumerate(exp["upsell_suggestions"], 1):
        print(f"    {i}. {s}")

    # 3h. Business insight
    bi = exp["business_insight"]
    print(f"\n  Business Insight:")
    print(f"    Segment         : {bi['customer_segment']}")
    print(f"    Retention Risk  : {bi['retention_risk']}")
    print(f"    Upsell Potential: {bi['upsell_potential']}")

    # ── 4. Simulate ──────────────────────────────────────────────────
    section("4. COUNTERFACTUAL SIMULATION")
    changes = {
        "Estimated_Annual_Income": 120000,
        "Deductible_Tier": "Tier_2_Mid_Ded",
        "Adult_Dependents": 3,
        "Child_Dependents": 3,
    }
    print(f"  Changes applied: {json.dumps(changes)}")

    r = requests.post(f"{API}/simulate", json={"base": SAMPLE_CUSTOMER, "changes": changes})
    sim = r.json()

    base_b = sim["base"]["predicted_bundle"]
    mod_b = sim["modified"]["predicted_bundle"]
    delta = sim["delta"]

    print(f"\n  Base prediction     : Bundle {base_b} ({sim['base']['confidence']:.1%})")
    print(f"  Modified prediction : Bundle {mod_b} ({sim['modified']['confidence']:.1%})")
    print(f"  Bundle changed?     : {'YES' if delta['bundle_changed'] else 'No'}")

    shift = delta["top_probability_shift"]
    print(f"  Largest shift       : Class {shift['class']} ({shift['from']:.3f} -> {shift['to']:.3f}, delta {shift['delta']:+.3f})")

    if sim["upsell_suggestions"]:
        print(f"\n  Modified-Scenario Upsell:")
        for i, s in enumerate(sim["upsell_suggestions"], 1):
            print(f"    {i}. {s}")

    # ── Done ─────────────────────────────────────────────────────────
    section("DEMO COMPLETE")
    print("  All 4 endpoints demonstrated successfully.")
    print(f"  Server: {API}")
    print(f"  Docs:   {API}/docs\n")


if __name__ == "__main__":
    main()
