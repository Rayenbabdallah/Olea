"""Simulate — compare base vs modified customer scenarios."""
from __future__ import annotations

from typing import Any

from .explain import generate_upsell, sensitivity_analysis


def run_simulation(
    base_raw: dict[str, Any],
    changes: dict[str, Any],
    model: Any,
) -> dict[str, Any]:
    """
    Run prediction on base and base+changes, return both results
    plus a delta summary and upsell suggestions for the modified scenario.
    """
    # ── base prediction ──────────────────────────────────────────────
    base_result = model.predict(base_raw)

    # ── modified prediction ──────────────────────────────────────────
    modified_raw = {**base_raw, **changes}
    mod_result = model.predict(modified_raw)

    # ── delta ────────────────────────────────────────────────────────
    bundle_changed = base_result["predicted_bundle"] != mod_result["predicted_bundle"]

    # find the class whose probability shifted the most
    base_probs = base_result["probabilities"]
    mod_probs = mod_result["probabilities"]
    deltas = [m - b for b, m in zip(base_probs, mod_probs)]
    abs_deltas = [abs(d) for d in deltas]
    max_idx = int(max(range(10), key=lambda i: abs_deltas[i]))

    # ── upsell for modified scenario ─────────────────────────────────
    influences = sensitivity_analysis(modified_raw, model, top_k=3)
    upsell = generate_upsell(modified_raw, mod_result["predicted_bundle"], influences)

    return {
        "base": base_result,
        "modified": mod_result,
        "delta": {
            "bundle_changed": bundle_changed,
            "top_probability_shift": {
                "class": max_idx,
                "from": round(base_probs[max_idx], 6),
                "to": round(mod_probs[max_idx], 6),
                "delta": round(deltas[max_idx], 6),
            },
        },
        "upsell_suggestions": upsell,
    }
