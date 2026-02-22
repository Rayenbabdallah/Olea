"""OLEA AI — Insurance Bundle Prediction API (Phase II)."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from core.agents import agent_panel
from core.bundle_space import bundle_space
from core.explain import full_explain
from core.model import model
from core.schemas import (
    BUNDLE_NAMES,
    ExplainRequest,
    ExplainResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
    SimulateRequest,
    SimulateResponse,
)
from core.simulate import run_simulation


# ── Startup / shutdown ───────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    model.load()
    bundle_space.build()
    agent_panel.train()   # multi-agent prediction layer
    yield


app = FastAPI(
    title="OLEA AI",
    description="Insurance coverage bundle prediction API",
    version="2.0.0",
    lifespan=lifespan,
)


# ── Routes ───────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok", model_loaded=model.is_loaded)


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    if not model.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    raw = req.model_dump()
    result = model.predict(raw)

    return PredictResponse(
        predicted_bundle=result["predicted_bundle"],
        bundle_name=BUNDLE_NAMES[result["predicted_bundle"]],
        confidence=result["confidence"],
        probabilities=result["probabilities"],
    )


@app.post("/explain", response_model=ExplainResponse)
async def explain(req: ExplainRequest):
    if not model.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    raw = req.model_dump()
    return full_explain(raw, model)


@app.post("/simulate", response_model=SimulateResponse)
async def simulate(req: SimulateRequest):
    if not model.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    base_raw = req.base.model_dump()
    result = run_simulation(base_raw, req.changes, model)
    return result
