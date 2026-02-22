"""OLEA AI — Insurance Bundle Prediction API (Phase II)."""
from __future__ import annotations

import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.routing import APIRouter
from fastapi.staticfiles import StaticFiles

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


# ── Background model loading (so uvicorn binds port immediately) ─────
def _init_models():
    """Heavy init in background thread — port stays open while loading."""
    model.load()
    bundle_space.build()
    agent_panel.train()


@asynccontextmanager
async def lifespan(app: FastAPI):
    thread = threading.Thread(target=_init_models, daemon=True)
    thread.start()
    yield


app = FastAPI(
    title="OLEA AI",
    description="Insurance coverage bundle prediction API",
    version="2.0.0",
    lifespan=lifespan,
)

# ── CORS (allow frontend dev server) ────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
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


# ── /api/* mirror (so frontend /api/explain works in production too) ─
api_router = APIRouter(prefix="/api")
api_router.add_api_route("/health", health, methods=["GET"])
api_router.add_api_route("/predict", predict, methods=["POST"])
api_router.add_api_route("/explain", explain, methods=["POST"])
api_router.add_api_route("/simulate", simulate, methods=["POST"])
app.include_router(api_router)


# ── Serve frontend in production (Docker build copies dist → static/) ─
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.is_dir():
    app.mount("/assets", StaticFiles(directory=STATIC_DIR / "assets"), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        """Catch-all: serve index.html for any non-API route (SPA routing)."""
        file = STATIC_DIR / full_path
        if file.is_file():
            return FileResponse(file)
        return FileResponse(STATIC_DIR / "index.html")
