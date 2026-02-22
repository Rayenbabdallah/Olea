"""Model loading and inference — loads the Phase I LightGBM model once."""
from __future__ import annotations

import os
import pickle
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from .preprocess import preprocess

_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_PATH = os.path.join(_DIR, "..", "model.pkl")


class Model:
    """Singleton-style wrapper around the trained LightGBM Booster."""

    def __init__(self) -> None:
        self.booster: lgb.Booster | None = None
        self.feature_cols: list[str] = []
        self.cat_cols: list[str] = []
        self._loaded = False

    # ── lifecycle ────────────────────────────────────────────────────
    def load(self, path: str = _MODEL_PATH) -> None:
        with open(path, "rb") as f:
            bundle: dict[str, Any] = pickle.load(f)
        self.booster = lgb.Booster(model_str=bundle["model_str"])
        self.feature_cols = bundle["feature_cols"]
        self.cat_cols = bundle["cat_cols"]
        self._loaded = True

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # ── inference ────────────────────────────────────────────────────
    def predict(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Run preprocessing + inference on a single customer record."""
        df = pd.DataFrame([raw])
        df = preprocess(df)

        X = df.reindex(columns=self.feature_cols)
        for col in self.cat_cols:
            if col in X.columns:
                X[col] = X[col].astype("category")

        probs: np.ndarray = self.booster.predict(X)[0]  # shape (10,)
        predicted = int(np.argmax(probs))

        return {
            "predicted_bundle": predicted,
            "confidence": round(float(probs[predicted]), 6),
            "probabilities": [round(float(p), 6) for p in probs],
        }


# Module-level singleton — imported by app.py
model = Model()
