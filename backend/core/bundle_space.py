"""Bundle Embedding Space — cosine-similarity between bundle centroids."""
from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd

from .preprocess import preprocess

_DIR = os.path.dirname(os.path.abspath(__file__))
_TRAIN_PATH = os.path.join(_DIR, "..", "..", "train.csv")

TARGET = "Purchased_Coverage_Bundle"
ID_COL = "User_ID"


class BundleSpace:
    """Computes per-bundle centroid vectors and cosine similarities."""

    def __init__(self) -> None:
        self.centroids: np.ndarray | None = None  # (10, n_features)
        self._ready = False

    # ── lifecycle ────────────────────────────────────────────────────
    def build(self, path: str = _TRAIN_PATH) -> None:
        df = pd.read_csv(path)
        labels = df[TARGET].astype(int)
        df = preprocess(df)

        # keep only numeric engineered columns
        drop = {TARGET, ID_COL}
        num_cols = [
            c for c in df.columns
            if c not in drop and pd.api.types.is_numeric_dtype(df[c])
        ]
        X = df[num_cols].fillna(0).values  # (N, d)

        # compute per-class centroid
        n_classes = 10
        centroids = np.zeros((n_classes, X.shape[1]))
        for k in range(n_classes):
            mask = labels.values == k
            if mask.sum() > 0:
                centroids[k] = X[mask].mean(axis=0)

        # normalise rows for cosine similarity
        norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        self.centroids = centroids / norms
        self._ready = True

    @property
    def is_ready(self) -> bool:
        return self._ready

    # ── queries ──────────────────────────────────────────────────────
    def nearest_bundles(
        self, bundle_id: int, top_k: int = 3
    ) -> list[dict[str, Any]]:
        """Return the top_k most similar bundles (excl. self) by cosine sim."""
        if not self._ready:
            return []
        sims = self.centroids @ self.centroids[bundle_id]  # (10,)
        ranked = np.argsort(-sims)
        results = []
        for idx in ranked:
            idx = int(idx)
            if idx == bundle_id:
                continue
            results.append({"bundle": idx, "similarity": round(float(sims[idx]), 4)})
            if len(results) >= top_k:
                break
        return results


# Module-level singleton
bundle_space = BundleSpace()
