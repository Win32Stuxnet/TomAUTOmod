from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from bot.ml.predictor import Prediction

if TYPE_CHECKING:
    from bot.aggression.embedder import Embedder

log = logging.getLogger(__name__)

AGGRESSION_THRESHOLD = 0.65


@runtime_checkable
class AggressionPredictor(Protocol):
    async def predict(self, embedding: np.ndarray) -> Prediction:
        ...

    async def load(self) -> None:
        ...


class SeedPredictor:
    def __init__(
        self,
        embedder: Embedder,
        *,
        seeds: list[str],
        threshold: float = AGGRESSION_THRESHOLD,
    ) -> None:
        self._embedder = embedder
        self._seeds = list(seeds)
        self._threshold = threshold
        self._seed_matrix: np.ndarray | None = None

    @property
    def seeds(self) -> tuple[str, ...]:
        return tuple(self._seeds)

    async def load(self) -> None:
        if not self._seeds:
            self._seed_matrix = np.empty((0, 0), dtype=np.float32)
            return

        seed_matrix = await self._embedder.embed_batch(self._seeds)
        if seed_matrix.ndim == 1:
            seed_matrix = seed_matrix.reshape(1, -1)
        self._seed_matrix = self._normalize_rows(seed_matrix)
        log.info("SeedPredictor loaded with %d seed phrases", len(self._seeds))

    async def predict(self, embedding: np.ndarray) -> Prediction:
        if self._seed_matrix is None:
            raise RuntimeError("SeedPredictor.load() must be called before predict().")
        if self._seed_matrix.size == 0:
            return Prediction(label="calm", confidence=0.0, features={})

        vector = self._normalize_vector(embedding)
        similarities = self._seed_matrix @ vector
        score = float(np.max(np.clip(similarities, 0.0, 1.0)))
        label = "aggressive" if score >= self._threshold else "calm"
        return Prediction(label=label, confidence=score, features={})

    def add_seed(self, phrase: str) -> None:
        self._seeds.append(phrase)

    def remove_seed(self, phrase: str) -> bool:
        if phrase not in self._seeds:
            return False
        self._seeds.remove(phrase)
        return True

    @staticmethod
    def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms

    @staticmethod
    def _normalize_vector(vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=np.float32)
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm


class TrainedAggressionPredictor:
    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)
        self._model = None

    async def load(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(f"Aggression model file not found: {self.model_path}")

        import joblib

        loop = asyncio.get_running_loop()
        bundle = await loop.run_in_executor(None, partial_joblib_load, self.model_path, joblib)
        self._model = bundle["model"]
        log.info("Loaded trained aggression model from %s", self.model_path)

    async def predict(self, embedding: np.ndarray) -> Prediction:
        if self._model is None:
            raise RuntimeError("TrainedAggressionPredictor.load() must be called before predict().")

        row = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        raw_label = self._model.predict(row)[0]

        score = 1.0 if raw_label == "aggressive" else 0.0
        if hasattr(self._model, "predict_proba"):
            probas = self._model.predict_proba(row)[0]
            classes = list(self._model.classes_)
            if "aggressive" in classes:
                score = float(probas[classes.index("aggressive")])
            else:
                score = float(probas[classes.index(raw_label)])

        label = "aggressive" if score >= 0.5 or raw_label == "aggressive" else "calm"
        return Prediction(label=label, confidence=score, features={})


def partial_joblib_load(model_path: Path, joblib_module):
    return joblib_module.load(model_path)
