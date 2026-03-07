from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class Prediction:
    label: str
    confidence: float
    features: dict


@runtime_checkable
class Predictor(Protocol):
    async def predict(self, features: dict) -> Prediction:
        ...

    async def load(self) -> None:
        ...


class HeuristicPredictor:
    async def load(self) -> None:
        pass

    async def predict(self, features: dict) -> Prediction:
        score = 0.0
        reasons: list[str] = []

        if features.get("caps_ratio", 0) > 0.7 and features.get("length", 0) > 20:
            score += 0.3
            reasons.append("caps")

        if features.get("mention_count", 0) >= 5:
            score += 0.4
            reasons.append("mention_spam")

        if features.get("repeated_chars_ratio", 0) > 0.3:
            score += 0.25
            reasons.append("char_spam")

        if features.get("unique_word_ratio", 1) < 0.3 and features.get("word_count", 0) > 10:
            score += 0.3
            reasons.append("repetitive")

        if features.get("newline_count", 0) > 10:
            score += 0.2
            reasons.append("newline_spam")

        confidence = min(score, 1.0)

        if confidence >= 0.6:
            label = "toxic"
        elif confidence >= 0.3:
            label = "flagged"
        else:
            label = "safe"

        return Prediction(label=label, confidence=confidence, features=features)


class TrainedPredictor:
    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)
        self._model = None
        self._scaler = None
        self._feature_keys: list[str] = []

    async def load(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        import joblib

        bundle = joblib.load(self.model_path)
        self._model = bundle["model"]
        self._scaler = bundle["scaler"]
        self._feature_keys = bundle["features"]
        log.info("Loaded trained model from %s", self.model_path)

    async def predict(self, features: dict) -> Prediction:
        row = np.array([[float(features.get(k, 0)) for k in self._feature_keys]])
        row_scaled = self._scaler.transform(row)

        label = self._model.predict(row_scaled)[0]
        probas = self._model.predict_proba(row_scaled)[0]
        class_index = list(self._model.classes_).index(label)
        confidence = float(probas[class_index])

        return Prediction(label=label, confidence=confidence, features=features)
