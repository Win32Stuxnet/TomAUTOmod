from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import TYPE_CHECKING

import discord

from pathlib import Path

from bot.ml.features import extract_features
from bot.ml.predictor import HeuristicPredictor, Prediction, Predictor, TrainedPredictor
from bot.models.ml_data import MLTrainingSample

MODEL_PATH = Path(__file__).parent.parent.parent / "model.joblib"

if TYPE_CHECKING:
    from bot.bot import ModBot

log = logging.getLogger(__name__)

_user_timestamps: dict[int, list[float]] = defaultdict(list)
_WINDOW_SECONDS = 60
_MAX_WINDOW = 50


class Collector:
    def __init__(self, bot: ModBot, *, predictor: Predictor | None = None) -> None:
        self.bot = bot
        self.db = bot.db

        if predictor:
            self.predictor: Predictor = predictor
        elif MODEL_PATH.exists():
            self.predictor = TrainedPredictor(MODEL_PATH)
        else:
            self.predictor = HeuristicPredictor()

    async def setup(self) -> None:
        await self.predictor.load()
        log.info("ML collector initialized with %s", type(self.predictor).__name__)

    async def process_message(self, message: discord.Message) -> Prediction | None:
        if not message.guild or message.author.bot:
            return None

        from bot.services.config_service import ConfigService
        config_svc = ConfigService(self.db)
        config = await config_svc.get(message.guild.id)
        if not config.ml_consent:
            return None

        features = extract_features(message.content)
        temporal = self._update_temporal(message.author.id)
        features.update(temporal)

        sample = MLTrainingSample(
            guild_id=message.guild.id,
            message_id=message.id,
            user_id=message.author.id,
            features=features,
        )
        await self.db.ml_training_data.insert_one(sample.to_doc())

        prediction = await self.predictor.predict(features)

        if prediction.label != "safe":
            return prediction
        return None

    def _update_temporal(self, user_id: int) -> dict:
        now = time.monotonic()
        window = _user_timestamps[user_id]

        cutoff = now - _WINDOW_SECONDS
        window[:] = [t for t in window if t > cutoff]
        window.append(now)

        if len(window) > _MAX_WINDOW:
            window[:] = window[-_MAX_WINDOW:]

        msgs_last_60s = len(window)
        if len(window) >= 2:
            intervals = [window[i] - window[i - 1] for i in range(1, len(window))]
            avg_interval = sum(intervals) / len(intervals)
        else:
            avg_interval = _WINDOW_SECONDS

        return {
            "messages_last_60s": msgs_last_60s,
            "avg_interval_seconds": round(avg_interval, 2),
        }
