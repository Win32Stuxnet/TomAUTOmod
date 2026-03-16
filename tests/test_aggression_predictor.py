from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from bot.ml.predictor import Prediction


@pytest.fixture
def mock_embedder():
    embedder = MagicMock()
    seed_embeddings = np.random.rand(3, 384).astype(np.float32)
    seed_embeddings = seed_embeddings / np.linalg.norm(seed_embeddings, axis=1, keepdims=True)
    embedder.embed_batch = AsyncMock(return_value=seed_embeddings)
    return embedder


@pytest.mark.asyncio
async def test_seed_predictor_load_embeds_seeds(mock_embedder):
    from bot.aggression.predictor import SeedPredictor

    predictor = SeedPredictor(mock_embedder, seeds=["shut up", "fight me", "hate you"])
    await predictor.load()

    mock_embedder.embed_batch.assert_called_once_with(["shut up", "fight me", "hate you"])


@pytest.mark.asyncio
async def test_seed_predictor_aggressive_message(mock_embedder):
    from bot.aggression.predictor import SeedPredictor

    predictor = SeedPredictor(mock_embedder, seeds=["shut up", "fight me", "hate you"])
    await predictor.load()

    seed_matrix = mock_embedder.embed_batch.return_value
    similar_embedding = seed_matrix[0] + np.random.rand(384).astype(np.float32) * 0.01
    similar_embedding = similar_embedding / np.linalg.norm(similar_embedding)

    result = await predictor.predict(similar_embedding)

    assert isinstance(result, Prediction)
    assert result.label == "aggressive"
    assert result.confidence > 0.65


@pytest.mark.asyncio
async def test_trained_predictor_maps_not_aggressive_to_calm():
    from bot.aggression.predictor import TrainedAggressionPredictor

    predictor = TrainedAggressionPredictor(Path("fake.joblib"))
    predictor._model = MagicMock()
    predictor._model.predict.return_value = np.array(["not_aggressive"])
    predictor._model.predict_proba.return_value = np.array([[0.8, 0.2]])
    predictor._model.classes_ = np.array(["not_aggressive", "aggressive"])

    result = await predictor.predict(np.ones(384, dtype=np.float32))

    assert result.label == "calm"
    assert result.confidence == pytest.approx(0.2)
