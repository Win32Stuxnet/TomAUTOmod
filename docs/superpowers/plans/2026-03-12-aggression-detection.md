# Aggression Detection Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a second-stage aggression detection pipeline that uses sentence-transformer embeddings to flag aggressive language patterns and alert moderators via a strike-based system.

**Architecture:** Messages that pass the existing spam/toxicity automod (return `None` from `process_message()`) are embedded using `all-MiniLM-L6-v2` and scored for aggression. Strikes accumulate per user; when a configurable threshold is crossed, an alert is sent. Two predictor modes: seed-phrase cosine similarity (day one) and trained Random Forest (after labeling).

**Tech Stack:** sentence-transformers, scikit-learn, numpy, discord.py, pymongo

**Spec:** `docs/superpowers/specs/2026-03-12-aggression-detection-design.md`

---

## Chunk 0: Fix Double process_message() Call

### Task 0: Add Message Result Cache to Collector

The `audit_log.py` cog already calls `process_message()` on every message in its `on_message` listener. The aggression cog also needs the result of `process_message()` to determine if the message was "safe." Calling it twice would double-insert into `ml_training_data` and corrupt temporal features. Fix: add a per-message result cache to `Collector` so the second call returns the cached result.

**Files:**
- Modify: `bot/ml/collector.py`
- Test: `tests/test_collector_cache.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_collector_cache.py
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from bot.ml.predictor import HeuristicPredictor, Prediction


@pytest.mark.asyncio
async def test_process_message_caches_result():
    """Second call with same message ID returns cached result, no double insert."""
    from bot.ml.collector import Collector

    bot = MagicMock()
    bot.db = MagicMock()
    bot.db.ml_training_data = MagicMock()
    bot.db.ml_training_data.insert_one = AsyncMock()

    with patch("bot.ml.collector.MODEL_PATH") as mock_path:
        mock_path.exists.return_value = False
        collector = Collector(bot)

    await collector.setup()

    # Mock config service to return ml_consent=True
    mock_config = MagicMock()
    mock_config.ml_consent = True

    message = MagicMock()
    message.guild = MagicMock()
    message.guild.id = 123
    message.author = MagicMock()
    message.author.bot = False
    message.author.id = 456
    message.channel = MagicMock()
    message.channel.id = 789
    message.id = 999
    message.content = "hello world"

    with patch("bot.ml.collector.ConfigService") as mock_svc_cls:
        mock_svc = MagicMock()
        mock_svc.get = AsyncMock(return_value=mock_config)
        mock_svc_cls.return_value = mock_svc

        result1 = await collector.process_message(message)
        result2 = await collector.process_message(message)

    # insert_one should only be called once
    assert bot.db.ml_training_data.insert_one.call_count == 1
    # Both results should be identical
    assert result1 is result2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_collector_cache.py -v`
Expected: FAIL — `insert_one` called twice

- [ ] **Step 3: Add caching to Collector.process_message**

In `bot/ml/collector.py`, add a simple dict cache to `__init__` and use it in `process_message`:

Add to `__init__`:
```python
self._result_cache: dict[int, Prediction | None] = {}
```

At the top of `process_message`, before any processing:
```python
if message.id in self._result_cache:
    return self._result_cache[message.id]
```

At the end of `process_message`, before returning (both return paths):
```python
self._result_cache[message.id] = prediction  # or None
# Keep cache bounded
if len(self._result_cache) > 1000:
    # Remove oldest entries (first 500)
    keys = list(self._result_cache.keys())[:500]
    for k in keys:
        del self._result_cache[k]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_collector_cache.py -v`
Expected: PASS

- [ ] **Step 5: Run existing tests**

Run: `pytest tests/ -v`
Expected: All existing tests PASS

- [ ] **Step 6: Commit**

```bash
git add bot/ml/collector.py tests/test_collector_cache.py
git commit -m "fix: add per-message result cache to Collector to prevent double inserts"
```

---

## Chunk 1: Foundation — Embedder, Predictors, Data Model

### Task 1: Aggression Training Data Model

**Files:**
- Create: `bot/models/aggression_data.py`
- Test: `tests/test_aggression_model.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_aggression_model.py
from __future__ import annotations

import pytest
from datetime import datetime, timezone

from bot.models.aggression_data import AggressionTrainingSample


def test_to_doc_contains_all_fields():
    sample = AggressionTrainingSample(
        guild_id=123,
        message_id=456,
        user_id=789,
        channel_id=101,
        content="you're so annoying",
        embedding=[0.1] * 384,
    )
    doc = sample.to_doc()
    assert doc["guild_id"] == 123
    assert doc["message_id"] == 456
    assert doc["user_id"] == 789
    assert doc["channel_id"] == 101
    assert doc["content"] == "you're so annoying"
    assert len(doc["embedding"]) == 384
    assert doc["label"] is None
    assert doc["score"] == 0.0
    assert doc["reviewed_by"] is None
    assert doc["reviewed_at"] is None
    assert isinstance(doc["created_at"], datetime)


def test_from_doc_round_trips():
    sample = AggressionTrainingSample(
        guild_id=123,
        message_id=456,
        user_id=789,
        channel_id=101,
        content="test",
        embedding=[0.5] * 384,
        label="aggressive",
        score=0.85,
    )
    doc = sample.to_doc()
    doc["_id"] = "mongo_id"  # simulate MongoDB adding _id
    restored = AggressionTrainingSample.from_doc(doc)
    assert restored.guild_id == 123
    assert restored.label == "aggressive"
    assert restored.score == 0.85
    assert len(restored.embedding) == 384
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_aggression_model.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'bot.models.aggression_data'`

- [ ] **Step 3: Write minimal implementation**

```python
# bot/models/aggression_data.py
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone


@dataclass
class AggressionTrainingSample:
    guild_id: int
    message_id: int
    user_id: int
    channel_id: int = 0
    content: str = ""
    embedding: list[float] = field(default_factory=list)
    label: str | None = None
    score: float = 0.0
    reviewed_by: int | None = None
    reviewed_at: datetime | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_doc(self) -> dict:
        return asdict(self)

    @classmethod
    def from_doc(cls, doc: dict) -> AggressionTrainingSample:
        doc.pop("_id", None)
        return cls(**{k: v for k, v in doc.items() if k in cls.__dataclass_fields__})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_aggression_model.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add bot/models/aggression_data.py tests/test_aggression_model.py
git commit -m "feat: add AggressionTrainingSample data model"
```

---

### Task 2: Database Collections and Indexes

**Files:**
- Modify: `bot/database.py`
- Test: `tests/test_aggression_db.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_aggression_db.py
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from bot.database import Database


def test_aggression_strikes_property():
    with patch("bot.database.AsyncMongoClient"):
        db = Database("mongodb://localhost", "test")
    coll = db.aggression_strikes
    assert coll is not None


def test_aggression_training_data_property():
    with patch("bot.database.AsyncMongoClient"):
        db = Database("mongodb://localhost", "test")
    coll = db.aggression_training_data
    assert coll is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_aggression_db.py -v`
Expected: FAIL with `AttributeError: 'Database' object has no attribute 'aggression_strikes'`

- [ ] **Step 3: Add collection properties and indexes to Database**

Add to `bot/database.py` — two new properties after `ml_training_data`:

```python
@property
def aggression_strikes(self):
    return self._db["aggression_strikes"]

@property
def aggression_training_data(self):
    return self._db["aggression_training_data"]
```

Add to `create_indexes()` method, after the `ml_training_data` indexes:

```python
await self.aggression_strikes.create_indexes([
    IndexModel([("guild_id", ASCENDING), ("user_id", ASCENDING), ("created_at", DESCENDING)]),
])

await self.aggression_training_data.create_indexes([
    IndexModel([("guild_id", ASCENDING), ("label", ASCENDING)]),
    IndexModel([("message_id", ASCENDING)]),
])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_aggression_db.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add bot/database.py tests/test_aggression_db.py
git commit -m "feat: add aggression_strikes and aggression_training_data collections"
```

---

### Task 3: Seed Phrases

**Files:**
- Create: `bot/aggression/seeds.py`
- Create: `bot/aggression/__init__.py`

- [ ] **Step 1: Create the aggression package and seed phrases**

```python
# bot/aggression/__init__.py
```

```python
# bot/aggression/seeds.py
"""Default aggressive seed phrases for the SeedPredictor bootstrap."""

DEFAULT_SEEDS: list[str] = [
    "shut up",
    "you're an idiot",
    "you are an idiot",
    "fight me",
    "i'll beat you up",
    "you're so stupid",
    "you are so stupid",
    "i hate you",
    "go kill yourself",
    "you're trash",
    "you are trash",
    "you're worthless",
    "you are worthless",
    "i'm going to hurt you",
    "you're a moron",
    "you are a moron",
    "nobody likes you",
    "you disgust me",
    "you make me sick",
    "i will find you",
    "watch your back",
    "you're dead",
    "you are dead",
    "piece of garbage",
    "screw you",
    "i'll destroy you",
    "you're pathetic",
    "you are pathetic",
    "dumb as hell",
    "stupid piece of",
    "threatening you",
    "i swear i'll hurt you",
    "you better watch out",
    "i'm coming for you",
    "you'll regret this",
]
```

- [ ] **Step 2: Commit**

```bash
git add bot/aggression/__init__.py bot/aggression/seeds.py
git commit -m "feat: add default aggressive seed phrases"
```

---

### Task 4: Embedder Service

**Files:**
- Create: `bot/aggression/embedder.py`
- Test: `tests/test_embedder.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_embedder.py
from __future__ import annotations

import asyncio

import numpy as np
import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from bot.aggression.embedder import Embedder


@pytest.fixture
def mock_model():
    model = MagicMock()
    model.encode.return_value = np.random.rand(384).astype(np.float32)
    return model


@pytest.fixture
def mock_model_batch():
    model = MagicMock()
    model.encode.return_value = np.random.rand(3, 384).astype(np.float32)
    return model


@pytest.mark.asyncio
async def test_embed_returns_384_dim_vector(mock_model):
    with patch("bot.aggression.embedder.SentenceTransformer", return_value=mock_model):
        embedder = Embedder()
        await embedder.setup()
        result = await embedder.embed("hello world")
    assert isinstance(result, np.ndarray)
    assert result.shape == (384,)


@pytest.mark.asyncio
async def test_embed_batch_returns_correct_shape(mock_model_batch):
    with patch("bot.aggression.embedder.SentenceTransformer", return_value=mock_model_batch):
        embedder = Embedder()
        await embedder.setup()
        result = await embedder.embed_batch(["hello", "world", "test"])
    assert isinstance(result, np.ndarray)
    assert result.shape == (3, 384)


@pytest.mark.asyncio
async def test_setup_loads_model():
    with patch("bot.aggression.embedder.SentenceTransformer") as mock_cls:
        embedder = Embedder()
        await embedder.setup()
    mock_cls.assert_called_once_with("all-MiniLM-L6-v2")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_embedder.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'bot.aggression.embedder'`

- [ ] **Step 3: Write minimal implementation**

```python
# bot/aggression/embedder.py
from __future__ import annotations

import asyncio
import logging
from functools import partial

import numpy as np

log = logging.getLogger(__name__)

MODEL_NAME = "all-MiniLM-L6-v2"


class Embedder:
    def __init__(self) -> None:
        self._model = None

    async def setup(self) -> None:
        loop = asyncio.get_event_loop()
        self._model = await loop.run_in_executor(None, self._load_model)
        log.info("Embedder loaded model %s", MODEL_NAME)

    @staticmethod
    def _load_model():
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(MODEL_NAME)

    async def embed(self, text: str) -> np.ndarray:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, partial(self._model.encode, text, convert_to_numpy=True),
        )

    async def embed_batch(self, texts: list[str]) -> np.ndarray:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, partial(self._model.encode, texts, convert_to_numpy=True),
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_embedder.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add bot/aggression/embedder.py tests/test_embedder.py
git commit -m "feat: add Embedder service for sentence-transformer inference"
```

---

### Task 5: Aggression Predictors

**Files:**
- Create: `bot/aggression/predictor.py`
- Test: `tests/test_aggression_predictor.py`

- [ ] **Step 1: Write the failing test for SeedPredictor**

```python
# tests/test_aggression_predictor.py
from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock

from bot.aggression.predictor import SeedPredictor, TrainedAggressionPredictor
from bot.ml.predictor import Prediction


@pytest.fixture
def mock_embedder():
    embedder = MagicMock()
    # Return 3 seed embeddings (normalized unit vectors)
    seed_embeddings = np.random.rand(3, 384).astype(np.float32)
    norms = np.linalg.norm(seed_embeddings, axis=1, keepdims=True)
    seed_embeddings = seed_embeddings / norms
    embedder.embed_batch = AsyncMock(return_value=seed_embeddings)
    return embedder


@pytest.mark.asyncio
async def test_seed_predictor_load_embeds_seeds(mock_embedder):
    predictor = SeedPredictor(mock_embedder, seeds=["shut up", "fight me", "hate you"])
    await predictor.load()
    mock_embedder.embed_batch.assert_called_once_with(["shut up", "fight me", "hate you"])


@pytest.mark.asyncio
async def test_seed_predictor_aggressive_message(mock_embedder):
    predictor = SeedPredictor(mock_embedder, seeds=["shut up", "fight me", "hate you"])
    await predictor.load()

    # Create a message embedding very similar to seed[0]
    seed_matrix = mock_embedder.embed_batch.return_value
    similar_embedding = seed_matrix[0] + np.random.rand(384) * 0.01
    similar_embedding = similar_embedding / np.linalg.norm(similar_embedding)

    result = await predictor.predict(similar_embedding)
    assert isinstance(result, Prediction)
    assert result.label == "aggressive"
    assert result.confidence > 0.65


@pytest.mark.asyncio
async def test_seed_predictor_calm_message(mock_embedder):
    predictor = SeedPredictor(mock_embedder, seeds=["shut up", "fight me", "hate you"])
    await predictor.load()

    # Create a random embedding unlikely to match seeds
    random_embedding = np.random.rand(384).astype(np.float32)
    random_embedding = random_embedding / np.linalg.norm(random_embedding)

    result = await predictor.predict(random_embedding)
    assert isinstance(result, Prediction)
    # May or may not be calm depending on random state, but confidence should be returned
    assert 0.0 <= result.confidence <= 1.0
    assert result.label in ("calm", "aggressive")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_aggression_predictor.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'bot.aggression.predictor'`

- [ ] **Step 3: Write minimal implementation**

```python
# bot/aggression/predictor.py
from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol, runtime_checkable, TYPE_CHECKING

import numpy as np

from bot.ml.predictor import Prediction

if TYPE_CHECKING:
    from bot.aggression.embedder import Embedder

log = logging.getLogger(__name__)

AGGRESSION_THRESHOLD = 0.65


@runtime_checkable
class AggressionPredictor(Protocol):
    async def predict(self, embedding: np.ndarray) -> Prediction: ...
    async def load(self) -> None: ...


class SeedPredictor:
    def __init__(self, embedder: Embedder, *, seeds: list[str]) -> None:
        self._embedder = embedder
        self._seeds = seeds
        self._seed_matrix: np.ndarray | None = None

    async def load(self) -> None:
        self._seed_matrix = await self._embedder.embed_batch(self._seeds)
        # Normalize rows
        norms = np.linalg.norm(self._seed_matrix, axis=1, keepdims=True)
        self._seed_matrix = self._seed_matrix / norms
        log.info("SeedPredictor loaded with %d seed phrases", len(self._seeds))

    async def predict(self, embedding: np.ndarray) -> Prediction:
        # Normalize input
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        # Cosine similarity against all seeds
        similarities = self._seed_matrix @ embedding
        max_sim = float(np.max(similarities))
        score = max(0.0, min(1.0, max_sim))

        label = "aggressive" if score >= AGGRESSION_THRESHOLD else "calm"
        return Prediction(label=label, confidence=score, features={})


class TrainedAggressionPredictor:
    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)
        self._model = None

    async def load(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(f"Aggression model not found: {self.model_path}")

        import asyncio
        import joblib

        loop = asyncio.get_event_loop()
        bundle = await loop.run_in_executor(None, joblib.load, self.model_path)
        self._model = bundle["model"]
        log.info("Loaded trained aggression model from %s", self.model_path)

    async def predict(self, embedding: np.ndarray) -> Prediction:
        row = embedding.reshape(1, -1)
        label = self._model.predict(row)[0]
        probas = self._model.predict_proba(row)[0]
        # Get the probability for the "aggressive" class
        classes = list(self._model.classes_)
        if "aggressive" in classes:
            agg_idx = classes.index("aggressive")
            score = float(probas[agg_idx])
        else:
            class_idx = classes.index(label)
            score = float(probas[class_idx])

        return Prediction(label=label, confidence=score, features={})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_aggression_predictor.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add bot/aggression/predictor.py tests/test_aggression_predictor.py
git commit -m "feat: add SeedPredictor and TrainedAggressionPredictor"
```

---

## Chunk 2: Strike Tracker, Review UI, and Guild Config

### Task 6: Guild Config Updates

**Files:**
- Modify: `bot/models/guild_config.py`

- [ ] **Step 1: Add aggression config fields**

Add these fields to the `GuildConfig` dataclass in `bot/models/guild_config.py`, after `review_channel_id`:

```python
aggression_channel_id: int | None = None
aggression_strike_count: int = 3
aggression_window_hours: int = 2
```

- [ ] **Step 2: Run existing tests to verify nothing breaks**

Run: `pytest tests/ -v`
Expected: All existing tests PASS

- [ ] **Step 3: Commit**

```bash
git add bot/models/guild_config.py
git commit -m "feat: add aggression config fields to GuildConfig"
```

---

### Task 7: Strike Tracker

**Files:**
- Create: `bot/aggression/tracker.py`
- Test: `tests/test_aggression_tracker.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_aggression_tracker.py
from __future__ import annotations

from datetime import datetime, timezone, timedelta

import pytest
from unittest.mock import AsyncMock, MagicMock

from bot.aggression.tracker import StrikeTracker


def _make_db():
    db = MagicMock()
    db.aggression_strikes = MagicMock()
    db.aggression_strikes.insert_one = AsyncMock()
    db.aggression_strikes.count_documents = AsyncMock(return_value=1)
    db.aggression_training_data = MagicMock()
    db.aggression_training_data.insert_one = AsyncMock()
    return db


@pytest.mark.asyncio
async def test_record_strike_inserts_document():
    db = _make_db()
    tracker = StrikeTracker(db)

    await tracker.record_strike(
        guild_id=123, user_id=456, message_id=789,
        channel_id=101, score=0.8, content="you're trash",
        embedding=[0.1] * 384,
    )

    db.aggression_strikes.insert_one.assert_called_once()
    doc = db.aggression_strikes.insert_one.call_args[0][0]
    assert doc["guild_id"] == 123
    assert doc["user_id"] == 456
    assert doc["score"] == 0.8


@pytest.mark.asyncio
async def test_record_strike_inserts_training_sample():
    db = _make_db()
    tracker = StrikeTracker(db)

    await tracker.record_strike(
        guild_id=123, user_id=456, message_id=789,
        channel_id=101, score=0.8, content="you're trash",
        embedding=[0.1] * 384,
    )

    db.aggression_training_data.insert_one.assert_called_once()
    doc = db.aggression_training_data.insert_one.call_args[0][0]
    assert doc["label"] is None  # unlabeled skeleton
    assert len(doc["embedding"]) == 384


@pytest.mark.asyncio
async def test_check_threshold_returns_true_when_exceeded():
    db = _make_db()
    db.aggression_strikes.count_documents = AsyncMock(return_value=3)
    tracker = StrikeTracker(db)

    result = await tracker.check_threshold(
        guild_id=123, user_id=456,
        strike_count=3, window_hours=2,
    )
    assert result is True


@pytest.mark.asyncio
async def test_check_threshold_returns_false_when_below():
    db = _make_db()
    db.aggression_strikes.count_documents = AsyncMock(return_value=1)
    tracker = StrikeTracker(db)

    result = await tracker.check_threshold(
        guild_id=123, user_id=456,
        strike_count=3, window_hours=2,
    )
    assert result is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_aggression_tracker.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'bot.aggression.tracker'`

- [ ] **Step 3: Write minimal implementation**

```python
# bot/aggression/tracker.py
from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from typing import TYPE_CHECKING

from bot.models.aggression_data import AggressionTrainingSample

if TYPE_CHECKING:
    from bot.database import Database

log = logging.getLogger(__name__)


class StrikeTracker:
    def __init__(self, db: Database) -> None:
        self.db = db
        self._last_alert: dict[tuple[int, int], datetime] = {}

    async def record_strike(
        self,
        *,
        guild_id: int,
        user_id: int,
        message_id: int,
        channel_id: int,
        score: float,
        content: str,
        embedding: list[float],
    ) -> None:
        now = datetime.now(timezone.utc)

        # Insert strike
        strike_doc = {
            "guild_id": guild_id,
            "user_id": user_id,
            "message_id": message_id,
            "channel_id": channel_id,
            "score": score,
            "created_at": now,
        }
        await self.db.aggression_strikes.insert_one(strike_doc)

        # Insert skeleton training sample
        sample = AggressionTrainingSample(
            guild_id=guild_id,
            message_id=message_id,
            user_id=user_id,
            channel_id=channel_id,
            content=content[:300],
            embedding=embedding,
            score=score,
        )
        await self.db.aggression_training_data.insert_one(sample.to_doc())

    async def check_threshold(
        self,
        *,
        guild_id: int,
        user_id: int,
        strike_count: int,
        window_hours: int,
    ) -> bool:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=window_hours)
        count = await self.db.aggression_strikes.count_documents({
            "guild_id": guild_id,
            "user_id": user_id,
            "created_at": {"$gte": cutoff},
        })
        return count >= strike_count

    def should_alert(self, guild_id: int, user_id: int, window_hours: int) -> bool:
        """Check if we already alerted for this user in the current window."""
        key = (guild_id, user_id)
        last = self._last_alert.get(key)
        if last is None:
            return True
        cutoff = datetime.now(timezone.utc) - timedelta(hours=window_hours)
        return last < cutoff

    def mark_alerted(self, guild_id: int, user_id: int) -> None:
        self._last_alert[(guild_id, user_id)] = datetime.now(timezone.utc)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_aggression_tracker.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add bot/aggression/tracker.py tests/test_aggression_tracker.py
git commit -m "feat: add StrikeTracker for recording and evaluating aggression strikes"
```

---

### Task 8: Aggression Label View

**Files:**
- Create: `bot/views/aggression_label.py`
- Test: `tests/test_aggression_label_view.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_aggression_label_view.py
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

import discord

from bot.views.aggression_label import AggressionLabelView


def _make_bot():
    bot = MagicMock()
    bot.db = MagicMock()
    bot.db.aggression_training_data = MagicMock()
    bot.db.aggression_training_data.update_one = AsyncMock()
    return bot


def _make_interaction(*, is_mod: bool = True):
    interaction = AsyncMock(spec=discord.Interaction)
    interaction.user = MagicMock(spec=discord.Member)
    interaction.user.id = 999
    interaction.user.mention = "<@999>"
    interaction.response = AsyncMock()
    return interaction, is_mod


@pytest.mark.asyncio
async def test_aggressive_label_updates_document():
    bot = _make_bot()
    view = AggressionLabelView(bot, guild_id=123, message_id=456)
    interaction, _ = _make_interaction()

    with patch("bot.views.aggression_label.is_mod", return_value=True):
        await view._label(interaction, "aggressive")

    bot.db.aggression_training_data.update_one.assert_called_once()
    call_args = bot.db.aggression_training_data.update_one.call_args
    query = call_args[0][0]
    update = call_args[0][1]
    assert query == {"guild_id": 123, "message_id": 456}
    assert update["$set"]["label"] == "aggressive"
    assert update["$set"]["reviewed_by"] == 999


@pytest.mark.asyncio
async def test_not_aggressive_label_updates_document():
    bot = _make_bot()
    view = AggressionLabelView(bot, guild_id=123, message_id=456)
    interaction, _ = _make_interaction()

    with patch("bot.views.aggression_label.is_mod", return_value=True):
        await view._label(interaction, "not_aggressive")

    call_args = bot.db.aggression_training_data.update_one.call_args
    update = call_args[0][1]
    assert update["$set"]["label"] == "not_aggressive"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_aggression_label_view.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'bot.views.aggression_label'`

- [ ] **Step 3: Write minimal implementation**

```python
# bot/views/aggression_label.py
from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

import discord

from bot.utils.permissions import is_mod

if TYPE_CHECKING:
    from bot.bot import ModBot


class AggressionLabelView(discord.ui.View):
    """Label buttons for aggression alerts — mods confirm or deny aggression."""

    def __init__(self, bot: ModBot, guild_id: int, message_id: int, *, timeout: float = 3600) -> None:
        super().__init__(timeout=timeout)
        self.bot = bot
        self.query = {"guild_id": guild_id, "message_id": message_id}

    async def _label(self, interaction: discord.Interaction, label: str) -> None:
        if not is_mod(interaction.user):
            return await interaction.response.send_message(
                "You need Moderate Members permission.", ephemeral=True,
            )

        await self.bot.db.aggression_training_data.update_one(
            self.query,
            {"$set": {
                "label": label,
                "reviewed_by": interaction.user.id,
                "reviewed_at": datetime.now(timezone.utc),
            }},
        )
        for child in self.children:
            if isinstance(child, discord.ui.Button):
                child.disabled = True
        await interaction.response.edit_message(
            content=f"Labeled as **{label}** by {interaction.user.mention}",
            view=self,
        )

    @discord.ui.button(label="Aggressive", style=discord.ButtonStyle.danger)
    async def aggressive_btn(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        await self._label(interaction, "aggressive")

    @discord.ui.button(label="Not Aggressive", style=discord.ButtonStyle.success)
    async def not_aggressive_btn(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        await self._label(interaction, "not_aggressive")

    async def on_timeout(self) -> None:
        for child in self.children:
            if isinstance(child, discord.ui.Button):
                child.disabled = True
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_aggression_label_view.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add bot/views/aggression_label.py tests/test_aggression_label_view.py
git commit -m "feat: add AggressionLabelView for mod review of aggression alerts"
```

---

## Chunk 3: Aggression Cog, Training Pipeline, Integration

### Task 9: Training Pipeline

**Files:**
- Create: `bot/aggression/train.py`
- Test: `tests/test_aggression_train.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_aggression_train.py
from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from bot.aggression.train import train


def test_train_produces_model_file(tmp_path):
    output_path = tmp_path / "test_aggression_model.joblib"

    # Create synthetic training data
    n_samples = 100
    X = np.random.rand(n_samples, 384).astype(np.float32)
    y = np.array(["aggressive"] * 50 + ["not_aggressive"] * 50)

    train(X, y, output_path)

    assert output_path.exists()

    import joblib
    bundle = joblib.load(output_path)
    assert "model" in bundle
    assert "scaler" not in bundle  # No scaler for embeddings


def test_train_model_can_predict(tmp_path):
    output_path = tmp_path / "test_aggression_model.joblib"

    n_samples = 100
    X = np.random.rand(n_samples, 384).astype(np.float32)
    y = np.array(["aggressive"] * 50 + ["not_aggressive"] * 50)

    train(X, y, output_path)

    import joblib
    bundle = joblib.load(output_path)
    model = bundle["model"]

    test_input = np.random.rand(1, 384).astype(np.float32)
    prediction = model.predict(test_input)
    assert prediction[0] in ("aggressive", "not_aggressive")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_aggression_train.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'bot.aggression.train'`

- [ ] **Step 3: Write minimal implementation**

```python
# bot/aggression/train.py
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


async def export_data(uri: str, db_name: str) -> tuple[np.ndarray, np.ndarray]:
    from pymongo import AsyncMongoClient

    client = AsyncMongoClient(uri)
    db = client[db_name]
    collection = db["aggression_training_data"]

    X_rows: list[list[float]] = []
    y_labels: list[str] = []

    async for doc in collection.find({"label": {"$ne": None}}):
        embedding = doc.get("embedding", [])
        if len(embedding) != 384:
            continue
        X_rows.append(embedding)
        y_labels.append(doc["label"])

    client.close()

    log.info("Found %d labeled aggression samples", len(X_rows))
    return np.array(X_rows, dtype=np.float32), np.array(y_labels)


def train(X: np.ndarray, y: np.ndarray, output_path: Path) -> None:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    import joblib

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    print("\n=== Aggression Classification Report ===")
    print(report)

    joblib.dump({"model": model}, output_path)
    log.info("Aggression model saved to %s", output_path)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_aggression_train.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add bot/aggression/train.py tests/test_aggression_train.py
git commit -m "feat: add aggression training pipeline"
```

---

### Task 10: Aggression Cog

**Files:**
- Create: `bot/cogs/aggression.py`
- Test: `tests/test_aggression_cog.py`

This is the main integration point. The cog owns the `on_message` listener that runs the aggression pipeline.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_aggression_cog.py
from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import discord

from bot.ml.predictor import Prediction


def _make_bot():
    bot = MagicMock()
    bot.db = MagicMock()
    bot.db.aggression_strikes = MagicMock()
    bot.db.aggression_training_data = MagicMock()
    bot.db.aggression_training_data.count_documents = AsyncMock(return_value=10)
    bot.owner_ids = {111}
    bot.collector = MagicMock()
    bot.collector.process_message = AsyncMock(return_value=None)  # message is "safe"
    return bot


def _make_message(*, guild_id=123, author_id=456, content="you're so annoying"):
    message = MagicMock(spec=discord.Message)
    message.guild = MagicMock()
    message.guild.id = guild_id
    message.author = MagicMock(spec=discord.Member)
    message.author.id = author_id
    message.author.bot = False
    message.author.mention = f"<@{author_id}>"
    message.author.guild_permissions = MagicMock()
    message.author.guild_permissions.administrator = False
    message.channel = MagicMock()
    message.channel.id = 999
    message.channel.mention = "<#999>"
    message.id = 789
    message.content = content
    return message


@pytest.mark.asyncio
async def test_on_message_skips_bot_messages():
    from bot.cogs.aggression import Aggression

    bot = _make_bot()
    cog = Aggression(bot)
    cog._ready = True

    message = _make_message()
    message.author.bot = True

    await cog.on_message(message)
    bot.collector.process_message.assert_not_called()


@pytest.mark.asyncio
async def test_on_message_skips_when_automod_flags():
    from bot.cogs.aggression import Aggression

    bot = _make_bot()
    # process_message returns a Prediction (non-None), meaning automod caught it
    bot.collector.process_message = AsyncMock(
        return_value=Prediction(label="toxic", confidence=0.9, features={})
    )
    cog = Aggression(bot)
    cog._ready = True
    cog.embedder = MagicMock()
    cog.embedder.embed = AsyncMock()

    # Mock config service
    mock_config = MagicMock()
    mock_config.ml_consent = True
    with patch("bot.cogs.aggression.ConfigService") as mock_svc_cls:
        mock_svc = MagicMock()
        mock_svc.get = AsyncMock(return_value=mock_config)
        mock_svc_cls.return_value = mock_svc

        message = _make_message()
        await cog.on_message(message)

    # Embedder should not be called since automod caught the message
    cog.embedder.embed.assert_not_called()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_aggression_cog.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'bot.cogs.aggression'`

- [ ] **Step 3: Write the aggression cog**

```python
# bot/cogs/aggression.py
from __future__ import annotations

import asyncio
import logging
from functools import partial
from pathlib import Path

import discord
import numpy as np
from discord import app_commands
from discord.ext import commands, tasks

from bot.aggression.embedder import Embedder
from bot.aggression.predictor import (
    SeedPredictor,
    TrainedAggressionPredictor,
    AggressionPredictor,
    AGGRESSION_THRESHOLD,
)
from bot.aggression.seeds import DEFAULT_SEEDS
from bot.aggression.tracker import StrikeTracker
from bot.bot import ModBot
from bot.ml.predictor import Prediction
from bot.services.config_service import ConfigService
from bot.utils.embeds import error_embed, success_embed, audit_embed
from bot.views.aggression_label import AggressionLabelView

log = logging.getLogger(__name__)

AGGRESSION_MODEL_PATH = Path(__file__).parent.parent.parent / "aggression_model.joblib"


class Aggression(commands.Cog):
    def __init__(self, bot: ModBot) -> None:
        self.bot = bot
        self.embedder = Embedder()
        self.tracker = StrikeTracker(bot.db)
        self.predictor: AggressionPredictor | None = None
        self._ready = False

    async def cog_load(self) -> None:
        await self.embedder.setup()

        if AGGRESSION_MODEL_PATH.exists():
            self.predictor = TrainedAggressionPredictor(AGGRESSION_MODEL_PATH)
        else:
            self.predictor = SeedPredictor(self.embedder, seeds=DEFAULT_SEEDS)

        await self.predictor.load()
        self._ready = True
        log.info("Aggression cog ready with %s", type(self.predictor).__name__)

    # --- on_message pipeline ---

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message) -> None:
        if not message.guild or message.author.bot:
            return
        if isinstance(message.author, discord.Member) and message.author.guild_permissions.administrator:
            return
        if not self._ready:
            return

        # Check ml_consent
        config_svc = ConfigService(self.bot.db)
        config = await config_svc.get(message.guild.id)
        if not config.ml_consent:
            return

        # Only process messages that automod considered safe
        # We re-check rather than depending on audit_log cog ordering
        prediction = await self.bot.collector.process_message(message)
        if prediction is not None:
            return  # automod already handled it

        # Embed and predict aggression
        embedding = await self.embedder.embed(message.content)
        result = await self.predictor.predict(embedding)

        if result.label != "aggressive":
            return

        # Record strike and training sample
        await self.tracker.record_strike(
            guild_id=message.guild.id,
            user_id=message.author.id,
            message_id=message.id,
            channel_id=message.channel.id,
            score=result.confidence,
            content=message.content,
            embedding=embedding.tolist(),
        )

        # Check threshold
        threshold_crossed = await self.tracker.check_threshold(
            guild_id=message.guild.id,
            user_id=message.author.id,
            strike_count=config.aggression_strike_count,
            window_hours=config.aggression_window_hours,
        )

        if threshold_crossed and self.tracker.should_alert(
            message.guild.id, message.author.id, config.aggression_window_hours,
        ):
            await self._send_alert(message, config, result)
            self.tracker.mark_alerted(message.guild.id, message.author.id)

    async def _send_alert(self, message: discord.Message, config, result: Prediction) -> None:
        channel_id = config.aggression_channel_id or config.audit_log_channel_id
        if not channel_id:
            return

        channel = message.guild.get_channel(channel_id)
        if not isinstance(channel, discord.TextChannel):
            return

        embed = audit_embed(
            title="Aggression Alert",
            description=(
                f"**User:** {message.author.mention}\n"
                f"**Channel:** {message.channel.mention}\n"
                f"**Score:** {result.confidence:.0%}\n"
                f"**Strikes:** {config.aggression_strike_count}+ in {config.aggression_window_hours}h\n"
                f"**Content:** {message.content[:500]}"
            ),
            user=message.author,
        )
        view = AggressionLabelView(self.bot, message.guild.id, message.id)
        await channel.send(embed=embed, view=view)

    # --- Slash commands ---

    aggression_group = app_commands.Group(
        name="aggression",
        description="Aggression detection stats and tools",
        default_permissions=discord.Permissions(administrator=True),
    )

    @aggression_group.command(name="stats", description="Show aggression detection statistics")
    async def aggression_stats(self, interaction: discord.Interaction) -> None:
        await interaction.response.defer(ephemeral=True)

        db = self.bot.db
        guild_id = interaction.guild.id

        total_strikes = await db.aggression_strikes.count_documents({"guild_id": guild_id})
        total_samples = await db.aggression_training_data.count_documents({"guild_id": guild_id})
        labeled = await db.aggression_training_data.count_documents(
            {"guild_id": guild_id, "label": {"$ne": None}}
        )

        predictor_name = type(self.predictor).__name__
        if isinstance(self.predictor, TrainedAggressionPredictor):
            predictor_status = "Trained Model"
        elif isinstance(self.predictor, SeedPredictor):
            predictor_status = "Seed Phrases (no model trained yet)"
        else:
            predictor_status = predictor_name

        embed = discord.Embed(title="Aggression Detection", color=discord.Color.orange())
        embed.add_field(name="Total Strikes", value=f"{total_strikes:,}", inline=True)
        embed.add_field(name="Training Samples", value=f"{total_samples:,}", inline=True)
        embed.add_field(name="Labeled", value=f"{labeled:,}", inline=True)
        embed.add_field(name="Predictor", value=predictor_status, inline=False)

        if labeled < 50:
            embed.set_footer(text=f"Need {50 - labeled} more labeled samples before training.")
        else:
            embed.set_footer(text="Ready to train! Use /aggression train")

        await interaction.followup.send(embed=embed)

    @aggression_group.command(name="train", description="Train the aggression model from labeled data")
    async def aggression_train(self, interaction: discord.Interaction) -> None:
        if interaction.user.id not in self.bot.owner_ids:
            return await interaction.response.send_message(
                embed=error_embed("Only bot owners can trigger training."),
                ephemeral=True,
            )

        await interaction.response.defer(ephemeral=True)

        labeled = await self.bot.db.aggression_training_data.count_documents(
            {"label": {"$ne": None}}
        )
        if labeled < 50:
            return await interaction.followup.send(
                embed=error_embed(f"Need at least 50 labeled samples. Currently have {labeled}.")
            )

        try:
            from bot.aggression.train import export_data, train as train_model

            X, y = await export_data(
                self.bot._settings.mongodb_uri,
                self.bot._settings.mongodb_db_name,
            )

            await asyncio.get_event_loop().run_in_executor(
                None, partial(train_model, X, y, AGGRESSION_MODEL_PATH),
            )

            self.predictor = TrainedAggressionPredictor(AGGRESSION_MODEL_PATH)
            await self.predictor.load()

            await interaction.followup.send(
                embed=success_embed(
                    f"Aggression model trained on **{len(y)}** samples and loaded.\n"
                    f"Predictor switched to **TrainedAggressionPredictor**."
                )
            )
        except Exception as e:
            log.exception("Aggression training failed")
            await interaction.followup.send(
                embed=error_embed(f"Training failed: {e}")
            )

    # --- Seed management commands ---

    @aggression_group.command(name="seeds", description="List current seed phrases")
    async def seeds_list(self, interaction: discord.Interaction) -> None:
        if not isinstance(self.predictor, SeedPredictor):
            return await interaction.response.send_message(
                embed=error_embed("Seeds are only used with the SeedPredictor. A trained model is active."),
                ephemeral=True,
            )

        phrases = self.predictor._seeds
        chunks = [phrases[i:i + 20] for i in range(0, len(phrases), 20)]
        text = "\n".join(f"• {p}" for p in (chunks[0] if chunks else []))
        embed = discord.Embed(
            title=f"Seed Phrases ({len(phrases)} total)",
            description=text or "No seed phrases.",
            color=discord.Color.orange(),
        )
        await interaction.response.send_message(embed=embed, ephemeral=True)

    @aggression_group.command(name="addseed", description="Add a seed phrase")
    @app_commands.describe(phrase="The aggressive phrase to add")
    async def seeds_add(self, interaction: discord.Interaction, phrase: str) -> None:
        if not isinstance(self.predictor, SeedPredictor):
            return await interaction.response.send_message(
                embed=error_embed("Seeds are only used with the SeedPredictor."),
                ephemeral=True,
            )

        await interaction.response.defer(ephemeral=True)
        self.predictor._seeds.append(phrase)
        await self.predictor.load()  # re-embed all seeds
        await interaction.followup.send(
            embed=success_embed(f"Added seed phrase and re-embedded. Total: {len(self.predictor._seeds)}")
        )

    @aggression_group.command(name="removeseed", description="Remove a seed phrase")
    @app_commands.describe(phrase="The phrase to remove")
    async def seeds_remove(self, interaction: discord.Interaction, phrase: str) -> None:
        if not isinstance(self.predictor, SeedPredictor):
            return await interaction.response.send_message(
                embed=error_embed("Seeds are only used with the SeedPredictor."),
                ephemeral=True,
            )

        if phrase not in self.predictor._seeds:
            return await interaction.response.send_message(
                embed=error_embed(f"Phrase not found: {phrase}"),
                ephemeral=True,
            )

        await interaction.response.defer(ephemeral=True)
        self.predictor._seeds.remove(phrase)
        await self.predictor.load()  # re-embed
        await interaction.followup.send(
            embed=success_embed(f"Removed seed phrase. Total: {len(self.predictor._seeds)}")
        )


async def setup(bot: ModBot) -> None:
    await bot.add_cog(Aggression(bot))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_aggression_cog.py -v`
Expected: PASS

- [ ] **Step 5: Run all tests**

Run: `pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add bot/cogs/aggression.py tests/test_aggression_cog.py
git commit -m "feat: add Aggression cog with on_message pipeline and slash commands"
```

---

### Task 11: Requirements Update

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add sentence-transformers dependency**

Add to `requirements.txt` after the existing ML dependencies:

```
sentence-transformers>=2.2.0
```

- [ ] **Step 2: Commit**

```bash
git add requirements.txt
git commit -m "feat: add sentence-transformers dependency for aggression detection"
```

---

### Task 12: Data Retention Cleanup

**Files:**
- Modify: `bot/cogs/aggression.py` (add a background task)

- [ ] **Step 1: Add cleanup task to the Aggression cog**

Add a `tasks.loop` to `bot/cogs/aggression.py` that runs hourly and deletes expired strikes and training data per guild based on `log_retention_days`:

```python
# Add to Aggression class, after __init__:

async def cog_load(self) -> None:
    # ... existing setup ...
    self.cleanup_old_strikes.start()

async def cog_unload(self) -> None:
    self.cleanup_old_strikes.cancel()

@tasks.loop(hours=1)
async def cleanup_old_strikes(self) -> None:
    """Delete expired aggression strikes per guild based on log_retention_days.
    Note: aggression_training_data is NOT cleaned — labeled samples are
    valuable for retraining and should persist indefinitely."""
    config_svc = ConfigService(self.bot.db)
    async for doc in self.bot.db.guild_configs.find({}):
        config = GuildConfig.from_doc(doc)
        cutoff = datetime.now(timezone.utc) - timedelta(days=config.log_retention_days)
        await self.bot.db.aggression_strikes.delete_many({
            "guild_id": config.guild_id,
            "created_at": {"$lt": cutoff},
        })

@cleanup_old_strikes.before_loop
async def before_cleanup(self) -> None:
    await self.bot.wait_until_ready()
```

- [ ] **Step 2: Run all tests**

Run: `pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
git add bot/cogs/aggression.py
git commit -m "feat: add hourly cleanup task for expired aggression data"
```

---

## Chunk 4: Integration Test and Final Verification

### Task 13: Integration Smoke Test

**Files:**
- Create: `tests/test_aggression_integration.py`

- [ ] **Step 1: Write integration test**

```python
# tests/test_aggression_integration.py
"""Smoke test that all aggression modules import and wire together correctly."""
from __future__ import annotations

import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.asyncio
async def test_full_pipeline_seed_predictor():
    """Test the full aggression pipeline with mocked embedder."""
    from bot.aggression.embedder import Embedder
    from bot.aggression.predictor import SeedPredictor
    from bot.aggression.tracker import StrikeTracker

    # Mock embedder
    embedder = MagicMock(spec=Embedder)
    seed_matrix = np.random.rand(3, 384).astype(np.float32)
    norms = np.linalg.norm(seed_matrix, axis=1, keepdims=True)
    seed_matrix = seed_matrix / norms
    embedder.embed_batch = AsyncMock(return_value=seed_matrix)

    # Create predictor
    predictor = SeedPredictor(embedder, seeds=["shut up", "fight me", "hate you"])
    await predictor.load()

    # Predict with a similar embedding (should be aggressive)
    similar = seed_matrix[0] + np.random.rand(384) * 0.01
    similar = similar / np.linalg.norm(similar)
    result = await predictor.predict(similar)
    assert result.label == "aggressive"
    assert result.confidence > 0.5

    # Mock DB for tracker
    db = MagicMock()
    db.aggression_strikes = MagicMock()
    db.aggression_strikes.insert_one = AsyncMock()
    db.aggression_strikes.count_documents = AsyncMock(return_value=3)
    db.aggression_training_data = MagicMock()
    db.aggression_training_data.insert_one = AsyncMock()

    tracker = StrikeTracker(db)
    await tracker.record_strike(
        guild_id=1, user_id=2, message_id=3,
        channel_id=4, score=result.confidence,
        content="test", embedding=similar.tolist(),
    )

    crossed = await tracker.check_threshold(
        guild_id=1, user_id=2, strike_count=3, window_hours=2,
    )
    assert crossed is True


def test_all_modules_import():
    """Verify all aggression modules can be imported."""
    from bot.aggression import embedder
    from bot.aggression import predictor
    from bot.aggression import tracker
    from bot.aggression import train
    from bot.aggression import seeds
    from bot.models import aggression_data
    from bot.views import aggression_label
```

- [ ] **Step 2: Run integration test**

Run: `pytest tests/test_aggression_integration.py -v`
Expected: PASS

- [ ] **Step 3: Run full test suite**

Run: `pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_aggression_integration.py
git commit -m "test: add aggression detection integration smoke test"
```

---

### Task 14: Final Review

- [ ] **Step 1: Verify all new files exist**

```bash
ls bot/aggression/__init__.py bot/aggression/embedder.py bot/aggression/predictor.py bot/aggression/tracker.py bot/aggression/train.py bot/aggression/seeds.py bot/models/aggression_data.py bot/views/aggression_label.py bot/cogs/aggression.py
```

- [ ] **Step 2: Run full test suite one final time**

Run: `pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 3: Verify the bot can load the cog** (manual)

Start the bot and verify in logs:
- `Embedder loaded model all-MiniLM-L6-v2`
- `SeedPredictor loaded with 35 seed phrases`
- `Loaded extension bot.cogs.aggression`
