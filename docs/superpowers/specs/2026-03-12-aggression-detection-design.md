# Aggression Detection System — Design Spec

## Overview

Add semantic aggression detection as a second-stage pipeline after the existing spam/toxicity automod. Uses `sentence-transformers` (`all-MiniLM-L6-v2`) to understand message meaning and detect hostile/aggressive language that surface-level features miss.

**Goal:** Alert moderators when a user exhibits a pattern of aggressive messages, enabling early intervention in arguments.

## Architecture

```
Message arrives
    │
    ▼
[Existing spam/toxicity pipeline] ──→ flagged/toxic → existing automod actions
    │
    returns None (message is "safe")
    ▼
[Aggression cog on_message listener]
    │
    checks ml_consent
    ▼
[Embedder] → 384-dim vector
    │
    ▼
[AggressionPredictor] → score 0.0–1.0
    │
    ▼
If score > threshold → record strike + store training sample
    │
    ▼
If strikes in window ≥ limit → emit alert (first crossing only)
```

- Two-stage: existing automod is untouched, aggression runs only on messages where `process_message()` returns `None`
- The aggression cog owns its own `on_message` listener — it calls `self.bot.collector.process_message()` result is checked, then runs the aggression pipeline. This avoids modifying the existing collector
- Hybrid predictor: seed phrases (day one) → trained classifier (after labeling)
- Strike-based alerting: count aggressive messages in a time window
- Gated by `ml_consent` — same consent flag as the existing ML pipeline

## Module Structure

```
bot/aggression/
    __init__.py
    embedder.py        # Shared sentence-transformer model
    predictor.py       # AggressionPredictor protocol + SeedPredictor + TrainedAggressionPredictor
    tracker.py         # Strike recording + threshold evaluation + alert emission
    train.py           # Training pipeline for aggression classifier
    seeds.py           # Curated aggressive seed phrases
```

## Components

### 1. Embedder (`bot/aggression/embedder.py`)

Shared singleton that loads `all-MiniLM-L6-v2` once at bot startup.

- `setup()` — loads model into memory (~80MB), must run in `run_in_executor` as model loading is blocking
- `embed(text: str) -> np.ndarray` — returns 384-dim vector, runs in `run_in_executor`
- `embed_batch(texts: list[str]) -> np.ndarray` — for seed phrases and training, runs in `run_in_executor`
- All transformer inference must use `run_in_executor` — this includes `SeedPredictor.load()` which calls `embed_batch()` on startup

### 2. Predictors (`bot/aggression/predictor.py`)

Reuses the existing `Prediction` dataclass from `bot.ml.predictor` with:
- `label: str` — "calm" or "aggressive"
- `confidence: float` — 0.0 to 1.0 aggression score
- `features: dict` — empty dict (embeddings not stored here)

**AggressionPredictor protocol:**
- `async predict(embedding: np.ndarray) -> Prediction`
- `async load() -> None`

**SeedPredictor (bootstrap, works day one):**
- On `load()`, embeds curated aggressive seed phrases via the shared embedder (via `run_in_executor`)
- On `predict()`, computes cosine similarity between message embedding and all seed vectors
- Max similarity = aggression score
- Threshold (default 0.65) determines calm vs aggressive
- Note: threshold needs empirical validation against real Discord messages; should be tunable per guild in future

**TrainedAggressionPredictor (after labeling):**
- Loads Random Forest from `aggression_model.joblib`
- Input: 384-dim embedding vector
- Output: aggression probability as score
- Bundle format: `{"model": RandomForestClassifier, "features": "embedding"}` — no scaler needed, Random Forests are scale-invariant and embeddings are already normalized by the sentence-transformer
- Same `joblib.dump/load` pattern as existing automod

### 3. Strike Tracker (`bot/aggression/tracker.py`)

**Call site:** The aggression cog's `on_message` listener handles the full flow:
1. Check `ml_consent`
2. Call `self.bot.collector.process_message(message)` — if result is not `None`, message was already handled by automod, skip
3. Call `embedder.embed(message.content)` via `run_in_executor`
4. Call `predictor.predict(embedding)`
5. If aggressive → pass to tracker

**Strike recording:**
- On aggressive score → insert strike document into `aggression_strikes` collection
- Strike document: `user_id`, `guild_id`, `message_id`, `channel_id`, `score`, `content` (first 300 chars), `created_at`
- Simultaneously insert a skeleton training document into `aggression_training_data` with `label: None` (consistent with existing ML pattern where docs are inserted at detection time, then updated when a mod labels)

**Strike evaluation:**
- Configurable per guild: strike count and time window (defaults: 3 strikes in 2 hours)
- On new strike, query strike count for user in window
- If count >= threshold AND no alert was sent for this user in the current window → emit alert
- Alert deduplication: only alert on first threshold crossing per user per window. Track last alert timestamp per user in the strike query — if an alert was sent within the current window, suppress

**Alert payload:**
- User mention
- Strike count and time window
- Links to flagged messages
- Aggression scores for each
- Sent to guild-configured aggression alert channel (channel config handled by dashboard, not this module)

**Data retention:** Uses a background task (same pattern as any existing cleanup) that periodically deletes strikes older than `log_retention_days` per guild. Cannot use MongoDB TTL indexes because retention is per-guild configurable. Cleanup runs on an interval (e.g., every hour), queries and deletes expired documents per guild.

### 4. Training Pipeline (`bot/aggression/train.py`)

**Labeling:**
- Training documents are inserted at detection time with `label: None` (see Strike Tracker above)
- Mods label via review buttons on alerts → `update_one` sets `label`, `reviewed_by`, `reviewed_at` on the existing document (consistent with existing `QuickLabelView` pattern in `bot/views/label.py`)
- Collection: `aggression_training_data`
- Document: `guild_id`, `message_id`, `user_id`, `content` (first 300 chars), `embedding` (384 floats), `label` (None → "aggressive" / "not_aggressive"), `reviewed_by`, `reviewed_at`, `created_at`

**Training:**
- `/aggression train` command (owner-only)
- Exports labeled embeddings from MongoDB
- Trains Random Forest on 384-dim vectors (no StandardScaler)
- Saves to `aggression_model.joblib`
- Swaps predictor from SeedPredictor → TrainedAggressionPredictor
- Minimum 50 labeled samples required

### 5. Review UI

- Extends existing `QuickLabelView` pattern from `bot/views/label.py`
- "Aggressive" / "Not Aggressive" buttons on alert embeds
- Button press calls `update_one` on the pre-inserted `aggression_training_data` document to set label, `reviewed_by`, `reviewed_at`
- View is attached to alert messages sent by the tracker

### 6. Aggression Cog (`bot/cogs/aggression.py`)

Commands:
- `/aggression stats` — show aggression detection statistics (total strikes, labeled samples, predictor type)
- `/aggression train` — trigger training (owner-only, same pattern as `/ml train`)
- `/aggression seeds list` — display current seed phrases
- `/aggression seeds add <phrase>` — add a custom seed phrase (re-embeds on add)
- `/aggression seeds remove <phrase>` — remove a seed phrase

`on_message` listener that owns the aggression pipeline flow (see Strike Tracker call site above).

## Database Collections

Register in `bot/database.py` as named properties, consistent with existing pattern.

**`aggression_strikes`:**
- Compound index: `(guild_id, user_id, created_at)` — supports the time-windowed strike count query
- Used for strike evaluation and alert deduplication

**`aggression_training_data`:**
- Compound index: `(guild_id, label)` — supports training data export
- Index: `(message_id)` — supports label updates from review UI

## Integration Points

- **Aggression cog `on_message`:** Calls `self.bot.collector.process_message()`, checks result is `None`, then runs aggression pipeline. Does NOT modify `collector.py`
- **`ml_consent` gating:** Aggression pipeline checks `config.ml_consent` via `ConfigService`, same as existing collector
- **Guild config (`bot/models/guild_config.py`):** Add `aggression_channel_id: int | None`, `aggression_strike_count: int` (default 3), `aggression_window_hours: int` (default 2). Dashboard-managed, added later
- **Requirements:** Add `sentence-transformers` to `requirements.txt`

## Predictor Switchover

Mirrors existing HeuristicPredictor → TrainedPredictor pattern:
1. Bot starts → checks if `aggression_model.joblib` exists
2. If yes → load TrainedAggressionPredictor
3. If no → load SeedPredictor with curated phrases
4. `/aggression train` retrains and swaps at runtime

## Dependencies

- `sentence-transformers>=2.2.0` (pulls in `torch`, `transformers`, `huggingface-hub`)
- Note: this adds ~1-2GB to the Docker image due to PyTorch
