from __future__ import annotations

import asyncio
import logging
from functools import partial

import numpy as np

log = logging.getLogger(__name__)

MODEL_NAME = "all-MiniLM-L6-v2"
SentenceTransformer = None


def _get_sentence_transformer():
    global SentenceTransformer
    if SentenceTransformer is None:
        from sentence_transformers import SentenceTransformer as sentence_transformer_cls

        SentenceTransformer = sentence_transformer_cls
    return SentenceTransformer


class Embedder:
    def __init__(self) -> None:
        self._model = None

    async def setup(self) -> None:
        if self._model is not None:
            return

        loop = asyncio.get_running_loop()
        self._model = await loop.run_in_executor(None, self._load_model)
        log.info("Embedder loaded model %s", MODEL_NAME)

    @staticmethod
    def _load_model():
        sentence_transformer_cls = _get_sentence_transformer()
        return sentence_transformer_cls(MODEL_NAME)

    async def embed(self, text: str) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Embedder.setup() must be called before embed().")

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, partial(self._encode, text))
        return np.asarray(result, dtype=np.float32)

    async def embed_batch(self, texts: list[str]) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Embedder.setup() must be called before embed_batch().")

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, partial(self._encode, texts))
        return np.asarray(result, dtype=np.float32)

    def _encode(self, inputs: str | list[str]):
        return self._model.encode(
            inputs,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
