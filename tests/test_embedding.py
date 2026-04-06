import asyncio
import numpy as np
import pytest

import src.services.embedding as embedding_module
from src.config import EmbeddingSettings
from src.services.embedding import SentenceTransformerEmbeddingService


@pytest.mark.asyncio
async def test_sentence_transformer_embedding(monkeypatch) -> None:
    pytest.importorskip("sentence_transformers")

    class FakeSentenceTransformer:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name
            self.encode_calls = 0

        def encode(self, text: str, convert_to_numpy: bool = True) -> np.ndarray:
            self.encode_calls += 1
            values = np.zeros(384, dtype=np.float32)
            seed = sum(ord(char) for char in text)
            values[seed % 384] = 1.0
            values[(seed * 7) % 384] = 0.5
            return values

    fake_model = FakeSentenceTransformer("all-MiniLM-L6-v2")
    monkeypatch.setattr(embedding_module, "SentenceTransformer", lambda name: fake_model)

    settings = EmbeddingSettings(dimensions=384, model_name="all-MiniLM-L6-v2")
    service = SentenceTransformerEmbeddingService(settings)

    await service.initialize()

    text = "Hello world"
    embedding = await service.embed(text)

    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (384,)
    assert not np.allclose(embedding, np.zeros(384))

    # Test caching - should not call encode again
    embedding2 = await service.embed(text)
    assert np.array_equal(embedding, embedding2)
    assert fake_model.encode_calls == 1

    # Test different text - should call encode again
    embedding3 = await service.embed("Different text")
    assert not np.array_equal(embedding, embedding3)
    assert fake_model.encode_calls == 2
