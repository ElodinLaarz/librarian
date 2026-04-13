import asyncio

import numpy as np
import pytest

from src.config import EmbeddingSettings
from src.services.embedding import OllamaEmbeddingService, SentenceTransformerEmbeddingService


@pytest.mark.asyncio
async def test_sentence_transformer_embedding(monkeypatch: pytest.MonkeyPatch) -> None:
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
    import sentence_transformers

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", lambda name: fake_model)

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


@pytest.mark.asyncio
async def test_sentence_transformer_embedding_cache_eviction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("sentence_transformers")

    class FakeSentenceTransformer:
        def __init__(self, model_name: str) -> None:
            self.encode_calls = 0

        def encode(self, text: str, convert_to_numpy: bool = True) -> np.ndarray:
            self.encode_calls += 1
            return np.zeros(384, dtype=np.float32)

    fake_model = FakeSentenceTransformer("all-MiniLM-L6-v2")
    import sentence_transformers

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", lambda name: fake_model)

    settings = EmbeddingSettings(dimensions=384, model_name="all-MiniLM-L6-v2", cache_size=2)
    service = SentenceTransformerEmbeddingService(settings)
    await service.initialize()

    await service.embed("text1")
    await service.embed("text2")
    assert fake_model.encode_calls == 2

    # text1 and text2 are in cache.
    # Now embed text3, should evict text1 (oldest).
    await service.embed("text3")
    assert fake_model.encode_calls == 3

    # Now embed text2 again, should NOT call encode because it was retained!
    await service.embed("text2")
    assert fake_model.encode_calls == 3

    # Now embed text1 again, should call encode because it was evicted!
    await service.embed("text1")
    assert fake_model.encode_calls == 4


@pytest.mark.asyncio
async def test_sentence_transformer_embedding_concurrency(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("sentence_transformers")

    class FakeSentenceTransformer:
        def __init__(self, model_name: str) -> None:
            self.encode_calls = 0

        def encode(self, text: str, convert_to_numpy: bool = True) -> np.ndarray:
            self.encode_calls += 1
            import time

            time.sleep(0.1)  # Simulate slow computation
            return np.zeros(384, dtype=np.float32)

    fake_model = FakeSentenceTransformer("all-MiniLM-L6-v2")
    import sentence_transformers

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", lambda name: fake_model)

    settings = EmbeddingSettings(dimensions=384, model_name="all-MiniLM-L6-v2")
    service = SentenceTransformerEmbeddingService(settings)
    await service.initialize()

    # Fire two concurrent requests for the same text
    t1 = asyncio.create_task(service.embed("concurrent_text"))
    t2 = asyncio.create_task(service.embed("concurrent_text"))

    await asyncio.gather(t1, t2)

    # In our current implementation, both will miss cache initially and call encode!
    # So encode_calls should be 2!
    assert fake_model.encode_calls == 2


@pytest.mark.asyncio
async def test_ollama_embed_real_scores(
    ollama_embedding_service: OllamaEmbeddingService,
) -> None:
    embed1 = await ollama_embedding_service.embed("The dog ran through the park.")
    embed2 = await ollama_embedding_service.embed("A puppy sprinted across the field.")
    embed3 = await ollama_embedding_service.embed("Quantum physics is a difficult subject.")

    def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    sim_related = cos_sim(embed1, embed2)
    sim_unrelated = cos_sim(embed1, embed3)

    assert sim_related > 0.0
    assert sim_unrelated > 0.0
    assert sim_related > sim_unrelated


@pytest.mark.asyncio
async def test_ollama_embed_raises_on_missing_embeddings_key() -> None:
    class FakeResponse:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict[str, object]:
            return {}

    settings = EmbeddingSettings(
        provider="ollama",
        model_name="nomic-embed-text",
        dimensions=3,
        cache_size=10,
    )
    service = OllamaEmbeddingService(settings)

    async def fake_post(url: str, json: dict[str, object]) -> FakeResponse:
        assert url == "/api/embed"
        assert json["input"] == "hello"
        return FakeResponse()

    service._client.post = fake_post  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="did not include a non-empty 'embeddings' array"):
        await service.embed("hello")

    await service.aclose()
