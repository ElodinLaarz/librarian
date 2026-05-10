import asyncio

import numpy as np
import pytest

from src.config import EmbeddingSettings
from src.services.embedding import (
    DummyEmbeddingService,
    OllamaEmbeddingService,
    SentenceTransformerEmbeddingService,
)


@pytest.mark.asyncio
async def test_sentence_transformer_embedding(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("sentence_transformers")

    class FakeSentenceTransformer:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name
            self.encode_calls = 0

        def get_sentence_embedding_dimension(self) -> int:
            return 384

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

        def get_sentence_embedding_dimension(self) -> int:
            return 384

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

        def get_sentence_embedding_dimension(self) -> int:
            return 384

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
async def test_sentence_transformer_dimensions_measured_from_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The service should expose the model's actual dimension after initialize()."""
    pytest.importorskip("sentence_transformers")

    class FakeSentenceTransformer:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

        def get_sentence_embedding_dimension(self) -> int:
            return 768

        def encode(self, text: str, convert_to_numpy: bool = True) -> np.ndarray:
            return np.zeros(768, dtype=np.float32)

    fake_model = FakeSentenceTransformer("fake-768-model")
    import sentence_transformers

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", lambda name: fake_model)

    # Use defaults — dimensions is NOT explicitly set
    settings = EmbeddingSettings(model_name="fake-768-model")
    service = SentenceTransformerEmbeddingService(settings)

    await service.initialize()

    assert service.dimensions == 768


@pytest.mark.asyncio
async def test_initialize_raises_on_dim_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """When config explicitly sets dimensions and model disagrees, initialize() should raise."""
    pytest.importorskip("sentence_transformers")

    class FakeSentenceTransformer:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

        def get_sentence_embedding_dimension(self) -> int:
            return 768

        def encode(self, text: str, convert_to_numpy: bool = True) -> np.ndarray:
            return np.zeros(768, dtype=np.float32)

    fake_model = FakeSentenceTransformer("fake-768-model")
    import sentence_transformers

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", lambda name: fake_model)

    # Explicitly set dimensions=384 — disagrees with the model's 768
    settings = EmbeddingSettings(dimensions=384, model_name="fake-768-model")
    service = SentenceTransformerEmbeddingService(settings)

    with pytest.raises(ValueError) as exc_info:
        await service.initialize()

    msg = str(exc_info.value)
    assert "384" in msg
    assert "768" in msg
    assert "fake-768-model" in msg


@pytest.mark.asyncio
async def test_initialize_uses_measured_when_setting_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If config dimensions is not explicitly set, accept and use the measured value."""
    pytest.importorskip("sentence_transformers")

    class FakeSentenceTransformer:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

        def get_sentence_embedding_dimension(self) -> int:
            return 768

        def encode(self, text: str, convert_to_numpy: bool = True) -> np.ndarray:
            return np.zeros(768, dtype=np.float32)

    fake_model = FakeSentenceTransformer("fake-768-model")
    import sentence_transformers

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", lambda name: fake_model)

    settings = EmbeddingSettings(model_name="fake-768-model")  # default dimensions, not set
    service = SentenceTransformerEmbeddingService(settings)

    # Should not raise
    await service.initialize()

    assert service.dimensions == 768


@pytest.mark.asyncio
async def test_dummy_keeps_settings_dimensions() -> None:
    """DummyEmbeddingService has no model — should keep config dimensions and not probe."""
    settings = EmbeddingSettings(dimensions=512, provider="dummy")
    service = DummyEmbeddingService(settings)

    await service.initialize()

    assert service.dimensions == 512
    # Sanity: embedding produced is 512-dim
    vec = await service.embed("hello")
    assert vec.shape == (512,)


@pytest.mark.asyncio
async def test_ollama_initialize_probes_model_for_dimensions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OllamaEmbeddingService.initialize should probe /api/embeddings to measure dimensions."""
    import httpx

    settings = EmbeddingSettings(
        provider="ollama",
        model_name="nomic-embed-text",
        ollama_url="http://localhost:11434",
    )
    service = OllamaEmbeddingService(settings)

    captured_calls: list[tuple[str, str]] = []

    async def fake_get(self: httpx.AsyncClient, url: str, *args, **kwargs):  # type: ignore[no-untyped-def]
        captured_calls.append(("GET", url))
        return httpx.Response(200, json={"models": []}, request=httpx.Request("GET", url))

    async def fake_post(self: httpx.AsyncClient, url: str, *args, **kwargs):  # type: ignore[no-untyped-def]
        captured_calls.append(("POST", url))
        # Return a 768-dim embedding
        return httpx.Response(
            200,
            json={"embedding": [0.1] * 768},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    await service.initialize()

    assert service.dimensions == 768
    assert any(call[0] == "POST" and "/api/embeddings" in call[1] for call in captured_calls)
    await service.aclose()


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
