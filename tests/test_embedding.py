import numpy as np
import pytest

from src.config import EmbeddingSettings
from src.services.embedding import SentenceTransformerEmbeddingService


@pytest.mark.asyncio
async def test_sentence_transformer_embedding() -> None:
    settings = EmbeddingSettings(dimensions=384, model_name="all-MiniLM-L6-v2")
    service = SentenceTransformerEmbeddingService(settings)

    await service.initialize()

    text = "Hello world"
    embedding = await service.embed(text)

    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (384,)
    assert not np.allclose(embedding, np.zeros(384))

    # Test caching
    embedding2 = await service.embed(text)
    assert np.array_equal(embedding, embedding2)

    # Test different text
    embedding3 = await service.embed("Different text")
    assert not np.array_equal(embedding, embedding3)
