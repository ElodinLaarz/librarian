from unittest.mock import AsyncMock, patch

import httpx
import pytest

from src.config import EmbeddingSettings
from src.services.embedding import (
    build_embedding_service,
)


@pytest.mark.asyncio
async def test_build_embedding_service_auto_fallback_to_ollama() -> None:
    settings = EmbeddingSettings(provider="auto", dimensions=384)

    # Mock SentenceTransformerEmbeddingService to fail during initialize
    with patch("src.services.embedding.SentenceTransformerEmbeddingService") as mock_st:
        mock_st_instance = mock_st.return_value
        mock_st_instance.initialize = AsyncMock(side_effect=ImportError("ST not found"))

        # Mock OllamaEmbeddingService to succeed
        with patch("src.services.embedding.OllamaEmbeddingService") as mock_ollama:
            mock_ollama_instance = mock_ollama.return_value
            mock_ollama_instance.initialize = AsyncMock()

            service = await build_embedding_service(settings)

            assert service == mock_ollama_instance
            mock_st_instance.initialize.assert_called_once()
            mock_ollama_instance.initialize.assert_called_once()


@pytest.mark.asyncio
async def test_build_embedding_service_auto_raises_when_no_real_provider() -> None:
    """'auto' must fail loudly when neither real provider initializes.

    A silent fallback to dummy (zero-vector) embeddings makes the server look
    healthy while search ranks on garbage; the factory instead raises an
    actionable error telling the operator how to fix it or opt in to dummy.
    """
    settings = EmbeddingSettings(provider="auto", dimensions=384)

    # Mock ST to fail
    with patch("src.services.embedding.SentenceTransformerEmbeddingService") as mock_st:
        mock_st_instance = mock_st.return_value
        mock_st_instance.initialize = AsyncMock(side_effect=ImportError("ST not found"))

        # Mock Ollama to fail
        with patch("src.services.embedding.OllamaEmbeddingService") as mock_ollama:
            mock_ollama_instance = mock_ollama.return_value
            mock_ollama_instance.initialize = AsyncMock(
                side_effect=httpx.ConnectError("Ollama down")
            )

            with pytest.raises(RuntimeError) as exc_info:
                await build_embedding_service(settings)

            mock_st_instance.initialize.assert_called_once()
            mock_ollama_instance.initialize.assert_called_once()

    message = str(exc_info.value)
    # The error must name both failures and how to resolve them.
    assert "ST not found" in message
    assert "Ollama down" in message
    assert "sentence-transformers" in message
    assert "dummy" in message


@pytest.mark.asyncio
async def test_build_embedding_service_explicit_dummy_still_works() -> None:
    """provider='dummy' remains a supported explicit opt-in."""
    settings = EmbeddingSettings(provider="dummy", dimensions=8)
    service = await build_embedding_service(settings)
    vec = await service.embed("anything")
    assert vec.shape == (8,)


@pytest.mark.asyncio
async def test_build_embedding_service_auto_fallback_on_value_error_and_os_error() -> None:
    settings = EmbeddingSettings(provider="auto", dimensions=384)

    # 1. Test SentenceTransformers falling back on ValueError
    with patch("src.services.embedding.SentenceTransformerEmbeddingService") as mock_st:
        mock_st_instance = mock_st.return_value
        mock_st_instance.initialize = AsyncMock(side_effect=ValueError("Bad model name"))

        # Mock OllamaEmbeddingService to succeed
        with patch("src.services.embedding.OllamaEmbeddingService") as mock_ollama:
            mock_ollama_instance = mock_ollama.return_value
            mock_ollama_instance.initialize = AsyncMock()

            service = await build_embedding_service(settings)

            assert service == mock_ollama_instance
            mock_st_instance.initialize.assert_called_once()
            mock_ollama_instance.initialize.assert_called_once()

    # 2. Test SentenceTransformers falling back on OSError
    with patch("src.services.embedding.SentenceTransformerEmbeddingService") as mock_st:
        mock_st_instance = mock_st.return_value
        mock_st_instance.initialize = AsyncMock(side_effect=OSError("Network issue"))

        # Mock OllamaEmbeddingService to succeed
        with patch("src.services.embedding.OllamaEmbeddingService") as mock_ollama:
            mock_ollama_instance = mock_ollama.return_value
            mock_ollama_instance.initialize = AsyncMock()

            service = await build_embedding_service(settings)

            assert service == mock_ollama_instance
            mock_st_instance.initialize.assert_called_once()
            mock_ollama_instance.initialize.assert_called_once()
