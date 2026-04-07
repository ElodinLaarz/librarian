from unittest.mock import AsyncMock, patch

import httpx
import pytest

from src.config import EmbeddingSettings
from src.services.embedding import (
    build_embedding_service,
)


@pytest.mark.asyncio
async def test_build_embedding_service_auto_fallback_to_ollama():
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
async def test_build_embedding_service_auto_fallback_to_dummy():
    settings = EmbeddingSettings(provider="auto", dimensions=384)

    # Mock ST to fail
    with patch("src.services.embedding.SentenceTransformerEmbeddingService") as mock_st:
        mock_st_instance = mock_st.return_value
        mock_st_instance.initialize = AsyncMock(side_effect=ImportError())

        # Mock Ollama to fail
        with patch("src.services.embedding.OllamaEmbeddingService") as mock_ollama:
            mock_ollama_instance = mock_ollama.return_value
            mock_ollama_instance.initialize = AsyncMock(
                side_effect=httpx.ConnectError("Ollama down")
            )

            # Mock Dummy to succeed
            with patch("src.services.embedding.DummyEmbeddingService") as mock_dummy:
                mock_dummy_instance = mock_dummy.return_value
                mock_dummy_instance.initialize = AsyncMock()

                service = await build_embedding_service(settings)

                assert service == mock_dummy_instance
                mock_st_instance.initialize.assert_called_once()
                mock_ollama_instance.initialize.assert_called_once()
                mock_dummy_instance.initialize.assert_called_once()
