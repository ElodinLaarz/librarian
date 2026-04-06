import pytest
from unittest.mock import AsyncMock, patch
from src.services.verifier import Verifier, ClaimResult, VerificationResult
from src.models.enums import VerificationVerdict
from src.config import LibrarianConfig, DatabaseSettings

@pytest.fixture
def test_config():
    return LibrarianConfig(
        database=DatabaseSettings(uri="mongodb://localhost:27017", tls_cert_path="/dev/null"),
        llm={"provider": "ollama", "model_name": "gemma:2b"},
        verification={"enabled": True, "mock_confidence": 0.6},
    )

@pytest.fixture
def verifier(test_config):
    return Verifier(test_config)

@pytest.mark.asyncio
async def test_aggregate_confidence(verifier):
    results = [
        ClaimResult(claim="c1", verdict=VerificationVerdict.SUPPORTED, evidence="e1"),
        ClaimResult(claim="c2", verdict=VerificationVerdict.CONTRADICTED, evidence="e2"),
        ClaimResult(claim="c3", verdict=VerificationVerdict.UNVERIFIABLE, evidence="e3"),
    ]
    # (1.0 + 0.0 + 0.5) / 3 = 1.5 / 3 = 0.5
    assert verifier._aggregate_confidence(results) == 0.5

@pytest.mark.asyncio
async def test_aggregate_confidence_empty(verifier):
    assert verifier._aggregate_confidence([]) == 0.6 # mock_confidence

@pytest.mark.asyncio
@patch("src.services.verifier.AsyncOpenAI")
@patch("instructor.from_openai")
async def test_extract_claims(mock_instructor_from_openai, mock_async_openai, test_config):
    mock_client = AsyncMock()
    mock_instructor_from_openai.return_value = mock_client
    
    mock_call = AsyncMock()
    mock_call.claims = ["claim 1", "claim 2"]
    mock_client.chat.completions.create.return_value = mock_call
    
    verifier = Verifier(test_config)
    claims = await verifier._extract_claims("some content")
    assert claims == ["claim 1", "claim 2"]


@pytest.mark.asyncio
@patch("src.services.verifier.AsyncOpenAI")
@patch("instructor.from_openai")
async def test_check_claim(mock_instructor_from_openai, mock_async_openai, test_config):
    mock_client = AsyncMock()
    mock_instructor_from_openai.return_value = mock_client
    
    mock_call = AsyncMock()
    mock_call.verdict = VerificationVerdict.SUPPORTED
    mock_call.evidence = "evidence"
    mock_client.chat.completions.create.return_value = mock_call
    
    verifier = Verifier(test_config)
    # Mock search client
    from unittest.mock import Mock
    verifier._search_client = Mock()
    verifier._search_client.is_available.return_value = True
    verifier._search_client.search = AsyncMock()


    
    from src.services.web_search import WebSearchResult
    verifier._search_client.search.return_value = [
        WebSearchResult(title="t1", url="u1", snippet="s1")
    ]
    
    result = await verifier._check_claim("some claim")
    assert result.claim == "some claim"
    assert result.verdict == VerificationVerdict.SUPPORTED
    assert result.evidence == "evidence"

@pytest.mark.asyncio
async def test_verify_disabled(test_config):
    test_config.verification.enabled = False
    verifier = Verifier(test_config)
    result = await verifier.verify("content")
    assert result.confidence == 1.0
    assert result.skipped == True
