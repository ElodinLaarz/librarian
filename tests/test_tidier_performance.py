from __future__ import annotations

from time import perf_counter

import pytest

from src.config import DatabaseSettings, LibrarianConfig, TidySettings
from src.services.tidier import Tidier
from tests.perf_utils import PerfTomeRepository, generate_tomes
from tests.stubs import StubEmbeddingService, StubIngestor, StubVerifier


def _make_tidier(total: int, duplicate_pairs: int) -> Tidier:
    config = LibrarianConfig(
        database=DatabaseSettings(uri="mongodb://localhost:27017"),
        tidy=TidySettings(
            threshold=0.95,
            min_shared_facts=2,
            min_fact_overlap=0.8,
            max_fact_frequency=32,
            semantic_planes=30,
            semantic_band_size=6,
            semantic_max_bucket_size=256,
            group_concurrency=4,
        ),
    )
    config.embedding.dimensions = 32
    config.ingest.build_concurrency = 8
    config.ingest.write_batch_size = 64
    tidy_settings = config.tidy
    repo = PerfTomeRepository(
        generate_tomes(total=total, duplicate_pairs=duplicate_pairs),
        tidy_settings,
    )
    ingestor = StubIngestor(
        config,
        StubEmbeddingService(dimensions=32),
        StubVerifier(),
        repo,
    )
    return Tidier(ingestor, repo, tidy_settings)


@pytest.mark.asyncio
async def test_duplicate_scan_regression_budget_1000_vectors() -> None:
    tidier = _make_tidier(total=1000, duplicate_pairs=50)

    started = perf_counter()
    report = await tidier.run_cleanup()
    elapsed = perf_counter() - started

    assert report["scanned"] == 1000
    assert elapsed < 10.0


@pytest.mark.asyncio
async def test_duplicate_scan_budget_10000_vectors_under_60_seconds() -> None:
    tidier = _make_tidier(total=10000, duplicate_pairs=100)

    started = perf_counter()
    report = await tidier.run_cleanup()
    elapsed = perf_counter() - started

    assert report["scanned"] == 10000
    assert elapsed < 60.0
