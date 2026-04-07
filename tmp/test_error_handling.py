import asyncio
import os
import shutil
from pathlib import Path
from uuid import uuid4

from src.config import DatabaseSettings
from src.storage.filesystem.fs_research_job_repository import FsResearchJobRepository
from src.storage.filesystem.fs_tome_repository import FsTomeRepository
from tests.stubs import StubEmbeddingService

async def test_error_handling():
    test_dir = Path("/tmp/librarian_test_error_handling")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True)

    try:
        settings = DatabaseSettings(uri=str(test_dir))
        embedding_service = StubEmbeddingService()
        
        job_repo = FsResearchJobRepository(settings)
        tome_repo = FsTomeRepository(settings, embedding_service)

        # 1. Test ResearchJob error handling
        job_id = uuid4()
        job_path = test_dir / "research_jobs" / f"{job_id}.json"
        job_path.parent.mkdir(parents=True, exist_ok=True)
        job_path.write_text("invalid json {")
        
        print(f"Testing ResearchJob with corrupted file at {job_path}...")
        job = await job_repo.get_by_id(job_id)
        assert job is None, "Expected None for corrupted ResearchJob file"
        print("✓ ResearchJob error handling verified.")

        # 2. Test Tome error handling
        tome_id = uuid4()
        tome_path = test_dir / "tomes" / f"{tome_id}.json"
        tome_path.parent.mkdir(parents=True, exist_ok=True)
        tome_path.write_text("invalid json {")
        
        print(f"Testing Tome with corrupted file at {tome_path}...")
        tome = await tome_repo.get_by_id(tome_id)
        assert tome is None, "Expected None for corrupted Tome file"
        print("✓ Tome error handling verified.")

    finally:
        if test_dir.exists():
            shutil.rmtree(test_dir)

if __name__ == "__main__":
    asyncio.run(test_error_handling())
