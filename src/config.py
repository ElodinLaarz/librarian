from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings

from src.models.enums import LogLevel, ResearchDepth


class DatabaseSettings(BaseSettings):
    uri: str = "localhost"
    database: str = "librarian"
    tomes_collection: str = "tomes"
    jobs_collection: str = "research_jobs"


class EmbeddingSettings(BaseSettings):
    dimensions: int = 768
    cache_size: int = 10_000


class SearchSettings(BaseSettings):
    default_top_k: int = 5
    max_top_k: int = 20
    min_confidence: float = 0.5
    use_keyword_prefilter: bool = True


class VerificationSettings(BaseSettings):
    enabled: bool = True
    reject_threshold: float = 0.3
    store_threshold: float = 0.7


class ResearcherSettings(BaseSettings):
    default_depth: ResearchDepth = ResearchDepth.STANDARD
    max_tomes_per_run: int = 10
    max_sources_per_query: int = 3
    async_default: bool = False


class ServerSettings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: LogLevel = LogLevel.INFO




class LibrarianConfig(BaseSettings):
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    search: SearchSettings = Field(default_factory=SearchSettings)
    verification: VerificationSettings = Field(default_factory=VerificationSettings)
    researcher: ResearcherSettings = Field(default_factory=ResearcherSettings)
    server: ServerSettings = Field(default_factory=ServerSettings)

    @classmethod
    def from_yaml(cls, path: Path) -> "LibrarianConfig":
        """Load configuration from a YAML file, with env-var overrides."""
        raise NotImplementedError
