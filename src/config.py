from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings

from src.models.enums import LogLevel


class DatabaseSettings(BaseSettings):
    uri: str = "localhost"
    database: str = "librarian"
    tomes_collection: str = "tomes"


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


class ServerSettings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: LogLevel = LogLevel.INFO


class LibrarianConfig(BaseSettings):
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    search: SearchSettings = Field(default_factory=SearchSettings)
    verification: VerificationSettings = Field(default_factory=VerificationSettings)
    server: ServerSettings = Field(default_factory=ServerSettings)

    @classmethod
    def from_yaml(cls, path: Path) -> "LibrarianConfig":
        """Load configuration from a YAML file, with env-var overrides."""
