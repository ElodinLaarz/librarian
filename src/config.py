from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from src import constants
from src.models.enums import LogLevel


class DatabaseSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LIBRARIAN_DATABASE_")
    uri: str = "localhost"
    database: str = "librarian"
    tomes_collection: str = "tomes"


class EmbeddingSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LIBRARIAN_EMBEDDING_")

    dimensions: int = 768
    cache_size: int = 10_000


class SearchSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LIBRARIAN_SEARCH_")

    default_top_k: int = 5
    max_top_k: int = 20
    min_confidence: float = 0.5
    use_keyword_prefilter: bool = True


class WebSearchSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LIBRARIAN_WEB_SEARCH_")

    default_max_results: int = constants.DEFAULT_MAX_RESULTS


class VerificationSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LIBRARIAN_VERIFICATION_")

    enabled: bool = True
    reject_threshold: float = 0.3
    store_threshold: float = 0.7
    mock_confidence: float = constants.DEFAULT_MOCK_CONFIDENCE
    noop_confidence: float = constants.DEFAULT_NOOP_CONFIDENCE


class IngestSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LIBRARIAN_INGEST_")

    shard_size: int = constants.DEFAULT_SHARD_SIZE
    shard_overlap: int = constants.DEFAULT_SHARD_OVERLAP
    summary_length: int = constants.DEFAULT_SUMMARY_LENGTH
    title_length: int = constants.TITLE_MAX_LENGTH
    unverified_confidence: float = constants.DEFAULT_UNVERIFIED_CONFIDENCE
    default_category: str = constants.DEFAULT_CATEGORY
    default_tags: list[str] = list(constants.DEFAULT_TAGS)


class ServerSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LIBRARIAN_SERVER_")

    host: str = "0.0.0.0"
    port: int = 8000
    log_level: LogLevel = LogLevel.INFO


class LibrarianConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LIBRARIAN_")

    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    search: SearchSettings = Field(default_factory=SearchSettings)
    web_search: WebSearchSettings = Field(default_factory=WebSearchSettings)
    verification: VerificationSettings = Field(default_factory=VerificationSettings)
    ingest: IngestSettings = Field(default_factory=IngestSettings)
    server: ServerSettings = Field(default_factory=ServerSettings)

    @classmethod
    def from_yaml(cls, path: Path) -> "LibrarianConfig":
        """Load configuration from a YAML file, with env-var overrides applied on top."""
        raw: dict[str, Any] = yaml.safe_load(path.read_text()) or {}

        # Build each sub-settings object from YAML values, then let pydantic-settings
        # apply env-var overrides on top via the prefixed env vars.
        return cls(
            database=DatabaseSettings(**raw.get("database", {})),
            embedding=EmbeddingSettings(**raw.get("embedding", {})),
            search=SearchSettings(**raw.get("search", {})),
            web_search=WebSearchSettings(**raw.get("web_search", {})),
            verification=VerificationSettings(**raw.get("verification", {})),
            ingest=IngestSettings(**raw.get("ingest", {})),
            server=ServerSettings(**raw.get("server", {})),
        )
