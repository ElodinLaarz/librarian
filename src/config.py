from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.models.enums import LogLevel


class DatabaseSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LIBRARIAN_DATABASE_")
    uri: str
    database: str = "library"
    tomes_collection: str = "tomes"
    tls_cert_path: str


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


class VerificationSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LIBRARIAN_VERIFICATION_")

    enabled: bool = True
    reject_threshold: float = 0.3
    store_threshold: float = 0.7


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
    verification: VerificationSettings = Field(default_factory=VerificationSettings)
    server: ServerSettings = Field(default_factory=ServerSettings)

    @classmethod
    def from_yaml(cls, path: Path) -> "LibrarianConfig":
        """Load configuration from a YAML file, with env-var overrides applied on top."""
        raw: dict[str, Any] = yaml.safe_load(path.read_text()) or {}

        # Build each sub-settings object from YAML values, then let pydantic-settings
        # apply env-var overrides on top via the prefixed env vars.
        section_classes: list[tuple[str, type[BaseSettings]]] = [
            ("database", DatabaseSettings),
            ("embedding", EmbeddingSettings),
            ("search", SearchSettings),
            ("verification", VerificationSettings),
            ("server", ServerSettings),
        ]
        sections: dict[str, BaseSettings] = {}
        for section, settings_cls in section_classes:
            try:
                sections[section] = settings_cls(**raw.get(section, {}))
            except ValidationError as exc:
                prefix = settings_cls.model_config.get("env_prefix", "")
                missing = [e["loc"][0] for e in exc.errors() if e["type"] == "missing"]
                hints = [f"  {prefix}{str(f).upper()}=<value>" for f in missing]
                raise ValueError(
                    f"Missing required config for [{section}] in {path.resolve()}.\n"
                    f"Set via YAML or environment variable:\n" + "\n".join(hints)
                ) from exc

        return cls(**sections)
