from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

from src import constants
from src.models.enums import LogLevel


# --- Nested config models ---
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
    default_tags: list[str] = Field(default_factory=lambda: list(constants.DEFAULT_TAGS))


class ServerSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LIBRARIAN_SERVER_")
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: LogLevel = LogLevel.INFO


# --- Top-level config ---
class LibrarianConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LIBRARIAN_", env_nested_delimiter="__")

    database: DatabaseSettings = Field(default_factory=DatabaseSettings)  # type: ignore
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    search: SearchSettings = Field(default_factory=SearchSettings)
    web_search: WebSearchSettings = Field(default_factory=WebSearchSettings)
    verification: VerificationSettings = Field(default_factory=VerificationSettings)
    ingest: IngestSettings = Field(default_factory=IngestSettings)
    server: ServerSettings = Field(default_factory=ServerSettings)

    @classmethod
    def from_yaml(cls, path: Path | str) -> "LibrarianConfig":
        """Load config from optional YAML file, validate nested sections, apply env overrides."""
        if TYPE_CHECKING:
            return cls()

        path = Path(path)
        raw: dict[str, Any] = {}

        # Load YAML if it exists
        if path.exists():
            raw = yaml.safe_load(path.read_text()) or {}

        validated_sections = {}
        errors = []

        # Validate each nested section separately to preserve nested errors
        for field_name, field_info in cls.model_fields.items():
            section_data = raw.get(field_name, {})
            section_type: type[BaseSettings] = field_info.annotation

            try:
                validated_sections[field_name] = section_type.model_validate(section_data)
            except ValidationError as exc:
                for err in exc.errors():
                    loc = [field_name, *err["loc"]]  # e.g., ['database', 'uri']
                    full_path = ".".join(str(p) for p in loc)

                    # Get the proper env prefix from nested model
                    env_prefix = section_type.model_config.get(
                        "env_prefix", field_name.upper()
                    ).rstrip("_")

                    if err["type"] == "missing":
                        missing_field = str(err["loc"][0])
                        env_var = f"{env_prefix}_{missing_field.upper()}"
                        errors.append(
                            f"Missing required config: [{full_path}]. "
                            f"Set in YAML or as env var: {env_var}"
                        )
                    else:
                        errors.append(f"{full_path}: {err['msg']}")

        if errors:
            full_msg = f"Configuration validation failed for {path}:\n" + "\n".join(errors)
            raise ValueError(full_msg)

        return cls(**validated_sections)
