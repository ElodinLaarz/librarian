import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

import yaml
from pydantic import Field, ValidationError, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from src import constants
from src.models.enums import LogLevel


def _load_dotenv() -> None:
    """Merge ``.env`` into the process env (does not override existing keys).

    Searches for ``.env`` in the current working directory, then upwards from
    the location of this file. See ``.env.example``.
    """
    # 1. Try CWD
    path = Path.cwd() / ".env"

    # 2. Try upwards from this file's directory
    if not path.is_file():
        current = Path(__file__).resolve().parent
        for _ in range(5):  # Look up to 5 levels up
            candidate = current / ".env"
            if candidate.is_file():
                path = candidate
                break
            if current.parent == current:  # Root reached
                break
            current = current.parent

    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value_raw = line.partition("=")
        key = key.strip()
        if not key:
            continue
        v = value_raw.strip()
        if v.startswith('"'):
            value = v[1:].partition('"')[0]
        elif v.startswith("'"):
            value = v[1:].partition("'")[0]
        else:
            value = v.partition("#")[0].strip()
        os.environ.setdefault(key, value)


# --- Nested config models ---
class DatabaseSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LIBRARIAN_DATABASE_")
    uri: str
    database: str = "library"
    tomes_collection: str = "tomes"
    jobs_collection: str = "research_jobs"
    tls: bool = False
    tls_cert_path: str = ""

    @model_validator(mode="after")
    def tls_requires_cert_path(self) -> Self:
        if self.tls and not self.tls_cert_path.strip():
            raise ValueError("tls_cert_path must be non-empty when tls is enabled")
        return self


class EmbeddingSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LIBRARIAN_EMBEDDING_")
    model_name: str = "all-MiniLM-L6-v2"
    dimensions: int = 384
    cache_size: int = 10_000
    provider: str = "auto"  # "auto", "sentence-transformers", "ollama", or "dummy"
    ollama_url: str = "http://localhost:11434"


class SearchSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LIBRARIAN_SEARCH_")
    default_top_k: int = 5
    max_top_k: int = 20
    min_confidence: float = 0.5
    use_keyword_prefilter: bool = True


class WebSearchSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LIBRARIAN_WEB_SEARCH_")
    default_max_results: int = constants.DEFAULT_MAX_RESULTS
    provider: str = "brave"
    api_key: str | None = None
    brave_api_base: str = "https://api.search.brave.com/res/v1/web/search"
    serper_api_base: str = "https://google.serper.dev"
    tavily_api_base: str = "https://api.tavily.com"


class VerificationSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LIBRARIAN_VERIFICATION_")
    enabled: bool = True
    reject_threshold: float = 0.3
    store_threshold: float = 0.7
    mock_confidence: float = constants.DEFAULT_MOCK_CONFIDENCE
    noop_confidence: float = constants.DEFAULT_NOOP_CONFIDENCE
    ollama_base_url: str = "http://localhost:11434"
    claim_model: str = "gemma2:2b"
    use_llm_claims: bool = True
    use_llm_verdict: bool = True
    verdict_model: str = ""


class IngestSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LIBRARIAN_INGEST_")
    shard_size: int = constants.DEFAULT_SHARD_SIZE
    shard_overlap: int = constants.DEFAULT_SHARD_OVERLAP
    summary_length: int = constants.DEFAULT_SUMMARY_LENGTH
    title_length: int = constants.TITLE_MAX_LENGTH
    unverified_confidence: float = constants.DEFAULT_UNVERIFIED_CONFIDENCE
    default_category: str = constants.DEFAULT_CATEGORY
    default_tags: list[str] = Field(default_factory=lambda: list(constants.DEFAULT_TAGS))
    use_llm_chunking: bool = True
    use_llm_classification: bool = True
    taxonomy: list[str] = Field(
        default_factory=lambda: [
            "Technology",
            "Science",
            "History",
            "Reference",
            "Uncategorized",
        ]
    )
    extraction_model: str = "gemma2:2b"
    ollama_base_url: str = "http://localhost:11434"


class ResearcherSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LIBRARIAN_RESEARCHER_")
    max_synthesis_chars: int = 24_000
    shallow_queries: int = 3
    standard_queries: int = 5
    deep_queries: int = 8
    shallow_urls_per_query: int = 2
    standard_urls_per_query: int = 3
    deep_urls_per_query: int = 4


class ServerSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LIBRARIAN_SERVER_")
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: LogLevel = LogLevel.INFO
    transport: str = "stdio"  # "stdio", "sse", or "streamable-http"


# --- Top-level config ---
class LibrarianConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LIBRARIAN_", env_nested_delimiter="__")

    database: DatabaseSettings = Field(default_factory=DatabaseSettings)  # type: ignore
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    search: SearchSettings = Field(default_factory=SearchSettings)
    web_search: WebSearchSettings = Field(default_factory=WebSearchSettings)
    verification: VerificationSettings = Field(default_factory=VerificationSettings)
    ingest: IngestSettings = Field(default_factory=IngestSettings)
    researcher: ResearcherSettings = Field(default_factory=ResearcherSettings)
    server: ServerSettings = Field(default_factory=ServerSettings)

    @classmethod
    def from_yaml(cls, path: Path | str) -> "LibrarianConfig":
        """Load config from optional YAML file, validate nested sections, apply env overrides."""
        if TYPE_CHECKING:
            return cls()

        _load_dotenv()
        path = Path(path)
        raw: dict[str, Any] = {}

        # Load YAML if it exists
        if path.exists():
            content = path.read_text(encoding="utf-8")
            # Basic env var interpolation: ${VAR} or ${VAR:-default}
            import re

            def _replace_env(match: re.Match[str]) -> str:
                var = match.group(1)
                default = match.group(2) if match.group(2) else ""
                return os.environ.get(var, default)

            # Match ${VAR} or ${VAR:-default}
            content = re.sub(r"\${([^}:-]+)(?::-(.*?))?}", _replace_env, content)
            raw = yaml.safe_load(content)
            if raw is None:
                raw = {}
            if not isinstance(raw, dict):
                raise ValueError(f"YAML root in {path} must be a mapping, got {type(raw).__name__}")

        validated_sections = {}
        errors = []

        # Validate each nested section separately to preserve nested errors.
        # Use the BaseSettings constructor (not model_validate) so each section
        # merges YAML values with its env_prefix sources (e.g. LIBRARIAN_DATABASE_URI).
        for field_name, field_info in cls.model_fields.items():
            section_raw = raw.get(field_name)
            if section_raw is None:
                section_data: dict[str, Any] = {}
            elif not isinstance(section_raw, dict):
                errors.append(
                    f"{field_name}: expected a mapping in YAML, got {type(section_raw).__name__}"
                )
                continue
            else:
                section_data = section_raw

            section_type: type[BaseSettings] = field_info.annotation

            try:
                validated_sections[field_name] = section_type(**section_data)
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
