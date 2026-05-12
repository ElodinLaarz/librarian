# Global constants for The Librarian

# Tome model constraints
TITLE_MAX_LENGTH = 120
# Per-tag character cap and per-tome tag-list cap. Keeps tag UIs and
# search indexes bounded regardless of how the LLM classifier behaves.
MAX_TAG_LENGTH = 60
MAX_TAGS_PER_TOME = 10

# Ingest defaults
DEFAULT_SHARD_SIZE = 2000
DEFAULT_SHARD_OVERLAP = 200
DEFAULT_SUMMARY_LENGTH = 200
DEFAULT_UNVERIFIED_CONFIDENCE = 0.5
DEFAULT_CATEGORY = "Uncategorized"
DEFAULT_TAGS = ("auto-tag",)  # Use tuple for immutability

# Max characters of input text passed to LLM prompts (keeps requests within
# typical model context limits and bounds latency/cost).
MAX_LLM_INPUT_CHARS = 8000

# Verifier defaults
DEFAULT_MOCK_CONFIDENCE = 0.6
DEFAULT_NOOP_CONFIDENCE = 1.0
MIN_CLAIMS = 3
MAX_CLAIMS = 7

# Web search defaults
DEFAULT_MAX_RESULTS = 3

# UI and Formatting
TRUNCATION_SUFFIX = "..."
CONTENT_SEPARATOR = "\n\n"
JOIN_SEPARATOR = "; "
ID_SEPARATOR = ", "
DOTENV_SEARCH_DEPTH = 5
