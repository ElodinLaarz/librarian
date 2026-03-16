# Global constants for The Librarian

# Tome model constraints
TITLE_MAX_LENGTH = 120

# Ingest defaults
DEFAULT_SHARD_SIZE = 400
DEFAULT_SHARD_OVERLAP = 100
DEFAULT_SUMMARY_LENGTH = 200
DEFAULT_UNVERIFIED_CONFIDENCE = 0.5
DEFAULT_CATEGORY = "Uncategorized"
DEFAULT_TAGS = ("auto-tag",)  # Use tuple for immutability

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
