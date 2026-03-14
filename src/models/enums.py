from enum import Enum


class SourceType(str, Enum):
    AGENT_INPUT = "agent_input"
    MANUAL = "manual"


class IngestStatus(str, Enum):
    STORED = "stored"
    REJECTED = "rejected"
    PARTIAL = "partial"


class LogLevel(str, Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class VerificationVerdict(str, Enum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    UNVERIFIABLE = "unverifiable"
