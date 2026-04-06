from enum import StrEnum


class SourceType(StrEnum):
    AGENT_INPUT = "agent_input"
    RESEARCHER = "researcher"
    MANUAL = "manual"


class IngestStatus(StrEnum):
    STORED = "stored"
    REJECTED = "rejected"
    PARTIAL = "partial"


class LogLevel(StrEnum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class VerificationVerdict(StrEnum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    UNVERIFIABLE = "unverifiable"


class ResearchJobStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ResearchDepth(StrEnum):
    SHALLOW = "shallow"
    STANDARD = "standard"
    DEEP = "deep"
