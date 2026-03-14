from enum import Enum


class SourceType(str, Enum):
    AGENT_INPUT = "agent_input"
    RESEARCHER = "researcher"
    MANUAL = "manual"


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ResearchDepth(str, Enum):
    SHALLOW = "shallow"
    STANDARD = "standard"
    DEEP = "deep"


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
