import uuid

import numpy as np

from src.models.enums import SourceType
from src.models.tome import Tome


def make_tome(content: str) -> Tome:
    """Helper to construct a Tome for testing."""
    return Tome(
        id=uuid.uuid4(),
        title="Tome",
        content=content,
        summary="Summary",
        category="general",
        tags=["stub"],
        source_url=None,
        source_type=SourceType.AGENT_INPUT,
        confidence=0.8,
        embedding=np.zeros(768, dtype=np.float32),
    )
