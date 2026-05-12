"""Factory for the shared Motor client used by all Mongo-backed repositories.

Centralising client construction here lets ``LibrarianServer.lifespan`` build a
single :class:`AsyncIOMotorClient` per process and pass it into both
:class:`~src.storage.mongo.mongo_tome_repository.MongoTomeRepository` and
:class:`~src.storage.mongo.mongo_research_job_repository.MongoResearchJobRepository`.
Previously each repo constructed its own client, doubling connection-pool size
and TLS handshakes per process (issue #25).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from motor.motor_asyncio import AsyncIOMotorClient

from src.config import DatabaseSettings

# Conservative defaults so a misconfigured / unreachable Mongo surfaces quickly
# instead of hanging the MCP server startup. Individual call sites can still
# override by passing different kwargs through ``extra``.
DEFAULT_SERVER_SELECTION_TIMEOUT_MS = 5_000
DEFAULT_CONNECT_TIMEOUT_MS = 5_000
DEFAULT_SOCKET_TIMEOUT_MS = 30_000


def build_motor_client(
    settings: DatabaseSettings,
    **extra: Any,
) -> AsyncIOMotorClient[Mapping[str, Any]]:
    """Construct a shared Motor client from :class:`DatabaseSettings`.

    Applies the project's standard UUID representation, TLS handling, and
    connection-timeout defaults. Any keyword in ``extra`` overrides the
    defaults — useful for tests that want a shorter ``serverSelectionTimeoutMS``
    or callers that need to flip ``uuidRepresentation``.
    """
    kwargs: dict[str, Any] = {
        "uuidRepresentation": "standard",
        "serverSelectionTimeoutMS": DEFAULT_SERVER_SELECTION_TIMEOUT_MS,
        "connectTimeoutMS": DEFAULT_CONNECT_TIMEOUT_MS,
        "socketTimeoutMS": DEFAULT_SOCKET_TIMEOUT_MS,
    }
    if settings.tls:
        kwargs["tls"] = True
        kwargs["tlsCertificateKeyFile"] = os.path.expanduser(settings.tls_cert_path)
    else:
        kwargs["tls"] = False

    kwargs.update(extra)

    return AsyncIOMotorClient(settings.uri, **kwargs)
