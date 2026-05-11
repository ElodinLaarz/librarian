"""Storage-layer exception taxonomy.

Service-layer code (``src/services/*``) must not import or reference
driver-specific exception types (``pymongo.errors.*``, ``OSError``, ...).
Repositories translate driver failures into the subclasses defined here so
callers can branch on storage semantics instead of backend implementation
details.

Always preserve the original cause with ``raise StorageError(...) from exc``
so debugging information is not lost.
"""

from __future__ import annotations


class StorageError(Exception):
    """Base class for all storage-layer failures.

    Catch this in services when any storage call may fail and the caller does
    not need to distinguish between specific failure modes.
    """


class NotFoundError(StorageError):
    """A requested record does not exist in the backing store.

    Raised by repository methods that semantically require an existing record
    (e.g. ``update``) when the target identifier is absent. ``get_by_id``
    callers should prefer returning ``None`` for "missing" rather than
    raising; this subclass is reserved for operations where the caller has
    asserted existence.
    """


class DuplicateError(StorageError):
    """A unique constraint was violated.

    Typically raised when ``insert`` collides with an existing primary key
    (Mongo ``DuplicateKeyError``) or when a filesystem path is already in
    use under a uniqueness invariant.
    """


class BackendUnavailableError(StorageError):
    """The storage backend cannot be reached or initialized.

    Raised when the driver cannot establish a connection (Mongo
    ``ServerSelectionTimeoutError``, ``ConnectionFailure``) or when the
    filesystem root is unreachable (missing directory, permission denied).
    """
