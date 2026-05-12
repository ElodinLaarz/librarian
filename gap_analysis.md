# Gap Analysis & Technical Debt Report

Based on a codebase scan, test suite execution, and architectural review, the following gaps in implementation, design flaws, and poor choices were identified:

## 1. Test Suite & Infrastructure Gaps

- **Unmocked External Dependencies in Tests:** `test_mongo_repository.py` attempts to connect unconditionally to a live MongoDB instance on `localhost:27017`. When unavailable, tests fatally crash with `pymongo.errors.ServerSelectionTimeoutError` instead of using a mock (like `mongomock`), skipping via `pytest.mark.skipif`, or using Testcontainers.
- **Placeholder Tests:** The file `tests/test_placeholder.py` contains a literal empty test block with the docstring `"Placeholder until real tests are added."`
- **Hardcoded Test Fixtures:** The `mongo_repo` fixture in `test_mongo_repository.py` hardcodes the database URI `mongodb://localhost:27017/?directConnection=true` rather than resolving it from an environment variable or test configuration.

## 2. Poor Dependency Management Patterns

- **Fragmented Dev Dependencies:** In `pyproject.toml`, development dependencies are split incorrectly between legacy `[project.optional-dependencies]` (containing `pytest`, `ruff`, `pre-commit`) and the modern PEP-735 `[dependency-groups]` (which only contains `mypy`). Because modern `uv` relies on dependency groups, running `uv sync` fails to install the testing and linting tools out of the box.
- **Pydantic V1 Incompatibility:** A warning is emitted during testing from `langchain_core` (pulled by `langchain-text-splitters`): `UserWarning: Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater.` Since the project specifies Python `>=3.11`, this is a ticking time bomb for future runtime upgrades.

## 3. Implementation Flaws & Code Smells

- **Deliberate Data Duplication on Failure:** In `src/services/ingestor.py` (`_dedup_and_store`), if a reshard operation successfully computes the new chunk but fails to delete the old duplicates from the storage layer, it swallows the deletion error and proceeds. The comment explicitly notes: *"we may end up with duplicate data in the library, but that seems like a better choice (IMO) than aborting."* This compromises the vector search index precision over time.
- **Missing Error Taxonomies for Storage:** While `SEARCH_API_UNAVAILABLE` is defined, the storage layer relies broadly on underlying OS or MongoDB exceptions rather than wrapping them in a unified `StorageError` or `RepositoryError` abstraction, leaking infrastructure details into the service layer.
