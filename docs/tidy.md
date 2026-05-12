# Tidy Design

`library_tidy` is the bulk duplicate-consolidation pass for Librarian. Its job is
to scan the full library, identify groups of tomes that represent the same
knowledge, and consolidate each group through `Ingestor.consolidate()`.

## Duplicate Semantics

Tidy uses three stages, in order:

1. Exact content groups.
1. Fact-overlap groups.
1. Semantic embedding groups.

Exact content groups are built from normalized tome content. If two tomes have
the same content after whitespace normalization, they are always treated as
duplicates.

Fact-overlap groups are built from normalized facts split on `constants.CONTENT_SEPARATOR` (the content separator). This
stage is intended to catch multi-fact duplicates without over-merging on shared
boilerplate. To avoid accidental transitive merges, a pair must:

- share at least `min_shared_facts` facts, and
- satisfy the configured overlap threshold.

Facts that appear in more than `max_fact_frequency` tomes are treated as
boilerplate and ignored for grouping.

Semantic groups are only built for tomes that were not already assigned to an
exact-content or fact-overlap group. This stage uses sparse candidate generation
plus exact cosine verification, not a dense all-pairs similarity matrix.

## Tidy Output

`library_tidy` returns these metrics:

- `scanned`: total tomes scanned.
- `groups_found`: duplicate groups returned by the repository scan.
- `groups_consolidated`: groups successfully consolidated.
- `tomes_removed`: original tomes removed after successful consolidation.
- `failed_groups`: groups that failed during consolidation.
- `skipped_groups`: groups skipped because of overlap protection or the per-run
  limit.
- `elapsed_ms`: wall-clock duration for the tidy run.

## Performance Target

The duplicate scan must support 10,000 vectors in under 60 seconds on the
target developer machine / CI benchmark environment.

To keep that target realistic:

- avoid dense `n x n` similarity matrices,
- cap work caused by repeated boilerplate facts,
- keep repository scans linear-ish in library size plus sparse candidate work,
- use bounded concurrency for consolidation writes and shard builds.

## Testing Strategy

The tidy test suite should include:

- repository tests for exact, fact-overlap, and semantic-only duplicates,
- service tests for limits, overlap protection, and failure reporting,
- a synthetic performance test path that exercises 10,000 vectors and checks the
  under-60-second target when performance tests are enabled.
