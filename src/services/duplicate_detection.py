from __future__ import annotations

import math
import re
from collections import defaultdict
from itertools import combinations
from typing import Any
from uuid import UUID

import numpy as np
from numpy.typing import NDArray

from src import constants
from src.config import TidySettings
from src.models.tome import Tome
from src.storage.tome_repository import DuplicateScanResult


class _UnionFind:
    def __init__(self, ids: list[UUID]) -> None:
        self._parent: dict[UUID, UUID] = {node_id: node_id for node_id in ids}
        self._size: dict[UUID, int] = {node_id: 1 for node_id in ids}

    def find(self, node_id: UUID) -> UUID:
        parent = self._parent[node_id]
        if parent != node_id:
            parent = self.find(parent)
            self._parent[node_id] = parent
        return parent

    def union(self, left: UUID, right: UUID) -> bool:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return False
        if self._size[root_left] < self._size[root_right]:
            root_left, root_right = root_right, root_left
        self._parent[root_right] = root_left
        self._size[root_left] += self._size[root_right]
        return True


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip()).lower()


def _extract_facts(content: str) -> set[str]:
    return {
        normalized
        for raw_fact in content.split(constants.CONTENT_SEPARATOR)
        if (normalized := _normalize_text(raw_fact))
    }


def _cosine_similarity(left: NDArray[np.floating[Any]], right: NDArray[np.floating[Any]]) -> float:
    if left.shape != right.shape:
        return 0.0
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom == 0.0:
        return 0.0
    return float(np.dot(left, right) / denom)


def _count_duplicate_groups(union_find: _UnionFind, ids: list[UUID]) -> int:
    counts: dict[UUID, int] = defaultdict(int)
    for node_id in ids:
        counts[union_find.find(node_id)] += 1
    return sum(1 for size in counts.values() if size > 1)


def _iter_semantic_candidate_pairs(
    vectors: dict[UUID, NDArray[np.float32]],
    settings: TidySettings,
) -> set[tuple[UUID, UUID]]:
    if not vectors:
        return set()

    ids = list(vectors)
    dimension = next(iter(vectors.values())).shape[0]
    plane_count = max(1, settings.semantic_planes)
    rng = np.random.default_rng(0)
    hyperplanes = rng.standard_normal((plane_count, dimension), dtype=np.float32)
    embeddings = np.stack([vectors[node_id] for node_id in ids]).astype(np.float32)
    signatures = embeddings @ hyperplanes.T >= 0

    candidate_pairs: set[tuple[UUID, UUID]] = set()
    band_size = min(max(1, settings.semantic_band_size), plane_count)

    for band_start in range(0, plane_count, band_size):
        band_end = min(band_start + band_size, plane_count)
        buckets: dict[tuple[bool, ...], list[UUID]] = defaultdict(list)

        for index, node_id in enumerate(ids):
            key = tuple(bool(bit) for bit in signatures[index, band_start:band_end])
            buckets[key].append(node_id)

        for bucket_ids in buckets.values():
            if len(bucket_ids) < 2:
                continue
            if len(bucket_ids) > settings.semantic_max_bucket_size:
                continue
            for left_id, right_id in combinations(sorted(bucket_ids), 2):
                candidate_pairs.add((left_id, right_id))

    return candidate_pairs


def build_duplicate_groups(tomes: list[Tome], settings: TidySettings) -> DuplicateScanResult:
    if not tomes:
        return DuplicateScanResult(groups=[], scanned=0)

    by_id = {tome.id: tome for tome in tomes}
    union_find = _UnionFind(list(by_id))

    exact_content_groups = 0
    fact_overlap_groups = 0
    semantic_groups = 0
    ignored_high_frequency_facts = 0

    normalized_content_to_ids: dict[str, list[UUID]] = defaultdict(list)
    fact_sets: dict[UUID, set[str]] = {}
    fact_to_ids: dict[str, list[UUID]] = defaultdict(list)

    for tome in tomes:
        normalized_content_to_ids[_normalize_text(tome.content)].append(tome.id)
        facts = _extract_facts(tome.content)
        fact_sets[tome.id] = facts
        for fact in facts:
            fact_to_ids[fact].append(tome.id)

    for group_ids in normalized_content_to_ids.values():
        if len(group_ids) < 2:
            continue
        for left_id, right_id in combinations(group_ids, 2):
            union_find.union(left_id, right_id)
    exact_content_groups = _count_duplicate_groups(union_find, list(by_id))

    shared_fact_counts: dict[tuple[UUID, UUID], int] = defaultdict(int)
    for group_ids in fact_to_ids.values():
        if len(group_ids) < 2:
            continue
        if len(group_ids) > settings.max_fact_frequency:
            ignored_high_frequency_facts += 1
            continue
        for left_id, right_id in combinations(sorted(group_ids), 2):
            shared_fact_counts[(left_id, right_id)] += 1

    for (left_id, right_id), shared_count in shared_fact_counts.items():
        if shared_count < settings.min_shared_facts:
            continue
        left_facts = fact_sets[left_id]
        right_facts = fact_sets[right_id]
        smallest_group = min(len(left_facts), len(right_facts))
        if smallest_group == 0:
            continue
        overlap = shared_count / smallest_group
        if overlap >= settings.min_fact_overlap:
            union_find.union(left_id, right_id)
    fact_overlap_groups = max(
        0,
        _count_duplicate_groups(union_find, list(by_id)) - exact_content_groups,
    )

    semantic_candidates: dict[UUID, NDArray[np.float32]] = {}
    for tome in tomes:
        if union_find.find(tome.id) != tome.id or tome.embedding is None:
            continue
        vector = np.asarray(tome.embedding, dtype=np.float32)
        if math.isclose(float(np.linalg.norm(vector)), 0.0):
            continue
        semantic_candidates[tome.id] = vector

    for left_id, right_id in _iter_semantic_candidate_pairs(semantic_candidates, settings):
        left_vec = semantic_candidates[left_id]
        right_vec = semantic_candidates[right_id]
        if _cosine_similarity(left_vec, right_vec) >= settings.threshold:
            union_find.union(left_id, right_id)
    semantic_groups = max(
        0,
        _count_duplicate_groups(union_find, list(by_id))
        - exact_content_groups
        - fact_overlap_groups,
    )

    grouped_ids: dict[UUID, list[UUID]] = defaultdict(list)
    for node_id in by_id:
        grouped_ids[union_find.find(node_id)].append(node_id)

    groups = [
        [by_id[node_id] for node_id in sorted(group, key=str)]
        for group in grouped_ids.values()
        if len(group) > 1
    ]
    groups.sort(key=lambda group: (-len(group), [str(tome.id) for tome in group]))

    return DuplicateScanResult(
        groups=groups,
        scanned=len(tomes),
        exact_content_groups=exact_content_groups,
        fact_overlap_groups=fact_overlap_groups,
        semantic_groups=semantic_groups,
        ignored_high_frequency_facts=ignored_high_frequency_facts,
    )
