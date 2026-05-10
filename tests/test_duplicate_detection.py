from __future__ import annotations

from uuid import uuid4

import numpy as np

from src.config import TidySettings
from src.services import duplicate_detection


def test_semantic_band_size_does_not_increase_plane_count(
    monkeypatch,
) -> None:
    captured_shape: tuple[int, int] | None = None
    expected_planes = 3

    class FakeRng:
        def standard_normal(
            self,
            shape: tuple[int, int],
            dtype: type[np.float32],
        ) -> np.ndarray:
            nonlocal captured_shape
            captured_shape = shape
            assert dtype is np.float32
            return np.ones(shape, dtype=np.float32)

    monkeypatch.setattr(duplicate_detection.np.random, "default_rng", lambda seed: FakeRng())

    vectors = {
        uuid4(): np.array([1.0, 0.0], dtype=np.float32),
        uuid4(): np.array([0.99, 0.01], dtype=np.float32),
    }
    settings = TidySettings(
        semantic_planes=expected_planes,
        semantic_band_size=10,
        threshold=0.9,
    )

    candidate_pairs = duplicate_detection._iter_semantic_candidate_pairs(vectors, settings)

    assert captured_shape == (expected_planes, 2)
    assert len(candidate_pairs) == 1
