from __future__ import annotations

import numpy as np
from absl.testing import absltest, parameterized

from rl_pipeline.core.eval.stats import (
    compute_basic_stats,
    scaled_same_convolve,
)


class ScaledSameConvolveTest(parameterized.TestCase):
    @parameterized.named_parameters(
        ("odd_window", 3),
        ("even_window", 4),
        ("window_size_one", 1),
        ("window_size_equal_length", 5),
    )
    def test_constant_array_preserves_value(self, window_size: int) -> None:
        data = np.full(5, 4.0)
        result = scaled_same_convolve(data, window_size)
        np.testing.assert_allclose(result, data)

    @parameterized.named_parameters(
        (
            "odd_window",
            np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            3,
            np.array([1.5, 2.0, 3.0, 4.0, 4.5]),
        ),
        (
            "size_one_identity",
            np.array([1.0, 2.0, 3.0]),
            1,
            np.array([1.0, 2.0, 3.0]),
        ),
    )
    def test_convolve_produces_expected_output(
        self,
        data: np.ndarray,
        window_size: int,
        expected: np.ndarray,
    ) -> None:
        result = scaled_same_convolve(data, window_size)
        np.testing.assert_allclose(result, expected)

    def test_compute_basic_stats(self) -> None:
        stats = compute_basic_stats([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertEqual(stats.mean, 3.0)
        self.assertEqual(stats.median, 3.0)
        self.assertEqual(stats.min, 1.0)
        self.assertEqual(stats.max, 5.0)


if __name__ == "__main__":
    absltest.main()
