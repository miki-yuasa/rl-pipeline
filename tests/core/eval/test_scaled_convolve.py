from __future__ import annotations

import numpy as np
from absl.testing import absltest, parameterized

from rl_pipeline.core.eval.stats import scaled_same_convolve


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

    def test_convolve_produces_expected_output(self) -> None:
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = scaled_same_convolve(data, 3)
        expected = np.array([1.5, 2.0, 3.0, 4.0, 4.5])
        np.testing.assert_allclose(result, expected)


if __name__ == "__main__":
    absltest.main()
