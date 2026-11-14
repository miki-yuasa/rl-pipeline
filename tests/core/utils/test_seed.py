from __future__ import annotations

import random

import numpy as np
from absl.testing import absltest

from rl_pipeline.core.utils.seed import (
    derive_replicate_seed,
    sample_base_seed,
)


class SeedUtilsTest(absltest.TestCase):
    def test_sample_base_seed_is_bounded_and_isolated(self) -> None:
        random.seed(123)
        np.random.seed(123)
        py_state = random.getstate()
        expected_np = int(np.random.randint(0, 10**6))

        np.random.seed(123)
        sampled = sample_base_seed()
        self.assertTrue(0 <= sampled < 2**31)
        self.assertEqual(random.getstate(), py_state)
        self.assertEqual(int(np.random.randint(0, 10**6)), expected_np)

    def test_derive_replicate_seed_is_deterministic_distinct_and_bounded(
        self,
    ) -> None:
        seeds = [derive_replicate_seed(17, rep_id) for rep_id in range(5)]
        self.assertEqual(seeds[3], derive_replicate_seed(17, 3))
        self.assertLen(set(seeds), 5)
        self.assertTrue(all(0 <= seed < 2**30 for seed in seeds))

    def test_derive_replicate_seed_rejects_negative_inputs(self) -> None:
        with self.assertRaises(ValueError):
            derive_replicate_seed(-1, 0)
        with self.assertRaises(ValueError):
            derive_replicate_seed(0, -1)


if __name__ == "__main__":
    absltest.main()
