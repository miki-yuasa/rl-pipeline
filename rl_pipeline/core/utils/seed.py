"""Seed sampling and deterministic per-replicate derivation helpers."""

from __future__ import annotations

import secrets

import numpy as np

_REPLICATE_SEED_MASK = 0x3FFFFFFF


def sample_base_seed() -> int:
    """Sample a non-negative 31-bit random base seed from OS entropy."""
    return secrets.randbits(31)


def derive_replicate_seed(base_seed: int, rep_id: int) -> int:
    """Derive an independent 30-bit seed for a replicate index.

    Args:
        base_seed: Non-negative root seed shared across replicates.
        rep_id: Non-negative absolute replicate identifier.

    Returns:
        Deterministic 30-bit seed in ``[0, 2**30)``.

    Raises:
        ValueError: If ``base_seed`` or ``rep_id`` is negative.
    """
    if base_seed < 0:
        raise ValueError(f"base_seed must be non-negative, got {base_seed}.")
    if rep_id < 0:
        raise ValueError(f"rep_id must be non-negative, got {rep_id}.")
    child_seq = np.random.SeedSequence(base_seed, spawn_key=(rep_id,))
    state = child_seq.generate_state(1, dtype=np.uint32)[0]
    return int(state & _REPLICATE_SEED_MASK)
