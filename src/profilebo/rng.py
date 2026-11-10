"""Deterministic seeds for a reproducible campaign.

Every random draw in a campaign (the acquisition step with its multi-start,
MC samples, GP-fit retries and resolution scan, and whatever randomness the
evaluator uses, such as the particle disperser and Serpent transport) gets
its own seed derived from the campaign seed and the case tag. Keying on the
tag rather than on a running counter means a restarted master regenerates
exactly the same numbers for the same step.
"""

from __future__ import annotations

import zlib

import numpy as np


def derive_seed(base: int, *keys: str | int) -> int:
    """A seed in [1, 2**31 - 1] determined by 'base' and 'keys'.

    31 bits because Serpent reads SERPENT_RNG_SEED with atoi and
    rejects a seed < 1.
    """
    entropy = [int(base)] + [zlib.crc32(str(k).encode()) for k in keys]
    state = np.random.SeedSequence(entropy).generate_state(1, dtype=np.uint64)[0]
    return int(state % np.uint64(2**31 - 1)) + 1
