"""seeds.py — deterministic seeding and apportionment.

Targets are split across units by largest-remainder apportionment, and each cell carries
seed = sha1(run_id|cell_id)."""
from __future__ import annotations

import hashlib
import random


def seed_int(*parts: str) -> int:
    """Integer seed from the first 12 hex digits of sha1 over the '|'-joined parts."""
    h = hashlib.sha1("|".join(parts).encode()).hexdigest()
    return int(h[:12], 16)


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()


def largest_remainder(weights: dict[str, float], n: int) -> dict[str, int]:
    """Exact total, proportional shares, no rounding drift.

    Ties are broken on the key, so the result does not depend on dict order."""
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("weights sum to zero")
    quotas = {k: n * w / total for k, w in weights.items()}
    counts = {k: int(q) for k, q in quotas.items()}
    short = n - sum(counts.values())
    order = sorted(quotas, key=lambda k: (-(quotas[k] - counts[k]), k))[:short]
    for k in order:
        counts[k] += 1
    assert sum(counts.values()) == n
    return counts


def weighted(rng: random.Random, weights: dict[str, float]) -> str:
    """Deterministic weighted pick consuming exactly one rng.random()."""
    keys = list(weights)
    total = sum(weights[k] for k in keys)
    x = rng.random() * total
    acc = 0.0
    for k in keys:
        acc += weights[k]
        if x < acc:
            return k
    return keys[-1]
