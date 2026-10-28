from __future__ import annotations
import numpy as np

__all__ = ["row_budget", "row_inclusion_probability", "sample_rows", "pair_inclusion_probability"]


def row_budget(n_ca: int) -> int:
    if n_ca <= 96:
        return int(n_ca)
    if n_ca < 300:
        return 64
    return 96


def row_inclusion_probability(n_ca: int) -> float:
    b = row_budget(n_ca)
    return 1.0 if b >= n_ca else b / float(n_ca)


def sample_rows(n_ca: int, rng: np.random.Generator) -> np.ndarray:
    b = row_budget(n_ca)
    if b >= n_ca:
        return np.arange(n_ca)
    return np.sort(rng.choice(n_ca, size=b, replace=False))


def pair_inclusion_probability(n_ca: int) -> float:
    b = row_budget(n_ca)
    if b >= n_ca:
        return 1.0
    return (b / n_ca) * ((b - 1) / (n_ca - 1))
