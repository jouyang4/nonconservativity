from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

EPS = 1e-30
DEFAULT_DIST_EDGES = (0.0, 8.0, 12.0, 20.0, 30.0, 1e18)
DEFAULT_SEP_EDGES = (1, 5, 17, 65, 1 << 30)


def _stratum_index(dist, sep, dist_edges, sep_edges):
    di = np.clip(np.digitize(dist, np.asarray(dist_edges[1:-1], float)), 0, len(dist_edges) - 2)
    si = np.clip(np.digitize(sep, np.asarray(sep_edges[1:-1], float)), 0, len(sep_edges) - 2)
    return di * (len(sep_edges) - 1) + si


def population_stratum_counts(ca_coords, res_ids, dist_edges=DEFAULT_DIST_EDGES, sep_edges=DEFAULT_SEP_EDGES):
    x = np.asarray(ca_coords, float)
    r = np.asarray(res_ids, float).reshape(-1)
    n = x.shape[0]
    if r.shape[0] != n:
        raise ValueError("res_ids length must match ca_coords")
    iu, ju = np.triu_indices(n, 1)
    d = np.linalg.norm(x[iu] - x[ju], axis=1)
    s = np.abs(r[iu] - r[ju])
    idx = _stratum_index(d, s, dist_edges, sep_edges)
    n_strata = (len(dist_edges) - 1) * (len(sep_edges) - 1)
    counts = np.bincount(idx, minlength=n_strata).astype(np.int64)
    return counts, n_strata


@dataclass
class StratifiedEstimate:
    lambda_offdiag: float
    off_normsq_total: float
    off_antisym_normsq_total: float
    n_pairs_possible: int
    n_pairs_sampled: int
    effective_weighted_sample_size: float
    strata: list = field(default_factory=list)
    dist_edges: tuple = DEFAULT_DIST_EDGES
    sep_edges: tuple = DEFAULT_SEP_EDGES

    def as_dict(self):
        return {
            "lambda_offdiag": self.lambda_offdiag,
            "estimated_offdiagonal_normsq": self.off_normsq_total,
            "estimated_offdiagonal_antisym_normsq": self.off_antisym_normsq_total,
            "n_pairs_possible": self.n_pairs_possible,
            "n_pairs_sampled": self.n_pairs_sampled,
            "effective_weighted_sample_size": self.effective_weighted_sample_size,
            "strata": self.strata,
            "dist_edges": [float(x) for x in self.dist_edges],
            "sep_edges": [float(x) for x in self.sep_edges],
        }


def stratified_offdiag_estimate(
    *,
    obs_distance: np.ndarray,
    obs_sep: np.ndarray,
    obs_pair_normsq: np.ndarray,
    obs_pair_antisym_normsq: np.ndarray,
    ca_coords: np.ndarray,
    res_ids: np.ndarray,
    dist_edges: Sequence[float] = DEFAULT_DIST_EDGES,
    sep_edges: Sequence[float] = DEFAULT_SEP_EDGES,
) -> StratifiedEstimate:
    d = np.asarray(obs_distance, float).reshape(-1)
    s = np.asarray(obs_sep, float).reshape(-1)
    pn = np.asarray(obs_pair_normsq, float).reshape(-1)
    pa = np.asarray(obs_pair_antisym_normsq, float).reshape(-1)
    if not (d.shape == s.shape == pn.shape == pa.shape):
        raise ValueError("observed arrays must share shape")
    Ns, n_strata = population_stratum_counts(ca_coords, res_ids, dist_edges, sep_edges)
    idx = _stratum_index(d, s, dist_edges, sep_edges)
    ns = np.bincount(idx, minlength=n_strata).astype(np.int64)
    sum_norm = np.bincount(idx, weights=pn, minlength=n_strata)
    sum_anti = np.bincount(idx, weights=pa, minlength=n_strata)

    off_total = 0.0
    off_anti = 0.0
    strata = []
    weights_per_obs = np.zeros(d.size)
    for k in range(n_strata):
        if ns[k] == 0:
            if Ns[k] > 0:
                strata.append({"stratum": int(k), "N_pop": int(Ns[k]), "n_obs": 0, "weight": None})
            continue
        w = Ns[k] / ns[k]
        off_total += w * sum_norm[k]
        off_anti += w * sum_anti[k]
        weights_per_obs[idx == k] = w
        strata.append({"stratum": int(k), "N_pop": int(Ns[k]), "n_obs": int(ns[k]),
                        "weight": float(w), "stratum_lambda": float(math.sqrt(max(sum_anti[k], 0.0) / max(sum_norm[k], EPS)))})
    n_eff = float((weights_per_obs.sum() ** 2) / max(np.sum(weights_per_obs ** 2), EPS))
    return StratifiedEstimate(
        lambda_offdiag=math.sqrt(max(off_anti, 0.0) / max(off_total, EPS)),
        off_normsq_total=float(off_total),
        off_antisym_normsq_total=float(off_anti),
        n_pairs_possible=int(Ns.sum()),
        n_pairs_sampled=int(d.size),
        effective_weighted_sample_size=n_eff,
        strata=strata,
        dist_edges=tuple(dist_edges),
        sep_edges=tuple(sep_edges),
    )
