import numpy as np
from nonconservativity.metrics import (
    lambda_off, lambda_off_from_full_jacobian, stratified_lambda_off, pair_terms, IID_BASELINE)

rng = np.random.default_rng(0)


def _pairs(P):
    return rng.standard_normal((P, 3, 3))


def test_symmetric_reciprocal_is_zero():
    Bij = _pairs(200)
    Bji = np.transpose(Bij, (0, 2, 1))
    assert lambda_off(Bij, Bji) < 1e-12


def test_transpose_antisymmetric_is_one():
    Bij = _pairs(200)
    Bji = -np.transpose(Bij, (0, 2, 1))
    assert abs(lambda_off(Bij, Bji) - 1.0) < 1e-12


def test_iid_random_approaches_one_over_sqrt2():
    Bij = _pairs(200000)
    Bji = _pairs(200000)
    assert abs(lambda_off(Bij, Bji) - IID_BASELINE) < 5e-3


def test_scale_invariance():
    Bij, Bji = _pairs(300), _pairs(300)
    base = lambda_off(Bij, Bji)
    for c in (2.0, 0.1, -7.0):
        assert abs(lambda_off(c * Bij, c * Bji) - base) < 1e-12


def test_full_jacobian_matches_pairwise():
    T = 12
    J = rng.standard_normal((T, T, 3, 3))
    iu, ju = np.triu_indices(T, 1)
    assert abs(lambda_off_from_full_jacobian(J) - lambda_off(J[iu, ju], J[ju, iu])) < 1e-12


def test_stratified_estimator_matches_exact_in_expectation():
    T = 60
    J = rng.standard_normal((T, T, 3, 3))
    iu, ju = np.triu_indices(T, 1)
    a = np.array([pair_terms(J[i, j], J[j, i])[0] for i, j in zip(iu, ju)])
    b = np.array([pair_terms(J[i, j], J[j, i])[1] for i, j in zip(iu, ju)])
    exact = float(np.sqrt(a.sum() / b.sum()))
    ests = []
    for s in range(400):
        r = np.random.default_rng(s)
        pi = 0.4
        keep = r.random(a.size) < pi
        ests.append(stratified_lambda_off(a[keep], b[keep], np.full(keep.sum(), pi)))
    assert abs(np.mean(ests) - exact) < 0.02


def test_stratified_exhaustive_limit_is_exact():
    a = rng.random(100) + 0.1
    b = rng.random(100) + 0.1
    exact = float(np.sqrt(a.sum() / b.sum()))
    assert abs(stratified_lambda_off(a, b, np.ones_like(a)) - exact) < 1e-12
