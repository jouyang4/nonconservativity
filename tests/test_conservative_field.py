import numpy as np
from nonconservativity.metrics import lambda_off_from_full_jacobian
from nonconservativity.paths import loop_work

rng = np.random.default_rng(1)


def _block_jacobian_of_linear_field(A, n):
    J = np.zeros((n, n, 3, 3))
    for i in range(n):
        for j in range(n):
            J[i, j] = A[3 * i:3 * i + 3, 3 * j:3 * j + 3]
    return J


def test_conservative_linear_field_has_zero_lambda_and_zero_loop():
    n = 8
    M = rng.standard_normal((3 * n, 3 * n))
    H = 0.5 * (M + M.T)
    c = rng.standard_normal(3 * n)
    J = _block_jacobian_of_linear_field(H, n)
    assert lambda_off_from_full_jacobian(J) < 1e-10

    def field(x):
        return (H @ x.reshape(-1) + c).reshape(x.shape)
    fwd = np.stack([rng.standard_normal((n, 3)) for _ in range(2)])
    a, b = fwd[0], fwd[1]
    p1 = np.stack([a, 0.5 * (a + b) + rng.standard_normal((n, 3)), b])
    p2 = np.stack([a, 0.5 * (a + b) - rng.standard_normal((n, 3)), b])
    assert abs(loop_work(field, p1, p2, n_intervals=64)) < 1e-8


def test_nonconservative_linear_field_has_positive_lambda_and_nonzero_loop():
    n = 8
    M = rng.standard_normal((3 * n, 3 * n))
    K = M - M.T
    H = 0.5 * (M + M.T) + 0.5 * K
    J = _block_jacobian_of_linear_field(H, n)
    assert lambda_off_from_full_jacobian(J) > 1e-3

    def field(x):
        return (H @ x.reshape(-1)).reshape(x.shape)
    a = rng.standard_normal((n, 3)); b = rng.standard_normal((n, 3))
    p1 = np.stack([a, 0.5 * (a + b) + np.ones((n, 3)), b])
    p2 = np.stack([a, 0.5 * (a + b) - np.ones((n, 3)), b])
    assert abs(loop_work(field, p1, p2, n_intervals=64)) > 1e-6
