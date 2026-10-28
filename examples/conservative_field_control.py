import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from nonconservativity.metrics import lambda_off_from_full_jacobian
from nonconservativity.paths import loop_work

rng = np.random.default_rng(0)
n = 8
M = rng.standard_normal((3 * n, 3 * n))
H_sym = 0.5 * (M + M.T)
K = M - M.T


def blocks(A):
    J = np.zeros((n, n, 3, 3))
    for i in range(n):
        for j in range(n):
            J[i, j] = A[3 * i:3 * i + 3, 3 * j:3 * j + 3]
    return J


def field(A):
    return lambda x: (A @ x.reshape(-1)).reshape(x.shape)


a, b = rng.standard_normal((n, 3)), rng.standard_normal((n, 3))
p1 = np.stack([a, 0.5 * (a + b) + np.ones((n, 3)), b])
p2 = np.stack([a, 0.5 * (a + b) - np.ones((n, 3)), b])

for label, A in [("symmetric", H_sym), ("symmetric+antisymmetric", H_sym + 0.5 * K)]:
    lo = lambda_off_from_full_jacobian(blocks(A))
    W = loop_work(field(A), p1, p2, n_intervals=128)
    print("%-24s lambda_off=%.2e loop_work=%.4e" % (label, lo, W))
