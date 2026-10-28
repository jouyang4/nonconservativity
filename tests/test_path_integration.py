import numpy as np
from nonconservativity.paths import midpoint_line_integral, loop_work

rng = np.random.default_rng(2)


def test_gradient_field_is_path_independent():
    S = rng.standard_normal((9, 9)); S = 0.5 * (S + S.T)

    def field(x):
        return (S @ x.reshape(-1)).reshape(x.shape)
    a = rng.standard_normal((3, 3)); b = rng.standard_normal((3, 3))
    straight = np.stack([a, b])
    bent = np.stack([a, 0.5 * (a + b) + rng.standard_normal((3, 3)), b])
    i1 = midpoint_line_integral(field, straight, n_intervals=256)
    i2 = midpoint_line_integral(field, bent, n_intervals=256)
    assert abs(i1 - i2) < 1e-6
    assert abs(loop_work(field, straight, bent, n_intervals=256)) < 1e-6


def test_midpoint_integration_converges():
    def field(x):
        return np.sin(x)
    a = np.array([0.0, 0.0, 0.0]); b = np.array([np.pi, 0.0, 0.0])
    path = np.stack([a, b])
    exact = 1 - np.cos(np.pi)
    errs = [abs(midpoint_line_integral(field, path, n_intervals=n) - exact) for n in (8, 32, 128, 512)]
    assert errs[-1] < errs[0]
    assert errs[-1] < 1e-3
