from __future__ import annotations
import numpy as np

__all__ = ["midpoint_line_integral", "loop_work", "detour_paths"]


def midpoint_line_integral(field, path, n_intervals=None):
    path = np.asarray(path, float)
    if n_intervals is not None and n_intervals > 1:
        pts = [path[0]]
        for k in range(len(path) - 1):
            for s in range(1, n_intervals + 1):
                pts.append(path[k] + (path[k + 1] - path[k]) * (s / n_intervals))
        path = np.stack(pts)
    total = 0.0
    for k in range(len(path) - 1):
        mid = 0.5 * (path[k] + path[k + 1])
        dx = path[k + 1] - path[k]
        total += float(np.sum(np.asarray(field(mid), float) * dx))
    return total


def loop_work(field, forward_path, return_path, n_intervals=128):
    wf = midpoint_line_integral(field, forward_path, n_intervals)
    wr = midpoint_line_integral(field, return_path, n_intervals)
    return wf - wr


def detour_paths(baseline, mode, amplitude):
    baseline = np.asarray(baseline, float); mode = np.asarray(mode, float)
    M = baseline.shape[0]
    profile = np.sin(np.linspace(0.0, np.pi, M))[:, None, None]
    bump = amplitude * profile * mode[None]
    return baseline, baseline + bump, baseline - bump
