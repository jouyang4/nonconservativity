from __future__ import annotations
import numpy as np

__all__ = ["compute_ca_jacobian_blocks"]


def compute_ca_jacobian_blocks(score_fn, x, rep_atom, rows=None):
    import torch
    rep_atom = np.asarray(rep_atom, int); T = rep_atom.shape[0]
    rows = np.arange(T) if rows is None else np.asarray(rows, int)
    J = np.zeros((T, T, 3, 3), np.float64)
    for i in rows:
        ai = int(rep_atom[i])
        xt = torch.as_tensor(np.asarray(x, np.float32)).clone().requires_grad_(True)
        s = score_fn(xt)
        for c in range(3):
            sel = torch.zeros_like(s); sel[ai, c] = 1.0
            grad = torch.autograd.grad((s * sel).sum(), xt, retain_graph=(c < 2))[0]
            J[i, :, c, :] = grad.detach().cpu().numpy()[rep_atom]
    return J
