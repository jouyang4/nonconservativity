import numpy as np


def ca_anm_modes(coords: np.ndarray, ca_idx: np.ndarray, cutoff: float = 12.0, count: int = 20) -> np.ndarray:
    xyz = np.asarray(coords, np.float64)[np.asarray(ca_idx, np.int64)]
    n = len(xyz)
    diff = xyz[:, None] - xyz[None, :]
    r2 = np.sum(diff * diff, axis=-1)
    contact = (r2 <= cutoff * cutoff) & (r2 > 1e-8)
    h = np.zeros((3*n, 3*n), np.float64)
    ii, jj = np.where(np.triu(contact, 1))
    for i, j in zip(ii.tolist(), jj.tolist()):
        d = diff[i, j]
        block = -(np.outer(d, d) / r2[i, j])
        h[3*i:3*i+3, 3*j:3*j+3] += block
        h[3*j:3*j+3, 3*i:3*i+3] += block
        h[3*i:3*i+3, 3*i:3*i+3] -= block
        h[3*j:3*j+3, 3*j:3*j+3] -= block
    w, v = np.linalg.eigh(h)
    order = np.argsort(w)
    return v[:, order[6:6+int(count)]].T.reshape(-1, n, 3)
