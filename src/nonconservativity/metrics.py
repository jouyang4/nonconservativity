from __future__ import annotations
import numpy as np

__all__ = ["pair_terms", "lambda_off", "lambda_off_from_full_jacobian", "stratified_lambda_off", "IID_BASELINE"]

IID_BASELINE = 1.0 / np.sqrt(2.0)


def pair_terms(J_ij: np.ndarray, J_ji: np.ndarray):
    J_ij = np.asarray(J_ij, float); J_ji = np.asarray(J_ji, float)
    a = 0.5 * float(((J_ij - J_ji.T) ** 2).sum())
    b = float((J_ij ** 2).sum() + (J_ji ** 2).sum())
    return a, b


def lambda_off(blocks_ij: np.ndarray, blocks_ji: np.ndarray) -> float:
    Bij = np.asarray(blocks_ij, float); Bji = np.asarray(blocks_ji, float)
    if Bij.shape != Bji.shape or Bij.ndim != 3 or Bij.shape[1:] != (3, 3):
        raise ValueError("blocks must both be (P, 3, 3)")
    diff = Bij - np.transpose(Bji, (0, 2, 1))
    A = 0.5 * (diff ** 2).sum()
    Bn = (Bij ** 2).sum() + (Bji ** 2).sum()
    return float(np.sqrt(A / Bn)) if Bn > 0 else float("nan")


def lambda_off_from_full_jacobian(J: np.ndarray) -> float:
    J = np.asarray(J, float)
    T = J.shape[0]
    iu, ju = np.triu_indices(T, 1)
    return lambda_off(J[iu, ju], J[ju, iu])


def stratified_lambda_off(a_p: np.ndarray, b_p: np.ndarray, inclusion_prob: np.ndarray) -> float:
    a_p = np.asarray(a_p, float); b_p = np.asarray(b_p, float); pi = np.asarray(inclusion_prob, float)
    if not (a_p.shape == b_p.shape == pi.shape):
        raise ValueError("a_p, b_p, inclusion_prob must have the same shape")
    if np.any(pi <= 0) or np.any(pi > 1):
        raise ValueError("inclusion probabilities must be in (0, 1]")
    A = np.sum(a_p / pi); B = np.sum(b_p / pi)
    return float(np.sqrt(A / B)) if B > 0 else float("nan")
