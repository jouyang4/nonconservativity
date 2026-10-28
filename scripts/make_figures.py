import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "data"); F = os.path.join(ROOT, "figures"); os.makedirs(F, exist_ok=True)


def save(fig, name):
    fig.savefig(os.path.join(F, name + ".png"), dpi=150, bbox_inches="tight")
    fig.savefig(os.path.join(F, name + ".pdf"), bbox_inches="tight"); plt.close(fig)


def fig_crossmodel():
    xm = pd.read_csv(os.path.join(D, "reciprocity", "reciprocity_matched.csv"))
    b, e = xm["lambda_off_boltz2_single_sequence"].values, xm["lambda_off_esmfold2_fast"].values
    r = stats.pearsonr(b, e)[0]; rho = stats.spearmanr(b, e).correlation
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(b, e, s=18, color="#245f9e", alpha=0.7)
    ax.set_xlabel(r"$\lambda_{off}$ Boltz-2 (single sequence)"); ax.set_ylabel(r"$\lambda_{off}$ ESMFold2-Fast")
    ax.set_title("Cross-model non-reciprocity\nn=%d  Pearson r=%.3f  Spearman=%.3f" % (len(xm), r, rho), fontsize=10)
    save(fig, "F_crossmodel_scatter"); print("F_crossmodel: n=%d r=%.4f rho=%.4f" % (len(xm), r, rho))


def fig_confidence():
    b = pd.read_csv(os.path.join(D, "reciprocity", "boltz2_single_sequence.csv")).dropna(subset=["mean_plddt"])
    rr = stats.spearmanr(b["mean_plddt"], b["lambda_off"])
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(b["mean_plddt"], b["lambda_off"], s=16, color="#245f9e", alpha=0.7)
    ax.set_xlabel("Mean pLDDT"); ax.set_ylabel(r"$\lambda_{off}$")
    ax.set_title(r"Boltz-2: $\rho$ = %.3f, n = %d" % (rr.correlation, len(b)), fontsize=11)
    fig.tight_layout(); save(fig, "F_confidence")
    print("F_confidence: rho=%.4f n=%d" % (rr.correlation, len(b)))

def fig_consequence():
    wr = pd.read_csv(os.path.join(D, "consequence", "consequence_W_vs_rmsd.csv"))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.4, 1]})
    ax[0].scatter(wr["ca_rmsd_A"], wr["exp_abs_W_boltz2"], s=32, color="#245f9e", label="Boltz-2")
    if "exp_abs_W_esmfold2" in wr:
        ax[0].scatter(wr["ca_rmsd_A"], wr["exp_abs_W_esmfold2"], s=18, color="#7ba3d0", marker="^", label="ESMFold2-Fast")
    ax[0].axhline(1.0, color="#888", ls=":", lw=1); ax[0].set_yscale("log")
    ax[0].set_xlabel("endpoint C$\\alpha$-RMSD (Å)"); ax[0].set_ylabel(r"$\exp|W|$")
    m = np.isfinite(wr["abs_W_boltz2"]) & np.isfinite(wr["ca_rmsd_A"])
    rho = stats.spearmanr(wr["abs_W_boltz2"][m], wr["ca_rmsd_A"][m]).correlation
    ax[0].set_title("All 25 pairs  (|W| vs RMSD Spearman=%.3f)" % rho, fontsize=10); ax[0].legend(fontsize=8)
    low = wr[wr["ca_rmsd_A"] < 1.7]
    ax[1].scatter(low["ca_rmsd_A"], low["exp_abs_W_boltz2"], s=42, color="#245f9e", label="Boltz-2")
    ax[1].scatter(low["ca_rmsd_A"], low["exp_abs_W_esmfold2"], s=24, color="#7ba3d0", marker="^", label="ESMFold2-Fast")
    ax[1].axhline(1.0, color="#888", ls=":", lw=1)
    ax[1].set_xlabel("endpoint C$\\alpha$-RMSD (Å)"); ax[1].set_ylabel(r"$\exp|W|$")
    ax[1].set_title("RMSD < 1.7 Å  (n=%d; median %.2f / %.2f)" %
                    (len(low), float(np.median(low["exp_abs_W_boltz2"])), float(np.median(low["exp_abs_W_esmfold2"].dropna()))), fontsize=9)
    ax[1].legend(fontsize=8)
    fig.tight_layout(); save(fig, "F_consequence_crossmodel_rmsd")
    print("F_consequence: spearman=%.4f low_rmsd_n=%d" % (rho, len(low)))


if __name__ == "__main__":
    fig_crossmodel(); fig_confidence(); fig_consequence()
