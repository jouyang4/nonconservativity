import os, sys, json
import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "data")


def load_expected(path):
    root = {}; last = None
    for line in open(path):
        if not line.strip():
            continue
        ind = len(line) - len(line.lstrip()); k, _, v = line.strip().partition(":"); v = v.strip()
        if ind == 0:
            if v == "":
                root[k] = {}; last = k
            else:
                root[k] = json.loads(v); last = None
        else:
            root[last][k] = json.loads(v)
    return root


EXP = load_expected(os.path.join(ROOT, "expected_results.yaml"))
drift = []


def rd(*p): return pd.read_csv(os.path.join(D, *p))


def sp(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float); m = np.isfinite(x) & np.isfinite(y)
    return float(stats.spearmanr(x[m], y[m]).correlation), int(m.sum())


def ck(label, val, exp, tol=5e-3):
    if isinstance(exp, str):
        ok = str(val) == exp
    else:
        ok = abs(float(val) - float(exp)) <= (abs(exp) * 0.02 if abs(exp) >= 1 else (1e-19 if abs(exp) < 1e-10 else tol))
    print("  %-40s = %-12s (exp %s) %s" % (label, ("%.5g" % val) if not isinstance(val, str) else val, exp, "OK" if ok else "DRIFT"))
    if not ok:
        drift.append(label)


def main():
    b = rd("reciprocity", "boltz2_single_sequence.csv"); e = rd("reciprocity", "esmfold2_fast.csv")
    xm = rd("reciprocity", "reciprocity_matched.csv"); msa = rd("reciprocity", "reciprocity_boltz2_msa_vs_single.csv")
    mm = rd("reciprocity", "boltz2_msa.csv"); alt = rd("reciprocity", "esmfold2_fast_alternate_revision_excluded.csv")
    wr = rd("consequence", "consequence_W_vs_rmsd.csv"); mp = rd("consequence", "path_dependence_msa_vs_single.csv")
    hodge = json.load(open(os.path.join(D, "controls", "adenylate_kinase_hodge.json")))

    print("=== RECIPROCITY ===")
    ck("Boltz-2 n", len(b), EXP["boltz2_single_sequence"]["n"])
    ck("Boltz-2 median lambda_off", float(np.median(b["lambda_off"])), EXP["boltz2_single_sequence"]["median_lambda_off"])
    ck("ESMFold2-Fast n", len(e), EXP["esmfold2_fast"]["n"])
    ck("ESMFold2-Fast median lambda_off", float(np.median(e["lambda_off"])), EXP["esmfold2_fast"]["median_lambda_off"])
    ck("ESM weights_revision", str(e["weights_revision"].iloc[0]), EXP["esmfold2_fast"]["weights_revision"])
    ck("ESM alt-revision excluded n", len(alt), EXP["esmfold2_fast"]["alternate_revision_excluded_n"])

    print("=== CROSS-MODEL ===")
    bx, ex = xm["lambda_off_boltz2_single_sequence"].values, xm["lambda_off_esmfold2_fast"].values
    ck("cross-model n", len(xm), EXP["cross_model"]["n"])
    ck("cross-model Pearson r", float(stats.pearsonr(bx, ex)[0]), EXP["cross_model"]["pearson_r"])
    ck("cross-model Spearman", float(stats.spearmanr(bx, ex).correlation), EXP["cross_model"]["spearman_rho"])

    print("=== CONFIDENCE ===")
    bc = b.dropna(subset=["mean_plddt"]); ec = e.dropna(subset=["mean_plddt"]); nc = b.dropna(subset=["near_contact_lambda_off", "mean_plddt"])
    ck("Boltz-2 pLDDT Spearman", sp(bc["mean_plddt"], bc["lambda_off"])[0], EXP["confidence"]["boltz2_plddt_spearman"])
    ck("Boltz-2 pLDDT n", sp(bc["mean_plddt"], bc["lambda_off"])[1], EXP["confidence"]["boltz2_plddt_n"])
    ck("ESMFold2-Fast pLDDT Spearman", sp(ec["mean_plddt"], ec["lambda_off"])[0], EXP["confidence"]["esmfold2_plddt_spearman"])
    ck("ESMFold2-Fast pLDDT n", sp(ec["mean_plddt"], ec["lambda_off"])[1], EXP["confidence"]["esmfold2_plddt_n"])
    ck("near-contact Spearman", sp(nc["near_contact_lambda_off"], nc["mean_plddt"])[0], EXP["confidence"]["near_contact_spearman"])

    print("=== MSA EFFECT ===")
    ss, ms = msa["lambda_off_single_sequence"].values, msa["lambda_off_msa"].values
    ck("MSA median single-seq", float(np.median(ss)), EXP["msa_effect"]["median_single_sequence"])
    ck("MSA median MSA", float(np.median(mm["lambda_off"])), EXP["msa_effect"]["median_msa"])
    ck("MSA median delta", float(np.median(ms - ss)), EXP["msa_effect"]["median_delta"])
    ck("MSA fraction decreased", float(np.mean(ms < ss)), EXP["msa_effect"]["fraction_decreased"])
    ck("MSA Wilcoxon p", float(stats.wilcoxon(ss, ms).pvalue), EXP["msa_effect"]["paired_wilcoxon_p"])
    ck("MSA paired Spearman", float(stats.spearmanr(ss, ms).correlation), EXP["msa_effect"]["paired_spearman"])

    print("=== PATH DEPENDENCE ===")
    ck("Boltz-2 path median exp|W|", float(np.median(wr["exp_abs_W_boltz2"])), EXP["path_dependence"]["boltz2_median_exp_abs_W"])
    ck("ESMFold2-Fast path median exp|W|", float(np.median(wr["exp_abs_W_esmfold2"].dropna())), EXP["path_dependence"]["esmfold2_median_exp_abs_W"])
    low = wr[wr["ca_rmsd_A"] < 1.7]
    ck("low-RMSD n", len(low), EXP["path_dependence"]["low_rmsd_n"])
    ck("low-RMSD Boltz-2 median", float(np.median(low["exp_abs_W_boltz2"])), EXP["path_dependence"]["low_rmsd_boltz2_median"])
    ck("low-RMSD ESMFold2-Fast median", float(np.median(low["exp_abs_W_esmfold2"].dropna())), EXP["path_dependence"]["low_rmsd_esmfold2_median"])
    wm = np.isfinite(wr["abs_W_boltz2"]) & np.isfinite(wr["ca_rmsd_A"])
    ck("|W| vs RMSD Spearman", float(stats.spearmanr(wr["abs_W_boltz2"][wm], wr["ca_rmsd_A"][wm]).correlation), EXP["path_dependence"]["abs_W_vs_rmsd_spearman"])

    print("=== MSA PATH EFFECT ===")
    sa, mb = mp["exp_abs_W_single_sequence"].values, mp["exp_abs_W_msa"].values
    ck("MSA path median single", float(np.median(sa)), EXP["msa_path_effect"]["median_single"])
    ck("MSA path median MSA", float(np.median(mb)), EXP["msa_path_effect"]["median_msa"])
    ck("MSA path Wilcoxon p", float(stats.wilcoxon(sa, mb).pvalue), EXP["msa_path_effect"]["paired_wilcoxon_p"])
    ck("MSA path median ratio", float(np.median(mb / sa)), EXP["msa_path_effect"]["median_ratio_msa_over_single"])
    ck("MSA path Spearman", float(stats.spearmanr(sa, mb).correlation), EXP["msa_path_effect"]["paired_spearman"])

    print("=== HODGE CONTROL (adenylate_kinase) ===")
    ck("Hodge sigma_A", hodge["sigma_A"], EXP["hodge_adenylate_kinase"]["sigma_A"])
    ck("Hodge gradient fraction", round(hodge["gradient_fraction_of_squared_field_norm"], 6), EXP["hodge_adenylate_kinase"]["gradient_fraction"])
    ck("Hodge raw circulation", round(hodge["raw_circulation"], 4), EXP["hodge_adenylate_kinase"]["raw_circulation"])

    print()
    if drift:
        print("FAIL %d %s" % (len(drift), drift)); sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
