import numpy as np
from nonconservativity.sampling import row_budget, row_inclusion_probability, sample_rows, pair_inclusion_probability


def test_row_budget_thresholds():
    assert row_budget(50) == 50
    assert row_budget(96) == 96
    assert row_budget(97) == 64
    assert row_budget(299) == 64
    assert row_budget(300) == 96
    assert row_budget(1000) == 96


def test_small_proteins_are_exhaustive():
    assert row_inclusion_probability(80) == 1.0
    assert pair_inclusion_probability(80) == 1.0
    rows = sample_rows(80, np.random.default_rng(0))
    assert len(rows) == 80


def test_sampled_budget_and_marginals():
    rows = sample_rows(500, np.random.default_rng(0))
    assert len(rows) == 96
    assert len(set(rows.tolist())) == 96
    assert abs(row_inclusion_probability(500) - 96 / 500) < 1e-12


def test_empirical_row_inclusion_matches_marginal():
    n = 200
    counts = np.zeros(n)
    trials = 4000
    for s in range(trials):
        counts[sample_rows(n, np.random.default_rng(s))] += 1
    emp = counts / trials
    assert abs(emp.mean() - row_inclusion_probability(n)) < 0.02
