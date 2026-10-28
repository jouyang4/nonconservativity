# nonconservativity

Reciprocity and path-integrability analysis of the Cartesian vector fields differentiated from
pretrained structure-prediction models (Boltz-2, ESMFold2-Fast).

Papers:

- Hidden non-conservativity in structure prediction models.
  https://openreview.net/forum?id=3bb2CB8WHx
- https://openreview.net/forum?id=EMnLEnwXbA

## Main quantity

`J_ij` is the 3x3 block `d s_i / d x_j` of the Cartesian field `s` at C-alpha representative
atoms. A field that is the gradient of a scalar potential satisfies the reciprocity
(integrability) condition

    J_ij = J_ji^T    for all i, j.

For each unordered pair `p = (i, j)`, `i < j`,

    a_p = 0.5 * || J_ij - J_ji^T ||_F^2
    b_p =       || J_ij ||_F^2 + || J_ji ||_F^2

    lambda_off = sqrt( sum_p a_p / sum_p b_p ).

`lambda_off = 0` for perfectly reciprocal blocks, `1` for transpose-antisymmetric pairing,
`1/sqrt(2)` for i.i.d. random blocks (`metrics.IID_BASELINE`), and is invariant to scaling every
block by a common nonzero scalar. Implemented once, in `src/nonconservativity/metrics.py`.

## Install

    pip install -r requirements.txt

or

    conda env create -f environment.yml

## Reproduce

    make test       # unit tests, incl. analytic conservative-field controls
    make stats      # recompute and assert 34 headline values against expected_results.yaml
    make figures    # regenerate figures/ from data/
    make examples   # numpy-only lambda_off and conservative-control examples
    make smoke      # all of the above
    make audit      # the above, writing release_audit.json

Level 1 and Level 2 (statistics, figures and `lambda_off` from stored blocks) run on CPU in
seconds. Regenerating the Jacobians from model weights requires a GPU and the model packages, and
is not shipped as runnable scripts here.

## Pinning

Boltz-2 `boltz==2.2.1`; single-sequence, deterministic, with augmentation, reverse-diffusion
alignment, steering and guidance disabled for field evaluation (`configs/`). ESMFold2-Fast pinned
to weights revision `c6c7958d63f5`; the 15 proteins run under the alternate revision are excluded
from the matched comparison and shipped separately. Per-protein RNG seed 20260725, with the sampled
row atom indices in `data/sampled_rows.json`. sigma = 4 Angstrom for the headline comparison.
PyTorch and CUDA versions for the archived model runs were not recorded; bitwise determinism is not
claimed.

## Layout

    src/nonconservativity/   metric, estimator, Jacobian blocks, ANM, path integration
    data/                    result tables
    scripts/                 reproduction entry points
    examples/                numpy-only examples
    configs/                 model-side evaluation configs
    expected_results.yaml    asserted values

## License

MIT. See `LICENSE`.
