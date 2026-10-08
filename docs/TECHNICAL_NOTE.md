# Methods and reproduction

## Cross-representation readout and geometry

The 768-image discovery result measures matching-image retrieval from a specified
TRIBE-v2-predicted cortical representation to DINOv2 features. Ridge
selection is nested inside the training partition. Each held-out query is scored
against its true image and 15 distractors from the same dataset-origin group
(COCO, ImageNet or Scene), sampled from the full discovery bank. These groups
define the candidate pool; distractors may include training images. The reported
95.57% Top-1 is therefore a discovery-stage, 16-candidate retrieval result.

The matched-image protocol uses two repeats of four outer folds on 64 matched images,
three-fold inner selection over targets and 13 ridge penalties, and retrieval
against the held-out fold. The discovery protocol uses its own configuration, seed derivation,
inner/outer candidate counts and nested selector. Their Top-1 rates refer to
different candidate universes and selection procedures.

Both normalize targets using training means and standardize predictors using
training moments. Predictors with training standard deviation at most 1e-8 are
excluded; fewer than two usable predictors is an error. The kernel is divided
by the retained predictor count. Held-out targets cannot change the fitted
prediction; gallery targets determine the evaluation score.

The discovery protocol ranks a correct target behind every tied distractor. Its category-null
permutations operate on fixed predictions, without refitting; the stratified
bootstrap resamples fixed query outcomes. These quantify evaluation
variability conditional on the fitted model. Retraining uncertainty would require
resampling the full pipeline. The inherited name `fwer_p` implements an add-one
upper-tail Monte Carlo p-value; multiple-testing correction is a separate step.
The matched-image protocol retains the historical NumPy argsort tie semantics.

Partial RSA ranks upper-triangle distances with average ties, projects out the
category-distance rank vector, and correlates the residual ranks. Constant or
numerically degenerate residuals are undefined and rejected. Neighborhood
overlap is mean rowwise Jaccard overlap; self is explicitly excluded, and equal
distances are ordered by stimulus index. Centered entropy rank uses eigenweights
proportional to squared singular values and the historical 1e-15 cutoff.

### Release-specific changes

The legacy self-neighbor bug is fixed only in the release copy; the later historical
local-scale estimator already used explicit diagonal exclusion. Non-finite
observed/null statistics, complex arrays, non-integer categories, repeated or
out-of-range indices, overlapping train/query sets, nonpositive regularization,
and undefined retrieval/RSA summaries are rejected. These supported-domain
changes are tested separately from valid-input numerical parity.

The release reproduces all 330 matched-image readout summary quantities across
88 folds from saved measurements. Historical matrix-level ranking parity requires
the original feature arrays and query-level outputs; see
[Reproduction scope](#reproduction-scope).

## Controlled shared-component intervention

The historical chart uses seven Gaussian latent coordinates Z, 28 bounded odd
features, and 28 bounded even features. The bounded odd function combines two
tanh responses with a quadrature-computed coefficient cancelling its Gaussian
linear moment. A seven-dimensional axial root frame supplies directions; the
odd spectrum and even trace mass are specified before sampling.

For centered shared features $S$ and latent controls $Q$ (named `z` in
the code), first solve the least-squares problem. Its residual has no component
in the span of the controls:

$$
B=\operatorname*{argmin}_B\|S-QB\|_F^2,\qquad R=S-QB,\qquad Q^\top R=0.
$$

The residual's covariance generally differs from the declared target $D$.
Whitening and recoloring restore that target:

$$
C=\frac{R^\top R}{n-1},\qquad T=RC^{-1/2}D^{1/2}.
$$

For full-column-rank $Q$ and positive-definite $C$ and $D$,
$Q^\top T=0$ and $T^\top T/(n-1)=D$ in exact arithmetic. Multiplication on the
right preserves the residual's orthogonality to the controls.
$D$ is the declared population diagonal covariance. Covariance-only
recoloring enforces the covariance identity alone. The combined intervention
operates on an entire cohort and controls these two properties; its effects on
the remaining geometry are measured separately below.

The portable implementation rejects rank-deficient or ill-conditioned controls
and residual covariance with minimum eigenvalue at most 1e-12 times its maximum.
The historical cross-degree implementation checked positivity alone. The release
adds the relative conditioning guard and evaluates the stated transform directly,
with failure on ill-conditioned inputs. Gaussian moments are evaluated by
numerical quadrature.

The `construction` command regenerates all eight n=512 and eight n=768 shared
charts using their saved seeds. It compares covariance-only and cross-degree
interventions and checks historical energy/CKA measurements. This calculation
isolates the shared chart. Full-source RSA also involves private features,
radial modulation, cells, carriers and rank calibration; those outcomes are
analyzed from saved records.

The `records` command separately reaggregates all 192 saved cross-degree-intervention RSA rows. At
carrier .02 the transformed/raw SD ratios are about .626 and .571; at carrier
.15 they are about .703-.705 and .693-.695. The corresponding mean RSA decreases.
Each sample size has eight cohorts measured at three repair settings, giving
paired repeated observations. The ratios measure standard deviation; variance
ratios are their squares. Gate success is evaluated separately below.

The cell-removal study's 24 paired ablation rows show mean micro losses near .0072 and mean meso
changes near zero, with nonzero individual meso changes. This supports a
scale-selective contribution with residual cross-scale effects. All 120 deltas
are checked against the saved, seed-matched cell-on controls.

The prospectively tested cross-degree branch passed all-repair gates in 6/8, 7/8 and
8/8 cohorts at n=256, n=512 and n=768. The n=256 route was rejected. A subsequent
sample-size-indexed construction retained the earlier untransformed n=256 branch;
a fresh validation tested it on untouched cohorts: 7/8 complete successes, 455/520 threshold-check
rows. The resulting size-indexed construction uses distinct branches for the
small and larger cohorts.

## Executable ideal-marginal entropy bound

The calculation bounds the expectation over iid samples from an ideal marginal.
For centered iid feature vectors $x$ with covariance $C$, let $t=\operatorname{tr}C$,
$R_2\geq\|x\|^2$, and $M_4=\mathbb{E}[\|x\|^2xx^\top]$.
Let $A$ be unbiased sample covariance from $n$ rows. The finite-sample
calculation begins with two identities:

$$
\mathbb{E}[(A-C)^2]=\frac{M_4}{n}
-\frac{(n-2)C^2}{n(n-1)}+\frac{tC}{n(n-1)},
$$

$$
\operatorname{Var}(\operatorname{tr}A)
=\frac{\operatorname{tr}M_4-t^2}{n}
+\frac{2\operatorname{tr}(C^2)}{n(n-1)}.
$$

These determine an upper bound $V$ on trace variance and a correction
$\epsilon=R_2/n+t/[n(n-1)]$. Bound the matrix-log term by $U$:

$$
\mathbb{E}[\operatorname{tr}(A\log A)]
\leq\operatorname{tr}\!\left(C\log(C+\epsilon I)\right)\leq U.
$$

The entropy bound then accounts for fluctuations in the trace used to normalize
the eigenvalues:

$$
\mathbb{E}[H(A)]\geq
\log t-\frac{U}{t}-\frac{\log(d)\sqrt{V}}{2t}.
$$

Here $d$ is the feature dimension and $H(A)$ is the entropy of
normalized eigenvalues, zero-extended at zero scatter. The covariance identity
accounts for sample centering. The final term accounts for random trace
normalization in the expected sample entropy.

To obtain the logarithmic bound without assuming commuting matrices, apply
$\log x\leq x-1$ to eigenvalue ratios with eigenbasis-overlap weights:

$$
\operatorname{tr}(A\log A)-\operatorname{tr}(A\log B)
\leq\operatorname{tr}(A^2B^{-1})-\operatorname{tr}A,\qquad B=C+\epsilon I.
$$

The second-moment bound cancels the right side in expectation. Jensen for
$T\log T$ with $T=\operatorname{tr}A$, then Cauchy-Schwarz and
$\operatorname{Var}(H)\leq\log(d)^2/4$, yields the entropy bound.

The selected model's fixed ideal marginal has exact rational state masses, stored direction
constants, uniform marginal copies, independent isotropic marks and radii 1/2
or 3/2 with equal probability. Its varying state10 group/fine fraction is
$f=43/176+u$, with the entire $u\in[-1/4096,1/4096]$ interval retained. Only
terms of the form $\lambda\log(\lambda+\epsilon)$ vary with $f$; their sum is convex
in $f$. The maximum is therefore attained at an endpoint, yielding a bound
over the entire interval.

The implementation uses rational arithmetic, integer-square-root enclosures,
and a 64-term positive atanh log series with an explicit remainder. A 117-sample
enumeration checks the covariance and trace-variance identities exactly. It
charges the eigenweight cutoff entropy loss and a positive-variance-domain
probability debit of `2^-4095`. The resulting conservative ideal expected-rank
lower bounds are 5.23 and 24.1 at n4096 for the ideal model.

`qualification --all-models` evaluates all seven recorded marginal models around their own centers
using the same half-width. Only the selected model has the frozen interval claim; the others
are labeled diagnostic extensions.
Run without `-O`: inherited proof assertions are part of the checker.

## Saved interval and execution checks

The controlled source-pair qualification's 2,048 global/local leaves and 16 ridge cells cover the exact decimal closed
hull `[.015,.020]` with rational endpoint comparisons and no gaps or overlaps.
Every saved lower bound and all 96 ridge fold-cell bounds must meet their
thresholds. The calculation verifies consistency and coverage of the saved
enclosures.

For each captured anchor-compatibility center execution, let r bound the Euclidean discrepancy
between saved singular values and ideal singular-value boxes, t lower-bound
the ideal Frobenius norm, and E bound input error. The comparison is

$$
\beta=\frac{r+E}{t-E},\qquad t>E,\qquad \beta\leq\frac{1}{16384}.
$$

The checker verifies all 768 values, including the tail, in each of four saved
executions. The tightest saved beta is approximately 2.342e-9 versus the
6.104e-5 threshold. Each check establishes a postcondition for that captured
center execution. Whole-interval exact-functional certification is a separate
part of the anchor-compatibility qualification.

## Reproduction scope

The package supports three levels of reproduction:

- **Recompute:** the shared-component construction and ideal-marginal entropy
  bound run from the supplied definitions, parameters and seeds.
- **Reaggregate:** readout summaries, full-source agreement measurements and
  paired ablation deltas are reconstructed from saved measurements. Historical
  matrix-level ranking parity remains unverified: the original feature arrays
  for both studies and the discovery query-level outputs are outside the package.
- **Check saved bounds:** interval coverage, threshold comparisons and captured
  singular-value errors are verified against supplied enclosures. Deriving the
  interval enclosures requires the original bound-generation code. Full SVD
  execution reproduction also requires the original binary, execution traces
  and matrix-residual calculations. The separate whole-interval exact-functional
  certificates were checked during the forensic audit; their dependency stack
  remains in the research archive.

The ideal-marginal expectation bound assumes the stated sampling law. Transfer
to the literal sampler and materialized arrays remains an open step toward the
complete population-expectation certificate. The captured SVD checks apply to
the four saved center executions.

The feature-extraction pipeline, model weights and empirical feature arrays
remain external dependencies for reproducing the source studies end to end.

`PROVENANCE.json` identifies the source snapshot, selected record fields and
exported files through their digests. Tests check numerical results, protocol
parity on fixtures, and failure behavior. The source snapshot includes the
completed controlled source-pair qualification and its sequential
success-probability certificate, together with the adopted anchor-compatibility
qualification. Construction and useful operation of the real-source joint
representation remain research objectives.

The release's input guards and neighbor-selection fix apply to its exported
implementations; the original research records and regression-test references
are preserved. The release is licensed under [Apache 2.0](../LICENSE).
Third-party models, weights and datasets retain their respective terms.

## Analyze matched arrays

Rows must refer to the same stimuli in both arrays. This example measures
full-bank geometry; train-only preprocessing and held-out readout evaluation
are handled by the separate readout protocols.

```python
import numpy as np
from genesis_core.analysis import (
    cosine_distance, l2_rows, partial_rsa, spectral_metrics,
    neighborhood_overlap,
)

x = np.load("source.npy", allow_pickle=False)
y = np.load("target.npy", allow_pickle=False)
labels = np.load("categories.npy", allow_pickle=False)
dx = cosine_distance(l2_rows(x - x.mean(axis=0)))
dy = cosine_distance(l2_rows(y - y.mean(axis=0)))
print(partial_rsa(dx, dy, labels))
print(neighborhood_overlap(dx, dy, k=5))
print(spectral_metrics(x))
```

Partial RSA correlates distance ranks after removing the category-distance
effect. Neighborhood overlap measures how many nearby stimuli the two
representations share. Spectral metrics describe how feature energy is
distributed across directions.

## Historical identifier crosswalk

These identifiers occur in the unchanged source records, regression-test
originals and provenance metadata. The table maps those source labels to the
methods described above.

| Descriptive name used here | Historical identifier |
|---|---|
| Matched-64-image readout | MP002C |
| Discovery readout | MP002D |
| Seed-matched cell-on controls / cell-removal ablation | F92 / F94 |
| Cross-degree intervention | F151 |
| Prospective cross-degree validation | F157 |
| Size-indexed construction / fresh small-cohort validation | F159 / F160 |
| Controlled source-pair qualification | Full F1 |
| Sequential success-probability certificate | Certificate E |
| Population-expectation certificate | Certificate P |
| Anchor-compatibility qualification | A01 |
| Selected ideal marginal model | Law 11 |
| Alternative ideal marginal models | Laws 10, 12–16 |

N and M denote the two source marginals in the mathematical bounds. Raw data
fields and original source filenames retain their historical identifiers for
traceability.
