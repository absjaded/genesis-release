"""Exact rational bounds on expected entropy rank under the ideal marginal model."""
import argparse
from datetime import datetime, timezone
from fractions import Fraction as F
from functools import lru_cache
from itertools import product
import hashlib
import json
from math import isqrt
from pathlib import Path
import time
N, D = (4096, 768)
F0, HALF = (F(43, 176), F(1, 4096))

def down(x, bits=80):
    return F((x.numerator << bits) // x.denominator, 1 << bits)

def up(x, bits=80):
    return -down(-x, bits)

def sqrt_box(x, bits=96):
    """Enclose sqrt(x) by adjacent dyadic rationals using integer arithmetic.
    
    The squared endpoints are checked against x using exact arithmetic.
    """
    assert x >= 0
    q = isqrt((x.numerator << 2 * bits) // x.denominator)
    lo = F(q, 1 << bits)
    hi = lo if lo * lo == x else F(q + 1, 1 << bits)
    assert lo * lo <= x <= hi * hi
    return (lo, hi)

def mantissa_log(m):
    """Positive atanh series for 1<=m<=2, with a geometric remainder."""
    assert 1 <= m <= 2
    y = (m - 1) / (m + 1)
    square, power, total = (y * y, y, F(0))
    for j in range(64):
        total += 2 * power / (2 * j + 1)
        power *= square
    return (total, total + 2 * power / (129 * (1 - square)))
LOG2 = mantissa_log(F(2))

@lru_cache(maxsize=128)
def log_box(x):
    """Enclose log(x) using range reduction and an explicit series remainder.
    
    Write x = 2**k * m, 1 <= m < 2. Bound log(m) through the positive
    atanh series, then combine with k*log(2), reversing endpoints for k<0.
    Dyadic rounding is outward at each reported boundary.
    """
    x = F(x)
    assert x > 0
    k = x.numerator.bit_length() - x.denominator.bit_length()
    if x < F(2) ** k:
        k -= 1
    m = x / F(2) ** k
    assert 1 <= m < 2
    ml, mh = (down(m), up(m))
    lo, hi = (mantissa_log(ml)[0], mantissa_log(mh)[1])
    lo += k * LOG2[0 if k >= 0 else 1]
    hi += k * LOG2[1 if k >= 0 else 0]
    return (down(lo), up(hi))

def dot(x, y):
    return sum((a * b for a, b in zip(x, y)), F(0))

def outer(x):
    return [[a * b for b in x] for a in x]

def mm(a, b):
    return [[sum((a[i][k] * b[k][j] for k in range(len(b))), F(0)) for j in range(len(b[0]))] for i in range(len(a))]

def tr(a):
    return sum((a[i][i] for i in range(len(a))), F(0))

def packed(x):
    """Exact dyadic outward reporting box; decimal display is not the proof."""
    lo, hi = (down(x), up(x))
    return {'lower': [str(lo.numerator), str(lo.denominator)], 'upper': [str(hi.numerator), str(hi.denominator)], 'display': float(x)}

def moment_identity_controls():
    """Enumerate all samples of a finite fixture; no Monte Carlo samples."""
    points = [[F(-1), F(0)], [F(1), F(1)], [F(0), F(-2)]]
    weights = [F(1, 2), F(1, 3), F(1, 6)]
    mean = [sum((p * x[j] for p, x in zip(weights, points)), F(0)) for j in range(2)]
    z = [[x[j] - mean[j] for j in range(2)] for x in points]
    c = [[sum((p * x[i] * x[j] for p, x in zip(weights, z)), F(0)) for j in range(2)] for i in range(2)]
    fourth = [[sum((p * dot(x, x) * x[i] * x[j] for p, x in zip(weights, z)), F(0)) for j in range(2)] for i in range(2)]
    c2, t = (mm(c, c), tr(c))
    checked = 0
    for n in (2, 3, 4):
        measured = [[F(0) for j in range(2)] for i in range(2)]
        variance, mass = (F(0), F(0))
        for indices in product(range(3), repeat=n):
            w = F(1)
            for k in indices:
                w *= weights[k]
            xs = [z[k] for k in indices]
            avg = [sum((x[j] for x in xs), F(0)) / n for j in range(2)]
            sample = [[sum(((x[i] - avg[i]) * (x[j] - avg[j]) for x in xs), F(0)) / (n - 1) for j in range(2)] for i in range(2)]
            delta = [[sample[i][j] - c[i][j] for j in range(2)] for i in range(2)]
            square = mm(delta, delta)
            for i in range(2):
                for j in range(2):
                    measured[i][j] += w * square[i][j]
            variance += w * (tr(sample) - t) ** 2
            mass += w
            checked += 1
        expected = [[fourth[i][j] / n - F(n - 2, n * (n - 1)) * c2[i][j] + t * c[i][j] / (n * (n - 1)) for j in range(2)] for i in range(2)]
        assert mass == 1 and measured == expected
        assert variance == (tr(fourth) - t * t) / n + 2 * tr(c2) / (n * (n - 1))
    assert log_box(F(1)) == (0, 0)
    for x in (F(1, 8), F(1, 3), F(2), F(3), F(256)):
        lo, hi = log_box(x)
        inv_lo, inv_hi = log_box(1 / x)
        assert lo <= -inv_lo and -inv_hi <= hi
        assert hi - lo <= F(1, 1 << 74)
    assert sqrt_box(F(9, 4)) == (F(3, 2), F(3, 2))
    return {'enumerated_fixture_samples': checked, 'matrix_second_moment_and_trace_variance_identities_exact': True, 'log_reciprocity_width_and_square_root_controls': True}

def bound(parent, law, slot, center=F0):
    """Bound expected cutoff entropy rank for one ideal source marginal.
    
    slot selects N or M; sample size is N=4096 and feature dimension D=768.
    The calculation charges sample centering and random trace normalization.
    Convexity gives a whole-interval bound from center +/- HALF endpoints.
    The final margin includes cutoff loss
    and the zero-extension debit before comparison with 5.23 or 24.1.
    Outputs include rational enclosures and the positive comparison margin.
    Only the selected model's frozen interval has the adopted claim.
    """
    p = [F(k, parent['mass_denominator_exact']) for k in parent['mass_numerators']]
    assert sum(p) == 1
    u = [[F(x) for x in row] for row in parent['unit_directions'][slot]]
    gamma, eta = (F(law['gamma'][slot]), F(law['eta'][slot]))
    scale = F(law['feature_scale'][slot])
    width = len(u[0])
    assert width + sum((c - 1 for c in parent['copies'])) + 2 == D
    v = [sum((p[s] * u[s][i] for s in range(len(p))), F(0)) for i in range(width)]
    mean_square = (1 - gamma) * dot(v, v)
    macro = [[(1 - gamma) * (F(5, 4) * sum((p[s] * u[s][i] * u[s][j] for s in range(len(p))), F(0)) - v[i] * v[j]) for j in range(width)] for i in range(width)]
    a = tr(macro)
    b = sum((x * x for row in macro for x in row), F(0))
    t = a + F(5, 4) * (gamma + eta)
    row_energy = [(1 - gamma) * dot(x, x) + gamma + eta for x in u]
    radii = (F(1, 2), F(3, 2))
    centered_energy = [[r * r * row_energy[s] + mean_square - 2 * r * (1 - gamma) * dot(u[s], v) for s in range(len(p))] for r in radii]
    assert t == sum((p[s] * centered_energy[j][s] / 2 for j in range(2) for s in range(len(p))), F(0))
    r2 = max((x for row in centered_energy for x in row))
    m4 = sum((p[s] * centered_energy[j][s] ** 2 / 2 for j in range(2) for s in range(len(p))), F(0))
    variance = (m4 - t * t) / N + 2 * t * t / (N * (N - 1))
    epsilon = r2 / N + t / (N * (N - 1))
    assert a > 0 and b > 0 and (t > 0) and (variance > 0)
    endpoints = []
    for f in (center - HALF, center + HALF):
        terms = [(a, b / a + epsilon)]
        copy_trace = F(0)
        for s, count in enumerate(parent['copies']):
            blocks = [(count - 1, F(5, 4) * p[s] * gamma / (count - 1))] if s != 10 else [(8, F(5, 4) * p[s] * gamma * f / 8), (36, F(5, 4) * p[s] * gamma * (1 - f) / 36)]
            for multiplicity, eigenvalue in blocks:
                terms.append((multiplicity * eigenvalue, eigenvalue + epsilon))
                copy_trace += multiplicity * eigenvalue
        assert copy_trace == F(5, 4) * gamma
        mark = F(5, 8) * eta
        terms.append((2 * mark, mark + epsilon))
        upper = sum((weight * log_box(argument)[1] for weight, argument in terms), F(0))
        lower = log_box(t)[0] - upper / t - log_box(F(D))[1] * sqrt_box(variance)[1] / (2 * t)
        endpoints.append({'f': str(f), 'trace_log_upper': packed(upper), 'entropy_expectation_lower': packed(lower)})
    h_lower = min((F(*map(int, x['entropy_expectation_lower']['lower'])) for x in endpoints))
    cutoff = F(float(1e-15))
    cutoff_loss = D * cutoff * log_box(1 / cutoff)[1]
    gap = scale * (F(3, 2) * sqrt_box(min(row_energy))[0] - F(1, 2) * sqrt_box(max(row_energy))[1])
    assert gap > F(1, 4)
    scatter_floor = F(N - 1, N) * gap * gap
    assert scatter_floor > F(float(1e-20))
    omitted_probability = F(1, 1 << N - 1)
    target = F(523, 100) if slot == 'N' else F(241, 10)
    margin = h_lower - cutoff_loss - log_box(target + D * omitted_probability)[1]
    assert margin > 0
    return {'slot': slot, 'dimension': D, 'n': N, 'trace': packed(t), 'macro_trace': packed(a), 'macro_trace_square': packed(b), 'centered_row_energy_ceiling': packed(r2), 'central_fourth_norm_moment': packed(m4), 'trace_variance_upper': packed(variance), 'epsilon': packed(epsilon), 'endpoints': endpoints, 'uniform_entropy_expectation_lower': packed(h_lower), 'cutoff_entropy_loss_upper': packed(cutoff_loss), 'mixed_radius_scatter_floor': packed(scatter_floor), 'invalid_variance_domain_probability_upper': '2^-4095', 'zero_extended_ideal_reference_rank_expectation_lower': str(target), 'log_comparison_positive_margin': packed(margin), 'literal_P_entropy_expectation_certified': False}
if not __debug__:
    raise RuntimeError('qualification checks require assertions; do not run Python -O')
