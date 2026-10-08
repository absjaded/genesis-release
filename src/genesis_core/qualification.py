"""Saved interval coverage and exact a-posteriori singular-value arithmetic.

The captured-output comparison is conditional on the supplied saved enclosures.
See docs/TECHNICAL_NOTE.md for the dependencies of full execution reproduction.
"""
from fractions import Fraction as F
from .data import load
from .intervals import inspect_leaves, inspect_ridge


def rational(record):
    if set(record) != {'numerator', 'denominator'} or int(record['denominator']) <= 0:
        raise ValueError('invalid rational record')
    return F(int(record['numerator']), int(record['denominator']))


def check_capture(record):
    """Check beta = (r + E)/(t - E) <= 1/16384 using exact saved arithmetic.

    r encloses the spectral discrepancy, t lower-bounds the ideal Frobenius
    norm, and E bounds input error. All 768 singular values, including the tail,
    contribute. The supplied enclosures are inputs to the check.
    """
    saved = record['saved']
    values = [F(float.fromhex(v)) for v in record['values_hex']]
    if len(values) != 768 or len(record['boxes']) != 768 or values != sorted(values, reverse=True) or min(values) < 0:
        raise ValueError('invalid saved singular values')
    square, norm2 = F(0), F(0)
    for index, (value, box) in enumerate(zip(values, record['boxes']), 1):
        lo, hi = rational(box['center_root_lower']), rational(box['center_root_upper'])
        eigen_lo, eigen_hi = rational(box['center_lambda_lower']), rational(box['center_lambda_upper'])
        if box['index'] != index or not 0 <= lo <= hi or not lo*lo <= eigen_lo <= eigen_hi <= hi*hi:
            raise ValueError('invalid spectral enclosure')
        if index > saved['rank'] and (lo != 0 or hi != 0):
            raise ValueError('nonzero certified tail')
        square += (abs(value-(lo+hi)/2) + (hi-lo)/2)**2
        norm2 += lo*lo
    r, t, error, beta, target = [rational(saved[k]) for k in ('comparison_root_upper', 'ideal_norm_root_lower', 'input_error', 'relative_upper', 'target')]
    if (r < 0 or t <= error or error < 0 or r*r < square or t*t > norm2
            or square != rational(saved['ideal_comparison_squared'])
            or norm2 != rational(saved['ideal_norm_squared_lower'])
            or beta*(t-error) != r+error or target != F(1,16384)
            or not 0 < beta <= target or saved['values_checked'] != 768
            or rational(saved['execution_error_upper']) != r+error
            or rational(saved['literal_norm_lower']) != t-error
            or saved['below_target'] is not True):
        raise ValueError('captured-output accuracy comparison failed')
    return {'id': saved['id'], 'relative_error_upper': float(beta), 'target': float(target),
            'target_to_bound_ratio': float(target/beta), 'values_checked': 768}


def saved_bounds():
    data = load('qualification')
    captures = [check_capture(r) for r in data['captures']]
    if len(captures) != 4 or len({r['id'] for r in captures}) != 4:
        raise ValueError('four distinct captured executions required')
    return {'scope': 'SAVED_BOUND_ARITHMETIC_NOT_FULL_CERTIFICATION_REEXECUTION',
            'construction_interval': inspect_leaves(data['f1_leaves']), 'ridge': inspect_ridge(data['ridge_cells']),
            'captured_svd': captures, 'execution_provenance_replayed': False,
            'whole_interval_SVD_solver_certified': False}
