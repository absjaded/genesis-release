"""Finite-sample cross-degree orthogonalization and covariance restoration.

The population diagonal covariance is specified before sampling.
See docs/TECHNICAL_NOTE.md for identities, numerical guards and scope.
"""
from __future__ import annotations

import numpy as np


def matrix(value):
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError('real matrices required')
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or min(array.shape) < 1 or array.shape[0] < 2:
        raise ValueError('a nonempty matrix with at least two rows is required')
    if not np.all(np.isfinite(array)):
        raise ValueError('nonfinite input')
    return array


def center(value):
    array = matrix(value)
    return array - array.mean(axis=0, keepdims=True)


def _whiten_recolor(residual, target):
    """Map centered residual R to T = R @ C^(-1/2) @ D^(1/2).
    
    C = R.T @ R / (n-1), D = diag(target). The symmetric inverse square
    root is formed spectrally. Rejecting an ill-conditioned C preserves
    the stated transform; eigenvalue clipping would define another method.
    """
    raw = np.asarray(target)
    if np.iscomplexobj(raw):
        raise ValueError('real target spectrum required')
    target = np.asarray(target, dtype=np.float64)
    if (target.shape != (residual.shape[1],) or not np.all(np.isfinite(target))
            or np.any(target <= 0)):
        raise ValueError('one finite positive target variance per column is required')
    covariance = residual.T @ residual / (len(residual) - 1)
    covariance = (covariance + covariance.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    # A deliberate packaging guard: never silently clip or regularize a singular
    # covariance and then claim the unregularized algebra still holds.
    if eigenvalues[0] <= max(0.0, 1e-12 * eigenvalues[-1]):
        raise ValueError('residual covariance is singular or too ill-conditioned')
    inverse_root = (eigenvectors * eigenvalues[None, :] ** -0.5) @ eigenvectors.T
    transformed = residual @ inverse_root @ np.diag(np.sqrt(target))
    if not np.all(np.isfinite(transformed)):
        raise ValueError('nonfinite transformed output')
    return transformed


def recolor(shared, target):
    """Marginal-only comparison: restore covariance without removing coupling."""
    return _whiten_recolor(center(shared), target)


def orthogonal_recolor(controls, shared, target):
    """Enforce control orthogonality and declared covariance on one cohort.
    
    controls Q: (n,k); shared S: (n,d); target diagonal: (d,).
    Center both inputs, solve B = argmin ||S-QB||_F^2, and form R = S-QB.
    Right multiplication by C^(-1/2) D^(1/2) restores covariance while
    preserving Q.T @ R = 0. Thus Q.T @ T = 0 and T.T @ T/(n-1) = D
    in exact arithmetic, given full-rank controls and positive residual
    covariance. The implementation uses z for Q and returns energy receipts.
    Q contains known latent constructor coordinates. The transform is fitted
    to the supplied cohort; applying it to unseen rows requires a separate API.
    """
    z, s = center(controls), center(shared)
    if len(z) != len(s):
        raise ValueError('row counts differ')
    if len(z) - 1 - z.shape[1] < s.shape[1]:
        raise ValueError('insufficient residual dimension for positive covariance')
    coefficients, _, rank, singular = np.linalg.lstsq(z, s, rcond=None)
    if rank != z.shape[1] or singular[-1] <= 1e-12 * singular[0]:
        raise ValueError('controls lack a numerically full-rank coordinate system')
    projection = z @ coefficients
    residual = s - projection
    transformed = _whiten_recolor(residual, target)
    total = float(np.sum(s * s))
    return transformed, {
        'projection_energy_fraction': float(np.sum(projection * projection)) / total,
        'residual_energy_fraction': float(np.sum(residual * residual)) / total,
        'latent_rank': int(rank),
    }


def linear_cka(left, right):
    x, y = center(left), center(right)
    if len(x) != len(y):
        raise ValueError('row counts differ')
    cross = x.T @ y
    denominator = np.linalg.norm(x.T @ x) * np.linalg.norm(y.T @ y)
    if not denominator > 0:
        raise ValueError('degenerate CKA denominator')
    return float(np.sum(cross * cross) / denominator)


def diagnostics(controls, shared, target):
    """Check the two algebraic targets on a realized feature block.
    
    cross_moment_max_abs = max(abs(Q.T @ S / (n-1))).
    covariance_max_abs_error = max(abs(S.T @ S / (n-1) - D)).
    Both diagnostics report maximum absolute entrywise errors.
    """
    z, s = center(controls), center(shared)
    if len(z) != len(s):
        raise ValueError('row counts differ')
    return {
        'cross_moment_max_abs': float(np.max(np.abs(z.T @ s / (len(s) - 1)))),
        'covariance_max_abs_error': float(np.max(np.abs(s.T @ s / (len(s) - 1) - np.diag(target)))),
        'centered_linear_CKA': linear_cka(z, s),
    }
