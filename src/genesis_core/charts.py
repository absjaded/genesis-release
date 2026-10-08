"""Bounded shared-component chart: 7 latent, 28 bounded odd, 28 bounded even axes.

Adapted from the original shared-chart construction functions. Population moments below
are evaluated by numerical quadrature.
Private features, radial factors, cells, carriers and rank calibration omitted.
"""
from __future__ import annotations

from itertools import combinations
import math
import numpy as np
from numpy.polynomial.hermite import hermgauss


def gaussian_quadrature(order):
    if type(order) is not int or not 8 <= order <= 128:
        raise ValueError('quadrature order must be an integer from 8 through 128')
    nodes, weights = hermgauss(order)
    return np.sqrt(2.0) * nodes, weights / np.sqrt(np.pi)


def axial_directions(gain):
    if not np.isfinite(gain) or gain <= 0:
        raise ValueError('positive finite gain required')
    helmert = np.zeros((8, 7))
    for column in range(7):
        k = column + 1
        helmert[:k, column] = 1 / math.sqrt(k * (k + 1))
        helmert[k, column] = -k / math.sqrt(k * (k + 1))
    roots = np.array([(helmert[i] - helmert[j]) / math.sqrt(2)
                      for i, j in combinations(range(8), 2)])
    generic = np.array([1., 2., 3., 5., 7., 11., 13.])
    generic /= np.linalg.norm(generic)
    first = np.zeros(7)
    first[0] = 1
    vector = generic - first
    householder = np.eye(7) - 2 * np.outer(vector, vector) / (vector @ vector)
    directions = roots @ householder.T
    directions[:, 0] *= gain
    return directions / np.linalg.norm(directions, axis=1)[:, None]


def bounded_frame(beta_low, beta_high, gain, order):
    """Construct bounded odd features with a cancelled Gaussian linear moment.
    
    g(t) = tanh(beta_low*t) - c*tanh(beta_high*t), where c is the ratio
    E[t*tanh(beta_low*t)] / E[t*tanh(beta_high*t)] for standard Gaussian t.
    Moments are evaluated by quadrature. The returned whitening map
    normalizes the resulting 28-direction feature covariance.
    """
    if not 0 < beta_low < beta_high or not np.isfinite(beta_high):
        raise ValueError('ordered positive finite beta pair required')
    z, probability = gaussian_quadrature(order)
    low = np.sum(probability * z * np.tanh(beta_low * z))
    high = np.sum(probability * z * np.tanh(beta_high * z))
    coefficient = float(low / high)

    def bounded(values):
        return np.tanh(beta_low * values) - coefficient * np.tanh(beta_high * values)

    directions = axial_directions(gain)
    g = bounded(z)
    variance = float(np.sum(probability * g * g))
    correlations = directions @ directions.T
    covariance = np.empty_like(correlations)
    cache = {1.: variance, -1.: -variance}
    x = z[:, None]
    weights = probability[:, None] * probability[None, :]
    for index in np.ndindex(correlations.shape):
        rho = float(correlations[index])
        key = round(rho, 12)
        if key not in cache:
            y = rho * x + math.sqrt(max(0., 1 - rho * rho)) * z[None, :]
            cache[key] = float(np.sum(weights * g[:, None] * bounded(y)))
        covariance[index] = cache[key]
    values, vectors = np.linalg.eigh((covariance + covariance.T) / 2)
    indices = np.argsort(values)[::-1]
    values, vectors = values[indices], vectors[:, indices]
    if values[-1] <= max(1e-14, 1e-11 * values[0]):
        raise ValueError('rank deficient bounded frame')
    return directions, coefficient, values, vectors / np.sqrt(values)[None, :]


def shared_chart(n, seed, *, beta_low, beta_high, axial_gain,
                 spectral_temperature_M, parity, order, dominant_mass_N):
    """Generate the shared blocks from the historical Gaussian latent substream.
    
    Returns z: (n,7), shared_n: (n,7), odd/even blocks: (n,28), shared:
    (n,56), and target: (56,). The latter specifies diagonal covariance D;
    it is specified by the construction before sampling the cohort.
    Finite-sample leakage can remain despite population moment cancellation.
    The generator omits the private, radial, cell and carrier components.
    """
    if type(n) is not int or not 65 <= n <= 4096:
        raise ValueError('bounded example supports integer n from 65 through 4096')
    if not 0 < parity < 1 or not 1 / 7 <= dominant_mass_N < 1:
        raise ValueError('invalid parity or N spectrum')
    if not np.isfinite(spectral_temperature_M) or spectral_temperature_M < 0:
        raise ValueError('nonnegative finite temperature required')
    # Exactly the historical latent substream; omitted streams are not consumed.
    latent_stream = np.random.SeedSequence(seed).spawn(5)[0]
    z = np.random.default_rng(latent_stream).standard_normal((n, 7))
    directions, coefficient, eigenvalues, whitening = bounded_frame(
        beta_low, beta_high, axial_gain, order)
    projection = z @ directions.T
    odd = np.tanh(beta_low * projection) - coefficient * np.tanh(beta_high * projection)
    relative = np.round(eigenvalues / np.max(eigenvalues), 12)
    spectrum = np.ones_like(relative) if spectral_temperature_M == 0 else relative ** spectral_temperature_M
    spectrum /= spectrum.sum()
    odd = (odd @ whitening) * np.sqrt(spectrum)[None, :]
    nodes, probability = gaussian_quadrature(order)
    values = np.tanh(0.25 * nodes)
    second = float(np.sum(probability * values ** 2))
    diagonal_variance = float(np.sum(probability * values ** 4)) - second ** 2
    u = np.tanh(0.25 * z)
    columns = [(u[:, i] ** 2 - second) / math.sqrt(diagonal_variance) for i in range(7)]
    columns += [(u[:, i] * u[:, j]) / second for i, j in combinations(range(7), 2)]
    even = np.column_stack(columns)
    shared = np.concatenate((np.sqrt(1 - parity) * odd, np.sqrt(parity / 28) * even), axis=1)
    target = np.concatenate(((1 - parity) * spectrum, np.full(28, parity / 28)))
    n_spectrum = np.array([dominant_mass_N] + [(1 - dominant_mass_N) / 6] * 6)
    return {'z': z, 'shared_n': z * np.sqrt(n_spectrum)[None, :],
            'shared_odd': odd, 'shared_even': even, 'shared': shared, 'target': target}
