"""Array-based cross-representation estimators, adapted from historical Genesis code.
Release guards reject invalid/degenerate inputs; neighbor selection explicitly excludes self."""
from __future__ import annotations
import itertools
from dataclasses import dataclass
from typing import Any, Iterable
import numpy as np
from numpy.typing import NDArray
from scipy.stats import rankdata, spearmanr
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import StratifiedKFold
Array = NDArray[np.float64]
Index = NDArray[np.int64]

@dataclass(frozen=True)
class PreparedFold:
    train: Index
    test: Index
    n_train: Array
    n_test: Array
    m_train: Array
    m_test: Array

def require_matrix(value, name):
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError(f'{name} must be real')
    result = np.asarray(raw, dtype=np.float64)
    if result.ndim != 2 or min(result.shape) < 2 or (not np.isfinite(result).all()):
        raise ValueError(f'{name} must be a finite, nontrivial 2-D matrix')
    return result

def require_categories(categories, n_samples):
    raw = np.asarray(categories)
    if raw.shape != (n_samples,) or raw.dtype.kind not in 'iu' or raw.dtype.kind == 'b':
        raise ValueError('categories must be a one-dimensional integer array')
    if raw.dtype.kind == 'u' and np.any(raw > np.iinfo(np.int64).max):
        raise ValueError('category label exceeds int64')
    result = raw.astype(np.int64)
    if len(np.unique(result)) < 2:
        raise ValueError('at least two categories are required')
    return result

def l2_rows(value: Array) -> Array:
    data = require_matrix(value, 'value')
    return data / np.maximum(np.linalg.norm(data, axis=1, keepdims=True), 1e-12)

def fit_center_l2(train: Array, test: Array) -> tuple[Array, Array]:
    train = require_matrix(train, 'train')
    test = require_matrix(test, 'test')
    if train.shape[1] != test.shape[1]:
        raise ValueError('train/test feature widths differ')
    mean = train.mean(axis=0, keepdims=True)
    return (l2_rows(train - mean), l2_rows(test - mean))

def fit_standardize(train: Array, test: Array) -> tuple[Array, Array, NDArray[np.bool_]]:
    train = require_matrix(train, 'train')
    test = require_matrix(test, 'test')
    if train.shape[1] != test.shape[1]:
        raise ValueError('train/test feature widths differ')
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, ddof=1, keepdims=True)
    valid = std[0] > 1e-08
    if np.count_nonzero(valid) < 2:
        raise ValueError('fewer than two nonconstant training predictors')
    return ((train[:, valid] - mean[:, valid]) / std[:, valid], (test[:, valid] - mean[:, valid]) / std[:, valid], valid)

def cosine_distance(normalized: Array) -> Array:
    data = require_matrix(normalized, 'normalized')
    result = 1.0 - np.clip(data @ data.T, -1.0, 1.0)
    np.fill_diagonal(result, 0.0)
    return result

def upper(square: Array) -> Array:
    square = np.asarray(square, dtype=np.float64)
    if square.ndim != 2 or square.shape[0] != square.shape[1]:
        raise ValueError('expected a square matrix')
    indices = np.triu_indices(square.shape[0], 1)
    return square[indices]

def residualized_rank_vector(distance, categories):
    distance = require_matrix(distance, 'distance')
    if distance.shape[0] != distance.shape[1]:
        raise ValueError('square distance matrix required')
    categories = require_categories(categories, len(distance))
    values = rankdata(upper(distance), method='average').astype(np.float64)
    control = rankdata(upper((categories[:, None] != categories[None, :]).astype(float)), method='average').astype(float)
    if values.std(ddof=1) <= 1e-12:
        raise ValueError('constant distance ranks')
    values = (values - values.mean()) / values.std(ddof=1)
    control = (control - control.mean()) / max(float(control.std(ddof=1)), 1e-12)
    values -= values @ control * control / max(float(control @ control), 1e-12)
    residual_std = float(values.std(ddof=1))
    if residual_std <= 1e-12:
        raise ValueError('category explains all distance rank variation')
    return (values - values.mean()) / residual_std

def partial_rsa(left_distance, right_distance, categories):
    """Correlate distance ranks after projecting out category distance.
    
    Both inputs are n-by-n distance matrices in identical stimulus order.
    Residualization includes an intercept; constant residuals are rejected.
    The result measures conditional geometric agreement.
    """
    left = residualized_rank_vector(left_distance, categories)
    right = residualized_rank_vector(right_distance, categories)
    if np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        raise ValueError('partial RSA is undefined for a constant residual rank vector')
    return finite_result(float(np.corrcoef(left, right)[0, 1]))

def neighborhood_overlap(left_distance, right_distance, k):
    """Mean rowwise Jaccard overlap of two k-neighbor sets.
    
    Self is excluded explicitly, even when duplicate rows tie at distance
    zero. Remaining ties are ordered by stimulus index. Both n-by-n distance
    matrices must refer to the same stimuli.
    """
    left = require_matrix(left_distance, 'left distance').copy()
    right = require_matrix(right_distance, 'right distance').copy()
    if left.shape != right.shape or left.shape[0] != left.shape[1]:
        raise ValueError('matched square distance matrices required')
    if type(k) is not int or not 1 <= k < len(left):
        raise ValueError('integer k must lie between one and n-1')
    np.fill_diagonal(left, np.inf)
    np.fill_diagonal(right, np.inf)
    a = np.argsort(left, axis=1, kind='stable')[:, :k]
    b = np.argsort(right, axis=1, kind='stable')[:, :k]
    return float(np.mean([len(set(x) & set(y)) / len(set(x) | set(y)) for x, y in zip(a, b, strict=True)]))

def spectral_metrics(value: Array) -> dict[str, float]:
    """Measure the spectrum of the centered feature matrix.
    
    With squared singular values lambda_i, p_i = lambda_i / sum(lambda).
    Entropy rank is exp(-sum(p_i * log(p_i))), using the recorded 1e-15
    cutoff without renormalizing retained weights. Participation and stable
    rank use the same spectrum but different concentration functionals.
    """
    data = require_matrix(value, 'value')
    centered = data - data.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False, full_matrices=False)
    eigen = singular ** 2
    total = float(eigen.sum())
    if total <= 1e-20:
        raise ValueError('zero total variance')
    probability = eigen / total
    nonzero = probability[probability > 1e-15]
    raw_normalized = l2_rows(data)
    pair_cosine = upper(raw_normalized @ raw_normalized.T)
    return {'effective_rank': float(np.exp(-np.sum(nonzero * np.log(nonzero)))), 'participation_ratio': float(total ** 2 / np.sum(eigen ** 2)), 'stable_rank': float(total / eigen.max()), 'anisotropy_mean_pair_cosine': float(pair_cosine.mean())}

def stratified_splits(categories: Index, n_splits: int, seed: int) -> list[tuple[Index, Index]]:
    categories = require_categories(categories, len(categories))
    minimum = min((int(np.count_nonzero(categories == value)) for value in np.unique(categories)))
    if n_splits > minimum:
        raise ValueError(f'n_splits={n_splits} exceeds smallest category count={minimum}')
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(seed % (2 ** 32 - 1)))
    return [(train.astype(np.int64), test.astype(np.int64)) for train, test in splitter.split(np.arange(len(categories)), categories)]

def prepare_fold(n_view: Array, m_view: Array, train: Index, test: Index) -> PreparedFold:
    n_train, n_test = fit_center_l2(n_view[train], n_view[test])
    m_train, m_test = fit_center_l2(m_view[train], m_view[test])
    return PreparedFold(train, test, n_train, n_test, m_train, m_test)

def permute_indices(rng: np.random.Generator, categories: Index, mode: str) -> Index:
    if mode == 'unrestricted':
        return rng.permutation(len(categories)).astype(np.int64)
    if mode != 'category_preserving':
        raise ValueError(mode)
    result = np.arange(len(categories), dtype=np.int64)
    for category in np.unique(categories):
        indices = np.flatnonzero(categories == category)
        result[indices] = rng.permutation(indices)
    return result

def fwer_p(observed, null):
    """Historical Monte Carlo upper-tail p; the name alone implies no correction."""
    values = np.asarray(list(null), dtype=np.float64)
    if not np.isfinite(observed) or values.ndim != 1 or (not len(values)) or (not np.isfinite(values).all()):
        raise ValueError('finite observed statistic and nonempty finite null required')
    return float((1 + np.count_nonzero(values >= observed)) / (len(values) + 1))

def crossfit_center_l2(value: Array, splits: list[tuple[Index, Index]]) -> Array:
    value = require_matrix(value, 'value')
    splits = validate_splits(splits, len(value))
    data = require_matrix(value, 'value')
    output = np.empty_like(data, dtype=np.float64)
    seen = np.zeros(len(data), dtype=np.int64)
    for train, test in splits:
        _, transformed = fit_center_l2(data[train], data[test])
        output[test] = transformed
        seen[test] += 1
    if not np.array_equal(seen, np.ones(len(data), dtype=np.int64)):
        raise ValueError('outer splits must cover every row exactly once')
    return output

def raw_geometry_summary(n_view: Array, m_view: Array, categories: Index, *, outer_splits: list[tuple[Index, Index]], primary_k: int, sensitivity_k: Iterable[int], permutations: int, seed: int) -> dict[str, Any]:
    n_view = require_matrix(n_view, 'n_view')
    m_view = require_matrix(m_view, 'm_view')
    if n_view.shape[0] != m_view.shape[0]:
        raise ValueError('source sample counts differ')
    categories = require_categories(categories, n_view.shape[0])
    if primary_k >= len(categories):
        raise ValueError('primary k exceeds cohort')
    n_crossfit = crossfit_center_l2(n_view, outer_splits)
    m_crossfit = crossfit_center_l2(m_view, outer_splits)
    n_distance = cosine_distance(n_crossfit)
    m_distance = cosine_distance(m_crossfit)
    observed_partial = partial_rsa(n_distance, m_distance, categories)
    observed_neighbor = neighborhood_overlap(n_distance, m_distance, primary_k)
    k_values = sorted(set([int(primary_k), *(int(value) for value in sensitivity_k)]))
    sensitivity = {str(k): neighborhood_overlap(n_distance, m_distance, k) for k in k_values if k < len(categories)}
    fold_partial = []
    for _, test in outer_splits:
        if len(test) >= 6 and len(np.unique(categories[test])) >= 2:
            fold_partial.append(partial_rsa(cosine_distance(n_crossfit[test]), cosine_distance(m_crossfit[test]), categories[test]))
    positive_fraction = float(np.mean(np.asarray(fold_partial) > 0)) if fold_partial else float('nan')
    rng = np.random.default_rng(seed)
    nulls: dict[str, dict[str, list[float]]] = {'unrestricted': {'partial': [], 'neighbor': []}, 'category_preserving': {'partial': [], 'neighbor': []}}
    for mode in nulls:
        for _ in range(permutations):
            permutation = permute_indices(rng, categories, mode)
            permuted = m_distance[permutation][:, permutation]
            nulls[mode]['partial'].append(partial_rsa(n_distance, permuted, categories))
            nulls[mode]['neighbor'].append(neighborhood_overlap(n_distance, permuted, primary_k))
    category_partial = np.asarray(nulls['category_preserving']['partial'], dtype=np.float64)
    category_neighbor = np.asarray(nulls['category_preserving']['neighbor'], dtype=np.float64)
    result: dict[str, Any] = {'partial_rsa': observed_partial, 'neighborhood_overlap': observed_neighbor, 'neighborhood_delta_vs_category_null': observed_neighbor - float(np.median(category_neighbor)), 'positive_fold_fraction': positive_fraction, 'unrestricted_partial_p': fwer_p(observed_partial, nulls['unrestricted']['partial']), 'category_partial_p': fwer_p(observed_partial, category_partial), 'category_neighborhood_p': fwer_p(observed_neighbor, category_neighbor), 'neighborhood_sensitivity': sensitivity, 'null_summary': {}}
    for mode, metrics in nulls.items():
        result['null_summary'][mode] = {}
        for metric, values in metrics.items():
            array = np.asarray(values, dtype=np.float64)
            result['null_summary'][mode][metric] = {'median': float(np.median(array)), 'p95': float(np.percentile(array, 95)), 'maximum': float(np.max(array))}
    return result

def bidirectional_reconstruction(n_view: Array, m_view: Array, outer_splits: list[tuple[Index, Index]], alpha: float=1e-06) -> dict[str, float]:
    n_view, m_view, _ = validate_pair(n_view, m_view)
    outer_splits = validate_splits(outer_splits, len(n_view))
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError('positive finite alpha required')
    for _, test in outer_splits:
        if any((float(np.var(v[test], axis=0).sum()) <= 1e-20 for v in (n_view, m_view))):
            raise ValueError('undefined reconstruction R2 for zero-variance targets')
    directional: dict[str, list[float]] = {'n_to_m': [], 'm_to_n': []}
    for train, test in outer_splits:
        for name, source, target in (('n_to_m', n_view, m_view), ('m_to_n', m_view, n_view)):
            model = Ridge(alpha=alpha, fit_intercept=True)
            model.fit(source[train], target[train])
            predicted = model.predict(source[test])
            directional[name].append(float(r2_score(target[test], predicted, multioutput='variance_weighted')))
    n_to_m = float(np.median(directional['n_to_m']))
    m_to_n = float(np.median(directional['m_to_n']))
    return {'n_to_m_r2': n_to_m, 'm_to_n_r2': m_to_n, 'bidirectional_min_r2': min(n_to_m, m_to_n), 'residual_nonredundancy': 1.0 - min(n_to_m, m_to_n)}

def normalize_target_gallery(train_target: Array, gallery_target: Array) -> tuple[Array, Array]:
    train_target = require_matrix(train_target, 'train_target')
    gallery_target = require_matrix(gallery_target, 'gallery_target')
    mean = train_target.mean(axis=0, keepdims=True)
    return (l2_rows(train_target - mean), l2_rows(gallery_target - mean))

def retrieval_metrics(prediction: Array, target_gallery: Array, query_indices: Index, correct_indices: Index, categories: Index, *, candidate_count: int, seed: int, gallery_indices: Index | None=None) -> dict[str, float]:
    prediction, target_gallery, query_indices, correct_indices, categories, gallery_indices = validate_retrieval(prediction, target_gallery, query_indices, correct_indices, categories, candidate_count, gallery_indices)
    prediction = l2_rows(prediction)
    target_gallery = l2_rows(target_gallery)
    query_indices = np.asarray(query_indices, dtype=np.int64)
    correct_indices = np.asarray(correct_indices, dtype=np.int64)
    categories = require_categories(categories, len(target_gallery))
    if prediction.shape[0] != len(query_indices) or len(query_indices) != len(correct_indices):
        raise ValueError('query, prediction, and correct-index counts differ')
    if gallery_indices is None:
        gallery_indices = np.arange(len(categories), dtype=np.int64)
    else:
        gallery_indices = np.asarray(gallery_indices, dtype=np.int64)
    ranks, pairwise, correct_cosine = ([], [], [])
    for row, (query, correct) in enumerate(zip(query_indices, correct_indices, strict=True)):
        if categories[query] != categories[correct]:
            raise ValueError('correct target must preserve query category')
        pool = gallery_indices[(categories[gallery_indices] == categories[query]) & (gallery_indices != correct)]
        if len(pool) < candidate_count - 1:
            raise ValueError(f'category {categories[query]} has only {len(pool)} distractors; need {candidate_count - 1}')
        query_seed = int((seed + 11400714819323198485 * (int(query) + 1)) % 2 ** 64)
        rng = np.random.default_rng(query_seed)
        distractors = rng.choice(pool, size=candidate_count - 1, replace=False)
        candidates = np.concatenate([[correct], np.sort(distractors)]).astype(np.int64)
        similarities = prediction[row] @ target_gallery[candidates].T
        correct_value = float(similarities[0])
        rank = 1 + int(np.count_nonzero(similarities[1:] >= correct_value))
        ranks.append(rank)
        pairwise.extend((correct_value > similarities[1:]).astype(np.float64).tolist())
        correct_cosine.append(correct_value)
    paired_targets = target_gallery[correct_indices]
    pred_distance = cosine_distance(prediction)
    target_distance = cosine_distance(paired_targets)
    rdm_spearman = float(spearmanr(upper(pred_distance), upper(target_distance)).statistic)
    category_partial = partial_rsa(pred_distance, target_distance, categories[query_indices])
    rank_array = np.asarray(ranks, dtype=np.int64)
    return finite_result({'top1': float(np.mean(rank_array <= 1)), 'top5': float(np.mean(rank_array <= min(5, candidate_count))), 'median_rank': float(np.median(rank_array)), 'paired_cosine': float(np.mean(correct_cosine)), 'pairwise_identification': float(np.mean(pairwise)), 'predicted_target_rdm_spearman': rdm_spearman, 'category_partial_rdm': category_partial})

def ridge_prediction(source: Array, target: Array, train: Index, query: Index, alpha: float) -> tuple[Array, Array]:
    """Predict query targets with a training-only scaled linear kernel.
    
    X has shape (n_train, p) after training standardization and filtering.
    K = X @ X.T / p; predictions = (X_query @ X.T / p) @ solve(K+alpha*I,Y).
    Y and the returned gallery share target centering fitted on training
    rows. The caller selects alpha using only the outer training partition.
    """
    source, target, train, query = validate_prediction_inputs(source, target, train, query, alpha)
    train_x, query_x, _ = fit_standardize(source[train], source[query])
    train_y, gallery_y = normalize_target_gallery(target[train], target)
    scale = float(train_x.shape[1])
    kernel = train_x @ train_x.T / scale
    cross = query_x @ train_x.T / scale
    coefficient = np.linalg.solve(kernel + float(alpha) * np.eye(len(train)), train_y)
    return (cross @ coefficient, gallery_y)

def pls_prediction(source: Array, target: Array, train: Index, query: Index, components: int) -> tuple[Array, Array]:
    if type(components) is not int:
        raise ValueError('integer components required')
    source, target, train, query = validate_prediction_inputs(source, target, train, query, components)
    train_x, query_x, _ = fit_standardize(source[train], source[query])
    train_y, gallery_y = normalize_target_gallery(target[train], target)
    maximum = min(train_x.shape[0] - 1, train_x.shape[1], train_y.shape[1])
    if components > maximum:
        raise ValueError(f'PLS components {components} exceed maximum {maximum}')
    model = PLSRegression(n_components=components, scale=False, max_iter=5000, tol=1e-06)
    model.fit(train_x, train_y)
    return (np.asarray(model.predict(query_x), dtype=np.float64), gallery_y)

def choose_ridge_alpha(source: Array, target: Array, categories: Index, outer_train: Index, *, alphas: Iterable[float], candidate_count: int, seed: int) -> dict[str, float]:
    inner_categories = categories[outer_train]
    inner_splits = stratified_splits(inner_categories, 3, seed)
    rows = []
    for alpha in alphas:
        metrics = []
        for fold, (local_train, local_valid) in enumerate(inner_splits):
            train = outer_train[local_train]
            valid = outer_train[local_valid]
            prediction, gallery = ridge_prediction(source, target, train, valid, float(alpha))
            metrics.append(retrieval_metrics(prediction, gallery, valid, valid, categories, candidate_count=candidate_count, seed=seed + fold, gallery_indices=outer_train))
        row = {'alpha': float(alpha), 'top1': float(np.mean([value['top1'] for value in metrics])), 'pairwise_identification': float(np.mean([value['pairwise_identification'] for value in metrics])), 'category_partial_rdm': float(np.mean([value['category_partial_rdm'] for value in metrics]))}
        rows.append(row)
    return max(rows, key=lambda row: (row['top1'], row['pairwise_identification'], row['category_partial_rdm'], -row['alpha']))

def choose_pls_components(source: Array, target: Array, categories: Index, outer_train: Index, *, components_grid: Iterable[int], candidate_count: int, seed: int) -> dict[str, float]:
    inner_categories = categories[outer_train]
    inner_splits = stratified_splits(inner_categories, 3, seed)
    rows = []
    for components in components_grid:
        metrics = []
        valid_components = True
        for fold, (local_train, local_valid) in enumerate(inner_splits):
            train = outer_train[local_train]
            valid = outer_train[local_valid]
            maximum = min(len(train) - 1, source.shape[1], target.shape[1])
            if int(components) > maximum:
                valid_components = False
                break
            prediction, gallery = pls_prediction(source, target, train, valid, int(components))
            metrics.append(retrieval_metrics(prediction, gallery, valid, valid, categories, candidate_count=candidate_count, seed=seed + fold, gallery_indices=outer_train))
        if not valid_components:
            continue
        rows.append({'components': int(components), 'top1': float(np.mean([value['top1'] for value in metrics])), 'pairwise_identification': float(np.mean([value['pairwise_identification'] for value in metrics])), 'category_partial_rdm': float(np.mean([value['category_partial_rdm'] for value in metrics]))})
    if not rows:
        raise ValueError('no valid PLS component candidate')
    return max(rows, key=lambda row: (row['top1'], row['pairwise_identification'], row['category_partial_rdm'], -row['components']))

def summarize_metric_folds(folds: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    if not folds:
        raise ValueError('no folds to summarize')
    return {metric: {'mean': float(np.mean([row[metric] for row in folds])), 'median': float(np.median([row[metric] for row in folds])), 'minimum': float(np.min([row[metric] for row in folds]))} for metric in folds[0]}

def indices(value, n, name):
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.dtype.kind not in 'iu' or (not len(raw)) or np.any(raw < 0) or np.any(raw >= n) or (len(np.unique(raw)) != len(raw)):
        raise ValueError(f'{name} must contain unique in-range integer indices')
    return raw.astype(np.int64)

def validate_pair(source, target, categories=None):
    source, target = (require_matrix(source, 'source'), require_matrix(target, 'target'))
    if len(source) != len(target):
        raise ValueError('source and target row counts differ')
    if categories is not None:
        categories = require_categories(categories, len(source))
    return (source, target, categories)

def validate_splits(splits, n):
    output, seen = ([], np.zeros(n, dtype=int))
    for train, test in splits:
        train, test = (indices(train, n, 'train'), indices(test, n, 'test'))
        if len(train) < 2 or len(test) < 2 or np.intersect1d(train, test).size:
            raise ValueError('folds require disjoint nontrivial train/test partitions')
        seen[test] += 1
        output.append((train, test))
    if not np.all(seen == 1):
        raise ValueError('outer tests must cover each row exactly once')
    return output

def finite_result(value):
    if isinstance(value, dict):
        for item in value.values():
            finite_result(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            finite_result(item)
    elif isinstance(value, (float, np.floating)) and (not np.isfinite(value)):
        raise ValueError('undefined or non-finite statistic; inspect degenerate inputs')
    return value

def validate_prediction_inputs(source, target, train, query, parameter):
    source, target, _ = validate_pair(source, target)
    train, query = (indices(train, len(source), 'train'), indices(query, len(source), 'query'))
    if np.intersect1d(train, query).size:
        raise ValueError('training and query rows must be disjoint')
    if not np.isfinite(parameter) or parameter <= 0:
        raise ValueError('positive finite regularization/component count required')
    return (source, target, train, query)

def validate_retrieval(prediction, target_gallery, query_indices, correct_indices, categories, candidate_count, gallery_indices):
    prediction, target_gallery = (require_matrix(prediction, 'prediction'), require_matrix(target_gallery, 'gallery'))
    if prediction.shape[1] != target_gallery.shape[1]:
        raise ValueError('prediction/gallery widths differ')
    categories = require_categories(categories, len(target_gallery))
    query_indices = indices(query_indices, len(target_gallery), 'query')
    correct_indices = indices(correct_indices, len(target_gallery), 'correct')
    gallery_indices = np.arange(len(target_gallery)) if gallery_indices is None else indices(gallery_indices, len(target_gallery), 'gallery')
    if type(candidate_count) is not int or candidate_count < 2:
        raise ValueError('at least two candidates required')
    if not np.isin(correct_indices, gallery_indices).all():
        raise ValueError('correct targets must belong to the gallery')
    return (prediction, target_gallery, query_indices, correct_indices, categories, gallery_indices)
