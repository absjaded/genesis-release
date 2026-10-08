"""Shared claim-bearing Gate-1 estimators for ORC-1 and MP-002D.

All fitting functions receive explicit training indices.  The module contains no
data loading, world-generation, threshold selection, or access to real feature
roots; callers own those concerns and manifest this file's SHA-256.
"""

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


def require_matrix(value: Array, name: str) -> Array:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or min(result.shape) < 2:
        raise ValueError(f"{name} must be a nontrivial two-dimensional matrix")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite values")
    return result


def require_categories(categories: Index, n_samples: int) -> Index:
    result = np.asarray(categories, dtype=np.int64)
    if result.shape != (n_samples,):
        raise ValueError(f"categories must have shape ({n_samples},)")
    if len(np.unique(result)) < 2:
        raise ValueError("at least two categories are required")
    return result


def l2_rows(value: Array) -> Array:
    data = require_matrix(value, "value")
    return data / np.maximum(np.linalg.norm(data, axis=1, keepdims=True), 1e-12)


def fit_center_l2(train: Array, test: Array) -> tuple[Array, Array]:
    train = require_matrix(train, "train")
    test = require_matrix(test, "test")
    if train.shape[1] != test.shape[1]:
        raise ValueError("train/test feature widths differ")
    mean = train.mean(axis=0, keepdims=True)
    return l2_rows(train - mean), l2_rows(test - mean)


def fit_standardize(train: Array, test: Array) -> tuple[Array, Array, NDArray[np.bool_]]:
    train = require_matrix(train, "train")
    test = require_matrix(test, "test")
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, ddof=1, keepdims=True)
    valid = std[0] > 1e-8
    if np.count_nonzero(valid) < 2:
        raise ValueError("fewer than two nonconstant training predictors")
    return (train[:, valid] - mean[:, valid]) / std[:, valid], (test[:, valid] - mean[:, valid]) / std[:, valid], valid


def cosine_distance(normalized: Array) -> Array:
    data = require_matrix(normalized, "normalized")
    result = 1.0 - np.clip(data @ data.T, -1.0, 1.0)
    np.fill_diagonal(result, 0.0)
    return result


def upper(square: Array) -> Array:
    square = np.asarray(square, dtype=np.float64)
    if square.ndim != 2 or square.shape[0] != square.shape[1]:
        raise ValueError("expected a square matrix")
    indices = np.triu_indices(square.shape[0], 1)
    return square[indices]


def residualized_rank_vector(distance: Array, categories: Index) -> Array:
    categories = require_categories(categories, distance.shape[0])
    values = rankdata(upper(distance), method="average").astype(np.float64)
    control_square = (categories[:, None] != categories[None, :]).astype(np.float64)
    control = rankdata(upper(control_square), method="average").astype(np.float64)
    values = (values - values.mean()) / max(float(values.std(ddof=1)), 1e-12)
    control = (control - control.mean()) / max(float(control.std(ddof=1)), 1e-12)
    values -= (values @ control) * control / max(float(control @ control), 1e-12)
    return (values - values.mean()) / max(float(values.std(ddof=1)), 1e-12)


def partial_rsa(left_distance: Array, right_distance: Array, categories: Index) -> float:
    left = residualized_rank_vector(left_distance, categories)
    right = residualized_rank_vector(right_distance, categories)
    return float(np.corrcoef(left, right)[0, 1])


def neighborhood_overlap(left_distance: Array, right_distance: Array, k: int) -> float:
    if not 1 <= k < left_distance.shape[0]:
        raise ValueError("k must lie between one and n-1")
    left_neighbors = np.argsort(left_distance, axis=1, kind="stable")[:, 1 : k + 1]
    right_neighbors = np.argsort(right_distance, axis=1, kind="stable")[:, 1 : k + 1]
    scores = []
    for left, right in zip(left_neighbors, right_neighbors, strict=True):
        a, b = set(map(int, left)), set(map(int, right))
        scores.append(len(a & b) / len(a | b))
    return float(np.mean(scores))


def spectral_metrics(value: Array) -> dict[str, float]:
    data = require_matrix(value, "value")
    centered = data - data.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False, full_matrices=False)
    eigen = singular**2
    total = float(eigen.sum())
    if total <= 1e-20:
        raise ValueError("zero total variance")
    probability = eigen / total
    nonzero = probability[probability > 1e-15]
    raw_normalized = l2_rows(data)
    pair_cosine = upper(raw_normalized @ raw_normalized.T)
    return {
        "effective_rank": float(np.exp(-np.sum(nonzero * np.log(nonzero)))),
        "participation_ratio": float(total**2 / np.sum(eigen**2)),
        "stable_rank": float(total / eigen.max()),
        "anisotropy_mean_pair_cosine": float(pair_cosine.mean()),
    }


def stratified_splits(categories: Index, n_splits: int, seed: int) -> list[tuple[Index, Index]]:
    categories = require_categories(categories, len(categories))
    minimum = min(int(np.count_nonzero(categories == value)) for value in np.unique(categories))
    if n_splits > minimum:
        raise ValueError(f"n_splits={n_splits} exceeds smallest category count={minimum}")
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(seed % (2**32 - 1)))
    return [
        (train.astype(np.int64), test.astype(np.int64))
        for train, test in splitter.split(np.arange(len(categories)), categories)
    ]


def prepare_fold(n_view: Array, m_view: Array, train: Index, test: Index) -> PreparedFold:
    n_train, n_test = fit_center_l2(n_view[train], n_view[test])
    m_train, m_test = fit_center_l2(m_view[train], m_view[test])
    return PreparedFold(train, test, n_train, n_test, m_train, m_test)


def permute_indices(rng: np.random.Generator, categories: Index, mode: str) -> Index:
    if mode == "unrestricted":
        return rng.permutation(len(categories)).astype(np.int64)
    if mode != "category_preserving":
        raise ValueError(mode)
    result = np.arange(len(categories), dtype=np.int64)
    for category in np.unique(categories):
        indices = np.flatnonzero(categories == category)
        result[indices] = rng.permutation(indices)
    return result


def fwer_p(observed: float, null: Iterable[float]) -> float:
    values = np.asarray(list(null), dtype=np.float64)
    if not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("null distribution is empty or non-finite")
    return float((1 + np.count_nonzero(values >= observed)) / (len(values) + 1))


def crossfit_center_l2(value: Array, splits: list[tuple[Index, Index]]) -> Array:
    data = require_matrix(value, "value")
    output = np.empty_like(data, dtype=np.float64)
    seen = np.zeros(len(data), dtype=np.int64)
    for train, test in splits:
        _, transformed = fit_center_l2(data[train], data[test])
        output[test] = transformed
        seen[test] += 1
    if not np.array_equal(seen, np.ones(len(data), dtype=np.int64)):
        raise ValueError("outer splits must cover every row exactly once")
    return output


def raw_geometry_summary(
    n_view: Array,
    m_view: Array,
    categories: Index,
    *,
    outer_splits: list[tuple[Index, Index]],
    primary_k: int,
    sensitivity_k: Iterable[int],
    permutations: int,
    seed: int,
) -> dict[str, Any]:
    n_view = require_matrix(n_view, "n_view")
    m_view = require_matrix(m_view, "m_view")
    if n_view.shape[0] != m_view.shape[0]:
        raise ValueError("source sample counts differ")
    categories = require_categories(categories, n_view.shape[0])
    if primary_k >= len(categories):
        raise ValueError("primary k exceeds cohort")

    n_crossfit = crossfit_center_l2(n_view, outer_splits)
    m_crossfit = crossfit_center_l2(m_view, outer_splits)
    n_distance = cosine_distance(n_crossfit)
    m_distance = cosine_distance(m_crossfit)
    observed_partial = partial_rsa(n_distance, m_distance, categories)
    observed_neighbor = neighborhood_overlap(n_distance, m_distance, primary_k)
    k_values = sorted(set([int(primary_k), *(int(value) for value in sensitivity_k)]))
    sensitivity = {
        str(k): neighborhood_overlap(n_distance, m_distance, k)
        for k in k_values
        if k < len(categories)
    }

    fold_partial = []
    for _, test in outer_splits:
        if len(test) >= 6 and len(np.unique(categories[test])) >= 2:
            fold_partial.append(
                partial_rsa(
                    cosine_distance(n_crossfit[test]),
                    cosine_distance(m_crossfit[test]),
                    categories[test],
                )
            )
    positive_fraction = float(np.mean(np.asarray(fold_partial) > 0)) if fold_partial else float("nan")

    rng = np.random.default_rng(seed)
    nulls: dict[str, dict[str, list[float]]] = {
        "unrestricted": {"partial": [], "neighbor": []},
        "category_preserving": {"partial": [], "neighbor": []},
    }
    for mode in nulls:
        for _ in range(permutations):
            permutation = permute_indices(rng, categories, mode)
            permuted = m_distance[permutation][:, permutation]
            nulls[mode]["partial"].append(partial_rsa(n_distance, permuted, categories))
            nulls[mode]["neighbor"].append(neighborhood_overlap(n_distance, permuted, primary_k))

    category_partial = np.asarray(nulls["category_preserving"]["partial"], dtype=np.float64)
    category_neighbor = np.asarray(nulls["category_preserving"]["neighbor"], dtype=np.float64)
    result: dict[str, Any] = {
        "partial_rsa": observed_partial,
        "neighborhood_overlap": observed_neighbor,
        "neighborhood_delta_vs_category_null": observed_neighbor - float(np.median(category_neighbor)),
        "positive_fold_fraction": positive_fraction,
        "unrestricted_partial_p": fwer_p(observed_partial, nulls["unrestricted"]["partial"]),
        "category_partial_p": fwer_p(observed_partial, category_partial),
        "category_neighborhood_p": fwer_p(observed_neighbor, category_neighbor),
        "neighborhood_sensitivity": sensitivity,
        "null_summary": {},
    }
    for mode, metrics in nulls.items():
        result["null_summary"][mode] = {}
        for metric, values in metrics.items():
            array = np.asarray(values, dtype=np.float64)
            result["null_summary"][mode][metric] = {
                "median": float(np.median(array)),
                "p95": float(np.percentile(array, 95)),
                "maximum": float(np.max(array)),
            }
    return result


def context_stability(contexts: Array) -> dict[str, float]:
    value = np.asarray(contexts, dtype=np.float64)
    if value.ndim != 3 or value.shape[0] < 2:
        raise ValueError("contexts must have shape (contexts, samples, features)")
    if not np.all(np.isfinite(value)):
        raise ValueError("contexts contain non-finite values")
    rdm_vectors = []
    identity_views = []
    for context in value:
        normalized = l2_rows(context - context.mean(axis=0, keepdims=True))
        rdm_vectors.append(rankdata(upper(cosine_distance(normalized)), method="average"))
        row_centered = context - context.mean(axis=1, keepdims=True)
        identity_views.append(l2_rows(row_centered))
    rdm_correlations = [
        float(spearmanr(rdm_vectors[i], rdm_vectors[j]).statistic)
        for i, j in itertools.combinations(range(len(rdm_vectors)), 2)
    ]
    within, between = [], []
    for i, j in itertools.combinations(range(len(identity_views)), 2):
        similarity = identity_views[i] @ identity_views[j].T
        within.extend(np.diag(similarity).tolist())
        between.extend(similarity[~np.eye(len(similarity), dtype=bool)].tolist())
    return {
        "rdm_reliability": float(np.median(rdm_correlations)),
        "identity_within_median": float(np.median(within)),
        "identity_within_p05": float(np.percentile(within, 5)),
        "identity_between_median": float(np.median(between)),
        "identity_margin": float(np.median(within) - np.median(between)),
    }


def bidirectional_reconstruction(
    n_view: Array,
    m_view: Array,
    outer_splits: list[tuple[Index, Index]],
    alpha: float = 1e-6,
) -> dict[str, float]:
    directional: dict[str, list[float]] = {"n_to_m": [], "m_to_n": []}
    for train, test in outer_splits:
        for name, source, target in (
            ("n_to_m", n_view, m_view),
            ("m_to_n", m_view, n_view),
        ):
            model = Ridge(alpha=alpha, fit_intercept=True)
            model.fit(source[train], target[train])
            predicted = model.predict(source[test])
            directional[name].append(
                float(r2_score(target[test], predicted, multioutput="variance_weighted"))
            )
    n_to_m = float(np.median(directional["n_to_m"]))
    m_to_n = float(np.median(directional["m_to_n"]))
    return {
        "n_to_m_r2": n_to_m,
        "m_to_n_r2": m_to_n,
        "bidirectional_min_r2": min(n_to_m, m_to_n),
        "residual_nonredundancy": 1.0 - min(n_to_m, m_to_n),
    }


def normalize_target_gallery(train_target: Array, gallery_target: Array) -> tuple[Array, Array]:
    train_target = require_matrix(train_target, "train_target")
    gallery_target = require_matrix(gallery_target, "gallery_target")
    mean = train_target.mean(axis=0, keepdims=True)
    return l2_rows(train_target - mean), l2_rows(gallery_target - mean)


def retrieval_metrics(
    prediction: Array,
    target_gallery: Array,
    query_indices: Index,
    correct_indices: Index,
    categories: Index,
    *,
    candidate_count: int,
    seed: int,
    gallery_indices: Index | None = None,
) -> dict[str, float]:
    prediction = l2_rows(prediction)
    target_gallery = l2_rows(target_gallery)
    query_indices = np.asarray(query_indices, dtype=np.int64)
    correct_indices = np.asarray(correct_indices, dtype=np.int64)
    categories = require_categories(categories, len(target_gallery))
    if prediction.shape[0] != len(query_indices) or len(query_indices) != len(correct_indices):
        raise ValueError("query, prediction, and correct-index counts differ")
    if gallery_indices is None:
        gallery_indices = np.arange(len(categories), dtype=np.int64)
    else:
        gallery_indices = np.asarray(gallery_indices, dtype=np.int64)

    ranks, pairwise, correct_cosine = [], [], []
    for row, (query, correct) in enumerate(zip(query_indices, correct_indices, strict=True)):
        if categories[query] != categories[correct]:
            raise ValueError("correct target must preserve query category")
        pool = gallery_indices[
            (categories[gallery_indices] == categories[query]) & (gallery_indices != correct)
        ]
        if len(pool) < candidate_count - 1:
            raise ValueError(
                f"category {categories[query]} has only {len(pool)} distractors; "
                f"need {candidate_count - 1}"
            )
        query_seed = int((seed + 0x9E3779B97F4A7C15 * (int(query) + 1)) % (2**64))
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
    return {
        "top1": float(np.mean(rank_array <= 1)),
        "top5": float(np.mean(rank_array <= min(5, candidate_count))),
        "median_rank": float(np.median(rank_array)),
        "paired_cosine": float(np.mean(correct_cosine)),
        "pairwise_identification": float(np.mean(pairwise)),
        "predicted_target_rdm_spearman": rdm_spearman,
        "category_partial_rdm": category_partial,
    }


def ridge_prediction(
    source: Array,
    target: Array,
    train: Index,
    query: Index,
    alpha: float,
) -> tuple[Array, Array]:
    train_x, query_x, _ = fit_standardize(source[train], source[query])
    train_y, gallery_y = normalize_target_gallery(target[train], target)
    scale = float(train_x.shape[1])
    kernel = (train_x @ train_x.T) / scale
    cross = (query_x @ train_x.T) / scale
    coefficient = np.linalg.solve(kernel + float(alpha) * np.eye(len(train)), train_y)
    return cross @ coefficient, gallery_y


def pls_prediction(
    source: Array,
    target: Array,
    train: Index,
    query: Index,
    components: int,
) -> tuple[Array, Array]:
    train_x, query_x, _ = fit_standardize(source[train], source[query])
    train_y, gallery_y = normalize_target_gallery(target[train], target)
    maximum = min(train_x.shape[0] - 1, train_x.shape[1], train_y.shape[1])
    if components > maximum:
        raise ValueError(f"PLS components {components} exceed maximum {maximum}")
    model = PLSRegression(n_components=components, scale=False, max_iter=5000, tol=1e-6)
    model.fit(train_x, train_y)
    return np.asarray(model.predict(query_x), dtype=np.float64), gallery_y


def choose_ridge_alpha(
    source: Array,
    target: Array,
    categories: Index,
    outer_train: Index,
    *,
    alphas: Iterable[float],
    candidate_count: int,
    seed: int,
) -> dict[str, float]:
    inner_categories = categories[outer_train]
    inner_splits = stratified_splits(inner_categories, 3, seed)
    rows = []
    for alpha in alphas:
        metrics = []
        for fold, (local_train, local_valid) in enumerate(inner_splits):
            train = outer_train[local_train]
            valid = outer_train[local_valid]
            prediction, gallery = ridge_prediction(source, target, train, valid, float(alpha))
            metrics.append(
                retrieval_metrics(
                    prediction, gallery, valid, valid, categories,
                    candidate_count=candidate_count,
                    seed=seed + fold,
                    gallery_indices=outer_train,
                )
            )
        row = {
            "alpha": float(alpha),
            "top1": float(np.mean([value["top1"] for value in metrics])),
            "pairwise_identification": float(np.mean([value["pairwise_identification"] for value in metrics])),
            "category_partial_rdm": float(np.mean([value["category_partial_rdm"] for value in metrics])),
        }
        rows.append(row)
    return max(
        rows,
        key=lambda row: (
            row["top1"], row["pairwise_identification"], row["category_partial_rdm"], -row["alpha"]
        ),
    )


def choose_pls_components(
    source: Array,
    target: Array,
    categories: Index,
    outer_train: Index,
    *,
    components_grid: Iterable[int],
    candidate_count: int,
    seed: int,
) -> dict[str, float]:
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
            metrics.append(
                retrieval_metrics(
                    prediction, gallery, valid, valid, categories,
                    candidate_count=candidate_count,
                    seed=seed + fold,
                    gallery_indices=outer_train,
                )
            )
        if not valid_components:
            continue
        rows.append({
            "components": int(components),
            "top1": float(np.mean([value["top1"] for value in metrics])),
            "pairwise_identification": float(np.mean([value["pairwise_identification"] for value in metrics])),
            "category_partial_rdm": float(np.mean([value["category_partial_rdm"] for value in metrics])),
        })
    if not rows:
        raise ValueError("no valid PLS component candidate")
    return max(
        rows,
        key=lambda row: (
            row["top1"], row["pairwise_identification"], row["category_partial_rdm"], -row["components"]
        ),
    )


def summarize_metric_folds(folds: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    if not folds:
        raise ValueError("no folds to summarize")
    return {
        metric: {
            "mean": float(np.mean([row[metric] for row in folds])),
            "median": float(np.median([row[metric] for row in folds])),
            "minimum": float(np.min([row[metric] for row in folds])),
        }
        for metric in folds[0]
    }


def nested_readout_summary(
    source: Array,
    target: Array,
    categories: Index,
    *,
    outer_splits: list[tuple[Index, Index]],
    alphas: Iterable[float],
    pls_components: Iterable[int],
    candidate_count: int,
    permutations: int,
    seed: int,
) -> dict[str, Any]:
    source = require_matrix(source, "source")
    target = require_matrix(target, "target")
    if source.shape[0] != target.shape[0]:
        raise ValueError("source and target sample counts differ")
    categories = require_categories(categories, len(source))
    all_indices = np.arange(len(categories), dtype=np.int64)
    ridge_folds: list[dict[str, float]] = []
    pls_folds: list[dict[str, float]] = []
    centroid_folds: list[dict[str, float]] = []
    ridge_state: list[tuple[Index, Array, Array]] = []
    pls_state: list[tuple[Index, Array, Array]] = []
    selections: list[dict[str, Any]] = []

    for fold, (train, test) in enumerate(outer_splits):
        fold_seed = int((seed + 104729 * (fold + 1)) % (2**64))
        ridge_choice = choose_ridge_alpha(
            source, target, categories, train,
            alphas=alphas, candidate_count=candidate_count, seed=fold_seed,
        )
        ridge_prediction_value, ridge_gallery = ridge_prediction(
            source, target, train, test, float(ridge_choice["alpha"])
        )
        ridge_metrics = retrieval_metrics(
            ridge_prediction_value, ridge_gallery, test, test, categories,
            candidate_count=candidate_count, seed=fold_seed, gallery_indices=all_indices,
        )
        ridge_folds.append(ridge_metrics)
        ridge_state.append((test, ridge_prediction_value, ridge_gallery))

        pls_choice = choose_pls_components(
            source, target, categories, train,
            components_grid=pls_components, candidate_count=candidate_count, seed=fold_seed + 1,
        )
        pls_prediction_value, pls_gallery = pls_prediction(
            source, target, train, test, int(pls_choice["components"])
        )
        pls_metrics = retrieval_metrics(
            pls_prediction_value, pls_gallery, test, test, categories,
            candidate_count=candidate_count, seed=fold_seed + 1, gallery_indices=all_indices,
        )
        pls_folds.append(pls_metrics)
        pls_state.append((test, pls_prediction_value, pls_gallery))

        centroid = np.stack([
            ridge_gallery[train][categories[train] == category].mean(axis=0)
            for category in categories[test]
        ])
        centroid_folds.append(
            retrieval_metrics(
                centroid, ridge_gallery, test, test, categories,
                candidate_count=candidate_count, seed=fold_seed, gallery_indices=all_indices,
            )
        )
        selections.append({
            "fold": fold,
            "ridge_alpha": float(ridge_choice["alpha"]),
            "ridge_inner_top1": float(ridge_choice["top1"]),
            "pls_components": int(pls_choice["components"]),
            "pls_inner_top1": float(pls_choice["top1"]),
        })

    rng = np.random.default_rng(seed + 2)
    null_top1: dict[str, list[float]] = {"ridge": [], "pls": []}
    for permutation_index in range(permutations):
        for label, states in (("ridge", ridge_state), ("pls", pls_state)):
            fold_values = []
            for fold, (test, prediction, gallery) in enumerate(states):
                local_permutation = permute_indices(rng, categories[test], "category_preserving")
                correct = test[local_permutation]
                metric = retrieval_metrics(
                    prediction, gallery, test, correct, categories,
                    candidate_count=candidate_count,
                    seed=seed + 1_000_003 * (permutation_index + 1) + fold,
                    gallery_indices=all_indices,
                )
                fold_values.append(metric["top1"])
            null_top1[label].append(float(np.mean(fold_values)))

    ridge_summary = summarize_metric_folds(ridge_folds)
    pls_summary = summarize_metric_folds(pls_folds)
    centroid_summary = summarize_metric_folds(centroid_folds)
    for label, summary in (("ridge", ridge_summary), ("pls", pls_summary)):
        observed = float(summary["top1"]["mean"])
        null = np.asarray(null_top1[label], dtype=np.float64)
        summary["top1_category_null_p"] = {"value": fwer_p(observed, null)}
        summary["top1_delta_vs_category_null"] = {"value": observed - float(np.median(null))}
        summary["top1_category_null_median"] = {"value": float(np.median(null))}
        summary["top1_category_null_p95"] = {"value": float(np.percentile(null, 95))}
    ridge_summary["top1_delta_vs_centroid"] = {
        "value": ridge_summary["top1"]["mean"] - centroid_summary["top1"]["mean"]
    }
    pls_summary["top1_delta_vs_centroid"] = {
        "value": pls_summary["top1"]["mean"] - centroid_summary["top1"]["mean"]
    }
    return {
        "ridge": ridge_summary,
        "pls": pls_summary,
        "category_centroid": centroid_summary,
        "selections": selections,
    }


def evaluate_gate1_estimators(
    n_view: Array,
    m_view: Array,
    n_contexts: Array,
    categories: Index,
    *,
    outer_folds: int,
    primary_k: int,
    sensitivity_k: Iterable[int],
    permutations: int,
    ridge_alphas: Iterable[float],
    pls_components: Iterable[int],
    retrieval_candidates: int,
    seed: int,
) -> dict[str, Any]:
    n_view = require_matrix(n_view, "n_view")
    m_view = require_matrix(m_view, "m_view")
    if n_view.shape[0] != m_view.shape[0]:
        raise ValueError("source sample counts differ")
    categories = require_categories(categories, len(n_view))
    splits = stratified_splits(categories, outer_folds, seed)
    result = {
        "n_samples": len(categories),
        "n_features": {"n": n_view.shape[1], "m": m_view.shape[1]},
        "raw": raw_geometry_summary(
            n_view, m_view, categories,
            outer_splits=splits,
            primary_k=primary_k,
            sensitivity_k=sensitivity_k,
            permutations=permutations,
            seed=seed + 11,
        ),
        "readout": nested_readout_summary(
            n_view, m_view, categories,
            outer_splits=splits,
            alphas=ridge_alphas,
            pls_components=pls_components,
            candidate_count=retrieval_candidates,
            permutations=permutations,
            seed=seed + 23,
        ),
        "redundancy": bidirectional_reconstruction(n_view, m_view, splits),
        "context": context_stability(n_contexts),
        "spectrum": {
            "n": spectral_metrics(n_view),
            "m": spectral_metrics(m_view),
        },
    }
    finite_values: list[float] = []

    def collect(value: Any) -> None:
        if isinstance(value, dict):
            for child in value.values():
                collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)
        elif isinstance(value, (float, np.floating)):
            finite_values.append(float(value))

    collect(result)
    if not np.all(np.isfinite(finite_values)):
        raise ValueError("estimator output contains non-finite values")
    return result
