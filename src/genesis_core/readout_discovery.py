"""Discovery readout: nested selection, category candidates and fixed-outcome nulls."""
import hashlib
import json
from typing import Any
import numpy as np
from . import analysis as gate
def oracle_modules():
    return gate, None
def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')

def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()

def derived_seed(master_seed: int, *parts: Any) -> int:
    payload = canonical_bytes([int(master_seed), *parts])
    return int(sha256_bytes(payload)[:16], 16) % (2 ** 63 - 1)

def query_top1_records(prediction: np.ndarray, target_gallery: np.ndarray, query_indices: np.ndarray, categories: np.ndarray, *, candidate_count: int, seed: int) -> list[dict[str, int]]:
    gate, _ = oracle_modules()
    prediction = gate.l2_rows(prediction)
    target_gallery = gate.l2_rows(target_gallery)
    all_indices = np.arange(len(categories), dtype=np.int64)
    output = []
    for row_index, query in enumerate(query_indices):
        query_value = int(query)
        pool = all_indices[(categories[all_indices] == categories[query_value]) & (all_indices != query_value)]
        query_seed = int((seed + 11400714819323198485 * (query_value + 1)) % 2 ** 64)
        rng = np.random.default_rng(query_seed)
        distractors = rng.choice(pool, size=candidate_count - 1, replace=False)
        candidates = np.concatenate([[query_value], np.sort(distractors)]).astype(np.int64)
        similarities = prediction[row_index] @ target_gallery[candidates].T
        rank = 1 + int(np.count_nonzero(similarities[1:] >= float(similarities[0])))
        output.append({'query_index': query_value, 'category': int(categories[query_value]), 'rank': rank, 'top1': int(rank == 1)})
    return output

def stratified_bootstrap_top1(records: list[dict[str, int]], config: dict[str, Any], seed: int) -> dict[str, Any]:
    values = np.asarray([row['top1'] for row in records], dtype=np.float64)
    categories = np.asarray([row['category'] for row in records], dtype=np.int64)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(int(config['bootstrap_resamples'])):
        selected = []
        for category in np.unique(categories):
            indices = np.flatnonzero(categories == category)
            selected.extend(rng.choice(indices, size=len(indices), replace=True).tolist())
        draws.append(float(values[np.asarray(selected, dtype=np.int64)].mean()))
    alpha = 1.0 - float(config['bootstrap_confidence'])
    array = np.asarray(draws, dtype=np.float64)
    return {'method': 'stratified stimulus bootstrap over fixed per-query candidate sets', 'resamples': int(config['bootstrap_resamples']), 'confidence': float(config['bootstrap_confidence']), 'lower': float(np.percentile(array, 100.0 * alpha / 2.0)), 'upper': float(np.percentile(array, 100.0 * (1.0 - alpha / 2.0))), 'median': float(np.median(array)), 'seed': int(seed)}

def _nested_readout_result(source: np.ndarray, target: np.ndarray, categories: np.ndarray, config: dict[str, Any], seed: int, family: str) -> dict[str, Any]:
    gate, _ = oracle_modules()
    outer_seed = int(config['outer_split_seed'])
    splits = gate.stratified_splits(categories, int(config['outer_folds']), outer_seed)
    all_indices = np.arange(len(categories), dtype=np.int64)
    folds: list[dict[str, float]] = []
    centroid_folds: list[dict[str, float]] = []
    states: list[tuple[np.ndarray, np.ndarray, np.ndarray, int]] = []
    selections: list[dict[str, Any]] = []
    query_records: list[dict[str, int]] = []
    inner_candidates = int(config['inner_retrieval_candidates'])
    outer_candidates = int(config['outer_retrieval_candidates'])
    for fold, (train, test) in enumerate(splits):
        smallest = min((int(np.count_nonzero(categories[train] == category)) for category in np.unique(categories)))
        if smallest < inner_candidates:
            raise RuntimeError(f'outer-training gallery supports {smallest}, not {inner_candidates}, candidates')
        fold_seed = outer_seed + 104729 * (fold + 1) + (1 if family == 'pls' else 0)
        if family == 'ridge':
            choice = gate.choose_ridge_alpha(source, target, categories, train, alphas=config['ridge_alphas'], candidate_count=inner_candidates, seed=fold_seed)
            prediction, gallery = gate.ridge_prediction(source, target, train, test, float(choice['alpha']))
            selected_value = {'ridge_alpha': float(choice['alpha'])}
        elif family == 'pls':
            choice = gate.choose_pls_components(source, target, categories, train, components_grid=config['pls_components'], candidate_count=inner_candidates, seed=fold_seed)
            prediction, gallery = gate.pls_prediction(source, target, train, test, int(choice['components']))
            selected_value = {'pls_components': int(choice['components'])}
        else:
            raise ValueError(family)
        metrics = gate.retrieval_metrics(prediction, gallery, test, test, categories, candidate_count=outer_candidates, seed=fold_seed, gallery_indices=all_indices)
        folds.append(metrics)
        records = query_top1_records(prediction, gallery, test, categories, candidate_count=outer_candidates, seed=fold_seed)
        if abs(float(np.mean([row['top1'] for row in records])) - metrics['top1']) > 1e-12:
            raise RuntimeError('query-level Top-1 audit differs from frozen retrieval estimator')
        query_records.extend(records)
        states.append((test, prediction, gallery, fold_seed))
        centroid = np.stack([gallery[train][categories[train] == category].mean(axis=0) for category in categories[test]])
        centroid_folds.append(gate.retrieval_metrics(centroid, gallery, test, test, categories, candidate_count=outer_candidates, seed=fold_seed, gallery_indices=all_indices))
        selections.append({'fold': fold, **selected_value, 'inner_top1': float(choice['top1']), 'inner_candidate_count': inner_candidates, 'outer_candidate_count': outer_candidates, 'seed': int(fold_seed)})
    summary = gate.summarize_metric_folds(folds)
    centroid_summary = gate.summarize_metric_folds(centroid_folds)
    rng = np.random.default_rng(derived_seed(seed, family, 'category-null'))
    null_top1 = []
    for permutation_index in range(int(config['permutations_per_mode'])):
        fold_values = []
        for fold, (test, prediction, gallery, fold_seed) in enumerate(states):
            local_permutation = gate.permute_indices(rng, categories[test], 'category_preserving')
            correct = test[local_permutation]
            metric = gate.retrieval_metrics(prediction, gallery, test, correct, categories, candidate_count=outer_candidates, seed=derived_seed(seed, family, 'null', permutation_index, fold, fold_seed), gallery_indices=all_indices)
            fold_values.append(metric['top1'])
        null_top1.append(float(np.mean(fold_values)))
    observed = float(summary['top1']['mean'])
    null = np.asarray(null_top1, dtype=np.float64)
    summary['top1_category_null_p'] = {'value': gate.fwer_p(observed, null)}
    summary['top1_delta_vs_category_null'] = {'value': observed - float(np.median(null))}
    summary['top1_category_null_median'] = {'value': float(np.median(null))}
    summary['top1_category_null_p95'] = {'value': float(np.percentile(null, 95))}
    summary['top1_delta_vs_centroid'] = {'value': observed - float(centroid_summary['top1']['mean'])}
    bootstrap_seed = derived_seed(seed, family, 'stimulus-bootstrap')
    return {'family': family, 'summary': summary, 'category_centroid': centroid_summary, 'fold_metrics': folds, 'selections': selections, 'query_top1_records': sorted(query_records, key=lambda row: row['query_index']), 'top1_bootstrap': stratified_bootstrap_top1(query_records, config, bootstrap_seed), 'category_null_top1': null_top1, 'outer_seed': outer_seed}

def nested_readout_result(source, target, categories, config, seed, family='ridge'):
    """Fit nested readouts and evaluate held-out matching-image retrieval.
    
    source and target have matched rows; categories define candidate groups.
    Historical groups are dataset origins: COCO, ImageNet and Scene.
    Distractors may come from the full bank. The null and bootstrap measure
    evaluation variability conditional on fitted predictions.
    """
    source, target, categories = gate.validate_pair(source, target, categories)
    for key in ('outer_folds', 'inner_retrieval_candidates', 'outer_retrieval_candidates', 'permutations_per_mode', 'bootstrap_resamples'):
        if type(config[key]) is not int or config[key] < (2 if key in ('outer_folds', 'inner_retrieval_candidates', 'outer_retrieval_candidates') else 1):
            raise ValueError('invalid ' + key)
    if not 0 < config['bootstrap_confidence'] < 1:
        raise ValueError('bootstrap confidence must be in (0,1)')
    return gate.finite_result(_nested_readout_result(source, target, categories, config, seed, family))
