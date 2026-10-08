"""Matched-image readout: two repeats of four held-out-fold evaluations."""
import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold
from .analysis import l2_rows, upper, validate_pair, finite_result
ALPHAS = np.logspace(-4, 4, 13)
OUTER_SEED = 26080731
def standardize_train_test(train_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, ddof=1, keepdims=True)
    valid = std[0] > 1e-08
    if np.count_nonzero(valid) < 2:
        raise RuntimeError('fewer than two nonconstant predictors')
    return ((train_x[:, valid] - mean[:, valid]) / std[:, valid], (test_x[:, valid] - mean[:, valid]) / std[:, valid])

def normalize_targets(train_y: np.ndarray, test_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train_y.mean(axis=0, keepdims=True)
    train = l2_rows(train_y - mean)
    test = l2_rows(test_y - mean)
    return (train, test)

def retrieval_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    prediction = l2_rows(prediction)
    target = l2_rows(target)
    similarity = prediction @ target.T
    order = np.argsort(-similarity, axis=1)
    correct = np.arange(len(target))
    ranks = np.asarray([int(np.flatnonzero(order[i] == i)[0]) + 1 for i in range(len(target))])
    diagonal = np.diag(similarity)
    pairwise = []
    for i in range(len(target)):
        pairwise.extend((diagonal[i] > np.delete(similarity[i], i)).astype(np.float64).tolist())
    pred_distance = 1.0 - np.clip(prediction @ prediction.T, -1.0, 1.0)
    true_distance = 1.0 - np.clip(target @ target.T, -1.0, 1.0)
    return {'top1': float(np.mean(ranks <= 1)), 'top5': float(np.mean(ranks <= min(5, len(target)))), 'median_rank': float(np.median(ranks)), 'paired_cosine': float(np.mean(diagonal)), 'pairwise_identification': float(np.mean(pairwise)), 'predicted_target_rdm_spearman': float(spearmanr(upper(pred_distance), upper(true_distance)).statistic)}
def kernel_design(train_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train_z, test_z = standardize_train_test(train_x, test_x)
    scale = float(train_z.shape[1])
    return (train_z @ train_z.T / scale, test_z @ train_z.T / scale)

def predict_design(kernel: np.ndarray, cross: np.ndarray, train_y: np.ndarray, alpha: float) -> np.ndarray:
    coefficient = np.linalg.solve(kernel + float(alpha) * np.eye(kernel.shape[0]), train_y)
    return cross @ coefficient

def choose(x: np.ndarray, targets: list[np.ndarray], categories: np.ndarray, outer_train: np.ndarray, candidate_indices: list[int]) -> dict:
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=OUTER_SEED + int(outer_train.sum()))
    scores = {(target_index, float(alpha)): [] for target_index in candidate_indices for alpha in ALPHAS}
    for inner_train_local, inner_valid_local in inner.split(outer_train, categories[outer_train]):
        train = outer_train[inner_train_local]
        valid = outer_train[inner_valid_local]
        kernel, cross = kernel_design(x[train], x[valid])
        for target_index in candidate_indices:
            y = targets[target_index]
            train_y, valid_y = normalize_targets(y[train], y[valid])
            for alpha in ALPHAS:
                prediction = predict_design(kernel, cross, train_y, float(alpha))
                scores[target_index, float(alpha)].append(retrieval_metrics(prediction, valid_y))
    best = None
    for (target_index, alpha), folds in scores.items():
        row = {'target_index': target_index, 'alpha': alpha, 'top1': float(np.mean([m['top1'] for m in folds])), 'pairwise_identification': float(np.mean([m['pairwise_identification'] for m in folds])), 'paired_cosine': float(np.mean([m['paired_cosine'] for m in folds]))}
        key = (row['top1'], row['pairwise_identification'], row['paired_cosine'], -np.log10(alpha))
        if best is None or key > best[0]:
            best = (key, row)
    assert best is not None
    return best[1]

def splits(categories: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    output = []
    for repeat in range(2):
        cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=OUTER_SEED + repeat)
        output.extend(((train, test) for train, test in cv.split(np.arange(len(categories)), categories)))
    return output

def _evaluate(name: str, x: np.ndarray, targets: list[np.ndarray], target_labels: list[str], categories: np.ndarray, candidate_indices: list[int]) -> dict:
    folds = []
    for fold_id, (train, test) in enumerate(splits(categories)):
        selected = choose(x, targets, categories, train, candidate_indices)
        target_index = int(selected['target_index'])
        y = targets[target_index]
        train_y, test_y = normalize_targets(y[train], y[test])
        kernel, cross = kernel_design(x[train], x[test])
        prediction = predict_design(kernel, cross, train_y, float(selected['alpha']))
        metrics = retrieval_metrics(prediction, test_y)
        centroid_prediction = np.stack([train_y[categories[train] == category].mean(axis=0) for category in categories[test]])
        folds.append({'fold_id': fold_id, 'train': train.tolist(), 'test': test.tolist(), 'selected_target': target_labels[target_index], 'selected_alpha': float(selected['alpha']), 'inner_selection': selected, 'metrics': metrics, 'category_centroid_baseline': retrieval_metrics(centroid_prediction, test_y)})
    summary = {}
    for metric in folds[0]['metrics']:
        values = np.asarray([fold['metrics'][metric] for fold in folds], dtype=np.float64)
        baseline = np.asarray([fold['category_centroid_baseline'][metric] for fold in folds], dtype=np.float64)
        summary[metric] = {'median': float(np.median(values)), 'mean': float(np.mean(values)), 'p05': float(np.percentile(values, 5)), 'category_centroid_mean': float(np.mean(baseline)), 'mean_delta_over_category_centroid': float(np.mean(values - baseline))}
    return {'name': name, 'shape': list(x.shape), 'summary': summary, 'folds': folds}

def evaluate(name, x, targets, target_labels, categories, candidate_indices):
    """Run the matched-64-image protocol with nested target/penalty selection.
    
    Each of two repeats uses four outer folds: 48 training and 16 test
    images. Retrieval ranks against the held-out fold. This candidate
    universe differs from the discovery protocol's dataset-origin groups.
    """
    if len(x) != 64 or not targets or len(targets) != len(target_labels):
        raise ValueError('Matched-image protocol requires 64 rows and named target banks')
    if not candidate_indices or any(type(i) is not int or not 0 <= i < len(targets) for i in candidate_indices):
        raise ValueError('invalid target candidate indices')
    validated = [validate_pair(x, y, categories) for y in targets]
    x, _, categories = validated[0]
    return finite_result(_evaluate(name, x, [v[1] for v in validated], target_labels, categories, candidate_indices))
