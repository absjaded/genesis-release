#!/usr/bin/env python3
"""Genesis MP-002C core attenuation and learned-readout audit.

This is diagnostic only. It preserves the MP-002 Gate 1 decision and does not
construct a third geometry.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.model_selection import StratifiedKFold


ROOT = Path(os.environ.get("GENESIS_ROOT", "/workspace/genesis"))
FEATURES = ROOT / "features/tribev2"
DINO_ROOT = ROOT / "features/dinov2"
ART = ROOT / "artifacts/mp002c"
RESULTS = ROOT / "results/mp002c"
MP002_PROJECT = ROOT / "project/mp002"

if str(MP002_PROJECT) not in sys.path:
    sys.path.insert(0, str(MP002_PROJECT))
import mp002  # noqa: E402


LAGS = {
    "plus4": "plus4",
    "plus5": "plus5_primary",
    "plus6": "plus6",
}
ATTEMPT3_LAGS = {
    "plus4": "plus4_primary",
    "plus5": "plus5",
    "plus6": "plus6",
}
ALPHAS = np.logspace(-4, 4, 13)
OUTER_SEED = 26080731


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_array(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    array = np.load(path)
    if array.shape != shape:
        raise RuntimeError(f"shape mismatch for {path}: {array.shape} != {shape}")
    if array.dtype != np.float32:
        raise RuntimeError(f"dtype mismatch for {path}: {array.dtype}")
    if not np.isfinite(array).all():
        raise RuntimeError(f"non-finite array: {path}")
    return array


def spectral_metrics(array: np.ndarray) -> dict[str, Any]:
    """Spectrum in stimulus space; efficient for N << features."""
    data = np.asarray(array, dtype=np.float64)
    data = data - data.mean(axis=0, keepdims=True)
    gram = data @ data.T
    eigen = np.linalg.eigvalsh((gram + gram.T) * 0.5)[::-1]
    eigen = np.clip(eigen, 0.0, None)
    total = float(eigen.sum())
    if total <= 1e-20:
        raise RuntimeError("zero source variance")
    probabilities = eigen / total
    nonzero = probabilities[probabilities > 1e-15]
    effective_rank = float(np.exp(-np.sum(nonzero * np.log(nonzero))))
    return {
        "effective_rank": effective_rank,
        "stable_rank": float(total / max(float(eigen[0]), 1e-20)),
        "participation_ratio": float(total**2 / max(float(np.sum(eigen**2)), 1e-20)),
        "pc1_variance_fraction": float(probabilities[0]),
        "pc4_cumulative_fraction": float(probabilities[:4].sum()),
        "pc8_cumulative_fraction": float(probabilities[:8].sum()),
        "positive_rank": int(np.count_nonzero(eigen > total * 1e-12)),
        "normalized_eigenvalues": probabilities.tolist(),
    }


def l2_rows(array: np.ndarray) -> np.ndarray:
    data = np.asarray(array, dtype=np.float64)
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    return data / np.maximum(norms, 1e-12)


def centered_l2(array: np.ndarray) -> np.ndarray:
    data = np.asarray(array, dtype=np.float64)
    return l2_rows(data - data.mean(axis=0, keepdims=True))


def distance_matrix(array: np.ndarray) -> np.ndarray:
    normalized = centered_l2(array)
    distance = 1.0 - np.clip(normalized @ normalized.T, -1.0, 1.0)
    np.fill_diagonal(distance, 0.0)
    return distance


def upper(square: np.ndarray) -> np.ndarray:
    i, j = np.triu_indices(square.shape[0], 1)
    return square[i, j]


def residualized_rank_vector(distance: np.ndarray, categories: np.ndarray) -> np.ndarray:
    values = rankdata(upper(distance), method="average")
    control = rankdata(
        upper((categories[:, None] != categories[None, :]).astype(np.float64)),
        method="average",
    )
    values = (values - values.mean()) / max(values.std(ddof=1), 1e-12)
    control = (control - control.mean()) / max(control.std(ddof=1), 1e-12)
    values = values - (values @ control) * control / max(control @ control, 1e-12)
    return (values - values.mean()) / max(values.std(ddof=1), 1e-12)


def partial_rsa(left_distance: np.ndarray, right_distance: np.ndarray, categories: np.ndarray) -> float:
    left = residualized_rank_vector(left_distance, categories)
    right = residualized_rank_vector(right_distance, categories)
    return float(np.corrcoef(left, right)[0, 1])


def neighborhood_overlap(left_distance: np.ndarray, right_distance: np.ndarray, k: int = 3) -> float:
    left = np.argsort(left_distance, axis=1)[:, 1 : k + 1]
    right = np.argsort(right_distance, axis=1)[:, 1 : k + 1]
    values = []
    for a, b in zip(left, right, strict=True):
        aa, bb = set(map(int, a)), set(map(int, b))
        values.append(len(aa & bb) / len(aa | bb))
    return float(np.mean(values))


def identity_reliability(contexts: np.ndarray) -> dict[str, float]:
    normalized = []
    for context in contexts:
        data = np.asarray(context, dtype=np.float64)
        data = data - data.mean(axis=1, keepdims=True)
        normalized.append(l2_rows(data))
    within, between = [], []
    for left, right in itertools.combinations(range(4), 2):
        matrix = normalized[left] @ normalized[right].T
        within.extend(np.diag(matrix).tolist())
        between.extend(matrix[~np.eye(matrix.shape[0], dtype=bool)].tolist())
    return {
        "within_median": float(np.median(within)),
        "within_p05": float(np.percentile(within, 5)),
        "between_median": float(np.median(between)),
        "identity_margin": float(np.median(within) - np.median(between)),
    }


def context_rdm_reliability(contexts: np.ndarray) -> float:
    vectors = [rankdata(upper(distance_matrix(context)), method="average") for context in contexts]
    correlations = [
        float(spearmanr(vectors[i], vectors[j]).statistic)
        for i, j in itertools.combinations(range(4), 2)
    ]
    return float(np.median(correlations))


def cross_context_signal_spectrum(contexts: np.ndarray) -> dict[str, Any]:
    centered = []
    for context in contexts:
        data = np.asarray(context, dtype=np.float64)
        data -= data.mean(axis=0, keepdims=True)
        data /= max(float(np.linalg.norm(data)), 1e-12)
        centered.append(data)
    gram = np.zeros((contexts.shape[1], contexts.shape[1]), dtype=np.float64)
    pairs = 0
    for i, j in itertools.combinations(range(4), 2):
        cross = centered[i] @ centered[j].T
        gram += (cross + cross.T) * 0.5
        pairs += 1
    gram /= pairs
    eigen = np.linalg.eigvalsh((gram + gram.T) * 0.5)[::-1]
    positive = np.clip(eigen, 0.0, None)
    negative = np.clip(-eigen, 0.0, None)
    total = float(positive.sum())
    probability = positive / max(total, 1e-20)
    nonzero = probability[probability > 1e-15]
    return {
        "positive_effective_rank": float(np.exp(-np.sum(nonzero * np.log(nonzero)))),
        "positive_rank": int(np.count_nonzero(positive > max(total, 1e-20) * 1e-12)),
        "negative_to_positive_mass": float(negative.sum() / max(total, 1e-20)),
        "positive_normalized_eigenvalues": probability.tolist(),
    }


def aggregate_contexts(contexts: np.ndarray, method: str) -> np.ndarray:
    if method == "mean":
        return contexts.mean(axis=0, dtype=np.float64)
    if method == "median":
        return np.median(contexts.astype(np.float64), axis=0)
    if method == "concat":
        return contexts.transpose(1, 0, 2).reshape(contexts.shape[1], -1).astype(np.float64)
    if method.startswith("context_"):
        return contexts[int(method.split("_")[1])].astype(np.float64)
    raise ValueError(method)


def dino_arrays() -> tuple[list[str], list[np.ndarray]]:
    cls = require_array(DINO_ROOT / "mp001_cls.npy", (64, 12, 768))
    patch = require_array(DINO_ROOT / "mp001_patch_mean.npy", (64, 12, 768))
    labels = [f"cls_block_{i:02d}" for i in range(12)] + [f"patch_mean_block_{i:02d}" for i in range(12)]
    arrays = [cls[:, i] for i in range(12)] + [patch[:, i] for i in range(12)]
    return labels, arrays


def load_sources() -> dict[str, dict[str, np.ndarray]]:
    result: dict[str, dict[str, np.ndarray]] = {
        "original_latent": {},
        "original_cortical": {},
        "balanced_cortical_raw": {},
        "balanced_cortical_centered": {},
    }
    for lag, suffix in LAGS.items():
        result["original_latent"][lag] = require_array(
            FEATURES / f"mp001_latent2048_{suffix}.npy", (4, 64, 2048)
        )
        result["original_cortical"][lag] = require_array(
            FEATURES / f"mp001_cortical_{suffix}.npy", (4, 64, 20484)
        )
    for lag, suffix in ATTEMPT3_LAGS.items():
        result["balanced_cortical_raw"][lag] = require_array(
            FEATURES / f"mp001_attempt3_cortical_raw_{suffix}.npy", (4, 64, 20484)
        )
        result["balanced_cortical_centered"][lag] = require_array(
            FEATURES / f"mp001_attempt3_cortical_centered_{suffix}.npy", (4, 64, 20484)
        )
    return result


def command_inventory(_: argparse.Namespace) -> None:
    sources = load_sources()
    validation = load_json(FEATURES / "mp001_tribe_validation.json")
    latent_validation = load_json(FEATURES / "mp001_tribe_latent_validation.json")
    balanced_validation = load_json(FEATURES / "mp001_tribe_attempt3_validation.json")
    paired = {
        "stimulus_order_identical": validation["stimulus_order"] == latent_validation["stimulus_order"],
        "alignment_identical": validation["alignment"] == latent_validation["alignment"],
        "checkpoint_identical": validation["checkpoint_sha256"] == latent_validation["checkpoint_sha256"],
        "vjepa_weights_identical": validation["vjepa2_weights_sha256"] == latent_validation["vjepa2_weights_sha256"],
        "balanced_stimulus_order_identical": validation["stimulus_order"] == balanced_validation["stimulus_order"],
    }
    if not all(paired.values()):
        raise RuntimeError(f"paired provenance failed: {paired}")
    inputs = []
    for source, lags in sources.items():
        for lag, array in lags.items():
            if source == "original_latent":
                suffix = LAGS[lag]
                path = FEATURES / f"mp001_latent2048_{suffix}.npy"
            elif source == "original_cortical":
                suffix = LAGS[lag]
                path = FEATURES / f"mp001_cortical_{suffix}.npy"
            elif source == "balanced_cortical_raw":
                suffix = ATTEMPT3_LAGS[lag]
                path = FEATURES / f"mp001_attempt3_cortical_raw_{suffix}.npy"
            else:
                suffix = ATTEMPT3_LAGS[lag]
                path = FEATURES / f"mp001_attempt3_cortical_centered_{suffix}.npy"
            inputs.append({
                "source": source,
                "lag": lag,
                "path": str(path),
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "sha256": sha256_file(path),
            })
    payload = {
        "created_utc": utc_now(),
        "paired_original_latent_cortical": paired,
        "raw_name_clarification": (
            "Attempt 3 raw arrays already subtract the mean predictions at onset-2 s and onset-1 s; "
            "raw means before across-stimulus context centering."
        ),
        "fixation_subtraction_isolatable": False,
        "inputs": inputs,
    }
    atomic_json(ART / "input_inventory.json", payload)
    print(ART / "input_inventory.json")


def best_raw_alignment(array: np.ndarray, categories: np.ndarray, labels: list[str], dino_distances: list[np.ndarray]) -> dict[str, Any]:
    source_distance = distance_matrix(array)
    rows = []
    for label, dino_distance in zip(labels, dino_distances, strict=True):
        rows.append({
            "dino": label,
            "partial_rsa": partial_rsa(source_distance, dino_distance, categories),
            "neighborhood_k3": neighborhood_overlap(source_distance, dino_distance, 3),
        })
    return {
        "best_partial_rsa": max(rows, key=lambda row: row["partial_rsa"]),
        "best_neighborhood_k3": max(rows, key=lambda row: row["neighborhood_k3"]),
    }


def command_stage(_: argparse.Namespace) -> None:
    sources = load_sources()
    categories = mp002.category_array()
    dino_labels, dino = dino_arrays()
    dino_distances = [distance_matrix(array) for array in dino]
    rows = []
    context_rows = []
    methods = ["context_0", "context_1", "context_2", "context_3", "mean", "median", "concat"]
    for source, lags in sources.items():
        if source == "balanced_cortical_centered":
            # Translation-equivalence is tested explicitly below; avoid duplicate heavy rows.
            continue
        for lag, contexts in lags.items():
            context_rows.append({
                "source": source,
                "lag": lag,
                "context_rdm_reliability": context_rdm_reliability(contexts),
                "identity_reliability": identity_reliability(contexts),
                "cross_context_signal_spectrum": cross_context_signal_spectrum(contexts),
                "per_context_effective_rank": [spectral_metrics(context)["effective_rank"] for context in contexts],
            })
            for method in methods:
                array = aggregate_contexts(contexts, method)
                rows.append({
                    "source": source,
                    "lag": lag,
                    "aggregation": method,
                    "feature_count": int(array.shape[1]),
                    "spectrum": spectral_metrics(array),
                    "raw_alignment": best_raw_alignment(array, categories, dino_labels, dino_distances),
                })

    equivalence = []
    for lag in ATTEMPT3_LAGS:
        raw = sources["balanced_cortical_raw"][lag]
        centered = sources["balanced_cortical_centered"][lag]
        raw_rank = spectral_metrics(raw.mean(axis=0))["effective_rank"]
        centered_rank = spectral_metrics(centered.mean(axis=0))["effective_rank"]
        raw_distance = distance_matrix(raw.mean(axis=0))
        centered_distance = distance_matrix(centered.mean(axis=0))
        equivalence.append({
            "lag": lag,
            "mean_effective_rank_raw": raw_rank,
            "mean_effective_rank_centered": centered_rank,
            "rank_absolute_difference": abs(raw_rank - centered_rank),
            "centered_cosine_rdm_max_absolute_difference": float(np.max(np.abs(raw_distance - centered_distance))),
        })

    lookup = {(r["source"], r["lag"], r["aggregation"]): r for r in rows}
    decoder = []
    protocol = []
    aggregation = []
    for lag in LAGS:
        for method in methods:
            latent = lookup[("original_latent", lag, method)]["spectrum"]["effective_rank"]
            cortical = lookup[("original_cortical", lag, method)]["spectrum"]["effective_rank"]
            decoder.append({
                "lag": lag,
                "aggregation": method,
                "latent_effective_rank": latent,
                "cortical_effective_rank": cortical,
                "absolute_change": cortical - latent,
                "retained_fraction": cortical / latent,
            })
            balanced = lookup[("balanced_cortical_raw", lag, method)]["spectrum"]["effective_rank"]
            protocol.append({
                "lag": lag,
                "aggregation": method,
                "original_cortical_effective_rank": cortical,
                "balanced_cortical_effective_rank": balanced,
                "absolute_change": balanced - cortical,
                "ratio": balanced / cortical,
            })
        for source in ("original_latent", "original_cortical", "balanced_cortical_raw"):
            individual = [lookup[(source, lag, f"context_{i}")]["spectrum"]["effective_rank"] for i in range(4)]
            mean_rank = lookup[(source, lag, "mean")]["spectrum"]["effective_rank"]
            concat_rank = lookup[(source, lag, "concat")]["spectrum"]["effective_rank"]
            aggregation.append({
                "source": source,
                "lag": lag,
                "median_individual_context_effective_rank": float(np.median(individual)),
                "mean_aggregation_effective_rank": mean_rank,
                "concat_aggregation_effective_rank": concat_rank,
                "mean_retained_fraction_vs_median_context": mean_rank / float(np.median(individual)),
                "concat_ratio_vs_median_context": concat_rank / float(np.median(individual)),
            })

    payload = {
        "created_utc": utc_now(),
        "rows": rows,
        "context_audit": context_rows,
        "paired_decoder_attenuation": decoder,
        "protocol_contrast": protocol,
        "aggregation_attenuation": aggregation,
        "attempt3_centering_equivalence": equivalence,
        "boundaries": {
            "decoder_contrast_is_paired": True,
            "protocol_contrast_changes_onset_schedule": True,
            "fixation_subtraction_isolated": False,
            "raw_alignment_is_descriptive_not_gate_rerun": True,
        },
    }
    atomic_json(ART / "stage_attenuation.json", payload)
    atomic_json(ART / "context_signal_audit.json", {"created_utc": utc_now(), "rows": context_rows})
    print(ART / "stage_attenuation.json")


def standardize_train_test(train_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, ddof=1, keepdims=True)
    valid = std[0] > 1e-8
    if np.count_nonzero(valid) < 2:
        raise RuntimeError("fewer than two nonconstant predictors")
    return (train_x[:, valid] - mean[:, valid]) / std[:, valid], (test_x[:, valid] - mean[:, valid]) / std[:, valid]


def kernel_ridge_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, alpha: float) -> np.ndarray:
    train_z, test_z = standardize_train_test(train_x, test_x)
    scale = float(train_z.shape[1])
    kernel = (train_z @ train_z.T) / scale
    cross = (test_z @ train_z.T) / scale
    coefficient = np.linalg.solve(kernel + float(alpha) * np.eye(kernel.shape[0]), train_y)
    return cross @ coefficient


def normalize_targets(train_y: np.ndarray, test_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train_y.mean(axis=0, keepdims=True)
    train = l2_rows(train_y - mean)
    test = l2_rows(test_y - mean)
    return train, test


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
    return {
        "top1": float(np.mean(ranks <= 1)),
        "top5": float(np.mean(ranks <= min(5, len(target)))),
        "median_rank": float(np.median(ranks)),
        "paired_cosine": float(np.mean(diagonal)),
        "pairwise_identification": float(np.mean(pairwise)),
        "predicted_target_rdm_spearman": float(spearmanr(upper(pred_distance), upper(true_distance)).statistic),
    }


def choose_hyperparameters(
    x: np.ndarray,
    targets: list[np.ndarray],
    categories: np.ndarray,
    outer_train: np.ndarray,
    candidate_indices: list[int],
) -> dict[str, Any]:
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=OUTER_SEED + int(outer_train.sum()))
    best = None
    for target_index in candidate_indices:
        scores = {float(alpha): [] for alpha in ALPHAS}
        y = targets[target_index]
        for inner_train_local, inner_valid_local in inner.split(outer_train, categories[outer_train]):
            train = outer_train[inner_train_local]
            valid = outer_train[inner_valid_local]
            train_y, valid_y = normalize_targets(y[train], y[valid])
            for alpha in ALPHAS:
                prediction = kernel_ridge_predict(x[train], train_y, x[valid], float(alpha))
                metric = retrieval_metrics(prediction, valid_y)
                scores[float(alpha)].append(metric)
        for alpha, folds in scores.items():
            row = {
                "target_index": target_index,
                "alpha": alpha,
                "top1": float(np.mean([m["top1"] for m in folds])),
                "pairwise_identification": float(np.mean([m["pairwise_identification"] for m in folds])),
                "paired_cosine": float(np.mean([m["paired_cosine"] for m in folds])),
            }
            key = (row["top1"], row["pairwise_identification"], row["paired_cosine"], -np.log10(alpha))
            if best is None or key > best[0]:
                best = (key, row)
    assert best is not None
    return best[1]


def outer_splits(categories: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    splits = []
    for repeat in range(4):
        cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=OUTER_SEED + repeat)
        splits.extend((train, test) for train, test in cv.split(np.arange(len(categories)), categories))
    return splits


def evaluate_readout(
    name: str,
    x: np.ndarray,
    targets: list[np.ndarray],
    target_labels: list[str],
    categories: np.ndarray,
    candidate_indices: list[int],
) -> dict[str, Any]:
    folds = []
    for fold_id, (train, test) in enumerate(outer_splits(categories)):
        selected = choose_hyperparameters(x, targets, categories, train, candidate_indices)
        target_index = int(selected["target_index"])
        y = targets[target_index]
        train_y, test_y = normalize_targets(y[train], y[test])
        prediction = kernel_ridge_predict(x[train], train_y, x[test], float(selected["alpha"]))
        metrics = retrieval_metrics(prediction, test_y)

        # Category-centroid baseline measures how much retrieval can be achieved without image identity.
        centroid_prediction = np.stack([
            train_y[categories[train] == category].mean(axis=0)
            for category in categories[test]
        ])
        centroid = retrieval_metrics(centroid_prediction, test_y)
        folds.append({
            "fold_id": fold_id,
            "train": train.tolist(),
            "test": test.tolist(),
            "selected_target": target_labels[target_index],
            "selected_alpha": float(selected["alpha"]),
            "inner_selection": selected,
            "metrics": metrics,
            "category_centroid_baseline": centroid,
        })
    summary = {}
    for metric in folds[0]["metrics"]:
        values = np.asarray([fold["metrics"][metric] for fold in folds], dtype=np.float64)
        baseline = np.asarray([fold["category_centroid_baseline"][metric] for fold in folds], dtype=np.float64)
        summary[metric] = {
            "median": float(np.median(values)),
            "mean": float(np.mean(values)),
            "p05": float(np.percentile(values, 5)),
            "category_centroid_mean": float(np.mean(baseline)),
            "mean_delta_over_category_centroid": float(np.mean(values - baseline)),
        }
    return {"name": name, "shape": list(x.shape), "summary": summary, "folds": folds}


def command_readout(_: argparse.Namespace) -> None:
    sources = load_sources()
    categories = mp002.category_array()
    labels, targets = dino_arrays()
    # Fixed, literature- and prior-audit-motivated candidates. Selection is nested within each outer fold.
    candidate_labels = [
        "cls_block_06", "cls_block_07", "cls_block_08", "cls_block_09", "cls_block_11",
        "patch_mean_block_06", "patch_mean_block_07", "patch_mean_block_08", "patch_mean_block_09", "patch_mean_block_11",
    ]
    candidate_indices = [labels.index(label) for label in candidate_labels]
    representations = {}
    for lag in LAGS:
        representations[f"original_latent_{lag}_mean"] = aggregate_contexts(sources["original_latent"][lag], "mean")
        representations[f"original_cortical_{lag}_mean"] = aggregate_contexts(sources["original_cortical"][lag], "mean")
        representations[f"balanced_cortical_{lag}_mean"] = aggregate_contexts(sources["balanced_cortical_raw"][lag], "mean")
    representations["balanced_cortical_plus4_median"] = aggregate_contexts(sources["balanced_cortical_raw"]["plus4"], "median")
    representations["balanced_cortical_plus4_concat"] = aggregate_contexts(sources["balanced_cortical_raw"]["plus4"], "concat")

    rows = []
    for name, array in representations.items():
        print(f"readout {name}", flush=True)
        rows.append(evaluate_readout(name, array, targets, labels, categories, candidate_indices))
        atomic_json(RESULTS / f"readout_{name}.json", rows[-1])
    payload = {
        "created_utc": utc_now(),
        "method": {
            "outer_cv": "4 repeats x category-stratified 4-fold",
            "inner_cv": "category-stratified 3-fold",
            "mapping": "dual-form ridge after train-only per-feature standardization",
            "alphas": ALPHAS.tolist(),
            "target_selection": "nested among predeclared DINO CLS and patch-mean blocks 6-9 and 11",
            "evaluation": "held-out retrieval, cosine, pairwise identification, and predicted-target RDM",
            "baseline": "train-fold DINO category centroid",
        },
        "rows": rows,
        "claim_boundary": (
            "A positive learned readout diagnoses recoverable cross-geometry signal. It is not raw isometry, "
            "does not change MP-002, and does not construct a third geometry."
        ),
    }
    atomic_json(ART / "learned_readout_audit.json", payload)
    print(ART / "learned_readout_audit.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("inventory").set_defaults(function=command_inventory)
    sub.add_parser("stage").set_defaults(function=command_stage)
    sub.add_parser("readout").set_defaults(function=command_readout)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
