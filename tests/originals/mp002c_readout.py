#!/usr/bin/env python3
"""Optimized learned-readout command for MP-002C.

Kept separate because the Windows workspace sandbox prevented an in-place edit
of the already-added core audit file.
"""

from __future__ import annotations

import argparse

import numpy as np
from sklearn.model_selection import StratifiedKFold

import mp002c_core as core


def kernel_design(train_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train_z, test_z = core.standardize_train_test(train_x, test_x)
    scale = float(train_z.shape[1])
    return (train_z @ train_z.T) / scale, (test_z @ train_z.T) / scale


def predict_design(kernel: np.ndarray, cross: np.ndarray, train_y: np.ndarray, alpha: float) -> np.ndarray:
    coefficient = np.linalg.solve(kernel + float(alpha) * np.eye(kernel.shape[0]), train_y)
    return cross @ coefficient


def choose(
    x: np.ndarray,
    targets: list[np.ndarray],
    categories: np.ndarray,
    outer_train: np.ndarray,
    candidate_indices: list[int],
) -> dict:
    inner = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=core.OUTER_SEED + int(outer_train.sum()),
    )
    scores = {
        (target_index, float(alpha)): []
        for target_index in candidate_indices
        for alpha in core.ALPHAS
    }
    for inner_train_local, inner_valid_local in inner.split(outer_train, categories[outer_train]):
        train = outer_train[inner_train_local]
        valid = outer_train[inner_valid_local]
        kernel, cross = kernel_design(x[train], x[valid])
        for target_index in candidate_indices:
            y = targets[target_index]
            train_y, valid_y = core.normalize_targets(y[train], y[valid])
            for alpha in core.ALPHAS:
                prediction = predict_design(kernel, cross, train_y, float(alpha))
                scores[(target_index, float(alpha))].append(core.retrieval_metrics(prediction, valid_y))
    best = None
    for (target_index, alpha), folds in scores.items():
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


def splits(categories: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    output = []
    for repeat in range(2):
        cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=core.OUTER_SEED + repeat)
        output.extend((train, test) for train, test in cv.split(np.arange(len(categories)), categories))
    return output


def evaluate(
    name: str,
    x: np.ndarray,
    targets: list[np.ndarray],
    target_labels: list[str],
    categories: np.ndarray,
    candidate_indices: list[int],
) -> dict:
    folds = []
    for fold_id, (train, test) in enumerate(splits(categories)):
        selected = choose(x, targets, categories, train, candidate_indices)
        target_index = int(selected["target_index"])
        y = targets[target_index]
        train_y, test_y = core.normalize_targets(y[train], y[test])
        kernel, cross = kernel_design(x[train], x[test])
        prediction = predict_design(kernel, cross, train_y, float(selected["alpha"]))
        metrics = core.retrieval_metrics(prediction, test_y)
        centroid_prediction = np.stack([
            train_y[categories[train] == category].mean(axis=0)
            for category in categories[test]
        ])
        folds.append({
            "fold_id": fold_id,
            "train": train.tolist(),
            "test": test.tolist(),
            "selected_target": target_labels[target_index],
            "selected_alpha": float(selected["alpha"]),
            "inner_selection": selected,
            "metrics": metrics,
            "category_centroid_baseline": core.retrieval_metrics(centroid_prediction, test_y),
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


def command_readout() -> None:
    sources = core.load_sources()
    categories = core.mp002.category_array()
    labels, targets = core.dino_arrays()
    candidate_labels = [
        "cls_block_07",
        "cls_block_11",
        "patch_mean_block_07",
        "patch_mean_block_11",
    ]
    candidate_indices = [labels.index(label) for label in candidate_labels]
    representations = {}
    for lag in core.LAGS:
        representations[f"original_latent_{lag}_mean"] = core.aggregate_contexts(
            sources["original_latent"][lag], "mean"
        )
        representations[f"original_cortical_{lag}_mean"] = core.aggregate_contexts(
            sources["original_cortical"][lag], "mean"
        )
        representations[f"balanced_cortical_{lag}_mean"] = core.aggregate_contexts(
            sources["balanced_cortical_raw"][lag], "mean"
        )
    representations["balanced_cortical_plus4_median"] = core.aggregate_contexts(
        sources["balanced_cortical_raw"]["plus4"], "median"
    )
    representations["balanced_cortical_plus4_concat"] = core.aggregate_contexts(
        sources["balanced_cortical_raw"]["plus4"], "concat"
    )
    rows = []
    for name, array in representations.items():
        print(f"readout {name}", flush=True)
        row = evaluate(name, array, targets, labels, categories, candidate_indices)
        rows.append(row)
        core.atomic_json(core.RESULTS / f"readout_{name}.json", row)
    payload = {
        "created_utc": core.utc_now(),
        "method": {
            "outer_cv": "2 repeats x category-stratified 4-fold",
            "inner_cv": "category-stratified 3-fold",
            "mapping": "dual-form ridge after train-only per-feature standardization",
            "alphas": core.ALPHAS.tolist(),
            "target_selection": "nested among DINO CLS/patch blocks 7 and 11",
            "evaluation": "held-out retrieval, cosine, pairwise identification, and predicted-target RDM",
            "baseline": "train-fold DINO category centroid",
        },
        "rows": rows,
        "claim_boundary": (
            "A positive learned readout diagnoses recoverable cross-geometry signal. It is not raw isometry, "
            "does not change MP-002, and does not construct a third geometry."
        ),
    }
    core.atomic_json(core.ART / "learned_readout_audit.json", payload)
    print(core.ART / "learned_readout_audit.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.parse_args()
    command_readout()
