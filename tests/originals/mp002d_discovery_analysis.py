#!/usr/bin/env python3
"""Sharded, confirmation-safe analysis of MP-002D discovery features.

The analysis boundary is deliberately separate from feature extraction:

    freeze -> inputs -> run-node/run-task -> verify-task -> assemble

Only ordered discovery identities 0:768 are readable.  This runner reports
discovery evidence but cannot write the Gate-1 specification or access the
sealed confirmation cohort.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

import mp002d_core as core


VERSION = "discovery_analysis_v1"
CONFIG_PATH = core.PROJECT / "mp002d_discovery_analysis_config.json"
ART_ROOT = core.ART / VERSION
RESULT_ROOT = core.RESULTS / VERSION
SHARD_ROOT = RESULT_ROOT / "shards"
EXECUTION_ROOT = ART_ROOT / "task_executions"
DETERMINISM_ROOT = ART_ROOT / "determinism"

SPEC_PATH = ART_ROOT / "MP002D_DISCOVERY_ANALYSIS_SPEC.json"
SPEC_SHA_PATH = ART_ROOT / "MP002D_DISCOVERY_ANALYSIS_SPEC.sha256"
SPEC_DONE = ART_ROOT / "MP002D_DISCOVERY_ANALYSIS_SPEC.done"
TASK_MANIFEST_PATH = ART_ROOT / "task_manifest.json"
INPUT_RECEIPT_PATH = ART_ROOT / "analysis_inputs.json"
INPUT_RECEIPT_SHA_PATH = ART_ROOT / "analysis_inputs.sha256"
INPUTS_DONE = ART_ROOT / "MP002D_DISCOVERY_ANALYSIS_INPUTS.done"
ANALYSIS_DONE = ART_ROOT / "MP002D_DISCOVERY_ANALYSIS.done"

DISCOVERY_ART = core.ART / "discovery_v1"
DISCOVERY_SPEC = DISCOVERY_ART / "MP002D_DISCOVERY_EXECUTION_SPEC.json"
DISCOVERY_SPEC_SHA = DISCOVERY_ART / "MP002D_DISCOVERY_EXECUTION_SPEC.sha256"
DISCOVERY_SPEC_DONE = DISCOVERY_ART / "MP002D_DISCOVERY_EXECUTION_SPEC.done"
DISCOVERY_VALIDATION = DISCOVERY_ART / "discovery_validation.json"
DISCOVERY_FEATURES_DONE = DISCOVERY_ART / "MP002D_DISCOVERY_FEATURES.done"
PILOT_ART = core.ART / "pilot_v1"
PILOT_EVALUATION = PILOT_ART / "pilot_evaluation.json"
PILOT_GATE_DONE = PILOT_ART / "MP002D_PILOT_GATE.done"
PILOT_SELECTION_SPEC = PILOT_ART / "MP002D_PILOT_SELECTION_SPEC.json"
PILOT_SELECTION_SHA = PILOT_ART / "MP002D_PILOT_SELECTION_SPEC.sha256"


def ensure_dirs() -> None:
    for path in (ART_ROOT, RESULT_ROOT, SHARD_ROOT, EXECUTION_ROOT, DETERMINISM_ROOT):
        path.mkdir(parents=True, exist_ok=True)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def artifact_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "sha256": core.sha256_file(path)}


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    core.atomic_json(path, value)


def assert_confirmation_closed() -> None:
    forbidden = (
        core.ART / "MP002D_GATE1_SPEC.json",
        core.ART / "MP002D_GATE1_SPEC.sha256",
        core.ART / "MP002D_GATE1_SPEC.done",
    )
    if any(path.exists() for path in forbidden):
        raise RuntimeError(
            "discovery analysis refuses to run after the confirmatory firewall is opened"
        )


def load_config() -> dict[str, Any]:
    config = core.load_json(CONFIG_PATH)
    required_exact = {
        "schema_version": "mp002d-discovery-analysis-v1",
        "prefixes": [64, 128, 256, 512, 768],
        "outer_folds": 6,
        "inner_folds": 3,
        "outer_split_seed": 26020731,
        "permutations_per_mode": 199,
        "outer_retrieval_candidates": 16,
        "inner_retrieval_candidates": 13,
        "neighborhood_k_micro": 5,
        "neighborhood_k_mesoscopic_rule": "max(5,floor(sqrt(n)+0.5))",
        "primary_pls_prefixes": [512, 768],
        "sensitivity_prefix": 768,
    }
    for key, expected in required_exact.items():
        if config.get(key) != expected:
            raise ValueError(f"analysis config {key}={config.get(key)!r} != {expected!r}")
    if config["primary_unit"] != {
        "unit_id": "lag4_patch7",
        "tribe_lag_s": 4,
        "dino_family": "patch_mean",
        "dino_block": 7,
    }:
        raise ValueError("primary unit differs from the prospective MP-002D contract")
    expected_sensitivities = {
        "lag5_patch7",
        "lag4_patch11",
        "lag4_cls7",
        "lag4_cls11",
    }
    actual_sensitivities = {row["unit_id"] for row in config["sensitivity_units"]}
    if actual_sensitivities != expected_sensitivities:
        raise ValueError("sensitivity unit set differs from the prospective contract")
    if int(config["bootstrap_resamples"]) < 1000:
        raise ValueError("discovery uncertainty requires at least 1,000 bootstrap resamples")
    return config


def build_tasks(config: dict[str, Any]) -> list[dict[str, Any]]:
    primary = dict(config["primary_unit"])
    tasks: list[dict[str, Any]] = []
    for prefix in config["prefixes"]:
        for kind in ("geometry", "ridge"):
            tasks.append(
                {
                    "task_id": f"primary_{kind}_n{int(prefix):03d}",
                    "role": "primary",
                    "kind": kind,
                    "prefix": int(prefix),
                    "unit": primary,
                }
            )
        if int(prefix) in set(map(int, config["primary_pls_prefixes"])):
            tasks.append(
                {
                    "task_id": f"primary_pls_n{int(prefix):03d}",
                    "role": "primary",
                    "kind": "pls",
                    "prefix": int(prefix),
                    "unit": primary,
                }
            )
    sensitivity_prefix = int(config["sensitivity_prefix"])
    for unit in config["sensitivity_units"]:
        for kind in ("geometry", "ridge"):
            tasks.append(
                {
                    "task_id": f"sensitivity_{unit['unit_id']}_{kind}_n{sensitivity_prefix:03d}",
                    "role": "sensitivity",
                    "kind": kind,
                    "prefix": sensitivity_prefix,
                    "unit": dict(unit),
                }
            )
    tasks.append(
        {
            "task_id": "context_pilot",
            "role": "protocol_qualification",
            "kind": "context",
            "prefix": 64,
            "unit": primary,
        }
    )
    identifiers = [row["task_id"] for row in tasks]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("analysis task IDs are not unique")
    return sorted(tasks, key=lambda row: row["task_id"])


def read_recorded_sha(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.read_text(encoding="utf-8").strip().split()[0]


def require_discovery_spec() -> dict[str, Any]:
    if not (DISCOVERY_SPEC.is_file() and DISCOVERY_SPEC_SHA.is_file() and DISCOVERY_SPEC_DONE.is_file()):
        raise RuntimeError("selected-protocol discovery execution specification is absent")
    actual = core.sha256_file(DISCOVERY_SPEC)
    if actual != read_recorded_sha(DISCOVERY_SPEC_SHA):
        raise RuntimeError("selected-protocol discovery execution specification changed")
    value = core.load_json(DISCOVERY_SPEC)
    scope = value.get("scope", {})
    if (
        value.get("status") != "FROZEN_BEFORE_EXPANDED_OUTCOMES"
        or int(scope.get("start_index", -1)) != 0
        or int(scope.get("stop_index_exclusive", -1)) != core.DISCOVERY_IMAGES
        or scope.get("confirmation_access") != "FORBIDDEN"
    ):
        raise RuntimeError("selected-protocol discovery scope is invalid")
    return value


def oracle_paths() -> dict[str, Path]:
    root = core.ROOT / "project/oracle"
    return {
        "gate1_estimators": root / "gate1_estimators.py",
        "gate1_estimators_v04": root / "gate1_estimators_v04.py",
        "orc_v05_config": root / "configs/orc_1_develop_v05_tranche_a.json",
    }


def command_freeze(_: argparse.Namespace) -> None:
    ensure_dirs()
    assert_confirmation_closed()
    config = load_config()
    discovery_spec = require_discovery_spec()
    if SPEC_PATH.exists() or SPEC_SHA_PATH.exists() or SPEC_DONE.exists():
        require_spec()
        print(SPEC_PATH)
        return
    if not (PILOT_SELECTION_SPEC.is_file() and PILOT_SELECTION_SHA.is_file()):
        raise RuntimeError("paired-pilot selection specification is absent")
    if core.sha256_file(PILOT_SELECTION_SPEC) != read_recorded_sha(PILOT_SELECTION_SHA):
        raise RuntimeError("paired-pilot selection specification changed")
    selected = discovery_spec["selected_protocol"]
    if (
        int(selected["duration_s"]) != 1
        or int(selected["primary_lag_s"]) != 4
        or int(selected["sensitivity_lag_s"]) != 5
        or int(selected["batch_size"]) != 24
    ):
        raise RuntimeError("selected protocol differs from the passed paired-pilot decision")

    tasks = build_tasks(config)
    task_manifest = {
        "schema_version": config["schema_version"],
        "run_label": config["run_label"],
        "config_sha256": core.sha256_file(CONFIG_PATH),
        "confirmation_access": "FORBIDDEN",
        "tasks": tasks,
    }
    write_json(TASK_MANIFEST_PATH, task_manifest)
    code = {
        "analysis": artifact_record(Path(__file__).resolve()),
        "analysis_config": artifact_record(CONFIG_PATH),
        "mp002d_core": artifact_record(core.PROJECT / "mp002d_core.py"),
    }
    for name, path in oracle_paths().items():
        code[name] = artifact_record(path)
    payload = {
        "created_utc": core.utc_now(),
        "status": "FROZEN_BEFORE_DISCOVERY_GEOMETRY_OUTCOMES",
        "version": VERSION,
        "scope": {
            "partition": "discovery",
            "start_index": 0,
            "stop_index_exclusive": core.DISCOVERY_IMAGES,
            "prefixes": list(map(int, config["prefixes"])),
            "confirmation_access": "FORBIDDEN",
        },
        "selected_protocol": selected,
        "analysis_contract": {
            "outer_folds": int(config["outer_folds"]),
            "inner_folds": int(config["inner_folds"]),
            "outer_candidates": int(config["outer_retrieval_candidates"]),
            "inner_candidates": int(config["inner_retrieval_candidates"]),
            "primary_unit": config["primary_unit"],
            "sensitivity_units": config["sensitivity_units"],
            "pls_role": "diagnostic_sensitivity",
            "multiplicity_and_practical_thresholds": "deferred_to_MP002D_GATE1_SPEC",
        },
        "code_and_config": code,
        "prospective_contract": artifact_record(core.PROJECT / "MP002D_PROSPECTIVE_CONTRACT.md"),
        "discovery_execution_spec": artifact_record(DISCOVERY_SPEC),
        "pilot_selection_spec": artifact_record(PILOT_SELECTION_SPEC),
        "stimulus_manifest": artifact_record(core.MANIFEST),
        "task_manifest": artifact_record(TASK_MANIFEST_PATH),
        "task_count": len(tasks),
        "orc_v05_mapping": config["orc_v05_mapping"],
        "claim_boundary": config["claim_boundary"],
        "confirmatory_firewall": "CLOSED",
    }
    write_json(SPEC_PATH, payload)
    digest = core.sha256_file(SPEC_PATH)
    atomic_text(SPEC_SHA_PATH, f"{digest}  {SPEC_PATH.name}\n")
    atomic_text(SPEC_DONE, core.utc_now() + "\n")
    print(SPEC_PATH)


def command_preflight(_: argparse.Namespace) -> None:
    ensure_dirs()
    assert_confirmation_closed()
    config = load_config()
    discovery_spec = require_discovery_spec()
    tasks = build_tasks(config)
    dependencies = {
        name: artifact_record(path) for name, path in oracle_paths().items()
    }
    if not (PILOT_SELECTION_SPEC.is_file() and PILOT_SELECTION_SHA.is_file()):
        raise RuntimeError("paired-pilot selection specification is absent")
    if core.sha256_file(PILOT_SELECTION_SPEC) != read_recorded_sha(PILOT_SELECTION_SHA):
        raise RuntimeError("paired-pilot selection specification changed")
    print(
        json.dumps(
            {
                "status": "ANALYSIS_PREFLIGHT_PASS",
                "task_count": len(tasks),
                "outer_split_seed": config["outer_split_seed"],
                "discovery_execution_spec_sha256": core.sha256_file(DISCOVERY_SPEC),
                "selected_protocol": discovery_spec["selected_protocol"],
                "dependencies": dependencies,
                "confirmatory_firewall": "CLOSED",
            },
            indent=2,
            sort_keys=True,
        )
    )


def require_spec() -> dict[str, Any]:
    if not (SPEC_PATH.is_file() and SPEC_SHA_PATH.is_file() and SPEC_DONE.is_file()):
        raise RuntimeError("discovery-analysis specification is not frozen")
    actual = core.sha256_file(SPEC_PATH)
    if actual != read_recorded_sha(SPEC_SHA_PATH):
        raise RuntimeError("discovery-analysis specification checksum mismatch")
    spec = core.load_json(SPEC_PATH)
    if spec.get("status") != "FROZEN_BEFORE_DISCOVERY_GEOMETRY_OUTCOMES":
        raise RuntimeError("discovery-analysis specification has an invalid state")
    for record in spec["code_and_config"].values():
        if core.sha256_file(Path(record["path"])) != record["sha256"]:
            raise RuntimeError(f"frozen analysis dependency changed: {record['path']}")
    for name in (
        "prospective_contract",
        "discovery_execution_spec",
        "pilot_selection_spec",
        "stimulus_manifest",
        "task_manifest",
    ):
        record = spec[name]
        if core.sha256_file(Path(record["path"])) != record["sha256"]:
            raise RuntimeError(f"frozen analysis input changed: {name}")
    assert_confirmation_closed()
    return spec


def validate_array(path: Path, expected_shape: tuple[int, ...]) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = np.load(path, mmap_mode="r", allow_pickle=False)
    if value.shape != expected_shape or value.dtype != np.float32:
        raise RuntimeError(f"invalid feature array {path}: shape={value.shape} dtype={value.dtype}")
    if not np.isfinite(value).all():
        raise RuntimeError(f"non-finite feature array: {path}")
    return {
        "path": str(path),
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": core.sha256_file(path),
    }


def discovery_rows() -> list[dict[str, Any]]:
    manifest = core.load_json(core.MANIFEST)
    rows = sorted(
        [
            row
            for row in manifest["rows"]
            if 0 <= int(row["mp002d_index"]) < core.DISCOVERY_IMAGES
        ],
        key=lambda row: int(row["mp002d_index"]),
    )
    if [int(row["mp002d_index"]) for row in rows] != list(range(core.DISCOVERY_IMAGES)):
        raise RuntimeError("analysis cohort is not exact discovery prefix 0:768")
    if any(row["partition"] != "discovery" for row in rows):
        raise RuntimeError("confirmatory identity entered the analysis cohort")
    return rows


def command_inputs(_: argparse.Namespace) -> None:
    ensure_dirs()
    spec = require_spec()
    if INPUTS_DONE.is_file() or INPUT_RECEIPT_PATH.exists() or INPUT_RECEIPT_SHA_PATH.exists():
        require_inputs()
        print(INPUT_RECEIPT_PATH)
        return
    if not (DISCOVERY_VALIDATION.is_file() and DISCOVERY_FEATURES_DONE.is_file()):
        raise RuntimeError("analysis inputs remain blocked until discovery feature validation passes")
    validation = core.load_json(DISCOVERY_VALIDATION)
    if not validation.get("passed") or validation.get("confirmatory_firewall") != "CLOSED":
        raise RuntimeError("discovery feature validation did not pass with a closed firewall")
    if validation.get("prefixes") != [64, 128, 256, 512, 768]:
        raise RuntimeError("discovery validation prefixes changed")
    if validation.get("execution_spec_sha256") != spec["discovery_execution_spec"]["sha256"]:
        raise RuntimeError("discovery validation refers to another execution specification")

    cortical_records = {}
    for lag in (4, 5):
        record = validation["cortical"][f"lag{lag}"]
        actual = validate_array(
            Path(record["path"]), (core.DISCOVERY_IMAGES, core.EXPECTED_VERTICES)
        )
        if actual["sha256"] != record["sha256"]:
            raise RuntimeError(f"validated cortical lag {lag} checksum changed")
        cortical_records[f"lag{lag}"] = actual
    dino_records = {}
    for family in ("cls", "patch_mean"):
        record = validation["dino"][family]
        actual = validate_array(
            Path(record["path"]),
            (core.DISCOVERY_IMAGES, core.EXPECTED_DINO_BLOCKS, core.EXPECTED_DINO_DIM),
        )
        if actual["sha256"] != record["sha256"]:
            raise RuntimeError(f"validated DINO {family} checksum changed")
        dino_records[family] = actual

    if not (PILOT_EVALUATION.is_file() and PILOT_GATE_DONE.is_file()):
        raise RuntimeError("paired-pilot qualification artifacts are absent")
    pilot_evaluation = core.load_json(PILOT_EVALUATION)
    decision = pilot_evaluation["decision"]
    if (
        not decision.get("passed")
        or int(decision["selected_duration_s"]) != 1
        or int(decision["selected_lag_s"]) != 4
    ):
        raise RuntimeError("paired-pilot qualification no longer selects 1 s / +4 s")
    pilot_contexts = {}
    for lag in (4, 5):
        path = core.TRIBE_FEATURES / f"pilot_v1/cortical_d1_lag{lag}.npy"
        pilot_contexts[f"lag{lag}"] = validate_array(
            path, (2, 64, core.EXPECTED_VERTICES)
        )

    rows = discovery_rows()
    order = [
        {"mp002d_index": int(row["mp002d_index"]), "stimulus_id": row["stimulus_id"]}
        for row in rows
    ]
    categories = [row["source_category"] for row in rows]
    receipt = {
        "created_utc": core.utc_now(),
        "version": VERSION,
        "analysis_spec_sha256": core.sha256_file(SPEC_PATH),
        "discovery_validation": artifact_record(DISCOVERY_VALIDATION),
        "pilot_evaluation": artifact_record(PILOT_EVALUATION),
        "cortical": cortical_records,
        "dino": dino_records,
        "pilot_contexts": pilot_contexts,
        "stimulus_count": len(rows),
        "stimulus_order_sha256": sha256_bytes(canonical_bytes(order)),
        "category_counts": dict(Counter(categories)),
        "category_order_sha256": sha256_bytes(canonical_bytes(categories)),
        "confirmation_access": "FORBIDDEN",
        "confirmatory_firewall": "CLOSED",
    }
    write_json(INPUT_RECEIPT_PATH, receipt)
    digest = core.sha256_file(INPUT_RECEIPT_PATH)
    atomic_text(INPUT_RECEIPT_SHA_PATH, f"{digest}  {INPUT_RECEIPT_PATH.name}\n")
    atomic_text(INPUTS_DONE, core.utc_now() + "\n")
    print(INPUT_RECEIPT_PATH)


def require_inputs() -> dict[str, Any]:
    require_spec()
    if not (
        INPUT_RECEIPT_PATH.is_file()
        and INPUT_RECEIPT_SHA_PATH.is_file()
        and INPUTS_DONE.is_file()
    ):
        raise RuntimeError("validated discovery-analysis input receipt is absent")
    actual = core.sha256_file(INPUT_RECEIPT_PATH)
    if actual != read_recorded_sha(INPUT_RECEIPT_SHA_PATH):
        raise RuntimeError("discovery-analysis input receipt checksum mismatch")
    receipt = core.load_json(INPUT_RECEIPT_PATH)
    if (
        receipt.get("analysis_spec_sha256") != core.sha256_file(SPEC_PATH)
        or receipt.get("confirmation_access") != "FORBIDDEN"
        or receipt.get("confirmatory_firewall") != "CLOSED"
    ):
        raise RuntimeError("discovery-analysis input receipt has an invalid boundary")
    for group in ("cortical", "dino", "pilot_contexts"):
        for record in receipt[group].values():
            if core.sha256_file(Path(record["path"])) != record["sha256"]:
                raise RuntimeError(f"analysis feature changed: {record['path']}")
    assert_confirmation_closed()
    return receipt


def oracle_modules():
    root = core.ROOT / "project/oracle"
    sys.path.insert(0, str(root))
    import gate1_estimators as gate
    import gate1_estimators_v04 as local_scale

    return gate, local_scale


def category_array(prefix: int, receipt: dict[str, Any]) -> np.ndarray:
    if prefix not in (64, 128, 256, 512, 768):
        raise ValueError(f"forbidden analysis prefix: {prefix}")
    rows = discovery_rows()
    categories_text = [row["source_category"] for row in rows]
    if sha256_bytes(canonical_bytes(categories_text)) != receipt["category_order_sha256"]:
        raise RuntimeError("discovery category ordering changed")
    mapping = {name: index for index, name in enumerate(core.SOURCE_ORDER)}
    return np.asarray([mapping[name] for name in categories_text[:prefix]], dtype=np.int64)


def load_unit(
    task: dict[str, Any], receipt: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prefix = int(task["prefix"])
    if not 1 <= prefix <= core.DISCOVERY_IMAGES:
        raise RuntimeError("task attempted to leave discovery indices 0:768")
    unit = task["unit"]
    lag = int(unit["tribe_lag_s"])
    family = str(unit["dino_family"])
    block = int(unit["dino_block"])
    source_all = np.load(receipt["cortical"][f"lag{lag}"]["path"], mmap_mode="r")
    target_all = np.load(receipt["dino"][family]["path"], mmap_mode="r")
    source = np.asarray(source_all[:prefix], dtype=np.float64)
    target = np.asarray(target_all[:prefix, block], dtype=np.float64)
    categories = category_array(prefix, receipt)
    if (
        source.shape != (prefix, core.EXPECTED_VERTICES)
        or target.shape != (prefix, core.EXPECTED_DINO_DIM)
        or not np.isfinite(source).all()
        or not np.isfinite(target).all()
    ):
        raise RuntimeError("analysis task received an invalid exact unit")
    return source, target, categories


def derived_seed(master_seed: int, *parts: Any) -> int:
    payload = canonical_bytes([int(master_seed), *parts])
    return int(sha256_bytes(payload)[:16], 16) % (2**63 - 1)


def geometry_result(
    source: np.ndarray,
    target: np.ndarray,
    categories: np.ndarray,
    config: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    gate, local_scale = oracle_modules()
    outer_seed = int(config["outer_split_seed"])
    splits = gate.stratified_splits(categories, int(config["outer_folds"]), outer_seed)
    raw = gate.raw_geometry_summary(
        source,
        target,
        categories,
        outer_splits=splits,
        primary_k=int(config["neighborhood_k_micro"]),
        sensitivity_k=config["neighborhood_k_diagnostic_sensitivity"],
        permutations=int(config["permutations_per_mode"]),
        seed=derived_seed(seed, "raw"),
    )
    local = local_scale.local_scale_summary(
        source,
        target,
        categories,
        outer_folds=int(config["outer_folds"]),
        micro_k=int(config["neighborhood_k_micro"]),
        permutations=int(config["permutations_per_mode"]),
        seed=outer_seed,
    )
    return {
        "raw": raw,
        "declared_local_scales": local,
        "spectrum": {
            "tribe_cortical": gate.spectral_metrics(source),
            "dino_target": gate.spectral_metrics(target),
        },
        "redundancy": gate.bidirectional_reconstruction(source, target, splits),
        "outer_fold_sizes": [int(len(test)) for _, test in splits],
        "seeds": {
            "outer": outer_seed,
            "raw": int(derived_seed(seed, "raw")),
            "declared_local_scales": outer_seed,
        },
    }


def query_top1_records(
    prediction: np.ndarray,
    target_gallery: np.ndarray,
    query_indices: np.ndarray,
    categories: np.ndarray,
    *,
    candidate_count: int,
    seed: int,
) -> list[dict[str, int]]:
    gate, _ = oracle_modules()
    prediction = gate.l2_rows(prediction)
    target_gallery = gate.l2_rows(target_gallery)
    all_indices = np.arange(len(categories), dtype=np.int64)
    output = []
    for row_index, query in enumerate(query_indices):
        query_value = int(query)
        pool = all_indices[
            (categories[all_indices] == categories[query_value]) & (all_indices != query_value)
        ]
        query_seed = int((seed + 0x9E3779B97F4A7C15 * (query_value + 1)) % (2**64))
        rng = np.random.default_rng(query_seed)
        distractors = rng.choice(pool, size=candidate_count - 1, replace=False)
        candidates = np.concatenate([[query_value], np.sort(distractors)]).astype(np.int64)
        similarities = prediction[row_index] @ target_gallery[candidates].T
        rank = 1 + int(np.count_nonzero(similarities[1:] >= float(similarities[0])))
        output.append(
            {
                "query_index": query_value,
                "category": int(categories[query_value]),
                "rank": rank,
                "top1": int(rank == 1),
            }
        )
    return output


def stratified_bootstrap_top1(
    records: list[dict[str, int]], config: dict[str, Any], seed: int
) -> dict[str, Any]:
    values = np.asarray([row["top1"] for row in records], dtype=np.float64)
    categories = np.asarray([row["category"] for row in records], dtype=np.int64)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(int(config["bootstrap_resamples"])):
        selected = []
        for category in np.unique(categories):
            indices = np.flatnonzero(categories == category)
            selected.extend(rng.choice(indices, size=len(indices), replace=True).tolist())
        draws.append(float(values[np.asarray(selected, dtype=np.int64)].mean()))
    alpha = 1.0 - float(config["bootstrap_confidence"])
    array = np.asarray(draws, dtype=np.float64)
    return {
        "method": "stratified stimulus bootstrap over fixed per-query candidate sets",
        "resamples": int(config["bootstrap_resamples"]),
        "confidence": float(config["bootstrap_confidence"]),
        "lower": float(np.percentile(array, 100.0 * alpha / 2.0)),
        "upper": float(np.percentile(array, 100.0 * (1.0 - alpha / 2.0))),
        "median": float(np.median(array)),
        "seed": int(seed),
    }


def nested_readout_result(
    source: np.ndarray,
    target: np.ndarray,
    categories: np.ndarray,
    config: dict[str, Any],
    seed: int,
    family: str,
) -> dict[str, Any]:
    gate, _ = oracle_modules()
    outer_seed = int(config["outer_split_seed"])
    splits = gate.stratified_splits(categories, int(config["outer_folds"]), outer_seed)
    all_indices = np.arange(len(categories), dtype=np.int64)
    folds: list[dict[str, float]] = []
    centroid_folds: list[dict[str, float]] = []
    states: list[tuple[np.ndarray, np.ndarray, np.ndarray, int]] = []
    selections: list[dict[str, Any]] = []
    query_records: list[dict[str, int]] = []
    inner_candidates = int(config["inner_retrieval_candidates"])
    outer_candidates = int(config["outer_retrieval_candidates"])

    for fold, (train, test) in enumerate(splits):
        smallest = min(
            int(np.count_nonzero(categories[train] == category))
            for category in np.unique(categories)
        )
        if smallest < inner_candidates:
            raise RuntimeError(
                f"outer-training gallery supports {smallest}, not {inner_candidates}, candidates"
            )
        fold_seed = outer_seed + 104729 * (fold + 1) + (1 if family == "pls" else 0)
        if family == "ridge":
            choice = gate.choose_ridge_alpha(
                source,
                target,
                categories,
                train,
                alphas=config["ridge_alphas"],
                candidate_count=inner_candidates,
                seed=fold_seed,
            )
            prediction, gallery = gate.ridge_prediction(
                source, target, train, test, float(choice["alpha"])
            )
            selected_value = {"ridge_alpha": float(choice["alpha"])}
        elif family == "pls":
            choice = gate.choose_pls_components(
                source,
                target,
                categories,
                train,
                components_grid=config["pls_components"],
                candidate_count=inner_candidates,
                seed=fold_seed,
            )
            prediction, gallery = gate.pls_prediction(
                source, target, train, test, int(choice["components"])
            )
            selected_value = {"pls_components": int(choice["components"])}
        else:
            raise ValueError(family)
        metrics = gate.retrieval_metrics(
            prediction,
            gallery,
            test,
            test,
            categories,
            candidate_count=outer_candidates,
            seed=fold_seed,
            gallery_indices=all_indices,
        )
        folds.append(metrics)
        records = query_top1_records(
            prediction,
            gallery,
            test,
            categories,
            candidate_count=outer_candidates,
            seed=fold_seed,
        )
        if abs(float(np.mean([row["top1"] for row in records])) - metrics["top1"]) > 1e-12:
            raise RuntimeError("query-level Top-1 audit differs from frozen retrieval estimator")
        query_records.extend(records)
        states.append((test, prediction, gallery, fold_seed))

        centroid = np.stack(
            [
                gallery[train][categories[train] == category].mean(axis=0)
                for category in categories[test]
            ]
        )
        centroid_folds.append(
            gate.retrieval_metrics(
                centroid,
                gallery,
                test,
                test,
                categories,
                candidate_count=outer_candidates,
                seed=fold_seed,
                gallery_indices=all_indices,
            )
        )
        selections.append(
            {
                "fold": fold,
                **selected_value,
                "inner_top1": float(choice["top1"]),
                "inner_candidate_count": inner_candidates,
                "outer_candidate_count": outer_candidates,
                "seed": int(fold_seed),
            }
        )

    summary = gate.summarize_metric_folds(folds)
    centroid_summary = gate.summarize_metric_folds(centroid_folds)
    rng = np.random.default_rng(derived_seed(seed, family, "category-null"))
    null_top1 = []
    for permutation_index in range(int(config["permutations_per_mode"])):
        fold_values = []
        for fold, (test, prediction, gallery, fold_seed) in enumerate(states):
            local_permutation = gate.permute_indices(
                rng, categories[test], "category_preserving"
            )
            correct = test[local_permutation]
            metric = gate.retrieval_metrics(
                prediction,
                gallery,
                test,
                correct,
                categories,
                candidate_count=outer_candidates,
                seed=derived_seed(seed, family, "null", permutation_index, fold, fold_seed),
                gallery_indices=all_indices,
            )
            fold_values.append(metric["top1"])
        null_top1.append(float(np.mean(fold_values)))
    observed = float(summary["top1"]["mean"])
    null = np.asarray(null_top1, dtype=np.float64)
    summary["top1_category_null_p"] = {"value": gate.fwer_p(observed, null)}
    summary["top1_delta_vs_category_null"] = {
        "value": observed - float(np.median(null))
    }
    summary["top1_category_null_median"] = {"value": float(np.median(null))}
    summary["top1_category_null_p95"] = {"value": float(np.percentile(null, 95))}
    summary["top1_delta_vs_centroid"] = {
        "value": observed - float(centroid_summary["top1"]["mean"])
    }
    bootstrap_seed = derived_seed(seed, family, "stimulus-bootstrap")
    return {
        "family": family,
        "summary": summary,
        "category_centroid": centroid_summary,
        "fold_metrics": folds,
        "selections": selections,
        "query_top1_records": sorted(query_records, key=lambda row: row["query_index"]),
        "top1_bootstrap": stratified_bootstrap_top1(
            query_records, config, bootstrap_seed
        ),
        "category_null_top1": null_top1,
        "outer_seed": outer_seed,
    }


def context_result(receipt: dict[str, Any]) -> dict[str, Any]:
    gate, _ = oracle_modules()
    values = {}
    for lag in (4, 5):
        array = np.load(receipt["pilot_contexts"][f"lag{lag}"]["path"], allow_pickle=False)
        values[f"lag{lag}"] = gate.context_stability(array.astype(np.float64))
    return {
        "duration_s": 1,
        "contexts": 2,
        "images": 64,
        "primary_lag_s": 4,
        "primary": values["lag4"],
        "sensitivity_lag_s": 5,
        "sensitivity": values["lag5"],
        "interpretation": (
            "protocol qualification carried from the paired pilot; production discovery "
            "has one context and is not duplicated into a synthetic context axis"
        ),
    }


def task_by_id(task_id: str) -> dict[str, Any]:
    spec = require_spec()
    manifest = core.load_json(Path(spec["task_manifest"]["path"]))
    matches = [row for row in manifest["tasks"] if row["task_id"] == task_id]
    if len(matches) != 1:
        raise ValueError(f"unknown analysis task: {task_id}")
    return matches[0]


def task_paths(task_id: str) -> tuple[Path, Path, Path]:
    return (
        SHARD_ROOT / f"{task_id}.json",
        SHARD_ROOT / f"{task_id}.sha256",
        SHARD_ROOT / f"{task_id}.done",
    )


def valid_task_result(task_id: str) -> bool:
    result_path, digest_path, marker = task_paths(task_id)
    if not (result_path.is_file() and digest_path.is_file() and marker.is_file()):
        return False
    if core.sha256_file(result_path) != read_recorded_sha(digest_path):
        return False
    payload = core.load_json(result_path)
    return bool(
        payload.get("task", {}).get("task_id") == task_id
        and payload.get("analysis_spec_sha256") == core.sha256_file(SPEC_PATH)
        and payload.get("input_receipt_sha256") == core.sha256_file(INPUT_RECEIPT_PATH)
        and payload.get("confirmation_access") == "FORBIDDEN"
    )


def compute_task_payload(task: dict[str, Any], receipt: dict[str, Any]) -> dict[str, Any]:
    config = load_config()
    seed = derived_seed(int(config["master_seed"]), task["task_id"])
    if task["kind"] == "context":
        metrics = context_result(receipt)
    else:
        source, target, categories = load_unit(task, receipt)
        if task["kind"] == "geometry":
            metrics = geometry_result(source, target, categories, config, seed)
        elif task["kind"] in ("ridge", "pls"):
            metrics = nested_readout_result(
                source, target, categories, config, seed, str(task["kind"])
            )
        else:
            raise ValueError(f"unknown task kind: {task['kind']}")
    return {
        "schema_version": config["schema_version"],
        "run_label": config["run_label"],
        "analysis_spec_sha256": core.sha256_file(SPEC_PATH),
        "input_receipt_sha256": core.sha256_file(INPUT_RECEIPT_PATH),
        "task": task,
        "task_seed": int(seed),
        "metrics": metrics,
        "confirmation_access": "FORBIDDEN",
        "claim_boundary": config["claim_boundary"],
    }


def run_task(task_id: str) -> Path:
    ensure_dirs()
    receipt = require_inputs()
    task = task_by_id(task_id)
    result_path, digest_path, marker = task_paths(task_id)
    if valid_task_result(task_id):
        print(f"SKIP {task_id}")
        return result_path
    if result_path.exists() or digest_path.exists() or marker.exists():
        raise RuntimeError(f"incomplete or corrupt existing task output: {task_id}")
    payload = compute_task_payload(task, receipt)
    write_json(result_path, payload)
    digest = core.sha256_file(result_path)
    atomic_text(digest_path, f"{digest}  {result_path.name}\n")
    atomic_text(marker, core.utc_now() + "\n")
    write_json(
        EXECUTION_ROOT / f"{task_id}.json",
        {
            "created_utc": core.utc_now(),
            "task_id": task_id,
            "result_sha256": digest,
            "hostname": platform.node(),
            "pid": os.getpid(),
            "thread_environment": {
                name: os.environ.get(name)
                for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
            },
        },
    )
    print(f"COMPLETE {task_id} {digest}")
    return result_path


def numeric_max_difference(left: Any, right: Any, path: str = "root") -> float:
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            raise RuntimeError(f"determinism key mismatch at {path}")
        return max(
            (numeric_max_difference(left[key], right[key], f"{path}.{key}") for key in left),
            default=0.0,
        )
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            raise RuntimeError(f"determinism length mismatch at {path}")
        return max(
            (numeric_max_difference(a, b, f"{path}[{index}]") for index, (a, b) in enumerate(zip(left, right, strict=True))),
            default=0.0,
        )
    if isinstance(left, bool) or isinstance(right, bool):
        if left != right:
            raise RuntimeError(f"determinism boolean mismatch at {path}")
        return 0.0
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right))
    if left != right:
        raise RuntimeError(f"determinism value mismatch at {path}: {left!r} != {right!r}")
    return 0.0


def command_run_task(args: argparse.Namespace) -> None:
    run_task(args.task_id)


def command_verify_task(args: argparse.Namespace) -> None:
    task_id = args.task_id
    config = load_config()
    if task_id not in config["determinism_tasks"]:
        raise ValueError(f"task is not in the frozen determinism set: {task_id}")
    if not valid_task_result(task_id):
        raise RuntimeError(f"determinism source result is absent: {task_id}")
    receipt = require_inputs()
    first = core.load_json(task_paths(task_id)[0])
    second = compute_task_payload(task_by_id(task_id), receipt)
    maximum = numeric_max_difference(first, second)
    passed = bool(maximum <= 1e-10)
    output = {
        "created_utc": core.utc_now(),
        "task_id": task_id,
        "passed": passed,
        "maximum_absolute_numeric_difference": maximum,
        "tolerance": 1e-10,
        "first_sha256": core.sha256_file(task_paths(task_id)[0]),
        "rerun_canonical_sha256": sha256_bytes(canonical_bytes(second)),
        "hostname": platform.node(),
    }
    path = DETERMINISM_ROOT / f"{task_id}.json"
    write_json(path, output)
    if not passed:
        raise RuntimeError(f"analysis determinism failed: {output}")
    atomic_text(DETERMINISM_ROOT / f"{task_id}.done", core.utc_now() + "\n")
    print(path)


def command_environment(args: argparse.Namespace) -> None:
    ensure_dirs()
    require_spec()
    node_index = int(args.node_index)
    node_count = int(args.node_count)
    if not 0 <= node_index < node_count:
        raise ValueError("invalid node index/count")
    commands = {
        "lscpu": ["lscpu"],
        "free": ["free", "-h"],
        "df_workspace": ["df", "-h", "/workspace"],
    }
    outputs = {}
    for name, command in commands.items():
        outputs[name] = subprocess.run(
            command, check=True, text=True, capture_output=True
        ).stdout
    import scipy
    import sklearn

    payload = {
        "created_utc": core.utc_now(),
        "node_index": node_index,
        "node_count": node_count,
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "executable": sys.executable,
        "cpu_count": os.cpu_count(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "thread_environment": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
        "outputs": outputs,
    }
    path = ART_ROOT / f"environment_node_{node_index:02d}_of_{node_count:02d}.json"
    write_json(path, payload)
    print(path)


def command_run_node(args: argparse.Namespace) -> None:
    node_index = int(args.node_index)
    node_count = int(args.node_count)
    if not 0 <= node_index < node_count:
        raise ValueError("invalid node index/count")
    require_inputs()
    tasks = core.load_json(TASK_MANIFEST_PATH)["tasks"]
    assigned = [row for index, row in enumerate(tasks) if index % node_count == node_index]
    print(f"NODE {node_index}/{node_count}: {len(assigned)} tasks")
    for task in assigned:
        run_task(task["task_id"])


def load_completed(task_id: str) -> dict[str, Any]:
    if not valid_task_result(task_id):
        raise RuntimeError(f"required analysis task is incomplete: {task_id}")
    return core.load_json(task_paths(task_id)[0])


def compact_prefix(prefix: int) -> dict[str, Any]:
    geometry = load_completed(f"primary_geometry_n{prefix:03d}")["metrics"]
    ridge = load_completed(f"primary_ridge_n{prefix:03d}")["metrics"]
    result = {
        "n": prefix,
        "partial_rsa": geometry["raw"]["partial_rsa"],
        "partial_rsa_unrestricted_p": geometry["raw"]["unrestricted_partial_p"],
        "partial_rsa_category_p": geometry["raw"]["category_partial_p"],
        "positive_fold_fraction": geometry["raw"]["positive_fold_fraction"],
        "micro_k": geometry["declared_local_scales"]["micro"]["k"],
        "micro_delta": geometry["declared_local_scales"]["micro"]["delta_vs_category_null"],
        "micro_p": geometry["declared_local_scales"]["micro"]["category_p"],
        "mesoscopic_k": geometry["declared_local_scales"]["mesoscopic"]["k"],
        "mesoscopic_delta": geometry["declared_local_scales"]["mesoscopic"]["delta_vs_category_null"],
        "mesoscopic_p": geometry["declared_local_scales"]["mesoscopic"]["category_p"],
        "ridge_top1": ridge["summary"]["top1"]["mean"],
        "ridge_top1_ci_lower": ridge["top1_bootstrap"]["lower"],
        "ridge_top1_ci_upper": ridge["top1_bootstrap"]["upper"],
        "ridge_top1_null_delta": ridge["summary"]["top1_delta_vs_category_null"]["value"],
        "ridge_top1_null_p": ridge["summary"]["top1_category_null_p"]["value"],
        "ridge_top1_centroid_delta": ridge["summary"]["top1_delta_vs_centroid"]["value"],
        "tribe_effective_rank": geometry["spectrum"]["tribe_cortical"]["effective_rank"],
        "dino_effective_rank": geometry["spectrum"]["dino_target"]["effective_rank"],
        "bidirectional_min_r2": geometry["redundancy"]["bidirectional_min_r2"],
    }
    pls_id = f"primary_pls_n{prefix:03d}"
    if valid_task_result(pls_id):
        pls = load_completed(pls_id)["metrics"]
        result.update(
            {
                "pls_top1": pls["summary"]["top1"]["mean"],
                "pls_top1_ci_lower": pls["top1_bootstrap"]["lower"],
                "pls_top1_ci_upper": pls["top1_bootstrap"]["upper"],
                "pls_top1_null_delta": pls["summary"]["top1_delta_vs_category_null"]["value"],
                "pls_top1_null_p": pls["summary"]["top1_category_null_p"]["value"],
            }
        )
    return result


def report_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# MP-002D Discovery Analysis",
        "",
        f"Status: **{summary['status']}**",
        "",
        "This report contains discovery evidence only. The 256-image confirmatory cohort "
        "remains sealed, and this report does not make a Gate-1 decision.",
        "",
        "## Primary discovery curve",
        "",
        "| n | Partial RSA | Category p | Micro delta | Micro p | Meso k | Meso delta | Meso p | Ridge Top-1 | 95% bootstrap CI | Null delta | Null p |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["primary_prefixes"]:
        lines.append(
            "| {n} | {partial_rsa:.4f} | {partial_rsa_category_p:.4f} | "
            "{micro_delta:.4f} | {micro_p:.4f} | {mesoscopic_k} | "
            "{mesoscopic_delta:.4f} | {mesoscopic_p:.4f} | {ridge_top1:.4f} | "
            "[{ridge_top1_ci_lower:.4f}, {ridge_top1_ci_upper:.4f}] | "
            "{ridge_top1_null_delta:.4f} | {ridge_top1_null_p:.4f} |".format(**row)
        )
    context = summary["context_qualification"]["primary"]
    lines.extend(
        [
            "",
            "## Protocol qualification carried from the paired pilot",
            "",
            f"- RDM reliability: `{context['rdm_reliability']:.6f}`",
            f"- Median within-image identity correlation: `{context['identity_within_median']:.6f}`",
            f"- Fifth-percentile within-image correlation: `{context['identity_within_p05']:.6f}`",
            f"- Identity margin over between-image median: `{context['identity_margin']:.6f}`",
            "",
            "## Boundary and next decision",
            "",
            "The PLS and alternate-unit outputs are diagnostics. Practical-effect floors, "
            "multiplicity, the adjacent-prefix stability rule, the exact context metric, and "
            "the ORC-V05-to-real-data semantic mapping must be written and hashed in "
            "`MP002D_GATE1_SPEC.json` before any confirmatory feature is computed or scored.",
            "",
        ]
    )
    return "\n".join(lines)


def command_assemble(_: argparse.Namespace) -> None:
    ensure_dirs()
    spec = require_spec()
    receipt = require_inputs()
    config = load_config()
    tasks = core.load_json(TASK_MANIFEST_PATH)["tasks"]
    for task in tasks:
        load_completed(task["task_id"])
    for task_id in config["determinism_tasks"]:
        path = DETERMINISM_ROOT / f"{task_id}.json"
        marker = DETERMINISM_ROOT / f"{task_id}.done"
        if not (path.is_file() and marker.is_file() and core.load_json(path).get("passed")):
            raise RuntimeError(f"required deterministic rerun is incomplete: {task_id}")

    prefixes = [compact_prefix(int(prefix)) for prefix in config["prefixes"]]
    sensitivity = {}
    for unit in config["sensitivity_units"]:
        unit_id = unit["unit_id"]
        sensitivity[unit_id] = {
            "geometry": load_completed(
                f"sensitivity_{unit_id}_geometry_n{int(config['sensitivity_prefix']):03d}"
            )["metrics"],
            "ridge": load_completed(
                f"sensitivity_{unit_id}_ridge_n{int(config['sensitivity_prefix']):03d}"
            )["metrics"],
        }
    context = load_completed("context_pilot")["metrics"]
    adjacent = []
    for left, right in zip(prefixes, prefixes[1:], strict=True):
        adjacent.append(
            {
                "from_n": left["n"],
                "to_n": right["n"],
                "ridge_top1_improvement": right["ridge_top1"] - left["ridge_top1"],
                "individual_bootstrap_intervals_overlap": not (
                    right["ridge_top1_ci_lower"] > left["ridge_top1_ci_upper"]
                    or left["ridge_top1_ci_lower"] > right["ridge_top1_ci_upper"]
                ),
                "role": "descriptive_only_until_MP002D_GATE1_SPEC_freezes_the_adjacent_rule",
            }
        )
    summary = {
        "created_utc": core.utc_now(),
        "status": "DISCOVERY_EVIDENCE_COMPLETE_GATE1_UNDECIDED",
        "version": VERSION,
        "analysis_spec_sha256": core.sha256_file(SPEC_PATH),
        "input_receipt_sha256": core.sha256_file(INPUT_RECEIPT_PATH),
        "task_manifest_sha256": spec["task_manifest"]["sha256"],
        "primary_unit": config["primary_unit"],
        "primary_prefixes": prefixes,
        "adjacent_prefix_diagnostics": adjacent,
        "context_qualification": context,
        "sensitivity_units": sensitivity,
        "determinism": {
            task_id: core.load_json(DETERMINISM_ROOT / f"{task_id}.json")
            for task_id in config["determinism_tasks"]
        },
        "orc_v05_mapping": config["orc_v05_mapping"],
        "multiplicity": config["multiplicity"],
        "confirmatory_firewall": "CLOSED",
        "claim_boundary": config["claim_boundary"],
    }
    output = ART_ROOT / "discovery_analysis_summary.json"
    report = ART_ROOT / "MP002D_DISCOVERY_ANALYSIS_REPORT.md"
    write_json(output, summary)
    atomic_text(report, report_markdown(summary))
    checksums = [
        f"{core.sha256_file(path)}  {path.name}"
        for path in (SPEC_PATH, TASK_MANIFEST_PATH, INPUT_RECEIPT_PATH, output, report)
    ]
    atomic_text(ART_ROOT / "checksums.sha256", "\n".join(checksums) + "\n")
    atomic_text(ANALYSIS_DONE, core.utc_now() + "\n")
    print(report)


def command_status(_: argparse.Namespace) -> None:
    ensure_dirs()
    tasks = core.load_json(TASK_MANIFEST_PATH)["tasks"] if TASK_MANIFEST_PATH.is_file() else []
    completed = [row["task_id"] for row in tasks if valid_task_result(row["task_id"])] if INPUT_RECEIPT_PATH.is_file() else []
    config = load_config()
    determinism = {
        task_id: (DETERMINISM_ROOT / f"{task_id}.done").is_file()
        for task_id in config["determinism_tasks"]
    }
    print(
        json.dumps(
            {
                "created_utc": core.utc_now(),
                "version": VERSION,
                "spec_frozen": SPEC_DONE.is_file(),
                "inputs_validated": INPUTS_DONE.is_file(),
                "completed_tasks": len(completed),
                "total_tasks": len(tasks),
                "remaining_task_ids": [
                    row["task_id"] for row in tasks if row["task_id"] not in set(completed)
                ],
                "determinism": determinism,
                "analysis_complete": ANALYSIS_DONE.is_file(),
                "confirmatory_firewall": "CLOSED",
            },
            indent=2,
            sort_keys=True,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MP-002D discovery CPU analysis")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight")
    sub.add_parser("freeze")
    sub.add_parser("inputs")
    run_task_parser = sub.add_parser("run-task")
    run_task_parser.add_argument("--task-id", required=True)
    verify_parser = sub.add_parser("verify-task")
    verify_parser.add_argument("--task-id", required=True)
    environment_parser = sub.add_parser("environment")
    environment_parser.add_argument("--node-index", type=int, default=0)
    environment_parser.add_argument("--node-count", type=int, default=1)
    node_parser = sub.add_parser("run-node")
    node_parser.add_argument("--node-index", type=int, required=True)
    node_parser.add_argument("--node-count", type=int, required=True)
    sub.add_parser("assemble")
    sub.add_parser("status")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    commands = {
        "preflight": command_preflight,
        "freeze": command_freeze,
        "inputs": command_inputs,
        "run-task": command_run_task,
        "verify-task": command_verify_task,
        "environment": command_environment,
        "run-node": command_run_node,
        "assemble": command_assemble,
        "status": command_status,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
