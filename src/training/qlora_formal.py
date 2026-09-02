"""Stage C5 Candidate 01 formal QLoRA preparation and manual-run orchestration."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import yaml

from src.evaluation.contracts import validate_output
from src.training.qlora_smoke import (
    MODEL_ID,
    MODEL_REVISION,
    PROMPT_PATH,
    TARGET_KEYS,
    read_jsonl,
    sha256_file,
    validate_adapter_artifacts,
    validate_chat_rows,
    validate_reload_artifact,
    write_json,
    write_jsonl,
)


SEED = 42
PROMPT_SHA256 = "6b84135769b7348758e8cc21a3cb168465e00de5efaf59ff8a8459087db3dc3b"
TRAIN_SOURCE = "data/processed/train.jsonl"
VALID_SOURCE = "data/processed/validation.jsonl"
TRAINING_TRAIN = "data/training/train.jsonl"
TRAINING_VALID = "data/training/valid.jsonl"
CONFIG_PATH = "configs/qlora_candidate_01.yaml"
ARTIFACT_DIR = "artifacts/stage5"
CANDIDATE_DIR = "artifacts/stage5/candidate_01"
ADAPTER_PATH = "artifacts/stage5/candidate_01/adapter"
TRAIN_RE = re.compile(
    r"Iter (?P<iteration>\d+): Train loss (?P<loss>[-+\w.]+), Learning Rate (?P<lr>[-+\w.]+), "
    r"It/sec (?P<it_sec>[-+\w.]+), Tokens/sec (?P<tokens_sec>[-+\w.]+), "
    r"Trained Tokens (?P<trained_tokens>\d+), Peak mem (?P<peak_memory>[-+\w.]+) GB"
)
VAL_RE = re.compile(r"Iter (?P<iteration>\d+): Val loss (?P<loss>[-+\w.]+), Val took (?P<seconds>[-+\w.]+)s")


def percentile(values: Sequence[int], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def summarize_lengths(lengths: Sequence[int]) -> Dict[str, Any]:
    if not lengths:
        raise ValueError("Sequence audit requires non-empty lengths")
    over_1024 = sum(length > 1024 for length in lengths)
    over_1536 = sum(length > 1536 for length in lengths)
    count = len(lengths)
    return {
        "count": count,
        "min": min(lengths),
        "mean": round(statistics.fmean(lengths), 6),
        "median": round(statistics.median(lengths), 6),
        "p90": round(percentile(lengths, 0.90), 6),
        "p95": round(percentile(lengths, 0.95), 6),
        "p99": round(percentile(lengths, 0.99), 6),
        "max": max(lengths),
        "count_over_1024": over_1024,
        "percentage_over_1024": round(100.0 * over_1024 / count, 6),
        "count_over_1536": over_1536,
        "percentage_over_1536": round(100.0 * over_1536 / count, 6),
    }


def strict_target(record: Mapping[str, Any], policy: Mapping[str, bool]) -> str:
    target = record["target"]
    if policy.get(target["intent"]) is not target["needs_human"]:
        raise ValueError(f"C2 escalation mismatch for source_index={record['metadata']['source_index']}")
    ordered = {
        "intent": target["intent"],
        "category": target["category"],
        "needs_human": target["needs_human"],
        "response": target["response"],
    }
    serialized = json.dumps(ordered, ensure_ascii=False, separators=(",", ":"))
    if not validate_output(serialized).valid:
        raise ValueError(f"C2 target validation failed for source_index={record['metadata']['source_index']}")
    return serialized


def convert_records(
    records: Sequence[Mapping[str, Any]], prompt: str, policy: Mapping[str, bool]
) -> List[Dict[str, Any]]:
    return [
        {
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": record["instruction"]},
                {"role": "assistant", "content": strict_target(record, policy)},
            ]
        }
        for record in records
    ]


def load_config(repo_root: Path) -> Dict[str, Any]:
    config = yaml.safe_load((repo_root / CONFIG_PATH).read_text(encoding="utf-8"))
    expected = {
        "model": MODEL_ID,
        "train": True,
        "fine_tune_type": "lora",
        "data": "data/training",
        "seed": 42,
        "num_layers": 16,
        "batch_size": 1,
        "grad_accumulation_steps": 2,
        "iters": 1350,
        "learning_rate": 1e-5,
        "steps_per_report": 10,
        "steps_per_eval": 150,
        "val_batches": 50,
        "adapter_path": ADAPTER_PATH,
        "save_every": 150,
        "test": False,
        "grad_checkpoint": False,
        "mask_prompt": True,
    }
    for key, expected_value in expected.items():
        if config.get(key) != expected_value:
            raise ValueError(f"Formal config {key!r} must be {expected_value!r}; found {config.get(key)!r}")
    if config.get("max_seq_length") not in (1024, 1536):
        raise ValueError("Formal max_seq_length must be audit-selected 1024 or 1536")
    lora = config.get("lora_parameters", {})
    if tuple(lora.get("keys", ())) != TARGET_KEYS:
        raise ValueError("Formal config does not contain exactly all seven approved target keys")
    if (lora.get("rank"), lora.get("scale"), lora.get("dropout")) != (8, 16.0, 0.0):
        raise ValueError("Formal LoRA parameters must be rank=8, scale=16.0, dropout=0.0")
    if "alpha" in lora:
        raise ValueError("MLX-LM formal config must use scale, not alpha")
    return config


def prepare(repo_root: Path) -> Dict[str, Any]:
    prompt_path = repo_root / PROMPT_PATH
    if sha256_file(prompt_path) != PROMPT_SHA256:
        raise ValueError("Frozen C4 prompt SHA-256 changed")
    prompt = prompt_path.read_text(encoding="utf-8").strip()
    hash_manifest = json.loads((repo_root / "data/manifests/dataset_hashes.json").read_text(encoding="utf-8"))
    policy_payload = json.loads((repo_root / "configs/escalation_policy.json").read_text(encoding="utf-8"))
    policy = {entry["intent"]: entry["needs_human"] for entry in policy_payload["intents"]}

    train_hash = sha256_file(repo_root / TRAIN_SOURCE)
    valid_hash = sha256_file(repo_root / VALID_SOURCE)
    if train_hash != hash_manifest["files"]["train"]["sha256"]:
        raise ValueError("Frozen Train SHA-256 mismatch")
    if valid_hash != hash_manifest["files"]["validation"]["sha256"]:
        raise ValueError("Frozen Validation SHA-256 mismatch")
    train_records = read_jsonl(repo_root / TRAIN_SOURCE)
    valid_records = read_jsonl(repo_root / VALID_SOURCE)
    if len(train_records) != 2700 or len(valid_records) != 300:
        raise ValueError(f"Formal row counts must be 2700/300; found {len(train_records)}/{len(valid_records)}")
    train_indices = {int(row["metadata"]["source_index"]) for row in train_records}
    valid_indices = {int(row["metadata"]["source_index"]) for row in valid_records}
    if len(train_indices) != 2700 or len(valid_indices) != 300 or train_indices & valid_indices:
        raise ValueError("Formal Train/Validation source indices are not unique and disjoint")

    train_chat = convert_records(train_records, prompt, policy)
    valid_chat = convert_records(valid_records, prompt, policy)
    validate_chat_rows(train_chat, 2700)
    validate_chat_rows(valid_chat, 300)
    write_jsonl(repo_root / TRAINING_TRAIN, train_chat)
    write_jsonl(repo_root / TRAINING_VALID, valid_chat)

    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    snapshot = Path(snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_files_only=True))
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    train_lengths = [len(tokenizer.apply_chat_template(row["messages"], tokenize=True)) for row in train_chat]
    valid_lengths = [len(tokenizer.apply_chat_template(row["messages"], tokenize=True)) for row in valid_chat]
    train_summary = summarize_lengths(train_lengths)
    valid_summary = summarize_lengths(valid_lengths)
    recommended = 1536 if train_summary["percentage_over_1024"] > 1.0 else 1024
    audit = {
        "tokenizer_model_id": MODEL_ID,
        "tokenizer_revision": MODEL_REVISION,
        "chat_template_applied": True,
        "train": train_summary,
        "validation": valid_summary,
        "selection_rule": "Use 1536 only when more than 1% of Train examples exceed 1024 tokens; otherwise use 1024.",
        "recommended_max_seq_length": recommended,
    }
    write_json(repo_root / ARTIFACT_DIR / "sequence_length_summary.json", audit)
    config = load_config(repo_root)
    if config["max_seq_length"] != recommended:
        raise ValueError(
            f"Formal config max_seq_length={config['max_seq_length']} conflicts with audit recommendation={recommended}"
        )
    dataset_preflight = {
        "source_files_opened": [TRAIN_SOURCE, VALID_SOURCE],
        "disallowed_dataset_content_accessed": False,
        "provenance_rule": "All formal rows are direct transformations of the complete allowed frozen Train/Validation files; no other split content is opened.",
        "train": {"rows": 2700, "source_sha256": train_hash, "training_sha256": sha256_file(repo_root / TRAINING_TRAIN), "chat_valid": True, "schema_invalid_count": 0, "policy_mismatch_count": 0},
        "validation": {"rows": 300, "source_sha256": valid_hash, "training_sha256": sha256_file(repo_root / TRAINING_VALID), "chat_valid": True, "schema_invalid_count": 0, "policy_mismatch_count": 0},
        "source_index_overlap": 0,
        "prompt_sha256": PROMPT_SHA256,
        "config_sha256": sha256_file(repo_root / CONFIG_PATH),
        "max_seq_length": recommended,
    }
    write_json(repo_root / ARTIFACT_DIR / "dataset_preflight.json", dataset_preflight)
    return {"dataset_preflight": dataset_preflight, "sequence_length_audit": audit}


def model_preflight(repo_root: Path) -> Dict[str, Any]:
    config = load_config(repo_root)
    from huggingface_hub import snapshot_download
    from mlx.utils import tree_flatten
    from mlx_lm import load
    from mlx_lm.tuner.utils import linear_to_lora_layers
    from mlx_lm.utils import get_total_parameters

    snapshot = Path(snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_files_only=True))
    model, _, model_config = load(str(snapshot), adapter_path=None, return_config=True)
    quantization = model_config.get("quantization", {})
    if (quantization.get("bits"), quantization.get("group_size")) != (4, 64):
        raise ValueError(f"Unexpected base quantization: {quantization}")
    if len(model.layers) != 28:
        raise ValueError(f"Expected 28 transformer layers; found {len(model.layers)}")
    model.freeze()
    adapted_layer_indices = list(range(len(model.layers) - config["num_layers"], len(model.layers)))
    per_layer = []
    for layer_index in adapted_layer_indices:
        available = {name for name, _ in model.layers[layer_index].named_modules()}
        matched = [key for key in TARGET_KEYS if key in available]
        if tuple(matched) != TARGET_KEYS:
            raise ValueError(f"Target module mismatch in transformer layer {layer_index}: {matched}")
        per_layer.append({"layer_index": layer_index, "matched_keys": matched})
    linear_to_lora_layers(model, config["num_layers"], config["lora_parameters"], use_dora=False)
    trainable_parameters = model.trainable_parameters()
    trainable_count = sum(value.size for _, value in tree_flatten(trainable_parameters))
    total_count = get_total_parameters(model)
    names = sorted(name for name, _ in tree_flatten(trainable_parameters))
    if trainable_count <= 0 or len(names) != 16 * 7 * 2:
        raise ValueError(f"Unexpected formal trainable LoRA tensors: count={trainable_count}, tensors={len(names)}")
    payload = {
        "model_load_success": True,
        "repo_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "architecture": (model_config.get("architectures") or [model_config.get("model_type")])[0],
        "quantization": quantization,
        "model_layer_count": len(model.layers),
        "adapted_transformer_layers": 16,
        "adapted_layer_indices": adapted_layer_indices,
        "target_keys": list(TARGET_KEYS),
        "actual_matched_target_modules": sorted({key for row in per_layer for key in row["matched_keys"]}),
        "matched_module_instances": sum(len(row["matched_keys"]) for row in per_layer),
        "per_layer_matches": per_layer,
        "trainable_parameter_count": trainable_count,
        "total_parameter_count": total_count,
        "trainable_percentage": round(100.0 * trainable_count / total_count, 6),
        "trainable_tensor_count": len(names),
        "trainable_parameter_names": names,
    }
    write_json(repo_root / CANDIDATE_DIR / "model_preflight.json", payload)
    return payload


def parse_training_log(text: str) -> Dict[str, Any]:
    training = [
        {
            "iteration": int(match["iteration"]),
            "train_loss": float(match["loss"]),
            "learning_rate": float(match["lr"]),
            "iterations_per_second": float(match["it_sec"]),
            "tokens_per_second": float(match["tokens_sec"]),
            "trained_tokens": int(match["trained_tokens"]),
            "peak_memory_gb": float(match["peak_memory"]),
        }
        for match in TRAIN_RE.finditer(text)
    ]
    validation = [
        {"iteration": int(match["iteration"]), "validation_loss": float(match["loss"]), "validation_seconds": float(match["seconds"])}
        for match in VAL_RE.finditer(text)
    ]
    if not training or training[-1]["iteration"] != 1350 or len(training) != 135:
        raise ValueError(f"Formal training did not produce 135 reports through iteration 1350: {len(training)}")
    if not validation or validation[-1]["iteration"] != 1350:
        raise ValueError("Formal validation history does not reach iteration 1350")
    losses = [row["train_loss"] for row in training] + [row["validation_loss"] for row in validation]
    if not all(math.isfinite(value) for value in losses):
        raise ValueError("Formal training log contains NaN or Inf loss")
    return {
        "iterations_completed": 1350,
        "training_reports": training,
        "validation_reports": validation,
        "initial_training_loss": training[0]["train_loss"],
        "final_training_loss": training[-1]["train_loss"],
        "minimum_training_loss": min(row["train_loss"] for row in training),
        "final_validation_loss": validation[-1]["validation_loss"],
        "mean_tokens_per_second": round(statistics.fmean(row["tokens_per_second"] for row in training), 6),
        "mean_iterations_per_second": round(statistics.fmean(row["iterations_per_second"] for row in training), 6),
        "peak_memory_gb": max(row["peak_memory_gb"] for row in training),
        "finite_losses": True,
    }


def validate_training_manifest(payload: Mapping[str, Any]) -> None:
    required = {
        "model_id", "model_revision", "quantization", "train_rows", "validation_rows",
        "prompt_sha256", "config_sha256", "sequence_length_audit", "max_seq_length",
        "num_layers", "target_modules", "rank", "scale", "dropout", "mask_prompt",
        "learning_rate", "batch_size", "gradient_accumulation_steps", "effective_batch_size",
        "iterations", "adapter_path", "adapter_files", "checkpoint_files", "adapter_size_bytes",
        "training_success", "reload_success", "dataset_boundary", "behavioral_evaluation_performed",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"Formal training manifest missing keys: {sorted(missing)}")
    invariants = (
        payload["model_id"] == MODEL_ID,
        payload["model_revision"] == MODEL_REVISION,
        payload["quantization"] == {"bits": 4, "group_size": 64},
        (payload["train_rows"], payload["validation_rows"]) == (2700, 300),
        payload["prompt_sha256"] == PROMPT_SHA256,
        payload["max_seq_length"] in (1024, 1536),
        payload["num_layers"] == 16,
        tuple(payload["target_modules"]) == TARGET_KEYS,
        (payload["rank"], payload["scale"], payload["dropout"]) == (8, 16.0, 0.0),
        payload["mask_prompt"] is True,
        payload["learning_rate"] == 1e-5,
        (payload["batch_size"], payload["gradient_accumulation_steps"], payload["effective_batch_size"]) == (1, 2, 2),
        payload["iterations"] == 1350,
        payload["adapter_path"] == ADAPTER_PATH,
        payload["adapter_size_bytes"] > 0,
        payload["training_success"] is True,
        payload["reload_success"] is True,
        payload["dataset_boundary"] == {"dev_content_accessed": False, "locked_test_content_accessed": False},
        payload["behavioral_evaluation_performed"] is False,
    )
    if not all(invariants):
        raise ValueError("Formal training manifest violates one or more fixed Stage C5 invariants")


def reload_adapter(repo_root: Path) -> Dict[str, Any]:
    from huggingface_hub import snapshot_download
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler

    adapter_dir = repo_root / ADAPTER_PATH
    validate_adapter_artifacts(adapter_dir)
    snapshot = Path(snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_files_only=True))
    model, tokenizer = load(str(snapshot), adapter_path=str(adapter_dir))
    rows = read_jsonl(repo_root / TRAINING_VALID)
    chosen = sorted(rows, key=lambda row: hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest())[:3]
    samples = []
    sampler = make_sampler(temp=0.0)
    for row in chosen:
        messages = row["messages"][:2]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        raw_output = ""
        for chunk in stream_generate(model, tokenizer, prompt, max_tokens=256, sampler=sampler):
            raw_output += chunk.text
        samples.append({"instruction": messages[1]["content"], "raw_output": raw_output})
    payload = {
        "adapter_load_success": True,
        "generation_success": all(sample["raw_output"].strip() for sample in samples),
        "sample_count": len(samples),
        "samples": samples,
        "new_process": True,
        "source": "formal_validation_derived",
        "accuracy_evaluated": False,
        "behavioral_improvement_claimed": False,
    }
    validate_reload_artifact(payload)
    write_json(repo_root / ARTIFACT_DIR / "reload_test.json", payload)
    return payload


def save_loss_curve(repo_root: Path, metrics: Mapping[str, Any]) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(9, 5))
    training = metrics["training_reports"]
    validation = metrics["validation_reports"]
    axis.plot([row["iteration"] for row in training], [row["train_loss"] for row in training], label="Training Loss")
    axis.plot([row["iteration"] for row in validation], [row["validation_loss"] for row in validation], marker="o", label="Validation Loss")
    axis.set(title="Stage C5 Candidate 01 Optimization Diagnostics", xlabel="Iteration", ylabel="Loss")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(repo_root / ARTIFACT_DIR / "loss_curve.png", dpi=160)
    plt.close(figure)


def build_report(manifest: Mapping[str, Any], metrics: Mapping[str, Any]) -> str:
    audit = manifest["sequence_length_audit"]
    model = manifest["model_preflight"]
    reload = manifest["reload_test"]
    validation_history = ", ".join(
        f"iter {row['iteration']}: {row['validation_loss']:.3f}" for row in metrics["validation_reports"]
    )
    return f"""# Stage C5 Formal QLoRA Training

## Goal

Train the first reproducible formal QLoRA Candidate 01 on the complete frozen Train set, use frozen Validation only for optimization diagnostics, save the adapter, and verify reload in a fresh process. Stage C5 does not evaluate behavioral improvement.

## Base Model

- `{MODEL_ID}` at revision `{MODEL_REVISION}`
- `{model['architecture']}`, 4-bit quantization, group size 64
- Quantized Base + LoRA; no BF16 conversion, full fine-tuning, or alternative implementation

## Dataset

- Train: 2,700 rows; SHA-256 `{manifest['train_sha256']}`
- Validation: 300 rows; SHA-256 `{manifest['validation_sha256']}`
- Converted data: `data/training/train.jsonl`, `data/training/valid.jsonl`

## Sequence Length Audit

- Train: min {audit['train']['min']}, mean {audit['train']['mean']:.3f}, median {audit['train']['median']:.3f}, p90 {audit['train']['p90']:.3f}, p95 {audit['train']['p95']:.3f}, p99 {audit['train']['p99']:.3f}, max {audit['train']['max']}
- Train >1024: {audit['train']['count_over_1024']} ({audit['train']['percentage_over_1024']:.6f}%)
- Train >1536: {audit['train']['count_over_1536']} ({audit['train']['percentage_over_1536']:.6f}%)
- Validation >1024: {audit['validation']['count_over_1024']} ({audit['validation']['percentage_over_1024']:.6f}%)
- Selected `max_seq_length`: {manifest['max_seq_length']} according to the fixed >1% Train rule

## Training Format

MLX-LM chat JSONL with exact system/user/assistant roles. The assistant target is strict C2 JSON. The C4 frozen prompt is reused unchanged with SHA-256 `{PROMPT_SHA256}`.

## QLoRA Configuration

- Candidate: 01; `mask_prompt = true`
- Rank 8, scale 16.0, dropout 0.0; learning rate 1e-5
- Physical batch 1, gradient accumulation 2, effective batch 2
- 1,350 iterations; reports every 10; 50 validation batches every 150; checkpoints every 150

## Difference from Smoke Test

Formal training uses all 2,700/300 rows, `mask_prompt=true`, 16 adapted layers, and 1,350 iterations. The C5A smoke adapter remains separate and is not overwritten or treated as a candidate.

## Target Modules

All seven requested keys matched with no fallback: {', '.join(f'`{key}`' for key in model['actual_matched_target_modules'])}.

## Adapted Layers

Last 16 of 28 transformer layers, indices {model['adapted_layer_indices'][0]}–{model['adapted_layer_indices'][-1]}; {model['matched_module_instances']} matched module instances.

## Trainable Parameters

- Trainable: {model['trainable_parameter_count']:,}
- Total (MLX-LM quantization-aware): {model['total_parameter_count']:,}
- Trainable percentage: {model['trainable_percentage']:.6f}%

## Training Loss

- Initial reported: {metrics['initial_training_loss']:.3f}
- Final reported: {metrics['final_training_loss']:.3f}
- Minimum reported: {metrics['minimum_training_loss']:.3f}
- All recorded losses finite: true

## Validation Loss

{validation_history}

Final reported validation loss: {metrics['final_validation_loss']:.3f}. Lower validation loss does not prove behavioral improvement.

## Runtime

- Total official training command: {manifest['training_duration_seconds']:.3f} seconds
- Mean throughput: {metrics['mean_tokens_per_second']:.3f} tokens/sec; {metrics['mean_iterations_per_second']:.3f} iterations/sec

## Peak Memory

- {metrics['peak_memory_gb']:.3f} GB

## Checkpoints

{', '.join(f'`{name}`' for name in manifest['checkpoint_files'])}

No checkpoint was behaviorally selected in Stage C5.

## Final Adapter

- `{ADAPTER_PATH}/adapters.safetensors`
- Adapter bytes: {manifest['adapter_size_bytes']:,}

## Adapter Reload

- New process: {str(reload['new_process']).lower()}
- Load success: {str(reload['adapter_load_success']).lower()}
- Non-empty generation: {str(reload['generation_success']).lower()} ({reload['sample_count']} Validation-derived inputs)
- Accuracy evaluation: false

## Dataset Boundary Validation

- Dev content accessed: no
- Locked Test content accessed: no
- Dev/Locked inference or behavioral metrics: none

## Limitations

Loss curves are optimization diagnostics only. Stage C5 does not establish intent accuracy, JSON/schema improvement, response quality, superiority to Base, checkpoint preference, or promotion readiness.

## Stage C5 Conclusion

Stage C5 only proves that formal QLoRA training completed successfully and its adapter can be reloaded. Behavioral improvement remains unknown until Stage C6. Stage C6 was not started.
"""


def finalize(repo_root: Path, duration_seconds: float) -> Dict[str, Any]:
    artifact_dir = repo_root / ARTIFACT_DIR
    dataset = json.loads((artifact_dir / "dataset_preflight.json").read_text(encoding="utf-8"))
    audit = json.loads((artifact_dir / "sequence_length_summary.json").read_text(encoding="utf-8"))
    model = json.loads((repo_root / CANDIDATE_DIR / "model_preflight.json").read_text(encoding="utf-8"))
    reload = json.loads((artifact_dir / "reload_test.json").read_text(encoding="utf-8"))
    validate_reload_artifact(reload)
    metrics = parse_training_log((artifact_dir / "training.log").read_text(encoding="utf-8"))
    metrics["training_duration_seconds"] = round(duration_seconds, 6)
    write_json(artifact_dir / "training_metrics.json", metrics)
    save_loss_curve(repo_root, metrics)
    config = load_config(repo_root)
    adapter_dir = repo_root / ADAPTER_PATH
    adapter_files = validate_adapter_artifacts(adapter_dir)
    checkpoint_files = sorted(name for name in adapter_files if re.fullmatch(r"\d{7}_adapters\.safetensors", name))
    expected_checkpoints = [f"{iteration:07d}_adapters.safetensors" for iteration in range(150, 1351, 150)]
    if checkpoint_files != expected_checkpoints:
        raise ValueError(f"Formal checkpoint set mismatch: {checkpoint_files}")
    manifest = {
        "stage": "C5 Formal QLoRA Training Candidate 01",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "quantization": model["quantization"],
        "train_rows": 2700,
        "validation_rows": 300,
        "train_sha256": dataset["train"]["source_sha256"],
        "validation_sha256": dataset["validation"]["source_sha256"],
        "prompt_sha256": PROMPT_SHA256,
        "config_sha256": dataset["config_sha256"],
        "sequence_length_audit": audit,
        "max_seq_length": config["max_seq_length"],
        "num_layers": 16,
        "target_modules": list(TARGET_KEYS),
        "rank": 8,
        "scale": 16.0,
        "dropout": 0.0,
        "mask_prompt": True,
        "learning_rate": 1e-5,
        "batch_size": 1,
        "gradient_accumulation_steps": 2,
        "effective_batch_size": 2,
        "iterations": 1350,
        "adapter_path": ADAPTER_PATH,
        "adapter_files": adapter_files,
        "checkpoint_files": checkpoint_files,
        "adapter_size_bytes": (adapter_dir / "adapters.safetensors").stat().st_size,
        "training_success": True,
        "reload_success": True,
        "training_duration_seconds": round(duration_seconds, 6),
        "model_preflight": model,
        "reload_test": reload,
        "dataset_boundary": {"dev_content_accessed": False, "locked_test_content_accessed": False},
        "behavioral_evaluation_performed": False,
        "candidate_selected_by_behavior": False,
        "environment": {"python": sys.version.split()[0], "mlx": version("mlx"), "mlx_lm": version("mlx-lm"), "platform": platform.platform()},
    }
    validate_training_manifest(manifest)
    write_json(artifact_dir / "training_manifest.json", manifest)
    (repo_root / "reports/stage5_formal_qlora_training.md").write_text(build_report(manifest, metrics), encoding="utf-8")
    return manifest


def run_command(command: Sequence[str], repo_root: Path, log_path: Path) -> Tuple[int, float]:
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8", newline="\n") as log:
        process = subprocess.Popen(
            list(command),
            cwd=repo_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env={**os.environ, "PYTHONPYCACHEPREFIX": "/tmp/project_c_pycache", "MPLCONFIGDIR": "/tmp/project_c_matplotlib", "HF_HUB_OFFLINE": "1"},
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        return_code = process.wait()
    return return_code, time.perf_counter() - started


def run_manual_pipeline(repo_root: Path) -> None:
    artifact_dir = repo_root / ARTIFACT_DIR
    candidate_dir = repo_root / CANDIDATE_DIR
    artifact_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir.mkdir(parents=True, exist_ok=True)
    prepare(repo_root)
    python = str(repo_root / ".venv/bin/python")
    preflight_code, _ = run_command(
        [python, "-m", "src.training.qlora_formal", "model-preflight"], repo_root, candidate_dir / "model_preflight.log"
    )
    if preflight_code:
        raise RuntimeError(f"Formal model preflight failed with exit code {preflight_code}")
    training_code, duration = run_command(
        [python, "-m", "mlx_lm", "lora", "--config", CONFIG_PATH], repo_root, artifact_dir / "training.log"
    )
    write_json(artifact_dir / "training_execution.json", {"exit_code": training_code, "duration_seconds": round(duration, 6)})
    if training_code:
        raise RuntimeError(f"Formal MLX-LM training failed with exit code {training_code}")
    reload_code, _ = run_command(
        [python, "-m", "src.training.qlora_formal", "reload"], repo_root, artifact_dir / "reload.log"
    )
    if reload_code:
        raise RuntimeError(f"Formal adapter reload failed with exit code {reload_code}")
    finalize(repo_root, duration)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage C5 formal QLoRA Candidate 01 pipeline")
    parser.add_argument("action", choices=("prepare", "model-preflight", "reload", "finalize", "run"))
    parser.add_argument("--training-duration-seconds", type=float, default=None)
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    if args.action == "prepare":
        result = prepare(repo_root)
    elif args.action == "model-preflight":
        result = model_preflight(repo_root)
    elif args.action == "reload":
        result = reload_adapter(repo_root)
    elif args.action == "finalize":
        if args.training_duration_seconds is None:
            execution = json.loads((repo_root / ARTIFACT_DIR / "training_execution.json").read_text())
            args.training_duration_seconds = execution["duration_seconds"]
        result = finalize(repo_root, args.training_duration_seconds)
    else:
        run_manual_pipeline(repo_root)
        result = {"stage": "C5", "candidate": "01", "success": True}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
