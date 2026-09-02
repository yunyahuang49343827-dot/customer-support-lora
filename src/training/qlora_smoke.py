"""Prepare and orchestrate the manually executed Stage C5A QLoRA smoke test.

The pure preparation/finalization paths never import MLX. Metal-dependent model
preflight, official MLX-LM training, and adapter reload run in separate processes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import yaml

from src.evaluation.contracts import validate_output


SEED = 42
MODEL_ID = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
MODEL_REVISION = "8b403126fc14f14cfc99bb4cfa72ecbc129ea677"
TRAIN_SOURCE = "data/processed/train.jsonl"
VALID_SOURCE = "data/processed/validation.jsonl"
SMOKE_TRAIN = "data/smoke/train.jsonl"
SMOKE_VALID = "data/smoke/valid.jsonl"
PROMPT_PATH = "prompts/base_system_prompt.txt"
CONFIG_PATH = "configs/qlora_smoke.yaml"
ARTIFACT_DIR = "artifacts/stage5a"
ADAPTER_PATH = "artifacts/stage5a/adapter"
TARGET_KEYS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)
REQUIRED_ADAPTER_FILES = ("adapter_config.json", "adapters.safetensors")
TRAIN_RE = re.compile(
    r"Iter (?P<iteration>\d+): Train loss (?P<loss>[-+\w.]+), Learning Rate (?P<lr>[-+\w.]+), "
    r"It/sec (?P<it_sec>[-+\w.]+), Tokens/sec (?P<tokens_sec>[-+\w.]+), "
    r"Trained Tokens (?P<trained_tokens>\d+), Peak mem (?P<peak_memory>[-+\w.]+) GB"
)
VAL_RE = re.compile(r"Iter (?P<iteration>\d+): Val loss (?P<loss>[-+\w.]+), Val took (?P<seconds>[-+\w.]+)s")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_rank(seed: int, split: str, source_index: int) -> str:
    return hashlib.sha256(f"{seed}\x1fstage5a\x1f{split}\x1f{source_index}".encode()).hexdigest()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def target_json(target: Mapping[str, Any]) -> str:
    ordered = {
        "intent": target["intent"],
        "category": target["category"],
        "needs_human": target["needs_human"],
        "response": target["response"],
    }
    serialized = json.dumps(ordered, ensure_ascii=False, separators=(",", ":"))
    if not validate_output(serialized).valid:
        raise ValueError("Frozen target failed the C2 output contract")
    return serialized


def sample_records(records: Sequence[Mapping[str, Any]], count: int, split: str) -> List[Mapping[str, Any]]:
    if len(records) < count:
        raise ValueError(f"Not enough {split} records for smoke sample: {len(records)} < {count}")
    return sorted(
        records,
        key=lambda row: stable_rank(SEED, split, int(row["metadata"]["source_index"])),
    )[:count]


def to_chat_record(record: Mapping[str, Any], prompt: str) -> Dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": record["instruction"]},
            {"role": "assistant", "content": target_json(record["target"])},
        ]
    }


def validate_chat_rows(rows: Sequence[Mapping[str, Any]], expected_count: int) -> Dict[str, Any]:
    if len(rows) != expected_count:
        raise ValueError(f"Expected {expected_count} chat rows; found {len(rows)}")
    for index, row in enumerate(rows):
        if set(row) != {"messages"} or not isinstance(row["messages"], list):
            raise ValueError(f"Row {index} must contain only a messages list")
        messages = row["messages"]
        if [message.get("role") for message in messages] != ["system", "user", "assistant"]:
            raise ValueError(f"Row {index} roles are not exactly system/user/assistant")
        if any(not isinstance(message.get("content"), str) or not message["content"].strip() for message in messages):
            raise ValueError(f"Row {index} contains empty/non-string content")
        assistant = messages[2]["content"]
        if assistant.startswith("```") or not validate_output(assistant).valid:
            raise ValueError(f"Row {index} assistant target is not strict C2 JSON")
    return {"row_count": len(rows), "chat_format_valid": True, "schema_invalid_count": 0}


def load_config(repo_root: Path) -> Dict[str, Any]:
    config = yaml.safe_load((repo_root / CONFIG_PATH).read_text(encoding="utf-8"))
    expected = {
        "model": MODEL_ID, "train": True, "fine_tune_type": "lora", "data": "data/smoke",
        "seed": 42, "num_layers": 8, "batch_size": 1, "grad_accumulation_steps": 2,
        "iters": 20, "val_batches": -1, "learning_rate": 1e-5, "steps_per_report": 1,
        "steps_per_eval": 10, "adapter_path": ADAPTER_PATH, "save_every": 20, "test": False,
        "max_seq_length": 1024, "grad_checkpoint": False,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Smoke config {key!r} must be {value!r}; found {config.get(key)!r}")
    lora = config.get("lora_parameters", {})
    if "alpha" in lora or lora.get("scale") != 16.0:
        raise ValueError("MLX-LM smoke config must use scale=16.0 and must not use alpha")
    if tuple(lora.get("keys", ())) != TARGET_KEYS:
        raise ValueError("Smoke target module keys differ from the approved list")
    if (lora.get("rank"), lora.get("dropout")) != (8, 0.0):
        raise ValueError("Smoke LoRA rank/dropout differ from the approved values")
    return config


def prepare(repo_root: Path) -> Dict[str, Any]:
    config = load_config(repo_root)
    hashes = json.loads((repo_root / "data/manifests/dataset_hashes.json").read_text(encoding="utf-8"))
    allowed = {"train": TRAIN_SOURCE, "validation": VALID_SOURCE}
    source_records: Dict[str, List[Dict[str, Any]]] = {}
    for split, relative_path in allowed.items():
        actual_hash = sha256_file(repo_root / relative_path)
        expected_hash = hashes["files"][split]["sha256"]
        if actual_hash != expected_hash:
            raise ValueError(f"Frozen {split} hash mismatch")
        source_records[split] = read_jsonl(repo_root / relative_path)

    prompt = (repo_root / PROMPT_PATH).read_text(encoding="utf-8").strip()
    chosen_train = sample_records(source_records["train"], 80, "train")
    chosen_valid = sample_records(source_records["validation"], 20, "validation")
    train_indices = [int(row["metadata"]["source_index"]) for row in chosen_train]
    valid_indices = [int(row["metadata"]["source_index"]) for row in chosen_valid]
    if len(set(train_indices)) != 80 or len(set(valid_indices)) != 20 or set(train_indices) & set(valid_indices):
        raise ValueError("Smoke source indices are not unique and split-disjoint")

    train_chat = [to_chat_record(row, prompt) for row in chosen_train]
    valid_chat = [to_chat_record(row, prompt) for row in chosen_valid]
    train_validation = validate_chat_rows(train_chat, 80)
    valid_validation = validate_chat_rows(valid_chat, 20)
    write_jsonl(repo_root / SMOKE_TRAIN, train_chat)
    write_jsonl(repo_root / SMOKE_VALID, valid_chat)

    audit = {
        "seed": SEED,
        "source_files_opened": [TRAIN_SOURCE, VALID_SOURCE],
        "disallowed_dataset_content_accessed": False,
        "provenance_rule": "Every smoke row is transformed directly from an allowed frozen source row; no other split content is opened.",
        "train": {**train_validation, "source_indices": train_indices, "source_sha256": hashes["files"]["train"]["sha256"], "smoke_sha256": sha256_file(repo_root / SMOKE_TRAIN)},
        "valid": {**valid_validation, "source_indices": valid_indices, "source_sha256": hashes["files"]["validation"]["sha256"], "smoke_sha256": sha256_file(repo_root / SMOKE_VALID)},
        "source_index_overlap": len(set(train_indices) & set(valid_indices)),
        "prompt_sha256": sha256_file(repo_root / PROMPT_PATH),
        "config_sha256": sha256_file(repo_root / CONFIG_PATH),
        "config_valid": bool(config),
    }
    write_json(repo_root / ARTIFACT_DIR / "dataset_preflight.json", audit)
    return audit


def validate_adapter_artifacts(adapter_dir: Path) -> List[str]:
    missing = [name for name in REQUIRED_ADAPTER_FILES if not (adapter_dir / name).is_file()]
    if missing:
        raise ValueError(f"Missing required adapter files: {missing}")
    if (adapter_dir / "adapters.safetensors").stat().st_size == 0:
        raise ValueError("Adapter weights file is empty")
    return sorted(path.name for path in adapter_dir.iterdir() if path.is_file())


def validate_reload_artifact(payload: Mapping[str, Any]) -> None:
    required = {"adapter_load_success", "generation_success", "sample_count", "samples", "new_process"}
    if set(payload) < required:
        raise ValueError(f"Reload artifact missing keys: {sorted(required - set(payload))}")
    if payload["sample_count"] not in (3, 4, 5) or len(payload["samples"]) != payload["sample_count"]:
        raise ValueError("Reload artifact must contain 3-5 samples")
    if not payload["adapter_load_success"] or not payload["generation_success"] or not payload["new_process"]:
        raise ValueError("Adapter reload validation did not succeed in a new process")
    if any(not isinstance(row.get("raw_output"), str) or not row["raw_output"].strip() for row in payload["samples"]):
        raise ValueError("Reload inference produced an empty output")


def parse_training_log(text: str) -> Dict[str, Any]:
    train_reports = []
    validation_reports = []
    for match in TRAIN_RE.finditer(text):
        row = {
            "iteration": int(match["iteration"]), "train_loss": float(match["loss"]),
            "learning_rate": float(match["lr"]), "iterations_per_second": float(match["it_sec"]),
            "tokens_per_second": float(match["tokens_sec"]), "trained_tokens": int(match["trained_tokens"]),
            "peak_memory_gb": float(match["peak_memory"]),
        }
        train_reports.append(row)
    for match in VAL_RE.finditer(text):
        validation_reports.append({
            "iteration": int(match["iteration"]), "validation_loss": float(match["loss"]),
            "validation_seconds": float(match["seconds"]),
        })
    if len(train_reports) != 20 or not validation_reports:
        raise ValueError(
            f"Expected 20 train reports and validation reports; found {len(train_reports)} and {len(validation_reports)}"
        )
    numeric_losses = [row["train_loss"] for row in train_reports] + [row["validation_loss"] for row in validation_reports]
    if not all(math.isfinite(value) for value in numeric_losses):
        raise ValueError("Training log contains NaN or Inf loss")
    return {
        "training_reports": train_reports,
        "validation_reports": validation_reports,
        "initial_training_loss": train_reports[0]["train_loss"],
        "final_training_loss": train_reports[-1]["train_loss"],
        "initial_validation_loss": validation_reports[0]["validation_loss"],
        "final_validation_loss": validation_reports[-1]["validation_loss"],
        "peak_memory_gb": max(row["peak_memory_gb"] for row in train_reports),
        "finite_losses": True,
    }


def model_preflight(repo_root: Path) -> Dict[str, Any]:
    config = load_config(repo_root)
    from huggingface_hub import snapshot_download
    from mlx.utils import tree_flatten
    from mlx_lm import load
    from mlx_lm.tuner.utils import linear_to_lora_layers
    from mlx_lm.utils import get_total_parameters

    snapshot = Path(snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_files_only=True))
    model, _, model_config = load(str(snapshot), adapter_path=None, return_config=True)
    model.freeze()
    layers = model.layers[-config["num_layers"] :]
    per_layer = []
    for offset, layer in enumerate(layers):
        available = {name for name, _ in layer.named_modules()}
        matched = [key for key in TARGET_KEYS if key in available]
        if tuple(matched) != TARGET_KEYS:
            raise ValueError(f"Target module mismatch in adapted layer offset {offset}: {matched}")
        per_layer.append({"adapted_layer_offset": offset, "matched_keys": matched})
    linear_to_lora_layers(model, config["num_layers"], config["lora_parameters"], use_dora=False)
    trainable = sum(value.size for _, value in tree_flatten(model.trainable_parameters()))
    total = get_total_parameters(model)
    lora_names = sorted(name for name, _ in tree_flatten(model.trainable_parameters()))
    if not lora_names or trainable <= 0:
        raise ValueError("No trainable LoRA parameters were created")
    payload = {
        "model_load_success": True, "repo_id": MODEL_ID, "revision": MODEL_REVISION,
        "architecture": (model_config.get("architectures") or [model_config.get("model_type")])[0],
        "quantization": model_config.get("quantization"), "model_layer_count": len(model.layers),
        "adapted_transformer_layers": config["num_layers"], "target_keys": list(TARGET_KEYS),
        "actual_matched_target_modules": sorted({key for row in per_layer for key in row["matched_keys"]}),
        "matched_module_instances": sum(len(row["matched_keys"]) for row in per_layer),
        "per_layer_matches": per_layer, "trainable_parameter_names": lora_names,
        "trainable_parameter_count": trainable, "total_parameter_count": total,
        "trainable_percentage": round(100.0 * trainable / total, 6),
    }
    write_json(repo_root / ARTIFACT_DIR / "model_preflight.json", payload)
    return payload


def reload_adapter(repo_root: Path) -> Dict[str, Any]:
    from huggingface_hub import snapshot_download
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler

    adapter_dir = repo_root / ADAPTER_PATH
    validate_adapter_artifacts(adapter_dir)
    snapshot = Path(snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_files_only=True))
    model, tokenizer = load(str(snapshot), adapter_path=str(adapter_dir))
    smoke_rows = read_jsonl(repo_root / SMOKE_TRAIN)
    chosen = sorted(smoke_rows, key=lambda row: hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest())[:3]
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
        "adapter_load_success": True, "generation_success": all(row["raw_output"].strip() for row in samples),
        "sample_count": len(samples), "samples": samples, "new_process": True,
        "accuracy_evaluated": False, "behavioral_improvement_claimed": False,
    }
    validate_reload_artifact(payload)
    write_json(repo_root / ARTIFACT_DIR / "reload_test.json", payload)
    return payload


def build_report(manifest: Mapping[str, Any], metrics: Mapping[str, Any]) -> str:
    model = manifest["model_preflight"]
    reload = manifest["reload_test"]
    return f"""# Stage C5A QLoRA Smoke Test

## Goal

Verify that the fixed MLX-LM QLoRA pipeline can load chat data, tokenize it, create and update LoRA layers, report finite losses, save an adapter, reload it in a new process, and generate non-empty output. This is not a behavioral evaluation.

## Model

- Base: `{MODEL_ID}`
- Revision: `{MODEL_REVISION}`
- Architecture: `{model['architecture']}`
- Quantization: {model['quantization']['bits']}-bit, group size {model['quantization']['group_size']}
- Fine-tune type: `lora` over a quantized base (QLoRA); no BF16 conversion or full fine-tuning

## Dataset

- Train smoke rows: 80 from frozen Train only
- Validation smoke rows: 20 from frozen Validation only
- Seed: 42
- Frozen source hashes and selected source indices are recorded in `artifacts/stage5a/dataset_preflight.json`.

## Chat Training Format

Each JSONL row contains exactly `messages` with roles `system`, `user`, and `assistant`. The assistant content is strict JSON serialization conforming to the C2 schema. The system content is the frozen C4 prompt; no per-row label is injected into the user message.

## QLoRA Configuration

- Layers: 8; rank: 8; scale: 16.0; dropout: 0.0
- Batch size: 1; gradient accumulation: 2; effective batch size: 2
- Iterations: 20; learning rate: 1e-5; maximum sequence length: 1024
- Config: `configs/qlora_smoke.yaml`

## Target Modules

Matched all seven requested module keys in each of the eight adapted transformer layers ({model['matched_module_instances']} module instances): {', '.join(f'`{key}`' for key in model['actual_matched_target_modules'])}.

## Trainable Parameters

- Trainable: {model['trainable_parameter_count']:,}
- Total parameter count (quantization-aware MLX-LM calculation): {model['total_parameter_count']:,}
- Trainable percentage: {model['trainable_percentage']:.6f}%

## Training Result

- Success: true
- Initial reported train loss: {metrics['initial_training_loss']:.3f}
- Final reported train loss: {metrics['final_training_loss']:.3f}
- Finite losses: true
- Runtime: {manifest['training_duration_seconds']:.3f} seconds

Loss values are pipeline diagnostics only and are not evidence of model improvement.

## Validation Result

- Initial validation loss: {metrics['initial_validation_loss']:.3f}
- Final validation loss: {metrics['final_validation_loss']:.3f}
- Validation batches: all 20 rows (`val_batches=-1`)

## Memory / Runtime

- Peak MLX memory: {metrics['peak_memory_gb']:.3f} GB
- Total training command duration: {manifest['training_duration_seconds']:.3f} seconds

## Adapter Save

- Path: `{ADAPTER_PATH}`
- Files: {', '.join(f'`{name}`' for name in manifest['adapter_files'])}

## Adapter Reload

- New process: {str(reload['new_process']).lower()}
- Adapter load success: {str(reload['adapter_load_success']).lower()}

## Inference Reload Test

- Samples: {reload['sample_count']}
- Non-empty generation success: {str(reload['generation_success']).lower()}
- Accuracy evaluated: false

## Dataset Boundary Validation

- Opened source content: `data/processed/train.jsonl`, `data/processed/validation.jsonl`
- Dev content accessed: no
- Locked Test content accessed: no
- Dev/Locked behavioral evaluation: none

## Limitations

Only 80/20 rows and 20 iterations were used. Loss direction and reload outputs do not establish response quality, accuracy, generalization, or candidate fitness. This smoke adapter is not a formal model candidate and must not be used for promotion.

## Stage C5A Conclusion

Stage C5A passed the mechanical QLoRA pipeline checks. Smoke Test loss and outputs do not represent behavioral improvement. Formal Stage C5 training was not started.
"""


def finalize(repo_root: Path, training_duration_seconds: float) -> Dict[str, Any]:
    artifact_dir = repo_root / ARTIFACT_DIR
    dataset = json.loads((artifact_dir / "dataset_preflight.json").read_text(encoding="utf-8"))
    model = json.loads((artifact_dir / "model_preflight.json").read_text(encoding="utf-8"))
    reload = json.loads((artifact_dir / "reload_test.json").read_text(encoding="utf-8"))
    validate_reload_artifact(reload)
    log_text = (artifact_dir / "training.log").read_text(encoding="utf-8")
    metrics = parse_training_log(log_text)
    metrics["training_duration_seconds"] = round(training_duration_seconds, 6)
    write_json(artifact_dir / "training_metrics.json", metrics)
    config = load_config(repo_root)
    adapter_files = validate_adapter_artifacts(repo_root / ADAPTER_PATH)
    manifest = {
        "stage": "C5A QLoRA Training Smoke Test", "base_model_id": MODEL_ID,
        "model_revision": MODEL_REVISION, "quantization": model["quantization"],
        "smoke_dataset_rows": {"train": 80, "validation": 20},
        "source_dataset_hashes": {"train": dataset["train"]["source_sha256"], "validation": dataset["valid"]["source_sha256"]},
        "config_hash": dataset["config_sha256"], "prompt_hash": dataset["prompt_sha256"],
        "lora": {"rank": 8, "scale": 16.0, "dropout": 0.0, "target_keys": list(TARGET_KEYS)},
        "num_layers": 8, "iterations": 20, "learning_rate": 1e-5, "batch_size": 1,
        "gradient_accumulation_steps": 2, "max_sequence_length": 1024,
        "adapter_path": ADAPTER_PATH, "adapter_files": adapter_files,
        "training_success": True, "reload_success": True,
        "training_duration_seconds": round(training_duration_seconds, 6),
        "model_preflight": model, "reload_test": reload,
        "dataset_boundary": {"dev_accessed": False, "locked_test_accessed": False},
        "smoke_adapter_is_formal_candidate": False,
        "environment": {"python": sys.version.split()[0], "mlx": version("mlx"), "mlx_lm": version("mlx-lm"), "platform": platform.platform()},
        "config": config,
    }
    write_json(artifact_dir / "smoke_manifest.json", manifest)
    (repo_root / "reports/stage5a_smoke_test.md").write_text(build_report(manifest, metrics), encoding="utf-8")
    return manifest


def run_command(command: Sequence[str], repo_root: Path, log_path: Path) -> Tuple[int, float]:
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8", newline="\n") as log:
        process = subprocess.Popen(
            list(command), cwd=repo_root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env={
                **os.environ,
                "PYTHONPYCACHEPREFIX": "/tmp/project_c_pycache",
                "MPLCONFIGDIR": "/tmp/project_c_matplotlib",
                "HF_HUB_OFFLINE": "1",
            },
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        return_code = process.wait()
    return return_code, time.perf_counter() - started


def run_manual_pipeline(repo_root: Path) -> None:
    artifact_dir = repo_root / ARTIFACT_DIR
    artifact_dir.mkdir(parents=True, exist_ok=True)
    prepare(repo_root)
    python = str(repo_root / ".venv/bin/python")
    preflight_code, _ = run_command(
        [python, "-m", "src.training.qlora_smoke", "model-preflight"], repo_root, artifact_dir / "model_preflight.log"
    )
    if preflight_code:
        raise RuntimeError(f"Model preflight failed with exit code {preflight_code}")
    training_code, duration = run_command(
        [python, "-m", "mlx_lm", "lora", "--config", CONFIG_PATH], repo_root, artifact_dir / "training.log"
    )
    write_json(artifact_dir / "training_execution.json", {"exit_code": training_code, "duration_seconds": round(duration, 6)})
    if training_code:
        raise RuntimeError(f"Official MLX-LM training failed with exit code {training_code}")
    reload_code, _ = run_command(
        [python, "-m", "src.training.qlora_smoke", "reload"], repo_root, artifact_dir / "reload.log"
    )
    if reload_code:
        raise RuntimeError(f"Adapter reload failed with exit code {reload_code}")
    finalize(repo_root, duration)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage C5A QLoRA smoke pipeline")
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
        result = {"stage": "C5A", "success": True}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
