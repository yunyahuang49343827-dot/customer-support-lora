"""Stage C4 deterministic Base Model evaluation on the frozen Dev split only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import statistics
import sys
import time
import traceback
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from src.evaluation.contracts import REQUIRED_KEYS, ContractVocabulary, load_vocabulary, validate_output


DEV_DATASET_RELATIVE_PATH = "data/processed/dev.jsonl"
DEFAULT_CONFIG_RELATIVE_PATH = "configs/base_inference.json"
STAGE4_ARTIFACT_RELATIVE_DIR = "artifacts/stage4"
ERROR_TAG_ORDER = (
    "wrong_intent",
    "wrong_category",
    "invalid_json",
    "missing_key",
    "extra_key",
    "invalid_enum",
    "intent_category_mismatch",
    "wrong_needs_human",
    "empty_response",
    "extra_text_before_json",
    "extra_text_after_json",
    "generation_truncated",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_rank(seed: int, *parts: Any) -> str:
    material = "\x1f".join([str(seed), *(str(part) for part in parts)]).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def _strict_json_object(raw_output: Any) -> Tuple[bool, Optional[Dict[str, Any]]]:
    if not isinstance(raw_output, str):
        return False, None
    try:
        parsed = json.loads(raw_output.strip())
    except json.JSONDecodeError:
        return False, None
    return (True, parsed) if isinstance(parsed, dict) else (False, None)


def _surrounding_text_tags(raw_output: Any) -> Tuple[str, ...]:
    if not isinstance(raw_output, str):
        return ()
    stripped = raw_output.strip()
    decoder = json.JSONDecoder()
    starts = [index for index, character in enumerate(stripped) if character == "{"]
    for start in starts:
        try:
            parsed, end = decoder.raw_decode(stripped[start:])
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict):
            continue
        tags = []
        if stripped[:start].strip():
            tags.append("extra_text_before_json")
        if stripped[start + end :].strip():
            tags.append("extra_text_after_json")
        return tuple(tags)
    return ()


def evaluate_prediction(
    raw_output: Any,
    ground_truth: Mapping[str, Any],
    latency_ms: float,
    generation_truncated: bool = False,
    vocabulary: Optional[ContractVocabulary] = None,
) -> Dict[str, Any]:
    """Evaluate one untouched model output against the fixed C2 contract."""
    vocabulary = vocabulary or load_vocabulary()
    json_valid, parsed = _strict_json_object(raw_output)
    validation = validate_output(raw_output, vocabulary)
    parsed_for_fields = parsed if parsed is not None else {}
    intent = parsed_for_fields.get("intent")
    category = parsed_for_fields.get("category")
    needs_human = parsed_for_fields.get("needs_human")
    intent_correct = intent == ground_truth["intent"]
    category_correct = category == ground_truth["category"]
    escalation_correct = type(needs_human) is bool and needs_human is ground_truth["needs_human"]

    tags = list(_surrounding_text_tags(raw_output))
    if not json_valid:
        tags.append("invalid_json")
    if parsed is not None:
        keys = set(parsed)
        if REQUIRED_KEYS - keys:
            tags.append("missing_key")
        if keys - REQUIRED_KEYS:
            tags.append("extra_key")
        if "invalid_intent" in validation.errors or "invalid_category" in validation.errors:
            tags.append("invalid_enum")
        if "intent_category_mismatch" in validation.errors:
            tags.append("intent_category_mismatch")
        if "response_empty" in validation.errors:
            tags.append("empty_response")
    if not intent_correct:
        tags.append("wrong_intent")
    if not category_correct:
        tags.append("wrong_category")
    if not escalation_correct:
        tags.append("wrong_needs_human")
    if generation_truncated:
        tags.append("generation_truncated")
    ordered_tags = [tag for tag in ERROR_TAG_ORDER if tag in set(tags)]

    return {
        "raw_model_output": raw_output,
        "parsed_output": parsed,
        "ground_truth_intent": ground_truth["intent"],
        "ground_truth_category": ground_truth["category"],
        "ground_truth_needs_human": ground_truth["needs_human"],
        "json_valid": json_valid,
        "schema_compliant": validation.valid,
        "intent_correct": intent_correct,
        "category_correct": category_correct,
        "escalation_correct": escalation_correct,
        "inference_latency_ms": round(float(latency_ms), 6),
        "generation_truncated": generation_truncated,
        "error_tags": ordered_tags,
    }


def percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("Latency aggregation requires at least one value")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def aggregate_metrics(predictions: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    total = len(predictions)
    if total == 0:
        raise ValueError("Cannot aggregate an empty prediction set")
    latencies = [float(row["inference_latency_ms"]) for row in predictions]
    rate = lambda key: round(100.0 * sum(bool(row[key]) for row in predictions) / total, 6)
    error_counts = Counter(tag for row in predictions for tag in row["error_tags"])
    truth_true = [row for row in predictions if row["ground_truth_needs_human"] is True]
    truth_false = [row for row in predictions if row["ground_truth_needs_human"] is False]
    predicted_needs_human = lambda row: (row["parsed_output"] or {}).get("needs_human")
    return {
        "evaluated_rows": total,
        "primary": {
            "intent_accuracy_percent": rate("intent_correct"),
            "json_valid_rate_percent": rate("json_valid"),
            "schema_compliance_percent": rate("schema_compliant"),
        },
        "secondary": {
            "category_accuracy_percent": rate("category_correct"),
            "escalation_accuracy_percent": rate("escalation_correct"),
            "escalation_false_negative_count": sum(predicted_needs_human(row) is False for row in truth_true),
            "escalation_false_positive_count": sum(predicted_needs_human(row) is True for row in truth_false),
            "escalation_invalid_or_missing_count": sum(
                type(predicted_needs_human(row)) is not bool for row in predictions
            ),
        },
        "operational": {
            "latency_sample_count": total,
            "mean_latency_ms": round(statistics.fmean(latencies), 6),
            "median_latency_ms": round(statistics.median(latencies), 6),
            "p95_latency_ms": round(percentile(latencies, 0.95), 6),
        },
        "error_tag_counts": dict(sorted(error_counts.items(), key=lambda item: (-item[1], item[0]))),
    }


def load_dev_records(repo_root: Path, configured_path: str) -> List[Dict[str, Any]]:
    if configured_path != DEV_DATASET_RELATIVE_PATH:
        raise ValueError(f"Stage C4 permits only {DEV_DATASET_RELATIVE_PATH}")
    path = repo_root / DEV_DATASET_RELATIVE_PATH
    with path.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    if len(records) != 300:
        raise ValueError(f"Frozen Dev must contain 300 rows; found {len(records)}")
    return records


def verify_dev_hash(repo_root: Path, config: Mapping[str, Any]) -> str:
    manifest_path = repo_root / config["dataset_hash_manifest_path"]
    hash_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = hash_manifest["files"]["dev"]["sha256"]
    actual = sha256_file(repo_root / DEV_DATASET_RELATIVE_PATH)
    if actual != expected:
        raise ValueError(f"Frozen Dev SHA-256 mismatch: expected {expected}, found {actual}")
    if not hash_manifest.get("locked_test", {}).get("frozen"):
        raise ValueError("Dataset hash manifest does not confirm a frozen final-evaluation split")
    return actual


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def write_error_cases(path: Path, predictions: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "source_index", "instruction", "ground_truth_intent", "predicted_intent",
        "ground_truth_category", "predicted_category", "ground_truth_needs_human",
        "predicted_needs_human", "json_valid", "schema_compliant", "generation_truncated",
        "error_tags", "raw_model_output",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in predictions:
            if not row["error_tags"]:
                continue
            parsed = row["parsed_output"] or {}
            writer.writerow({
                "source_index": row["source_index"], "instruction": row["instruction"],
                "ground_truth_intent": row["ground_truth_intent"], "predicted_intent": parsed.get("intent"),
                "ground_truth_category": row["ground_truth_category"], "predicted_category": parsed.get("category"),
                "ground_truth_needs_human": row["ground_truth_needs_human"],
                "predicted_needs_human": parsed.get("needs_human"), "json_valid": row["json_valid"],
                "schema_compliant": row["schema_compliant"], "generation_truncated": row["generation_truncated"],
                "error_tags": ";".join(row["error_tags"]), "raw_model_output": row["raw_model_output"],
            })


def write_confusion(path: Path, predictions: Sequence[Mapping[str, Any]], intents: Sequence[str]) -> List[Dict[str, Any]]:
    counts = Counter()
    allowed = set(intents)
    for row in predictions:
        predicted = (row["parsed_output"] or {}).get("intent")
        if predicted in allowed:
            counts[(row["ground_truth_intent"], predicted)] += 1
    rows = [
        {"ground_truth_intent": truth, "predicted_intent": predicted, "count": count}
        for (truth, predicted), count in sorted(counts.items())
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ground_truth_intent", "predicted_intent", "count"])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def select_manual_qa(predictions: Sequence[Mapping[str, Any]], seed: int, count: int = 30) -> List[Mapping[str, Any]]:
    ranked = lambda rows, label: sorted(
        rows, key=lambda row: stable_rank(seed, "manual_qa", label, row["source_index"])
    )
    selected: List[Mapping[str, Any]] = []
    seen = set()

    def take(rows: Sequence[Mapping[str, Any]], label: str, limit: int) -> None:
        taken = 0
        for row in ranked(rows, label):
            if row["source_index"] in seen:
                continue
            selected.append(row)
            seen.add(row["source_index"])
            taken += 1
            if taken >= limit:
                break

    take([row for row in predictions if not row["error_tags"]], "fully_correct", 8)
    take([row for row in predictions if not row["intent_correct"]], "wrong_intent", 8)
    take([row for row in predictions if not row["schema_compliant"]], "schema_failure", 8)
    take([row for row in predictions if row["ground_truth_needs_human"] is True], "needs_human_true", 5)
    take([row for row in predictions if row["ground_truth_needs_human"] is False], "needs_human_false", 5)
    for row in ranked(predictions, "fill"):
        if len(selected) >= count:
            break
        if row["source_index"] not in seen:
            selected.append(row)
            seen.add(row["source_index"])
    return selected[:count]


def write_manual_qa(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "instruction", "ground_truth_intent", "predicted_intent", "ground_truth_needs_human",
        "predicted_needs_human", "raw_model_output", "response", "error_tags",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            parsed = row["parsed_output"] or {}
            writer.writerow({
                "instruction": row["instruction"], "ground_truth_intent": row["ground_truth_intent"],
                "predicted_intent": parsed.get("intent"),
                "ground_truth_needs_human": row["ground_truth_needs_human"],
                "predicted_needs_human": parsed.get("needs_human"),
                "raw_model_output": row["raw_model_output"], "response": parsed.get("response"),
                "error_tags": ";".join(row["error_tags"]),
            })


def build_report(metrics: Mapping[str, Any], manifest: Mapping[str, Any], confusions: Sequence[Mapping[str, Any]]) -> str:
    errors = list(metrics["error_tag_counts"].items())[:12]
    confusion_errors = sorted(
        (row for row in confusions if row["ground_truth_intent"] != row["predicted_intent"]),
        key=lambda row: (-row["count"], row["ground_truth_intent"], row["predicted_intent"]),
    )[:12]
    error_lines = "\n".join(f"- `{tag}`: {count}" for tag, count in errors) or "- None"
    confusion_lines = "\n".join(
        f"- `{row['ground_truth_intent']}` → `{row['predicted_intent']}`: {row['count']}"
        for row in confusion_errors
    ) or "- No canonical wrong-intent predictions"
    primary = metrics["primary"]
    secondary = metrics["secondary"]
    latency = metrics["operational"]
    return f"""# Stage C4 Base Model Development Baseline

## Model

- Repository: `{manifest['model']['repo_id']}`
- Revision: `{manifest['model']['revision']}`
- Load success: `{str(manifest['model']['load_success']).lower()}`
- Architecture: `{manifest['model']['architecture']}`
- Quantization: {manifest['model']['quantization_bits']}-bit, group size {manifest['model']['quantization_group_size']}
- Parameter size metadata: {manifest['model']['parameter_size_label']}
- Adapter: none

## Environment

- Python: `{manifest['environment']['python_version']}`
- MLX: `{manifest['environment']['mlx_version']}`
- MLX-LM: `{manifest['environment']['mlx_lm_version']}`
- Platform: `{manifest['environment']['platform']}`

## Frozen Prompt

The system prompt is frozen at `prompts/base_system_prompt.txt` with SHA-256 `{manifest['prompt_sha256']}`. It contains the complete 27-intent and 11-category vocabularies, the four-key JSON contract, and response safety constraints. Ground-truth labels are not included in model inputs. The escalation mapping is not enumerated.

## Inference Configuration

Greedy deterministic decoding (`temperature=0`), seed {manifest['inference_config']['seed']}, maximum {manifest['inference_config']['max_generation_tokens']} generated tokens, concurrency 1, no adapters, and no per-example decoding changes.

## Dev Dataset

Evaluated all {metrics['evaluated_rows']} frozen Dev rows. Dev SHA-256: `{manifest['dev_sha256']}`. Model input contains only the frozen system prompt and customer instruction. The final-evaluation dataset content was not opened or evaluated.

## Primary Metrics

- Intent Accuracy: {primary['intent_accuracy_percent']:.6f}%
- JSON Valid Rate: {primary['json_valid_rate_percent']:.6f}%
- Schema Compliance: {primary['schema_compliance_percent']:.6f}%

## Secondary Metrics

- Category Accuracy: {secondary['category_accuracy_percent']:.6f}%
- Escalation Accuracy: {secondary['escalation_accuracy_percent']:.6f}%
- Escalation false negatives: {secondary['escalation_false_negative_count']}
- Escalation false positives: {secondary['escalation_false_positive_count']}

## Latency

- Mean: {latency['mean_latency_ms']:.3f} ms
- Median: {latency['median_latency_ms']:.3f} ms
- p95: {latency['p95_latency_ms']:.3f} ms
- Samples: {latency['latency_sample_count']}

## Error Breakdown

{error_lines}

## Intent Confusions

{confusion_lines}

The full canonical confusion table is `artifacts/stage4/intent_confusion.csv`.

## Manual QA Required

`artifacts/stage4/manual_qa_samples.csv` contains 30 deterministic seed-42 cases sampled across correct outputs, wrong intents, schema failures, and both escalation labels.

> **這一步需要你手動做**
>
> Review `artifacts/stage4/manual_qa_samples.csv` for relevance, unsupported action claims, fabricated policy, safety, unnecessary escalation, and missing escalation. Do not alter the frozen prompt or rerun this baseline to improve its score.

## Limitations

Automated metrics measure exact labels and output-contract behavior, not response relevance. Latency is specific to this machine and sequential MLX execution. This is a Base development baseline only; no LoRA comparison or improvement claim is made.

## Stage C4 Conclusion

The frozen Base Model baseline completed on all 300 Dev rows with a fixed prompt and deterministic decoding. No adapter, training, model modification, prompt optimization, or final-evaluation behavioral use occurred. Stage C5 was not started.
"""


def run_baseline(repo_root: Path) -> Dict[str, Any]:
    config_path = repo_root / DEFAULT_CONFIG_RELATIVE_PATH
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["adapter_path"] is not None or config["temperature"] != 0 or config["warmup_runs"] != 0:
        raise ValueError("Stage C4 requires no adapter, greedy decoding, and zero warm-up runs")
    prompt_path = repo_root / config["prompt_path"]
    prompt_text = prompt_path.read_text(encoding="utf-8").strip()
    dev_sha256 = verify_dev_hash(repo_root, config)
    dev_records = load_dev_records(repo_root, config["dev_dataset_path"])
    vocabulary = load_vocabulary(repo_root / "configs")

    from huggingface_hub import snapshot_download
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler
    import mlx.core as mx

    model_path = Path(snapshot_download(
        repo_id=config["model_id"], revision=config["model_revision"], local_files_only=True
    ))
    load_started = time.perf_counter()
    model, tokenizer, model_config = load(str(model_path), adapter_path=None, return_config=True)
    model_load_seconds = time.perf_counter() - load_started
    mx.random.seed(config["seed"])
    sampler = make_sampler(temp=config["temperature"])

    artifact_dir = repo_root / STAGE4_ARTIFACT_RELATIVE_DIR
    artifact_dir.mkdir(parents=True, exist_ok=True)
    predictions: List[Dict[str, Any]] = []
    try:
        for number, record in enumerate(dev_records, start=1):
            messages = [
                {"role": "system", "content": prompt_text},
                {"role": "user", "content": record["instruction"]},
            ]
            model_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            started = time.perf_counter()
            raw_output = ""
            finish_reason = None
            for chunk in stream_generate(
                model, tokenizer, model_prompt, max_tokens=config["max_generation_tokens"], sampler=sampler
            ):
                raw_output += chunk.text
                finish_reason = chunk.finish_reason or finish_reason
            latency_ms = (time.perf_counter() - started) * 1000.0
            ground_truth = {
                "intent": record["target"]["intent"],
                "category": record["target"]["category"],
                "needs_human": record["target"]["needs_human"],
            }
            evaluated = evaluate_prediction(
                raw_output, ground_truth, latency_ms, finish_reason == "length", vocabulary
            )
            evaluated.update({
                "source_index": record["metadata"]["source_index"],
                "stable_id": record["metadata"]["group_id"],
                "instruction": record["instruction"],
            })
            predictions.append(evaluated)
            if number % 10 == 0 or number == len(dev_records):
                print(f"Stage C4 Dev inference: {number}/{len(dev_records)}", flush=True)
    except BaseException:
        (artifact_dir / "base_inference_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise

    metrics = aggregate_metrics(predictions)
    confusion = write_confusion(artifact_dir / "intent_confusion.csv", predictions, sorted(vocabulary.intents))
    _write_jsonl(artifact_dir / "base_dev_predictions.jsonl", predictions)
    _write_json(artifact_dir / "base_metrics.json", metrics)
    write_error_cases(artifact_dir / "base_error_cases.csv", predictions)
    write_manual_qa(
        artifact_dir / "manual_qa_samples.csv", select_manual_qa(predictions, config["seed"], 30)
    )

    quantization = model_config.get("quantization", {})
    manifest = {
        "stage": "C4 Base Model Development Baseline",
        "model": {
            "repo_id": config["model_id"], "revision": config["model_revision"], "load_success": True,
            "local_snapshot": str(model_path), "architecture": (model_config.get("architectures") or [model_config.get("model_type")])[0],
            "parameter_size_label": "1.5B", "quantization_bits": quantization.get("bits"),
            "quantization_group_size": quantization.get("group_size"), "model_load_seconds": round(model_load_seconds, 6),
            "adapter_loaded": False,
        },
        "environment": {
            "python_version": sys.version.split()[0], "mlx_version": version("mlx"),
            "mlx_lm_version": version("mlx-lm"), "platform": platform.platform(),
        },
        "prompt_path": config["prompt_path"], "prompt_sha256": sha256_file(prompt_path),
        "dev_dataset_path": DEV_DATASET_RELATIVE_PATH, "dev_sha256": dev_sha256,
        "inference_config_path": DEFAULT_CONFIG_RELATIVE_PATH,
        "inference_config_sha256": sha256_file(config_path), "inference_config": config,
        "evaluated_rows": len(predictions),
        "artifacts": {
            "predictions": "artifacts/stage4/base_dev_predictions.jsonl",
            "metrics": "artifacts/stage4/base_metrics.json", "errors": "artifacts/stage4/base_error_cases.csv",
            "confusion": "artifacts/stage4/intent_confusion.csv", "manual_qa": "artifacts/stage4/manual_qa_samples.csv",
        },
        "final_evaluation_hash_confirmed_in_manifest": True,
        "final_evaluation_content_accessed": False,
    }
    _write_json(artifact_dir / "base_baseline_manifest.json", manifest)
    report = build_report(metrics, manifest, confusion)
    report_path = repo_root / "reports/base_baseline_report.md"
    report_path.write_text(report, encoding="utf-8")
    return {"metrics": metrics, "manifest": manifest}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the frozen Stage C4 Base Model Dev baseline.")
    return parser.parse_args()


def main() -> None:
    parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    result = run_baseline(repo_root)
    print(json.dumps({
        "model_load_success": result["manifest"]["model"]["load_success"],
        "evaluated_rows": result["metrics"]["evaluated_rows"],
        "metrics": result["metrics"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
