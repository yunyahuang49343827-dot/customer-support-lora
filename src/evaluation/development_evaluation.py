"""Stage C6 fair Dev comparison for the frozen Base and Candidate 01."""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from src.evaluation.base_baseline import (
    DEV_DATASET_RELATIVE_PATH,
    ERROR_TAG_ORDER,
    aggregate_metrics,
    evaluate_prediction,
    load_dev_records,
    sha256_file,
    stable_rank,
    verify_dev_hash,
    write_confusion,
    write_error_cases,
)
from src.evaluation.contracts import load_vocabulary


BASE_CONFIG_PATH = "configs/base_inference.json"
LORA_CONFIG_PATH = "configs/lora_dev_inference.json"
BASE_MANIFEST_PATH = "artifacts/stage4/base_baseline_manifest.json"
BASE_METRICS_PATH = "artifacts/stage4/base_metrics.json"
BASE_PREDICTIONS_PATH = "artifacts/stage4/base_dev_predictions.jsonl"
TRAINING_CONFIG_PATH = "configs/qlora_candidate_01.yaml"
TRAINING_MANIFEST_PATH = "artifacts/stage5/training_manifest.json"
STAGE6_DIR = "artifacts/stage6"
REPORT_PATH = "reports/stage6_development_evaluation.md"
RECOMMENDATION_PATH = "reports/stage6_controlled_iteration_recommendation.md"
ADAPTER_PATH = "artifacts/stage5/candidate_01/adapter"
PROMPT_SHA256 = "6b84135769b7348758e8cc21a3cb168465e00de5efaf59ff8a8459087db3dc3b"
MODEL_ID = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
MODEL_REVISION = "8b403126fc14f14cfc99bb4cfa72ecbc129ea677"
SHARED_CONFIG_KEYS = (
    "model_id", "model_revision", "framework", "seed", "temperature",
    "max_generation_tokens", "prompt_path", "dev_dataset_path",
    "dataset_hash_manifest_path", "warmup_runs", "concurrency", "local_files_only",
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def escalation_metrics(predictions: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Compute conservative positive-class metrics; invalid predictions count as FN for true labels."""
    tp = fp = fn = tn = invalid = 0
    for row in predictions:
        truth = row["ground_truth_needs_human"] is True
        predicted = (row.get("parsed_output") or {}).get("needs_human")
        if type(predicted) is not bool:
            invalid += 1
        if truth and predicted is True:
            tp += 1
        elif truth:
            fn += 1
        elif predicted is True:
            fp += 1
        elif predicted is False:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positive": tp, "false_positive": fp, "false_negative": fn, "true_negative": tn,
        "invalid_or_missing": invalid,
        "precision_percent": round(100 * precision, 6),
        "recall_percent": round(100 * recall, 6),
        "f1_percent": round(100 * f1, 6),
    }


def metric_delta(base: float, lora: float, unit: str = "percentage_points") -> Dict[str, Any]:
    return {"base": base, "lora": lora, "absolute_delta": round(lora - base, 6), "unit": unit}


def augment_metrics(metrics: Mapping[str, Any], predictions: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    result = json.loads(json.dumps(metrics))
    result["secondary"]["escalation_positive_class"] = escalation_metrics(predictions)
    return result


def compare_metrics(
    base: Mapping[str, Any], lora: Mapping[str, Any],
    base_predictions: Sequence[Mapping[str, Any]], lora_predictions: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    bp, lp = base["primary"], lora["primary"]
    bs, ls = base["secondary"], lora["secondary"]
    bo, lo = base["operational"], lora["operational"]
    bpr, lpr = bs["escalation_positive_class"], ls["escalation_positive_class"]
    metrics = {
        "intent_accuracy": metric_delta(bp["intent_accuracy_percent"], lp["intent_accuracy_percent"]),
        "category_accuracy": metric_delta(bs["category_accuracy_percent"], ls["category_accuracy_percent"]),
        "json_valid_rate": metric_delta(bp["json_valid_rate_percent"], lp["json_valid_rate_percent"]),
        "schema_compliance": metric_delta(bp["schema_compliance_percent"], lp["schema_compliance_percent"]),
        "escalation_accuracy": metric_delta(bs["escalation_accuracy_percent"], ls["escalation_accuracy_percent"]),
        "escalation_precision": metric_delta(bpr["precision_percent"], lpr["precision_percent"]),
        "escalation_recall": metric_delta(bpr["recall_percent"], lpr["recall_percent"]),
        "escalation_f1": metric_delta(bpr["f1_percent"], lpr["f1_percent"]),
        "mean_latency": metric_delta(bo["mean_latency_ms"], lo["mean_latency_ms"], "milliseconds"),
        "p95_latency": metric_delta(bo["p95_latency_ms"], lo["p95_latency_ms"], "milliseconds"),
    }
    base_errors = Counter(tag for row in base_predictions for tag in row["error_tags"])
    lora_errors = Counter(tag for row in lora_predictions for tag in row["error_tags"])
    errors = {
        tag: {"base": base_errors[tag], "lora": lora_errors[tag], "delta": lora_errors[tag] - base_errors[tag]}
        for tag in ERROR_TAG_ORDER
    }
    return {"evaluated_rows": len(lora_predictions), "metrics": metrics, "error_tag_comparison": errors}


RISK_PATTERNS: Tuple[Tuple[str, re.Pattern[str]], ...] = (
    ("fabricated_contact_details", re.compile(r"(?:[\w.+-]+@[\w.-]+\.[a-z]{2,}|(?:\+?\d[\d ()-]{7,}\d)|https?://|www\.)", re.I)),
    ("fabricated_24_7_availability", re.compile(r"\b(?:24\s*/\s*7|24 hours a day|around the clock)\b", re.I)),
    ("fabricated_fees_or_timelines", re.compile(r"(?:[$€£]\s*\d|\b\d+\s*(?:business\s+)?(?:minutes?|hours?|days?|weeks?)\b|\bfee (?:is|of)\b)", re.I)),
    ("unsupported_action_completion", re.compile(r"\b(?:we(?:'ve| have)?\s+)?(?:processed|completed|issued|cancelled|canceled|refunded)\b", re.I)),
    ("unsupported_update_claim", re.compile(r"\b(?:has been|was|is now)\s+(?:updated|changed|shipped|delivered|activated|deleted)\b", re.I)),
    ("asks_for_sensitive_secret", re.compile(r"\b(?:provide|share|send|enter)\b.{0,35}\b(?:password|authentication code|verification code|full (?:credit |debit )?card|cvv|cvc)\b", re.I)),
    ("unsupported_guarantee", re.compile(r"\b(?:guarantee|guaranteed|definitely|assure you that)\b", re.I)),
)
ESCALATION_LANGUAGE = re.compile(r"\b(?:human agent|customer service|support (?:agent|team)|representative)\b", re.I)
RISK_FLAG_NAMES = tuple(name for name, _ in RISK_PATTERNS) + ("unnecessary_escalation",)


def screen_response(response: Any, truth_needs_human: bool, predicted_needs_human: Any) -> List[str]:
    text = response if isinstance(response, str) else ""
    flags = [name for name, pattern in RISK_PATTERNS if pattern.search(text)]
    if not truth_needs_human and (predicted_needs_human is True or ESCALATION_LANGUAGE.search(text)):
        flags.append("unnecessary_escalation")
    return flags


def write_risk_qa(
    path: Path, base_predictions: Sequence[Mapping[str, Any]], lora_predictions: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    base_by_id = {row["source_index"]: row for row in base_predictions}
    fields = [
        "source_index", "instruction", "ground_truth_intent", "ground_truth_needs_human",
        "base_response", "base_risk_flags", "lora_response", "lora_risk_flags", "requires_manual_review",
    ]
    fields.extend(f"base_{name}" for name in RISK_FLAG_NAMES)
    fields.extend(f"lora_{name}" for name in RISK_FLAG_NAMES)
    summary = {
        "base": Counter({name: 0 for name in RISK_FLAG_NAMES}),
        "lora": Counter({name: 0 for name in RISK_FLAG_NAMES}),
        "rows": len(lora_predictions),
    }
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for lora in lora_predictions:
            base = base_by_id[lora["source_index"]]
            base_parsed, lora_parsed = base.get("parsed_output") or {}, lora.get("parsed_output") or {}
            base_flags = screen_response(base_parsed.get("response"), base["ground_truth_needs_human"], base_parsed.get("needs_human"))
            lora_flags = screen_response(lora_parsed.get("response"), lora["ground_truth_needs_human"], lora_parsed.get("needs_human"))
            summary["base"].update(base_flags)
            summary["lora"].update(lora_flags)
            output_row = {
                "source_index": lora["source_index"], "instruction": lora["instruction"],
                "ground_truth_intent": lora["ground_truth_intent"],
                "ground_truth_needs_human": lora["ground_truth_needs_human"],
                "base_response": base_parsed.get("response", ""), "base_risk_flags": ";".join(base_flags),
                "lora_response": lora_parsed.get("response", ""), "lora_risk_flags": ";".join(lora_flags),
                "requires_manual_review": bool(base_flags or lora_flags),
            }
            output_row.update({f"base_{name}": name in base_flags for name in RISK_FLAG_NAMES})
            output_row.update({f"lora_{name}": name in lora_flags for name in RISK_FLAG_NAMES})
            writer.writerow(output_row)
    return {
        "screened_rows": len(lora_predictions),
        "base_flag_counts": dict(sorted(summary["base"].items())),
        "lora_flag_counts": dict(sorted(summary["lora"].items())),
        "automated_screening_is_not_complete_quality_judgment": True,
    }


def select_paired_qa(
    base_predictions: Sequence[Mapping[str, Any]], lora_predictions: Sequence[Mapping[str, Any]], seed: int, count: int = 30,
) -> List[Tuple[Mapping[str, Any], Mapping[str, Any]]]:
    base_by_id = {row["source_index"]: row for row in base_predictions}
    pairs = [(base_by_id[row["source_index"]], row) for row in lora_predictions]
    selected: List[Tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    seen = set()

    def take(candidates: Sequence[Tuple[Mapping[str, Any], Mapping[str, Any]]], label: str, limit: int) -> None:
        ordered = sorted(candidates, key=lambda pair: stable_rank(seed, "stage6_qa", label, pair[1]["source_index"]))
        for pair in ordered:
            key = pair[1]["source_index"]
            if key in seen:
                continue
            selected.append(pair)
            seen.add(key)
            if sum(1 for chosen in selected if chosen[1]["source_index"] in seen) >= count:
                break
            limit -= 1
            if limit == 0:
                break

    take([p for p in pairs if not p[1]["error_tags"]], "lora_correct", 7)
    take([p for p in pairs if not p[1]["intent_correct"]], "lora_wrong_intent", 7)
    take([p for p in pairs if p[1]["ground_truth_needs_human"] is True], "escalation_true", 5)
    take([p for p in pairs if p[1]["ground_truth_needs_human"] is False], "escalation_false", 5)
    take([p for p in pairs if not p[1]["schema_compliant"]], "schema_failure", 4)
    take([p for p in pairs if not p[0]["intent_correct"] and p[1]["intent_correct"]], "intent_reduced", 4)
    for pair in sorted(pairs, key=lambda p: stable_rank(seed, "stage6_qa", "fill", p[1]["source_index"])):
        if len(selected) >= count:
            break
        if pair[1]["source_index"] not in seen:
            selected.append(pair)
            seen.add(pair[1]["source_index"])
    return selected[:count]


def write_manual_qa(path: Path, pairs: Sequence[Tuple[Mapping[str, Any], Mapping[str, Any]]]) -> None:
    fields = [
        "source_index", "instruction", "ground_truth_intent", "ground_truth_category", "ground_truth_needs_human",
        "base_output", "lora_output", "base_error_tags", "lora_error_tags",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for base, lora in pairs:
            writer.writerow({
                "source_index": lora["source_index"], "instruction": lora["instruction"],
                "ground_truth_intent": lora["ground_truth_intent"], "ground_truth_category": lora["ground_truth_category"],
                "ground_truth_needs_human": lora["ground_truth_needs_human"],
                "base_output": base["raw_model_output"], "lora_output": lora["raw_model_output"],
                "base_error_tags": ";".join(base["error_tags"]), "lora_error_tags": ";".join(lora["error_tags"]),
            })


def top_confusions(predictions: Sequence[Mapping[str, Any]], limit: int = 10) -> List[Dict[str, Any]]:
    allowed = load_vocabulary().intents
    counts = Counter()
    for row in predictions:
        predicted = (row.get("parsed_output") or {}).get("intent")
        if predicted in allowed and predicted != row["ground_truth_intent"]:
            counts[(row["ground_truth_intent"], predicted)] += 1
    return [
        {"ground_truth_intent": truth, "predicted_intent": predicted, "count": count}
        for (truth, predicted), count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]
    ]


def validate_inputs(repo_root: Path) -> Dict[str, Any]:
    base_config = read_json(repo_root / BASE_CONFIG_PATH)
    lora_config = read_json(repo_root / LORA_CONFIG_PATH)
    base_manifest = read_json(repo_root / BASE_MANIFEST_PATH)
    training_manifest = read_json(repo_root / TRAINING_MANIFEST_PATH)
    if any(base_config[key] != lora_config[key] for key in SHARED_CONFIG_KEYS):
        raise ValueError("Base and LoRA inference contracts differ")
    if base_config["adapter_path"] is not None or lora_config["adapter_path"] != ADAPTER_PATH:
        raise ValueError("Adapter paths violate the Stage C6 contract")
    if (lora_config["model_id"], lora_config["model_revision"]) != (MODEL_ID, MODEL_REVISION):
        raise ValueError("Candidate model identity or revision changed")
    prompt_hash = sha256_file(repo_root / lora_config["prompt_path"])
    if prompt_hash != PROMPT_SHA256 or base_manifest["prompt_sha256"] != prompt_hash or training_manifest["prompt_sha256"] != prompt_hash:
        raise ValueError("Frozen prompt SHA mismatch")
    if base_manifest["inference_config_sha256"] != sha256_file(repo_root / BASE_CONFIG_PATH):
        raise ValueError("C4 Base config changed after baseline")
    if training_manifest["model_id"] != MODEL_ID or training_manifest["model_revision"] != MODEL_REVISION:
        raise ValueError("Candidate training model identity mismatch")
    if training_manifest["adapter_path"] != ADAPTER_PATH or not training_manifest["training_success"] or not training_manifest["reload_success"]:
        raise ValueError("Candidate 01 training manifest is not successful or uses another adapter")
    if training_manifest["config_sha256"] != sha256_file(repo_root / TRAINING_CONFIG_PATH):
        raise ValueError("Candidate 01 training config no longer matches its manifest")
    adapter = repo_root / ADAPTER_PATH
    if not (adapter / "adapter_config.json").is_file() or not (adapter / "adapters.safetensors").is_file():
        raise ValueError("Candidate 01 adapter artifacts are incomplete")
    dev_hash = verify_dev_hash(repo_root, lora_config)
    dev_records = load_dev_records(repo_root, lora_config["dev_dataset_path"])
    base_predictions = read_jsonl(repo_root / BASE_PREDICTIONS_PATH)
    base_metrics = read_json(repo_root / BASE_METRICS_PATH)
    if len(base_predictions) != 300 or len({row["source_index"] for row in base_predictions}) != 300:
        raise ValueError("C4 Base predictions must contain exactly 300 unique Dev rows")
    if aggregate_metrics(base_predictions) != base_metrics:
        raise ValueError("C4 Base metrics do not reproduce from frozen predictions")
    expected_ids = {row["metadata"]["source_index"] for row in dev_records}
    if {row["source_index"] for row in base_predictions} != expected_ids:
        raise ValueError("C4 Base predictions do not match the current frozen Dev membership")
    return {
        "base_config": base_config, "lora_config": lora_config, "base_manifest": base_manifest,
        "training_manifest": training_manifest, "dev_hash": dev_hash, "dev_records": dev_records,
        "base_predictions": base_predictions, "base_metrics": augment_metrics(base_metrics, base_predictions),
    }


def development_decision(comparison: Mapping[str, Any]) -> Dict[str, Any]:
    metrics = comparison["metrics"]
    checks = {
        "intent_at_least_base_plus_3pp": metrics["intent_accuracy"]["lora"] >= 34.333333,
        "json_regression_within_1pp": metrics["json_valid_rate"]["absolute_delta"] >= -1.0,
        "schema_regression_within_1pp": metrics["schema_compliance"]["absolute_delta"] >= -1.0,
        "category_drop_less_than_3pp": metrics["category_accuracy"]["absolute_delta"] > -3.0,
        "escalation_drop_less_than_3pp": metrics["escalation_accuracy"]["absolute_delta"] > -3.0,
    }
    strong = all(checks.values())
    return {
        "decision": "candidate_01_strong_enough_to_freeze_pending_manual_response_qa" if strong else "controlled_iteration_justified",
        "controlled_iteration_needed": not strong,
        "checks": checks,
        "not_a_locked_promotion_decision": True,
        "manual_response_qa_still_required": True,
    }


def build_recommendation(comparison: Mapping[str, Any]) -> str:
    intent = comparison["metrics"]["intent_accuracy"]
    schema = comparison["metrics"]["schema_compliance"]
    return f"""# Stage C6 Controlled Iteration Recommendation

Candidate 01 reached {intent['lora']:.6f}% intent accuracy ({intent['absolute_delta']:+.6f} pp vs Base) and {schema['lora']:.6f}% schema compliance ({schema['absolute_delta']:+.6f} pp).

## Primary change

Increase LoRA rank from 8 to 16 while keeping the frozen prompt, decoding, data, learning rate, target modules, target layers, and all other settings unchanged. This is a single controlled capacity hypothesis and is not an automated sweep.

## Secondary change

None proposed. A second variable would confound the capacity hypothesis.

Candidate 02 must not be trained until a human explicitly approves this recommendation.
"""


def build_report(
    base: Mapping[str, Any], lora: Mapping[str, Any], comparison: Mapping[str, Any],
    decision: Mapping[str, Any], risk: Mapping[str, Any], base_confusions: Sequence[Mapping[str, Any]],
    lora_confusions: Sequence[Mapping[str, Any]], dev_hash: str,
) -> str:
    def metric_line(name: str) -> str:
        row = comparison["metrics"][name]
        suffix = "pp" if row["unit"] == "percentage_points" else "ms"
        return f"- {name}: Base {row['base']:.6f}, LoRA {row['lora']:.6f}, delta {row['absolute_delta']:+.6f} {suffix}"
    metric_lines = "\n".join(metric_line(name) for name in comparison["metrics"])
    errors = sorted(comparison["error_tag_comparison"].items(), key=lambda item: (item[1]["delta"], item[0]))
    error_lines = "\n".join(f"- `{tag}`: Base {v['base']}, LoRA {v['lora']}, delta {v['delta']:+d}" for tag, v in errors)
    confusion_lines = lambda rows: "\n".join(
        f"- `{r['ground_truth_intent']}` → `{r['predicted_intent']}`: {r['count']}" for r in rows
    ) or "- None"
    esc = lora["secondary"]["escalation_positive_class"]
    return f"""# Stage C6 Development Evaluation

## Goal

Fairly compare the frozen C4 Base artifacts with QLoRA Candidate 01 on all 300 Frozen Dev rows.

## Fair Comparison Contract

Base and LoRA use the same model revision, frozen system prompt (SHA-256 `{PROMPT_SHA256}`), customer instruction, greedy temperature 0 decoding, 512-token limit, evaluator, parser, schema, taxonomies, and escalation policy. The only inference difference is the Candidate 01 adapter.

## Dataset Boundary

Dev SHA-256: `{dev_hash}`. Dev rows: 300. Locked Test content accessed: no.

## Base Metrics

The immutable C4 predictions and metrics were reused and reproduced from all 300 stored predictions; Base inference was not rerun.

## LoRA Metrics

- Intent accuracy: {lora['primary']['intent_accuracy_percent']:.6f}%
- Category accuracy: {lora['secondary']['category_accuracy_percent']:.6f}%
- JSON valid rate: {lora['primary']['json_valid_rate_percent']:.6f}%
- Schema compliance: {lora['primary']['schema_compliance_percent']:.6f}%
- Escalation accuracy: {lora['secondary']['escalation_accuracy_percent']:.6f}%

## Metric Delta

{metric_lines}

## Escalation Precision / Recall / F1

- Precision: {esc['precision_percent']:.6f}%
- Recall: {esc['recall_percent']:.6f}%
- F1: {esc['f1_percent']:.6f}%
- TP / FP / FN / TN: {esc['true_positive']} / {esc['false_positive']} / {esc['false_negative']} / {esc['true_negative']}
- Invalid or missing boolean: {esc['invalid_or_missing']}

## Error Reduction

{error_lines}

## Intent Confusions

Top Base confusions:

{confusion_lines(base_confusions)}

Top LoRA confusions:

{confusion_lines(lora_confusions)}

## Response QA

`artifacts/stage6/manual_qa_samples.csv` contains 30 deterministic seed-42 paired Base/LoRA cases covering correct cases, wrong intents, both escalation labels, and schema failures when present. Manual review remains required.

## Response Risk QA

`artifacts/stage6/response_risk_qa.csv` screens all paired rows. Base flags: `{json.dumps(risk['base_flag_counts'], sort_keys=True)}`. LoRA flags: `{json.dumps(risk['lora_flag_counts'], sort_keys=True)}`. These conservative rules are only risk screening and cannot establish complete response quality.

## Latency

- Base mean / p95: {base['operational']['mean_latency_ms']:.3f} / {base['operational']['p95_latency_ms']:.3f} ms
- LoRA mean / p95: {lora['operational']['mean_latency_ms']:.3f} / {lora['operational']['p95_latency_ms']:.3f} ms

## Development Decision

`{decision['decision']}`. This is development-oriented and is not a Locked Promotion Gate decision.

## Controlled Iteration Needed?

`{str(decision['controlled_iteration_needed']).lower()}`. No retraining was started.

## Limitations

Automated exact-match metrics and heuristic risk flags do not replace blinded human response review. Latency is machine-specific. No Locked Test inspection or inference occurred.

## Stage C6 Conclusion

Candidate 01 evaluation is complete on the full Dev set. Stage C6.5 and Stage C7 were not entered.
"""


def finalize(repo_root: Path, lora_predictions: Sequence[Mapping[str, Any]], contract: Mapping[str, Any]) -> Dict[str, Any]:
    artifact_dir = repo_root / STAGE6_DIR
    artifact_dir.mkdir(parents=True, exist_ok=True)
    base_predictions = contract["base_predictions"]
    if len(lora_predictions) != 300 or {r["source_index"] for r in lora_predictions} != {r["source_index"] for r in base_predictions}:
        raise ValueError("LoRA predictions must cover exactly the same 300 Dev rows as Base")
    lora_metrics = augment_metrics(aggregate_metrics(lora_predictions), lora_predictions)
    base_metrics = contract["base_metrics"]
    comparison = compare_metrics(base_metrics, lora_metrics, base_predictions, lora_predictions)
    decision = development_decision(comparison)
    write_jsonl(artifact_dir / "lora_dev_predictions.jsonl", lora_predictions)
    write_json(artifact_dir / "lora_metrics.json", lora_metrics)
    write_json(artifact_dir / "base_vs_lora_comparison.json", comparison)
    write_error_cases(artifact_dir / "lora_error_cases.csv", lora_predictions)
    write_confusion(artifact_dir / "intent_confusion_lora.csv", lora_predictions, sorted(load_vocabulary(repo_root / "configs").intents))
    qa_pairs = select_paired_qa(base_predictions, lora_predictions, contract["lora_config"]["seed"], 30)
    write_manual_qa(artifact_dir / "manual_qa_samples.csv", qa_pairs)
    risk = write_risk_qa(artifact_dir / "response_risk_qa.csv", base_predictions, lora_predictions)
    write_json(artifact_dir / "response_risk_summary.json", risk)
    base_confusions, lora_confusions = top_confusions(base_predictions), top_confusions(lora_predictions)
    summary = {"base": base_confusions, "lora": lora_confusions}
    write_json(artifact_dir / "intent_confusion_summary.json", summary)
    manifest = {
        "stage": "C6 Development Evaluation", "dev_rows": 300, "dev_sha256": contract["dev_hash"],
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION, "adapter_path": ADAPTER_PATH,
        "prompt_sha256": PROMPT_SHA256, "base_config_sha256": sha256_file(repo_root / BASE_CONFIG_PATH),
        "lora_config_sha256": sha256_file(repo_root / LORA_CONFIG_PATH), "same_evaluator": True,
        "base_inference_rerun": False, "adapter_load_success": True,
        "locked_content_accessed": False, "locked_inference_performed": False,
        "decision": decision,
    }
    write_json(artifact_dir / "stage6_manifest.json", manifest)
    (repo_root / REPORT_PATH).write_text(
        build_report(base_metrics, lora_metrics, comparison, decision, risk, base_confusions, lora_confusions, contract["dev_hash"]),
        encoding="utf-8",
    )
    recommendation = repo_root / RECOMMENDATION_PATH
    if decision["controlled_iteration_needed"]:
        recommendation.write_text(build_recommendation(comparison), encoding="utf-8")
    elif recommendation.exists():
        recommendation.unlink()
    return {"base_metrics": base_metrics, "lora_metrics": lora_metrics, "comparison": comparison, "decision": decision, "risk": risk}


def run(repo_root: Path) -> Dict[str, Any]:
    contract = validate_inputs(repo_root)
    config = contract["lora_config"]
    prompt_text = (repo_root / config["prompt_path"]).read_text(encoding="utf-8").strip()
    vocabulary = load_vocabulary(repo_root / "configs")
    from huggingface_hub import snapshot_download
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler
    import mlx.core as mx

    snapshot = Path(snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_files_only=True))
    model, tokenizer = load(str(snapshot), adapter_path=str(repo_root / ADAPTER_PATH))
    mx.random.seed(config["seed"])
    sampler = make_sampler(temp=config["temperature"])
    predictions: List[Dict[str, Any]] = []
    artifact_dir = repo_root / STAGE6_DIR
    artifact_dir.mkdir(parents=True, exist_ok=True)
    try:
        for number, record in enumerate(contract["dev_records"], 1):
            prompt = tokenizer.apply_chat_template([
                {"role": "system", "content": prompt_text},
                {"role": "user", "content": record["instruction"]},
            ], tokenize=False, add_generation_prompt=True)
            started = time.perf_counter()
            raw_output, finish_reason = "", None
            for chunk in stream_generate(model, tokenizer, prompt, max_tokens=config["max_generation_tokens"], sampler=sampler):
                raw_output += chunk.text
                finish_reason = chunk.finish_reason or finish_reason
            ground_truth = {
                "intent": record["target"]["intent"], "category": record["target"]["category"],
                "needs_human": record["target"]["needs_human"],
            }
            evaluated = evaluate_prediction(raw_output, ground_truth, (time.perf_counter() - started) * 1000, finish_reason == "length", vocabulary)
            evaluated.update({
                "source_index": record["metadata"]["source_index"], "stable_id": record["metadata"]["group_id"],
                "instruction": record["instruction"],
            })
            predictions.append(evaluated)
            if number % 10 == 0 or number == 300:
                print(f"Stage C6 Candidate 01 Dev inference: {number}/300", flush=True)
    except BaseException:
        (artifact_dir / "lora_inference_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    return finalize(repo_root, predictions, contract)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage C6 Candidate 01 Dev evaluation")
    parser.parse_args()
    result = run(Path(__file__).resolve().parents[2])
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
