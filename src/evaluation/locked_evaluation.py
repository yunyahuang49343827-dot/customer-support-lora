"""Stage C7 one-time Locked Base versus frozen Candidate 01 evaluation."""

from __future__ import annotations

import csv
import gc
import json
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from src.evaluation.base_baseline import (
    ERROR_TAG_ORDER,
    aggregate_metrics,
    evaluate_prediction,
    sha256_file,
    stable_rank,
    write_error_cases,
)
from src.evaluation.contracts import load_vocabulary
from src.evaluation.development_evaluation import (
    RISK_FLAG_NAMES,
    augment_metrics,
    compare_metrics,
    escalation_metrics,
    screen_response,
    top_confusions,
    write_risk_qa,
)
from src.evaluation.freeze_stage6_5 import binary_line_count


STAGE7_DIR = "artifacts/stage7"
REPORT_PATH = "reports/stage7_locked_evaluation.md"
LOCKED_PATH = "data/processed/locked_test.jsonl"
FREEZE_MANIFEST_PATH = "artifacts/stage6_5/freeze_manifest.json"
FROZEN_COMPONENT_HASHES_PATH = "artifacts/stage6_5/frozen_component_hashes.json"
FROZEN_INFERENCE_PATH = "artifacts/stage6_5/frozen_inference_contract.json"
FREEZE_VALIDATION_PATH = "artifacts/stage6_5/freeze_validation.json"
ADAPTER_INVENTORY_PATH = "artifacts/stage6_5/adapter_inventory.json"
BASE_MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
BASE_REVISION = "8b403126fc14f14cfc99bb4cfa72ecbc129ea677"
CANDIDATE = "candidate_01"
ADAPTER_PATH = "artifacts/stage5/candidate_01/adapter"
ADAPTER_SHA256 = "da763e47f3c6051defb605345e9aaccd989a8768b804c802606a7f8317fc2c16"
PROMPT_PATH = "prompts/base_system_prompt.txt"
PROMPT_SHA256 = "6b84135769b7348758e8cc21a3cb168465e00de5efaf59ff8a8459087db3dc3b"
INFERENCE_SHA256 = "1225f7169ea6f2394e40478dcb1df572768d44c08f73f990866393b0d0b26752"
LOCKED_SHA256 = "b7f7af8c5e366c743fafd68c8c8f3e7a2b101dfce53e63bf1f7a8ead0bce1fac"
EXPECTED_EVALUATOR_HASHES = {
    "src/evaluation/base_baseline.py": "184ca998f1a29dcf99cc4bc48788d09ad0c177314a16ac7eb4f21c0caf64fb52",
    "src/evaluation/contracts.py": "e2f8bb620a3b7d44f98c5ca0a96e985d98b944fe7d68b0a265cd26aa425a31a3",
    "src/evaluation/development_evaluation.py": "2cb2f7f37f4b5b03837eb8b2cec17355a3df77cdc1df07939f990bfb38ba9a37",
}
EXPECTED_SCHEMA_HASHES = {
    "configs/output_schema.json": "6a3d0900b3485e5a24205ea5f7ae42360d598a6c7a7fc6d97cde2d8fde88daa2",
    "configs/intent_taxonomy.json": "8e99fdfcdd90a2bcc2dd733503e936d5f0785ef4548468fa5923b4d965e3422f",
    "configs/category_taxonomy.json": "694f2c4a56fe662d795a1315781ed7c86f68114012ea0012ead43cefc4a5ba79",
    "configs/escalation_policy.json": "c07898c29254bc584c944007bc2fd2785c9db1e70fedda0aeb7c0ec7c2ef0f2d",
}
KNOWN_LIMITATION_EN = (
    "QLoRA significantly improved structured classification behavior, but manual QA found that generated "
    "responses can still contain unsupported policy or capability claims. Therefore, the fine-tuned model "
    "should not be treated as an enterprise factual authority."
)
KNOWN_LIMITATION_ZH = (
    "QLoRA 顯著改善 structured classification behavior，但人工 QA 發現生成式 response 仍可能產生 "
    "unsupported policy/capability claims，因此模型不應直接被視為企業 factual authority。"
)
HUMAN_FIELDS = (
    "review_relevance", "review_fabrication", "review_unsupported_action",
    "review_sensitive_data", "review_unnecessary_escalation", "review_overall", "review_note",
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def check_item(name: str, expected: Any, actual: Any) -> Dict[str, Any]:
    return {"name": name, "expected": expected, "actual": actual, "status": "PASS" if actual == expected else "FAIL"}


def stage8_outputs(repo_root: Path) -> List[str]:
    found = []
    for pattern in ("artifacts/stage8*", "reports/stage8*", "**/promotion_decision.json", "**/promotion_report.md"):
        found.extend(path.relative_to(repo_root).as_posix() for path in repo_root.glob(pattern))
    return sorted(set(found))


def verify_freeze(repo_root: Path) -> Dict[str, Any]:
    """Verify all critical hashes before any Locked JSON parsing."""
    manifest = read_json(repo_root / FREEZE_MANIFEST_PATH)
    hashes = read_json(repo_root / FROZEN_COMPONENT_HASHES_PATH)
    inference = read_json(repo_root / FROZEN_INFERENCE_PATH)
    validation = read_json(repo_root / FREEZE_VALIDATION_PATH)
    inventory = read_json(repo_root / ADAPTER_INVENTORY_PATH)
    items = [
        check_item("freeze_status", "PASS", manifest["freeze_status"]),
        check_item("freeze_validation_fail_count", 0, validation["fail_count"]),
        check_item("candidate", CANDIDATE, manifest["candidate"]),
        check_item("base_model", BASE_MODEL, manifest["base_model"]),
        check_item("base_revision", BASE_REVISION, manifest["base_revision"]),
        check_item("adapter_path", ADAPTER_PATH, manifest["adapter"]["path"]),
        check_item("adapter_sha256_manifest", ADAPTER_SHA256, manifest["adapter"]["final_adapter_sha256"]),
        check_item("adapter_sha256_current", ADAPTER_SHA256, sha256_file(repo_root / ADAPTER_PATH / "adapters.safetensors")),
        check_item("adapter_inventory", inventory, manifest["adapter"]),
        check_item("prompt_sha256", PROMPT_SHA256, sha256_file(repo_root / PROMPT_PATH)),
        check_item("inference_contract_sha256", INFERENCE_SHA256, sha256_file(repo_root / FROZEN_INFERENCE_PATH)),
        check_item("locked_sha256", LOCKED_SHA256, sha256_file(repo_root / LOCKED_PATH)),
        check_item("locked_rows_binary", 300, binary_line_count(repo_root / LOCKED_PATH)),
        check_item("inference_model", BASE_MODEL, inference["base_model"]),
        check_item("inference_revision", BASE_REVISION, inference["base_revision"]),
        check_item("inference_adapter", ADAPTER_PATH, inference["candidate_adapter"]),
        check_item("inference_decoding", {
            "concurrency": 1, "max_generated_tokens": 512, "seed": 42,
            "strategy": "deterministic_greedy", "temperature": 0.0, "warmup_runs": 0,
        }, inference["decoding"]),
        check_item("promotion_gate_sha256", hashes["promotion_gate"]["sha256"], sha256_file(repo_root / hashes["promotion_gate"]["path"])),
        check_item("stage8_outputs_before_run", [], stage8_outputs(repo_root)),
    ]
    frozen_evaluators = {entry["path"]: entry["sha256"] for entry in hashes["evaluator_sources"]}
    frozen_schemas = {entry["path"]: entry["sha256"] for entry in hashes["schema_taxonomy_escalation"]}
    for path, expected in EXPECTED_EVALUATOR_HASHES.items():
        items.append(check_item(f"evaluator_manifest:{path}", expected, frozen_evaluators.get(path)))
        items.append(check_item(f"evaluator_current:{path}", expected, sha256_file(repo_root / path)))
    for path, expected in EXPECTED_SCHEMA_HASHES.items():
        items.append(check_item(f"schema_manifest:{path}", expected, frozen_schemas.get(path)))
        items.append(check_item(f"schema_current:{path}", expected, sha256_file(repo_root / path)))
    failed = [item for item in items if item["status"] == "FAIL"]
    result = {
        "status": "PASS" if not failed else "ABORTED",
        "checked_before_locked_json_parse": True,
        "pass_count": len(items) - len(failed), "fail_count": len(failed), "items": items,
        "freeze_manifest_sha256": sha256_file(repo_root / FREEZE_MANIFEST_PATH),
    }
    return {"result": result, "manifest": manifest, "hashes": hashes, "inference": inference}


def load_locked_records(repo_root: Path) -> List[Dict[str, Any]]:
    rows = read_jsonl(repo_root / LOCKED_PATH)
    if len(rows) != 300:
        raise ValueError(f"Locked Test must contain exactly 300 records; found {len(rows)}")
    if len({row["metadata"]["source_index"] for row in rows}) != 300:
        raise ValueError("Locked Test source indices must be unique")
    return rows


def run_model(
    repo_root: Path,
    role: str,
    model_path: Path,
    adapter_path: Optional[str],
    records: Sequence[Mapping[str, Any]],
    prompt_text: str,
    inference: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler
    import mlx.core as mx

    model, tokenizer = load(str(model_path), adapter_path=adapter_path)
    mx.random.seed(inference["decoding"]["seed"])
    sampler = make_sampler(temp=inference["decoding"]["temperature"])
    stage_dir = repo_root / STAGE7_DIR
    in_progress = stage_dir / f".{role}_locked_predictions.inprogress.jsonl"
    final_path = stage_dir / f"{role}_locked_predictions.jsonl"
    if final_path.exists() or in_progress.exists():
        raise RuntimeError(f"Refusing to rerun or overwrite existing {role} Locked attempts")
    predictions: List[Dict[str, Any]] = []
    vocabulary = load_vocabulary(repo_root / "configs")
    with in_progress.open("w", encoding="utf-8", newline="\n") as handle:
        for number, record in enumerate(records, 1):
            messages = [
                {"role": "system", "content": prompt_text},
                {"role": "user", "content": record["instruction"]},
            ]
            model_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            started = time.perf_counter()
            raw_output, finish_reason, generation_error = "", None, None
            try:
                for chunk in stream_generate(
                    model, tokenizer, model_prompt,
                    max_tokens=inference["decoding"]["max_generated_tokens"], sampler=sampler,
                ):
                    raw_output += chunk.text
                    finish_reason = chunk.finish_reason or finish_reason
            except Exception as error:  # keep every attempted row in the denominator
                generation_error = f"{type(error).__name__}: {error}"
            latency_ms = (time.perf_counter() - started) * 1000.0
            ground_truth = {
                "intent": record["target"]["intent"], "category": record["target"]["category"],
                "needs_human": record["target"]["needs_human"],
            }
            evaluated = evaluate_prediction(
                raw_output, ground_truth, latency_ms, finish_reason == "length", vocabulary
            )
            evaluated.update({
                "model_role": role,
                "source_index": record["metadata"]["source_index"],
                "stable_id": record["metadata"]["group_id"],
                "instruction": record["instruction"],
                "expected_intent": ground_truth["intent"],
                "expected_category": ground_truth["category"],
                "expected_needs_human": ground_truth["needs_human"],
                "latency_ms": evaluated["inference_latency_ms"],
                "generation_error": generation_error,
            })
            predictions.append(evaluated)
            handle.write(json.dumps(evaluated, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            if number % 10 == 0 or number == len(records):
                print(f"Stage C7 {role} Locked inference: {number}/{len(records)}", flush=True)
    in_progress.replace(final_path)
    del model
    gc.collect()
    if hasattr(mx, "clear_cache"):
        mx.clear_cache()
    return predictions


def metric_delta(base: float, lora: float, unit: str = "percentage_points") -> Dict[str, Any]:
    return {"base": base, "lora": lora, "absolute_delta": round(lora - base, 6), "unit": unit}


def escalation_comparison(base: Mapping[str, Any], lora: Mapping[str, Any]) -> Dict[str, Any]:
    count_keys = ("true_positive", "false_positive", "false_negative", "true_negative", "invalid_or_missing")
    rate_keys = ("precision_percent", "recall_percent", "f1_percent")
    return {
        "positive_class": "needs_human=true",
        "base": dict(base), "lora": dict(lora),
        "delta": {
            **{key: lora[key] - base[key] for key in count_keys},
            **{key: round(lora[key] - base[key], 6) for key in rate_keys},
        },
        "all_attempted_rows_preserved_in_denominator": True,
    }


def pair_flags(base: Mapping[str, Any], lora: Mapping[str, Any]) -> Tuple[List[str], List[str]]:
    base_parsed, lora_parsed = base.get("parsed_output") or {}, lora.get("parsed_output") or {}
    return (
        screen_response(base_parsed.get("response"), base["ground_truth_needs_human"], base_parsed.get("needs_human")),
        screen_response(lora_parsed.get("response"), lora["ground_truth_needs_human"], lora_parsed.get("needs_human")),
    )


def select_manual_pairs(
    base_predictions: Sequence[Mapping[str, Any]], lora_predictions: Sequence[Mapping[str, Any]], count: int = 30,
) -> List[Tuple[Mapping[str, Any], Mapping[str, Any], List[str], List[str]]]:
    base_by_id = {row["source_index"]: row for row in base_predictions}
    pairs = []
    for lora in lora_predictions:
        base = base_by_id[lora["source_index"]]
        base_flags, lora_flags = pair_flags(base, lora)
        pairs.append((base, lora, base_flags, lora_flags))
    selected, seen = [], set()

    def take(candidates: Sequence[Tuple[Mapping[str, Any], Mapping[str, Any], List[str], List[str]]], label: str, limit: int) -> None:
        ordered = sorted(candidates, key=lambda pair: stable_rank(42, "stage7_manual_qa", label, pair[1]["source_index"]))
        for pair in ordered:
            key = pair[1]["source_index"]
            if key in seen:
                continue
            selected.append(pair)
            seen.add(key)
            limit -= 1
            if limit == 0 or len(selected) >= count:
                break

    fabricated = {"fabricated_contact_details", "fabricated_24_7_availability", "fabricated_fees_or_timelines", "unsupported_guarantee"}
    take([p for p in pairs if "unsupported_action_completion" in p[2] or "unsupported_action_completion" in p[3]], "unsupported_action", 3)
    take([p for p in pairs if fabricated & (set(p[2]) | set(p[3]))], "fabricated_policy_timeline", 3)
    take([p for p in pairs if not p[1]["intent_correct"]], "lora_error", 6)
    take([p for p in pairs if not p[0]["intent_correct"] and p[1]["intent_correct"]], "base_error_lora_correct", 5)
    take([p for p in pairs if p[1]["ground_truth_needs_human"] is True], "escalation_true", 3)
    take([p for p in pairs if p[1]["ground_truth_needs_human"] is False], "escalation_false", 3)
    take([p for p in pairs if not p[1]["error_tags"]], "lora_correct", 4)
    take([p for p in pairs if p[2] or p[3]], "risk_flagged", 3)
    for pair in sorted(pairs, key=lambda p: stable_rank(42, "stage7_manual_qa", "fill", p[1]["source_index"])):
        if len(selected) >= count:
            break
        if pair[1]["source_index"] not in seen:
            selected.append(pair)
            seen.add(pair[1]["source_index"])
    return selected[:count]


def write_manual_samples(path: Path, pairs: Sequence[Tuple[Mapping[str, Any], Mapping[str, Any], List[str], List[str]]]) -> None:
    fields = (
        "source_index", "stable_id", "instruction", "ground_truth_intent", "ground_truth_category",
        "ground_truth_needs_human", "base_output", "lora_output", "base_error_tags", "lora_error_tags",
        "base_risk_flags", "lora_risk_flags",
    )
    rows = []
    for base, lora, base_flags, lora_flags in pairs:
        rows.append({
            "source_index": lora["source_index"], "stable_id": lora["stable_id"], "instruction": lora["instruction"],
            "ground_truth_intent": lora["ground_truth_intent"], "ground_truth_category": lora["ground_truth_category"],
            "ground_truth_needs_human": lora["ground_truth_needs_human"],
            "base_output": base["raw_model_output"], "lora_output": lora["raw_model_output"],
            "base_error_tags": ";".join(base["error_tags"]), "lora_error_tags": ";".join(lora["error_tags"]),
            "base_risk_flags": ";".join(base_flags), "lora_risk_flags": ";".join(lora_flags),
        })
    write_csv(path, fields, rows)


def write_manual_review(
    path: Path,
    all_pairs: Sequence[Tuple[Mapping[str, Any], Mapping[str, Any], List[str], List[str]]],
    sample_ids: Sequence[int],
) -> int:
    selected = []
    sample_set = set(sample_ids)
    for base, lora, base_flags, lora_flags in all_pairs:
        if lora["source_index"] not in sample_set and not base_flags and not lora_flags:
            continue
        row = {
            "source_index": lora["source_index"], "stable_id": lora["stable_id"], "instruction": lora["instruction"],
            "ground_truth_intent": lora["ground_truth_intent"], "ground_truth_category": lora["ground_truth_category"],
            "ground_truth_needs_human": lora["ground_truth_needs_human"],
            "base_output": base["raw_model_output"], "lora_output": lora["raw_model_output"],
            "base_risk_flags": ";".join(base_flags), "lora_risk_flags": ";".join(lora_flags),
        }
        row.update({field: "" for field in HUMAN_FIELDS})
        selected.append(row)
    selected.sort(key=lambda row: int(row["source_index"]))
    fields = (
        "source_index", "stable_id", "instruction", "ground_truth_intent", "ground_truth_category",
        "ground_truth_needs_human", "base_output", "lora_output", "base_risk_flags", "lora_risk_flags",
        *HUMAN_FIELDS,
    )
    write_csv(path, fields, selected)
    return len(selected)


def build_promotion_inputs(comparison: Mapping[str, Any], risk: Mapping[str, Any]) -> Dict[str, Any]:
    metrics = comparison["metrics"]
    return {
        "stage": "C7_locked_inputs_only",
        "formal_promotion_decision_performed": False,
        "quantitative_inputs": {
            "intent_accuracy": {"threshold": "delta >= +3.0 pp", **metrics["intent_accuracy"]},
            "json_valid_rate": {"threshold": "delta >= -1.0 pp", **metrics["json_valid_rate"]},
            "schema_compliance": {"threshold": "delta >= -1.0 pp", **metrics["schema_compliance"]},
            "category_accuracy": {"threshold": "drop >= 3.0 pp is material", **metrics["category_accuracy"]},
            "escalation_accuracy": {"threshold": "drop >= 3.0 pp is material", **metrics["escalation_accuracy"]},
        },
        "critical_behavioral_regression": {"threshold": "not allowed", "base_result": None, "lora_result": None, "delta": None},
        "material_response_safety_regression": {
            "threshold": "not allowed", "base_result": risk["base_flag_counts"],
            "lora_result": risk["lora_flag_counts"], "delta": {
                key: risk["lora_flag_counts"].get(key, 0) - risk["base_flag_counts"].get(key, 0)
                for key in RISK_FLAG_NAMES
            },
        },
        "latency": {"threshold": "operational context only; not a frozen promotion blocker"},
    }


def build_report(
    base_metrics: Mapping[str, Any], lora_metrics: Mapping[str, Any], comparison: Mapping[str, Any],
    escalation: Mapping[str, Any], errors: Mapping[str, Any], confusions: Mapping[str, Any],
    risk: Mapping[str, Any], manual_review_rows: int, integrity: Mapping[str, Any],
) -> str:
    bp, lp = base_metrics["primary"], lora_metrics["primary"]
    bs, ls = base_metrics["secondary"], lora_metrics["secondary"]
    bo, lo = base_metrics["operational"], lora_metrics["operational"]
    delta_lines = "\n".join(
        f"- {name}: Base {row['base']:.6f}, LoRA {row['lora']:.6f}, delta {row['absolute_delta']:+.6f} {row['unit']}"
        for name, row in comparison["metrics"].items()
    )
    error_lines = "\n".join(
        f"- `{name}`: Base {row['base']}, LoRA {row['lora']}, delta {row['delta']:+d}"
        for name, row in sorted(errors.items(), key=lambda item: (item[1]["delta"], item[0]))
    )
    confusion_lines = "\n".join(
        f"- `{row['ground_truth_intent']}` → `{row['predicted_intent']}`: {row['count']}"
        for row in confusions["top_lora_confusions"]
    ) or "- None"
    return f"""# Stage C7 Locked Evaluation

## Goal

Evaluate the frozen Base model and frozen Candidate 01 once on all 300 Locked Test rows under the Stage C6.5 contract.

## Locked Evaluation Boundary

This stage produced evaluation evidence only. No training, tuning, checkpoint selection, configuration change, or formal promotion decision occurred.

## Freeze Integrity

Freeze preflight and post-run integrity: **{integrity['status']}**. All frozen hashes matched before Locked JSON parsing.

## Locked Dataset

Rows: 300. SHA-256: `{LOCKED_SHA256}`. Base attempts: 300. LoRA attempts: 300. No attempted row was excluded.

## Base Locked Metrics

- Intent: {bp['intent_accuracy_percent']:.6f}%
- Category: {bs['category_accuracy_percent']:.6f}%
- JSON valid: {bp['json_valid_rate_percent']:.6f}%
- Schema compliant: {bp['schema_compliance_percent']:.6f}%
- Escalation: {bs['escalation_accuracy_percent']:.6f}%

## LoRA Locked Metrics

- Intent: {lp['intent_accuracy_percent']:.6f}%
- Category: {ls['category_accuracy_percent']:.6f}%
- JSON valid: {lp['json_valid_rate_percent']:.6f}%
- Schema compliant: {lp['schema_compliance_percent']:.6f}%
- Escalation: {ls['escalation_accuracy_percent']:.6f}%

## Base vs LoRA Delta

{delta_lines}

## Escalation Precision / Recall / F1

- Base: precision {escalation['base']['precision_percent']:.6f}%, recall {escalation['base']['recall_percent']:.6f}%, F1 {escalation['base']['f1_percent']:.6f}%
- LoRA: precision {escalation['lora']['precision_percent']:.6f}%, recall {escalation['lora']['recall_percent']:.6f}%, F1 {escalation['lora']['f1_percent']:.6f}%

## Error Reduction

{error_lines}

## Remaining Intent Confusions

{confusion_lines}

Error analysis is descriptive only and must not be used to modify Candidate 01.

## Response Risk Screening

Base flags: `{json.dumps(risk['base_flag_counts'], sort_keys=True)}`. LoRA flags: `{json.dumps(risk['lora_flag_counts'], sort_keys=True)}`.

Automated risk screening is not a complete quality judgment.

## Manual QA Preparation

`locked_manual_qa_samples.csv` contains 30 deterministic paired samples. `locked_manual_response_review.csv` contains {manual_review_rows} deduplicated sampled/risk-flagged rows with every human field blank.

## Latency Trade-off

- Base mean / median / p95: {bo['mean_latency_ms']:.3f} / {bo['median_latency_ms']:.3f} / {bo['p95_latency_ms']:.3f} ms
- LoRA mean / median / p95: {lo['mean_latency_ms']:.3f} / {lo['median_latency_ms']:.3f} / {lo['p95_latency_ms']:.3f} ms

## Frozen Promotion Gate Inputs

`promotion_gate_inputs.json` records thresholds, Base results, LoRA results, and deltas only. Stage C7 makes no formal promotion decision.

## Known Response Limitations

{KNOWN_LIMITATION_EN}

{KNOWN_LIMITATION_ZH}

## Locked-Test Governance

Locked results are final-evaluation evidence and must not be used for training, tuning, evaluator changes, threshold changes, checkpoint selection, or candidate modification.

## Limitations

Exact-match metrics do not establish complete response quality. Risk screening is heuristic, manual QA remains required, and latency is specific to this machine.

## Stage C7 Conclusion

Stage C7 evaluation is complete on all 300 Locked rows. The metrics provide held-out generalization evidence under the frozen contract. Formal promotion decision is reserved for Stage C8.
"""


def finalize(
    repo_root: Path,
    base_predictions: Sequence[Mapping[str, Any]],
    lora_predictions: Sequence[Mapping[str, Any]],
    preflight: Mapping[str, Any],
    started_at: str,
) -> Dict[str, Any]:
    stage_dir = repo_root / STAGE7_DIR
    base_metrics = augment_metrics(aggregate_metrics(base_predictions), base_predictions)
    lora_metrics = augment_metrics(aggregate_metrics(lora_predictions), lora_predictions)
    comparison = compare_metrics(base_metrics, lora_metrics, base_predictions, lora_predictions)
    comparison["metrics"]["median_latency"] = metric_delta(
        base_metrics["operational"]["median_latency_ms"], lora_metrics["operational"]["median_latency_ms"], "milliseconds"
    )
    base_esc = base_metrics["secondary"]["escalation_positive_class"]
    lora_esc = lora_metrics["secondary"]["escalation_positive_class"]
    escalation = escalation_comparison(base_esc, lora_esc)
    errors = comparison["error_tag_comparison"]
    confusions = {
        "base_confusion_count": base_metrics["error_tag_counts"].get("wrong_intent", 0),
        "lora_confusion_count": lora_metrics["error_tag_counts"].get("wrong_intent", 0),
        "top_base_confusions": top_confusions(base_predictions, 15),
        "top_lora_confusions": top_confusions(lora_predictions, 15),
        "remaining_lora_taxonomy_boundaries": top_confusions(lora_predictions, 30),
        "descriptive_only_not_for_candidate_modification": True,
    }
    write_json(stage_dir / "base_locked_metrics.json", base_metrics)
    write_json(stage_dir / "lora_locked_metrics.json", lora_metrics)
    write_json(stage_dir / "locked_escalation_metrics.json", escalation)
    write_json(stage_dir / "base_vs_lora_locked_comparison.json", comparison)
    write_json(stage_dir / "locked_error_comparison.json", errors)
    write_error_cases(stage_dir / "lora_locked_error_cases.csv", lora_predictions)
    write_json(stage_dir / "locked_intent_confusions.json", confusions)
    risk = write_risk_qa(stage_dir / "locked_response_risk_qa.csv", base_predictions, lora_predictions)
    write_json(stage_dir / "locked_response_risk_summary.json", risk)

    manual_pairs = select_manual_pairs(base_predictions, lora_predictions, 30)
    write_manual_samples(stage_dir / "locked_manual_qa_samples.csv", manual_pairs)
    base_by_id = {row["source_index"]: row for row in base_predictions}
    all_pairs = []
    for lora in lora_predictions:
        base = base_by_id[lora["source_index"]]
        base_flags, lora_flags = pair_flags(base, lora)
        all_pairs.append((base, lora, base_flags, lora_flags))
    manual_review_rows = write_manual_review(
        stage_dir / "locked_manual_response_review.csv", all_pairs,
        [pair[1]["source_index"] for pair in manual_pairs],
    )
    promotion_inputs = build_promotion_inputs(comparison, risk)
    write_json(stage_dir / "promotion_gate_inputs.json", promotion_inputs)

    postflight = verify_freeze(repo_root)
    expected_ids = {row["source_index"] for row in base_predictions}
    same_ids = expected_ids == {row["source_index"] for row in lora_predictions} and len(expected_ids) == 300
    integrity_items = list(postflight["result"]["items"])
    integrity_items.extend([
        check_item("base_attempts", 300, len(base_predictions)),
        check_item("lora_attempts", 300, len(lora_predictions)),
        check_item("same_locked_membership", True, same_ids),
        check_item("same_prompt", PROMPT_SHA256, sha256_file(repo_root / PROMPT_PATH)),
        check_item("same_evaluator", True, all(sha256_file(repo_root / path) == expected for path, expected in EXPECTED_EVALUATOR_HASHES.items())),
        check_item("same_decoder", True, preflight["inference"]["decoding"] == postflight["inference"]["decoding"]),
        check_item("same_parser", preflight["inference"]["parser"], postflight["inference"]["parser"]),
        check_item("same_schema_taxonomy_policy", True, all(sha256_file(repo_root / path) == expected for path, expected in EXPECTED_SCHEMA_HASHES.items())),
        check_item("all_base_failures_retained", 300, len(base_predictions)),
        check_item("all_lora_failures_retained", 300, len(lora_predictions)),
        check_item("adapter_unchanged", ADAPTER_SHA256, sha256_file(repo_root / ADAPTER_PATH / "adapters.safetensors")),
        check_item("training_performed", False, False),
        check_item("prompt_modified", False, False),
        check_item("evaluator_modified", False, False),
        check_item("thresholds_modified", False, False),
        check_item("candidate_modified", False, False),
        check_item("stage8_outputs", [], stage8_outputs(repo_root)),
    ])
    failures = [item for item in integrity_items if item["status"] == "FAIL"]
    integrity = {
        "stage": "C7", "status": "PASS" if not failures else "FAIL",
        "pass_count": len(integrity_items) - len(failures), "fail_count": len(failures),
        "items": integrity_items,
    }
    write_json(stage_dir / "locked_evaluation_integrity.json", integrity)
    ended_at = datetime.now(timezone.utc).isoformat()
    manifest = {
        "stage": "C7 Locked Base vs LoRA Evaluation",
        "status": "EVALUATION_COMPLETE" if integrity["status"] == "PASS" else "INCOMPLETE",
        "candidate": CANDIDATE,
        "freeze_manifest_sha256": preflight["result"]["freeze_manifest_sha256"],
        "locked_dataset_hash": LOCKED_SHA256, "locked_rows": 300,
        "base_model": BASE_MODEL, "base_revision": BASE_REVISION,
        "adapter_path": ADAPTER_PATH, "adapter_hash": ADAPTER_SHA256,
        "prompt_hash": PROMPT_SHA256, "inference_contract_hash": INFERENCE_SHA256,
        "evaluator_hashes": EXPECTED_EVALUATOR_HASHES, "schema_hashes": EXPECTED_SCHEMA_HASHES,
        "start_timestamp": started_at, "end_timestamp": ended_at,
        "base_attempts": len(base_predictions), "lora_attempts": len(lora_predictions),
        "base_generation_failure_count": sum(row["generation_error"] is not None for row in base_predictions),
        "lora_generation_failure_count": sum(row["generation_error"] is not None for row in lora_predictions),
        "training_performed": False, "prompt_modified": False, "evaluator_modified": False,
        "thresholds_modified": False, "candidate_modified": False,
        "locked_inference_performed": True, "stage8_performed": False,
        "formal_promotion_decision_performed": False,
        "artifacts": sorted(path.relative_to(repo_root).as_posix() for path in stage_dir.iterdir() if path.is_file()),
    }
    write_json(stage_dir / "stage7_manifest.json", manifest)
    (repo_root / REPORT_PATH).write_text(
        build_report(base_metrics, lora_metrics, comparison, escalation, errors, confusions, risk, manual_review_rows, integrity),
        encoding="utf-8",
    )
    return {
        "manifest": manifest, "base_metrics": base_metrics, "lora_metrics": lora_metrics,
        "comparison": comparison, "escalation": escalation, "errors": errors,
        "confusions": confusions, "risk": risk, "integrity": integrity,
    }


def run(repo_root: Path) -> Dict[str, Any]:
    stage_dir = repo_root / STAGE7_DIR
    stage_dir.mkdir(parents=True, exist_ok=True)
    if any((stage_dir / name).exists() for name in ("base_locked_predictions.jsonl", "lora_locked_predictions.jsonl", "stage7_manifest.json")):
        raise RuntimeError("Refusing to rerun an existing Stage C7 Locked evaluation")
    started_at = datetime.now(timezone.utc).isoformat()
    preflight = verify_freeze(repo_root)
    write_json(stage_dir / "freeze_preflight.json", preflight["result"])
    if preflight["result"]["status"] != "PASS":
        raise RuntimeError("Stage C7 ABORTED: critical frozen component mismatch")

    records = load_locked_records(repo_root)  # first JSON parse occurs only after PASS preflight
    prompt_text = (repo_root / PROMPT_PATH).read_text(encoding="utf-8").strip()
    from huggingface_hub import snapshot_download
    model_path = Path(snapshot_download(BASE_MODEL, revision=BASE_REVISION, local_files_only=True))
    try:
        base = run_model(repo_root, "base", model_path, None, records, prompt_text, preflight["inference"])
        lora = run_model(repo_root, "lora", model_path, str(repo_root / ADAPTER_PATH), records, prompt_text, preflight["inference"])
        return finalize(repo_root, base, lora, preflight, started_at)
    except BaseException:
        (stage_dir / "stage7_execution_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    result = run(repo_root)
    print(json.dumps({
        "status": result["manifest"]["status"],
        "base_attempts": result["manifest"]["base_attempts"],
        "lora_attempts": result["manifest"]["lora_attempts"],
        "base_metrics": result["base_metrics"], "lora_metrics": result["lora_metrics"],
        "comparison": result["comparison"], "integrity": result["integrity"]["status"],
        "formal_promotion_decision_performed": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
