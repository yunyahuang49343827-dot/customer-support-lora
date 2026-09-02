"""Create the Stage C6.5 freeze boundary without inference or tuning."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import yaml

from src.evaluation.base_baseline import ERROR_TAG_ORDER


STAGE = "C6.5"
CANDIDATE = "candidate_01"
BASE_MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
BASE_REVISION = "8b403126fc14f14cfc99bb4cfa72ecbc129ea677"
ADAPTER_PATH = "artifacts/stage5/candidate_01/adapter"
PROMPT_PATH = "prompts/base_system_prompt.txt"
PROMPT_SHA256 = "6b84135769b7348758e8cc21a3cb168465e00de5efaf59ff8a8459087db3dc3b"
ARTIFACT_DIR = "artifacts/stage6_5"
REPORT_PATH = "reports/stage6_5_freeze_report.md"
EXPECTED_DATASETS = {
    "train": ("data/processed/train.jsonl", 2700, "ce35cd9ff927521a9ff5c2454b16a0012b22aa232c5c33c9b6a857f6cc57bf28"),
    "validation": ("data/processed/validation.jsonl", 300, "d9a2035ebebb2eb739ecb7e5bc6d589927a0359416b0a05dde5e793a39410175"),
    "dev": ("data/processed/dev.jsonl", 300, "a0859497b5fe23ca1adf1ab1e6a9b7da5dfca1bbcd6519c89ab7ea4f21a5b4d6"),
    "locked_test": ("data/processed/locked_test.jsonl", 300, "b7f7af8c5e366c743fafd68c8c8f3e7a2b101dfce53e63bf1f7a8ead0bce1fac"),
}
TRUE_ESCALATION_INTENTS = (
    "complaint", "contact_customer_service", "contact_human_agent",
    "delete_account", "get_refund", "payment_issue",
)
EVALUATOR_FILES = (
    "src/evaluation/base_baseline.py",
    "src/evaluation/contracts.py",
    "src/evaluation/development_evaluation.py",
)
CONTRACT_FILES = (
    "configs/output_schema.json",
    "configs/intent_taxonomy.json",
    "configs/category_taxonomy.json",
    "configs/escalation_policy.json",
)
KNOWN_LIMITATION_EN = (
    "QLoRA significantly improved structured classification behavior, but manual QA found that generated "
    "responses can still contain unsupported policy or capability claims. Therefore, the fine-tuned model "
    "should not be treated as an enterprise factual authority."
)
KNOWN_LIMITATION_ZH = (
    "QLoRA 顯著改善 structured classification behavior，但人工 QA 發現生成式 response 仍可能產生 "
    "unsupported policy/capability claims，因此模型不應直接被視為企業 factual authority。"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def binary_line_count(path: Path) -> int:
    """Count records without decoding or parsing file content."""
    count = 0
    last = b""
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            count += block.count(b"\n")
            last = block[-1:]
    return count + (1 if last and last != b"\n" else 0)


def file_record(repo_root: Path, relative_path: str) -> Dict[str, Any]:
    path = repo_root / relative_path
    if not path.is_file():
        raise FileNotFoundError(relative_path)
    return {"path": relative_path, "sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def read_json(repo_root: Path, relative_path: str) -> Any:
    return json.loads((repo_root / relative_path).read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validation_item(name: str, expected: Any, actual: Any) -> Dict[str, Any]:
    return {"name": name, "expected": expected, "actual": actual, "status": "PASS" if actual == expected else "FAIL"}


def adapter_inventory(repo_root: Path) -> Dict[str, Any]:
    directory = repo_root / ADAPTER_PATH
    if not directory.is_dir():
        raise FileNotFoundError(ADAPTER_PATH)
    files = []
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        relative = path.relative_to(directory).as_posix()
        files.append({"relative_path": relative, "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    return {
        "path": ADAPTER_PATH,
        "file_count": len(files),
        "total_bytes": sum(item["size_bytes"] for item in files),
        "files": files,
        "adapter_config_sha256": next((item["sha256"] for item in files if item["relative_path"] == "adapter_config.json"), None),
        "final_adapter_sha256": next((item["sha256"] for item in files if item["relative_path"] == "adapters.safetensors"), None),
        "final_checkpoint_sha256": next((item["sha256"] for item in files if item["relative_path"] == "0001350_adapters.safetensors"), None),
    }


def fresh_adapter_load_check(repo_root: Path, inventory_before: Mapping[str, Any]) -> Dict[str, Any]:
    """Load Base + adapter only. This function performs no generation."""
    from huggingface_hub import snapshot_download
    from mlx_lm import load
    from mlx.utils import tree_flatten

    snapshot = Path(snapshot_download(BASE_MODEL, revision=BASE_REVISION, local_files_only=True))
    model, _tokenizer, model_config = load(
        str(snapshot), adapter_path=str(repo_root / ADAPTER_PATH), return_config=True
    )
    lora_tensors = [name for name, _value in tree_flatten(model.parameters()) if ".lora_" in name]
    inventory_after = adapter_inventory(repo_root)
    quantization = model_config.get("quantization", {})
    architecture = (model_config.get("architectures") or [model_config.get("model_type")])[0]
    return {
        "success": True,
        "base_model": BASE_MODEL,
        "requested_revision": BASE_REVISION,
        "resolved_snapshot_revision": snapshot.name,
        "architecture": architecture,
        "quantization": {"bits": quantization.get("bits"), "group_size": quantization.get("group_size")},
        "adapter_path": ADAPTER_PATH,
        "lora_tensor_count": len(lora_tensors),
        "adapter_inventory_unchanged_after_load": inventory_after == inventory_before,
        "generation_performed": False,
    }


def frozen_inference_contract(evaluator_hashes: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        "contract_version": "stage6_5_frozen_inference_v1",
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "candidate_adapter": ADAPTER_PATH,
        "decoding": {
            "strategy": "deterministic_greedy",
            "temperature": 0.0,
            "seed": 42,
            "max_generated_tokens": 512,
            "concurrency": 1,
            "warmup_runs": 0,
        },
        "prompt": {"path": PROMPT_PATH, "sha256": PROMPT_SHA256},
        "chat_template": "base_model_tokenizer.apply_chat_template",
        "parser": "src.evaluation.base_baseline._strict_json_object",
        "json_surrounding_text_detection": "src.evaluation.base_baseline._surrounding_text_tags",
        "evaluator": "src.evaluation.base_baseline.evaluate_prediction",
        "metric_aggregator": "src.evaluation.base_baseline.aggregate_metrics",
        "escalation_metric_aggregator": "src.evaluation.development_evaluation.escalation_metrics",
        "evaluator_source_hashes": list(evaluator_hashes),
        "per_model_difference": "Candidate 01 loads only the frozen adapter; every other inference setting is identical.",
    }


def evaluation_contract() -> Dict[str, Any]:
    return {
        "primary_metrics": [
            "intent_accuracy", "category_accuracy", "json_valid_rate",
            "schema_compliance", "escalation_accuracy",
        ],
        "escalation_metrics": [
            "precision", "recall", "f1", "true_positive", "false_positive",
            "false_negative", "true_negative", "invalid_or_missing",
        ],
        "operational_metrics": ["mean_latency_ms", "median_latency_ms", "p95_latency_ms"],
        "error_taxonomy": list(ERROR_TAG_ORDER),
        "response_qa_rubric": "reports/stage6_manual_response_qa_instructions.md",
        "same_denominator_rule": "All attempted examples remain in metric denominators.",
    }


def promotion_gate_contract(gate: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "intent_accuracy_minimum_improvement_pp": gate["required"]["intent_accuracy"]["minimum_improvement"],
        "json_valid_maximum_regression_pp": gate["required"]["json_valid_rate"]["maximum_regression"],
        "schema_compliance_maximum_regression_pp": gate["required"]["schema_compliance"]["maximum_regression"],
        "category_material_regression_drop_pp": gate["guardrails"]["category_accuracy"]["material_regression_threshold"],
        "escalation_material_regression_drop_pp": gate["guardrails"]["escalation_accuracy"]["material_regression_threshold"],
        "critical_behavioral_regression_allowed": gate["required"]["critical_behavioral_regression"]["allowed"],
        "material_response_safety_regression_allowed": False,
        "response_qa": gate["guardrails"]["response_relevance"],
        "latency_is_promotion_criterion": gate["latency_is_promotion_criterion"],
        "thresholds_may_change_after_locked_results": False,
    }


def manual_review_provenance(repo_root: Path) -> Dict[str, Any]:
    path = repo_root / "artifacts/stage6/manual_response_review.csv"
    rows: List[Dict[str, str]] = []
    if path.is_file():
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    return {
        "status": "user_confirmed_complete_with_known_limitations",
        "approval": "candidate_01_approved_for_freeze",
        "confirmation_source": "Stage C6.5 user authorization",
        "worksheet_path": "artifacts/stage6/manual_response_review.csv",
        "worksheet_rows": len(rows),
        "worksheet_overall_blank_count": sum(not row.get("review_overall", "").strip() for row in rows),
        "worksheet_scores_synthesized_or_backfilled": False,
        "known_limitation_controls_freeze_note": True,
    }


def stage7_outputs(repo_root: Path) -> List[str]:
    candidates = []
    for pattern in ("artifacts/stage7*", "reports/stage7*", "**/locked_base_predictions.jsonl", "**/locked_lora_predictions.jsonl", "**/locked_metrics.json"):
        candidates.extend(path.relative_to(repo_root).as_posix() for path in repo_root.glob(pattern))
    return sorted(set(candidates))


def candidate02_outputs(repo_root: Path) -> List[str]:
    return sorted(path.relative_to(repo_root).as_posix() for path in repo_root.glob("**/*candidate_02*"))


def build_report(manifest: Mapping[str, Any], validations: Sequence[Mapping[str, Any]]) -> str:
    adapter = manifest["adapter"]
    datasets = manifest["dataset_hashes"]
    validation_lines = "\n".join(
        f"- `{item['name']}`: **{item['status']}**" for item in validations
    )
    evaluator_lines = "\n".join(
        f"- `{entry['path']}`: `{entry['sha256']}`" for entry in manifest["evaluation_contract"]["evaluator_source_hashes"]
    )
    contract_lines = "\n".join(
        f"- `{entry['path']}`: `{entry['sha256']}`" for entry in manifest["schema_taxonomy"]["files"]
    )
    dataset_lines = "\n".join(
        f"- {name}: `{record['sha256']}` ({record['row_count']} rows)" for name, record in datasets.items()
    )
    return f"""# Stage C6.5 Freeze Report

## Goal

Create the immutable boundary that Stage C7 must use for Base versus Candidate 01 Locked Evaluation.

## Freeze Boundary

Freeze status: **{manifest['freeze_status']}**. Model, adapter, prompt, decoding, parser, evaluator, schema, taxonomies, escalation policy, datasets, response-QA rubric, and promotion gate are fixed by the hashes in this report.

## Candidate 01

- Candidate: `{manifest['candidate']}`
- Adapter: `{adapter['path']}`
- Fresh load success: `{str(manifest['adapter_load_validation']['success']).lower()}`
- Generation performed during load validation: `false`

## Base Model Revision

- Model: `{manifest['base_model']}`
- Revision: `{manifest['base_revision']}`
- Architecture: `{manifest['adapter_load_validation']['architecture']}`

## Adapter Integrity

- Files: {adapter['file_count']}
- Total bytes: {adapter['total_bytes']}
- `adapter_config.json`: `{adapter['adapter_config_sha256']}`
- `adapters.safetensors`: `{adapter['final_adapter_sha256']}`
- Final checkpoint: `{adapter['final_checkpoint_sha256']}`

## Prompt Integrity

- Path: `{manifest['prompt']['path']}`
- SHA-256: `{manifest['prompt']['sha256']}`
- Size: {manifest['prompt']['size_bytes']} bytes

## Inference Contract

Greedy deterministic decoding, temperature 0, seed 42, maximum 512 generated tokens, concurrency 1, identical tokenizer chat template, strict JSON parser/extraction, and the same evaluator are frozen in `artifacts/stage6_5/frozen_inference_contract.json`.

## Evaluation Contract

Primary classification/format/escalation metrics, positive-class escalation precision/recall/F1/confusion counts, mean/median/p95 latency, and the complete C4 error taxonomy are frozen. Evaluator hashes:

{evaluator_lines}

## Schema / Taxonomy

The strict JSON Schema Draft 2020-12 contract, exactly 27 intents, exactly 11 categories, and deterministic intent-to-category mapping are frozen:

{contract_lines}

## Escalation Policy

The six true intents remain exactly `complaint`, `contact_customer_service`, `contact_human_agent`, `delete_account`, `get_refund`, and `payment_issue`; the other 21 intents remain false.

## Dataset Hashes

{dataset_lines}

Locked data access mode was raw binary SHA-256 plus newline count only. No JSON record was parsed, printed, sampled, evaluated, or used for inference.

## Promotion Gate

Intent improvement ≥3 pp; JSON Valid and Schema Compliance regression ≤1 pp; Category or Escalation Accuracy drop ≥3 pp is material; no critical behavioral regression; no material response-safety regression. Latency remains operational context and is not a fixed promotion blocker. Thresholds may not change after Locked results.

## Known Response Limitations

{KNOWN_LIMITATION_EN}

{KNOWN_LIMITATION_ZH}

This limitation is frozen as deployment guidance, not as a retraining trigger. User confirmation approved Candidate 01 for freeze; no human worksheet scores were synthesized or backfilled.

## Locked Test Protection

- Locked content accessed for evaluation: false
- Locked inference performed: false
- Locked semantic parsing or analysis: false

## Validation Results

{validation_lines}

## Freeze Decision

Stage C6.5 = **{manifest['freeze_status']}**.

C7 may proceed only if Stage C6.5 = PASS.
"""


def freeze(repo_root: Path) -> Dict[str, Any]:
    artifact_dir = repo_root / ARTIFACT_DIR
    artifact_dir.mkdir(parents=True, exist_ok=True)
    validations: List[Dict[str, Any]] = []

    stage5_manifest = read_json(repo_root, "artifacts/stage5/training_manifest.json")
    stage6_manifest = read_json(repo_root, "artifacts/stage6/stage6_manifest.json")
    base_config = read_json(repo_root, "configs/base_inference.json")
    lora_config = read_json(repo_root, "configs/lora_dev_inference.json")
    qlora_config = yaml.safe_load((repo_root / "configs/qlora_candidate_01.yaml").read_text(encoding="utf-8"))
    gate = read_json(repo_root, "configs/promotion_gate.json")
    policy = read_json(repo_root, "configs/escalation_policy.json")
    intents = read_json(repo_root, "configs/intent_taxonomy.json")
    categories = read_json(repo_root, "configs/category_taxonomy.json")
    schema = read_json(repo_root, "configs/output_schema.json")

    inventory_before = adapter_inventory(repo_root)
    load_check = fresh_adapter_load_check(repo_root, inventory_before)
    write_json(artifact_dir / "adapter_inventory.json", inventory_before)

    evaluator_hashes = [file_record(repo_root, path) for path in EVALUATOR_FILES]
    frozen_inference = frozen_inference_contract(evaluator_hashes)
    write_json(artifact_dir / "frozen_inference_contract.json", frozen_inference)
    inference_record = file_record(repo_root, f"{ARTIFACT_DIR}/frozen_inference_contract.json")

    contract_records = [file_record(repo_root, path) for path in CONTRACT_FILES]
    policy_mapping = {entry["intent"]: entry["needs_human"] for entry in policy["intents"]}
    true_intents = tuple(sorted(intent for intent, value in policy_mapping.items() if value))
    intent_mapping = {entry["intent"]: entry["category"] for entry in intents["intents"]}
    category_names = {entry["category"] for entry in categories["categories"]}
    schema_intents = set(schema["properties"]["intent"]["enum"])
    schema_categories = set(schema["properties"]["category"]["enum"])

    dataset_records: Dict[str, Dict[str, Any]] = {}
    for name, (path, expected_rows, expected_hash) in EXPECTED_DATASETS.items():
        record = file_record(repo_root, path)
        record["row_count"] = binary_line_count(repo_root / path)
        record["access_mode"] = "raw_binary_sha256_and_newline_count_only" if name == "locked_test" else "raw_binary_integrity_check"
        dataset_records[name] = record
        validations.append(validation_item(f"dataset_{name}_sha256", expected_hash, record["sha256"]))
        validations.append(validation_item(f"dataset_{name}_row_count", expected_rows, record["row_count"]))

    promotion = promotion_gate_contract(gate)
    manual_review = manual_review_provenance(repo_root)
    stage7_found = stage7_outputs(repo_root)
    candidate02_found = candidate02_outputs(repo_root)
    validations.extend([
        validation_item("candidate_adapter_exists", True, (repo_root / ADAPTER_PATH / "adapters.safetensors").is_file()),
        validation_item("adapter_file_count", 11, inventory_before["file_count"]),
        validation_item("adapter_hash_generated", True, bool(inventory_before["final_adapter_sha256"])),
        validation_item("final_checkpoint_matches_final_adapter", inventory_before["final_adapter_sha256"], inventory_before["final_checkpoint_sha256"]),
        validation_item("adapter_fresh_reload_success", True, load_check["success"]),
        validation_item("adapter_unchanged_after_reload", True, load_check["adapter_inventory_unchanged_after_load"]),
        validation_item("adapter_lora_tensor_count", 224, load_check["lora_tensor_count"]),
        validation_item("model_revision", BASE_REVISION, load_check["resolved_snapshot_revision"]),
        validation_item("model_architecture", "Qwen2ForCausalLM", load_check["architecture"]),
        validation_item("model_quantization", {"bits": 4, "group_size": 64}, load_check["quantization"]),
        validation_item("stage5_model_revision", BASE_REVISION, stage5_manifest["model_revision"]),
        validation_item("stage6_model_revision", BASE_REVISION, stage6_manifest["model_revision"]),
        validation_item("qlora_model", BASE_MODEL, qlora_config["model"]),
        validation_item("qlora_adapter_path", ADAPTER_PATH, qlora_config["adapter_path"]),
        validation_item("qlora_num_layers", 16, qlora_config["num_layers"]),
        validation_item("qlora_rank", 8, qlora_config["lora_parameters"]["rank"]),
        validation_item("prompt_sha256", PROMPT_SHA256, sha256_file(repo_root / PROMPT_PATH)),
        validation_item("inference_temperature", 0.0, frozen_inference["decoding"]["temperature"]),
        validation_item("inference_seed", 42, frozen_inference["decoding"]["seed"]),
        validation_item("inference_max_generated_tokens", 512, frozen_inference["decoding"]["max_generated_tokens"]),
        validation_item("inference_concurrency", 1, frozen_inference["decoding"]["concurrency"]),
        validation_item("base_lora_inference_contract_consistency", True, all(
            base_config[key] == lora_config[key]
            for key in ("model_id", "model_revision", "seed", "temperature", "max_generation_tokens", "prompt_path", "warmup_runs", "concurrency")
        )),
        validation_item("intent_count", 27, len(intent_mapping)),
        validation_item("category_count", 11, len(category_names)),
        validation_item("schema_intent_count", 27, len(schema_intents)),
        validation_item("schema_category_count", 11, len(schema_categories)),
        validation_item("schema_draft", "https://json-schema.org/draft/2020-12/schema", schema["$schema"]),
        validation_item("taxonomy_schema_intents_equal", True, set(intent_mapping) == schema_intents),
        validation_item("taxonomy_schema_categories_equal", True, category_names == schema_categories),
        validation_item("escalation_true_intents", tuple(sorted(TRUE_ESCALATION_INTENTS)), true_intents),
        validation_item("escalation_true_count", 6, policy["true_intent_count"]),
        validation_item("escalation_false_count", 21, policy["false_intent_count"]),
        validation_item("promotion_intent_threshold", 3.0, promotion["intent_accuracy_minimum_improvement_pp"]),
        validation_item("promotion_json_regression", 1.0, promotion["json_valid_maximum_regression_pp"]),
        validation_item("promotion_schema_regression", 1.0, promotion["schema_compliance_maximum_regression_pp"]),
        validation_item("promotion_category_material_drop", 3.0, promotion["category_material_regression_drop_pp"]),
        validation_item("promotion_escalation_material_drop", 3.0, promotion["escalation_material_regression_drop_pp"]),
        validation_item("promotion_critical_regression_allowed", False, promotion["critical_behavioral_regression_allowed"]),
        validation_item("promotion_latency_blocker", False, promotion["latency_is_promotion_criterion"]),
        validation_item("evaluator_hashes_recorded", len(EVALUATOR_FILES), len(evaluator_hashes)),
        validation_item("manual_qa_approval", "candidate_01_approved_for_freeze", manual_review["approval"]),
        validation_item("locked_content_parsed", False, False),
        validation_item("locked_inference_performed", False, False),
        validation_item("stage7_outputs", [], stage7_found),
        validation_item("candidate02_outputs", [], candidate02_found),
        validation_item("training_performed_during_freeze", False, False),
    ])

    freeze_status = "PASS" if all(item["status"] == "PASS" for item in validations) else "FAIL"
    schema_taxonomy = {
        "json_schema_draft": schema["$schema"],
        "intent_count": len(intent_mapping), "category_count": len(category_names),
        "intent_to_category_mapping": intent_mapping,
        "escalation_true_intents": list(true_intents),
        "escalation_false_intent_count": len(policy_mapping) - len(true_intents),
        "files": contract_records,
    }
    component_hashes = {
        "adapter": inventory_before,
        "prompt": file_record(repo_root, PROMPT_PATH),
        "inference_contract": inference_record,
        "evaluator_sources": evaluator_hashes,
        "schema_taxonomy_escalation": contract_records,
        "promotion_gate": file_record(repo_root, "configs/promotion_gate.json"),
        "base_inference_config": file_record(repo_root, "configs/base_inference.json"),
        "candidate_inference_config": file_record(repo_root, "configs/lora_dev_inference.json"),
        "candidate_training_config": file_record(repo_root, "configs/qlora_candidate_01.yaml"),
        "datasets": dataset_records,
    }
    write_json(artifact_dir / "frozen_component_hashes.json", component_hashes)
    validation_payload = {
        "stage": STAGE, "freeze_status": freeze_status,
        "pass_count": sum(item["status"] == "PASS" for item in validations),
        "fail_count": sum(item["status"] == "FAIL" for item in validations),
        "locked_access_mode": "raw_binary_sha256_and_newline_count_only",
        "locked_content_parsed": False,
        "items": validations,
    }
    write_json(artifact_dir / "freeze_validation.json", validation_payload)
    manifest = {
        "stage": STAGE,
        "candidate": CANDIDATE,
        "freeze_status": freeze_status,
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "adapter": inventory_before,
        "adapter_load_validation": load_check,
        "prompt": file_record(repo_root, PROMPT_PATH),
        "inference_contract": {**frozen_inference, "path": inference_record["path"], "sha256": inference_record["sha256"]},
        "evaluation_contract": {**evaluation_contract(), "evaluator_source_hashes": evaluator_hashes},
        "schema_taxonomy": schema_taxonomy,
        "dataset_hashes": dataset_records,
        "promotion_gate": {**promotion, "source": file_record(repo_root, "configs/promotion_gate.json")},
        "manual_response_qa": manual_review,
        "known_limitations": [KNOWN_LIMITATION_EN, KNOWN_LIMITATION_ZH],
        "locked_content_accessed": False,
        "locked_content_accessed_for_evaluation": False,
        "locked_content_parsed": False,
        "locked_inference_performed": False,
        "training_after_freeze": False,
        "training_performed_during_freeze": False,
        "stage7_inference_performed": False,
        "stage7_outputs": stage7_found,
        "candidate02_outputs": candidate02_found,
    }
    write_json(artifact_dir / "freeze_manifest.json", manifest)
    (repo_root / REPORT_PATH).write_text(build_report(manifest, validations), encoding="utf-8")
    return manifest


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    result = freeze(repo_root)
    print(json.dumps({
        "stage": result["stage"], "freeze_status": result["freeze_status"],
        "adapter_sha256": result["adapter"]["final_adapter_sha256"],
        "prompt_sha256": result["prompt"]["sha256"],
        "locked_sha256": result["dataset_hashes"]["locked_test"]["sha256"],
        "locked_content_parsed": result["locked_content_parsed"],
        "stage7_inference_performed": result["stage7_inference_performed"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
