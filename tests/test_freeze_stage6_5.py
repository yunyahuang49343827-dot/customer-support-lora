import json
from pathlib import Path

from src.evaluation.freeze_stage6_5 import (
    ADAPTER_PATH,
    ARTIFACT_DIR,
    BASE_MODEL,
    BASE_REVISION,
    CONTRACT_FILES,
    EVALUATOR_FILES,
    EXPECTED_DATASETS,
    PROMPT_PATH,
    PROMPT_SHA256,
    TRUE_ESCALATION_INTENTS,
    binary_line_count,
    sha256_file,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_artifact(name):
    return json.loads((REPO_ROOT / ARTIFACT_DIR / name).read_text(encoding="utf-8"))


def test_candidate_adapter_and_prompt_are_frozen_exactly():
    manifest = load_artifact("freeze_manifest.json")
    assert manifest["base_model"] == BASE_MODEL
    assert manifest["base_revision"] == BASE_REVISION
    assert manifest["adapter"]["path"] == ADAPTER_PATH
    assert manifest["adapter"]["final_adapter_sha256"] == sha256_file(REPO_ROOT / ADAPTER_PATH / "adapters.safetensors")
    assert manifest["adapter"]["final_checkpoint_sha256"] == manifest["adapter"]["final_adapter_sha256"]
    assert manifest["prompt"]["path"] == PROMPT_PATH
    assert manifest["prompt"]["sha256"] == PROMPT_SHA256 == sha256_file(REPO_ROOT / PROMPT_PATH)
    assert manifest["adapter_load_validation"]["success"] is True
    assert manifest["adapter_load_validation"]["generation_performed"] is False


def test_frozen_inference_contract_is_exact():
    contract = load_artifact("frozen_inference_contract.json")
    assert contract["base_revision"] == BASE_REVISION
    assert contract["candidate_adapter"] == ADAPTER_PATH
    assert contract["decoding"] == {
        "strategy": "deterministic_greedy", "temperature": 0.0, "seed": 42,
        "max_generated_tokens": 512, "concurrency": 1, "warmup_runs": 0,
    }
    assert contract["parser"] == "src.evaluation.base_baseline._strict_json_object"
    assert contract["evaluator"] == "src.evaluation.base_baseline.evaluate_prediction"


def test_evaluator_schema_taxonomy_and_policy_hashes_are_recorded_and_current():
    hashes = load_artifact("frozen_component_hashes.json")
    evaluator = {entry["path"]: entry for entry in hashes["evaluator_sources"]}
    contracts = {entry["path"]: entry for entry in hashes["schema_taxonomy_escalation"]}
    assert set(evaluator) == set(EVALUATOR_FILES)
    assert set(contracts) == set(CONTRACT_FILES)
    assert all(entry["sha256"] == sha256_file(REPO_ROOT / path) for path, entry in evaluator.items())
    assert all(entry["sha256"] == sha256_file(REPO_ROOT / path) for path, entry in contracts.items())
    manifest = load_artifact("freeze_manifest.json")
    assert manifest["schema_taxonomy"]["intent_count"] == 27
    assert manifest["schema_taxonomy"]["category_count"] == 11
    assert manifest["schema_taxonomy"]["escalation_true_intents"] == sorted(TRUE_ESCALATION_INTENTS)
    assert manifest["schema_taxonomy"]["escalation_false_intent_count"] == 21


def test_all_dataset_hashes_and_raw_row_counts_are_exact():
    manifest = load_artifact("freeze_manifest.json")
    for name, (path, expected_rows, expected_hash) in EXPECTED_DATASETS.items():
        record = manifest["dataset_hashes"][name]
        assert record["sha256"] == expected_hash == sha256_file(REPO_ROOT / path)
        assert record["row_count"] == expected_rows == binary_line_count(REPO_ROOT / path)
    assert manifest["dataset_hashes"]["locked_test"]["access_mode"] == "raw_binary_sha256_and_newline_count_only"
    assert manifest["locked_content_parsed"] is False
    assert manifest["locked_content_accessed_for_evaluation"] is False


def test_promotion_gate_and_known_limitations_are_frozen():
    manifest = load_artifact("freeze_manifest.json")
    gate = manifest["promotion_gate"]
    assert gate["intent_accuracy_minimum_improvement_pp"] == 3.0
    assert gate["json_valid_maximum_regression_pp"] == 1.0
    assert gate["schema_compliance_maximum_regression_pp"] == 1.0
    assert gate["category_material_regression_drop_pp"] == 3.0
    assert gate["escalation_material_regression_drop_pp"] == 3.0
    assert gate["critical_behavioral_regression_allowed"] is False
    assert gate["latency_is_promotion_criterion"] is False
    assert len(manifest["known_limitations"]) == 2
    assert "enterprise factual authority" in manifest["known_limitations"][0]


def test_freeze_validation_passes_and_records_no_stage7_activity_during_freeze():
    manifest = load_artifact("freeze_manifest.json")
    validation = load_artifact("freeze_validation.json")
    assert manifest["freeze_status"] == "PASS"
    assert validation["freeze_status"] == "PASS"
    assert validation["fail_count"] == 0
    assert manifest["training_performed_during_freeze"] is False
    assert manifest["training_after_freeze"] is False
    assert manifest["stage7_inference_performed"] is False
    assert manifest["stage7_outputs"] == []
    assert manifest["candidate02_outputs"] == []
    # The freeze manifest is an immutable record of C6.5. A later, explicitly
    # authorized C7 run may now exist without changing that historical fact.
    stage7_manifest_path = REPO_ROOT / "artifacts/stage7/stage7_manifest.json"
    if stage7_manifest_path.exists():
        stage7_manifest = json.loads(stage7_manifest_path.read_text(encoding="utf-8"))
        assert stage7_manifest["status"] == "EVALUATION_COMPLETE"
        assert stage7_manifest["training_performed"] is False
        assert stage7_manifest["stage8_performed"] is False


def test_freeze_source_never_parses_locked_records():
    source = (REPO_ROOT / "src/evaluation/freeze_stage6_5.py").read_text(encoding="utf-8")
    forbidden = "json.loads" + "(locked"
    assert forbidden not in source
    assert "stream_generate" not in source
