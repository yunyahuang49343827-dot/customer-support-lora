import csv
import json
from pathlib import Path

from src.evaluation.base_baseline import aggregate_metrics
from src.evaluation.development_evaluation import RISK_FLAG_NAMES, augment_metrics, screen_response
from src.evaluation.locked_evaluation import (
    ADAPTER_PATH,
    ADAPTER_SHA256,
    BASE_REVISION,
    EXPECTED_EVALUATOR_HASHES,
    EXPECTED_SCHEMA_HASHES,
    HUMAN_FIELDS,
    INFERENCE_SHA256,
    LOCKED_PATH,
    LOCKED_SHA256,
    PROMPT_PATH,
    PROMPT_SHA256,
    STAGE7_DIR,
    escalation_comparison,
    metric_delta,
    pair_flags,
    read_jsonl,
    select_manual_pairs,
    sha256_file,
    stage8_outputs,
    verify_freeze,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def artifact(name):
    return json.loads((REPO_ROOT / STAGE7_DIR / name).read_text(encoding="utf-8"))


def test_stage6_5_preflight_and_all_frozen_hashes_are_exact():
    recorded_preflight = artifact("freeze_preflight.json")
    freeze_manifest = json.loads((REPO_ROOT / "artifacts/stage6_5/freeze_manifest.json").read_text())
    assert recorded_preflight["status"] == "PASS"
    assert freeze_manifest["freeze_status"] == "PASS"
    assert sha256_file(REPO_ROOT / ADAPTER_PATH / "adapters.safetensors") == ADAPTER_SHA256
    assert sha256_file(REPO_ROOT / PROMPT_PATH) == PROMPT_SHA256
    assert sha256_file(REPO_ROOT / "artifacts/stage6_5/frozen_inference_contract.json") == INFERENCE_SHA256
    assert sha256_file(REPO_ROOT / LOCKED_PATH) == LOCKED_SHA256
    assert freeze_manifest["base_revision"] == BASE_REVISION
    assert all(sha256_file(REPO_ROOT / path) == expected for path, expected in EXPECTED_EVALUATOR_HASHES.items())
    assert all(sha256_file(REPO_ROOT / path) == expected for path, expected in EXPECTED_SCHEMA_HASHES.items())


def test_locked_attempts_are_exactly_300_same_membership_and_no_exclusions():
    base = read_jsonl(REPO_ROOT / STAGE7_DIR / "base_locked_predictions.jsonl")
    lora = read_jsonl(REPO_ROOT / STAGE7_DIR / "lora_locked_predictions.jsonl")
    assert len(base) == len(lora) == 300
    assert len({row["source_index"] for row in base}) == 300
    assert {row["source_index"] for row in base} == {row["source_index"] for row in lora}
    assert all("generation_error" in row for row in base + lora)
    manifest = artifact("stage7_manifest.json")
    assert manifest["base_attempts"] == manifest["lora_attempts"] == 300
    assert manifest["base_generation_failure_count"] == sum(row["generation_error"] is not None for row in base)
    assert manifest["lora_generation_failure_count"] == sum(row["generation_error"] is not None for row in lora)


def test_locked_metrics_reproduce_with_frozen_evaluator_and_denominator():
    for role in ("base", "lora"):
        rows = read_jsonl(REPO_ROOT / STAGE7_DIR / f"{role}_locked_predictions.jsonl")
        expected = augment_metrics(aggregate_metrics(rows), rows)
        assert artifact(f"{role}_locked_metrics.json") == expected
        assert expected["evaluated_rows"] == 300


def test_escalation_and_delta_calculations_are_exact():
    base = {"true_positive": 1, "false_positive": 2, "false_negative": 3, "true_negative": 4, "invalid_or_missing": 0, "precision_percent": 33.333333, "recall_percent": 25.0, "f1_percent": 28.571429}
    lora = {"true_positive": 3, "false_positive": 1, "false_negative": 1, "true_negative": 5, "invalid_or_missing": 0, "precision_percent": 75.0, "recall_percent": 75.0, "f1_percent": 75.0}
    result = escalation_comparison(base, lora)
    assert result["delta"]["true_positive"] == 2
    assert result["delta"]["f1_percent"] == 46.428571
    assert metric_delta(31.0, 40.0)["absolute_delta"] == 9.0


def test_base_and_lora_use_same_frozen_inference_settings_and_denominator_rule():
    manifest = artifact("stage7_manifest.json")
    inference = json.loads((REPO_ROOT / "artifacts/stage6_5/frozen_inference_contract.json").read_text())
    assert manifest["prompt_hash"] == inference["prompt"]["sha256"] == PROMPT_SHA256
    assert manifest["inference_contract_hash"] == INFERENCE_SHA256
    assert inference["decoding"] == {"concurrency": 1, "max_generated_tokens": 512, "seed": 42, "strategy": "deterministic_greedy", "temperature": 0.0, "warmup_runs": 0}
    assert manifest["base_attempts"] == manifest["lora_attempts"] == manifest["locked_rows"]


def test_response_screening_uses_frozen_rules_and_manual_review_is_blank():
    flags = screen_response("Call +1 555 123 4567 and share your password.", False, True)
    assert {"fabricated_contact_details", "asks_for_sensitive_secret", "unnecessary_escalation"} <= set(flags)
    summary = artifact("locked_response_risk_summary.json")
    assert set(summary["base_flag_counts"]) == set(RISK_FLAG_NAMES)
    assert set(summary["lora_flag_counts"]) == set(RISK_FLAG_NAMES)
    with (REPO_ROOT / STAGE7_DIR / "locked_manual_response_review.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert len({row["source_index"] for row in rows}) == len(rows)
    assert all(row[field] == "" for row in rows for field in HUMAN_FIELDS)


def test_manual_qa_is_deterministic_seed42_and_has_30_pairs():
    with (REPO_ROOT / STAGE7_DIR / "locked_manual_qa_samples.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 30
    assert len({row["source_index"] for row in rows}) == 30
    assert all(row["base_output"] and row["lora_output"] for row in rows)

    base = read_jsonl(REPO_ROOT / STAGE7_DIR / "base_locked_predictions.jsonl")
    lora = read_jsonl(REPO_ROOT / STAGE7_DIR / "lora_locked_predictions.jsonl")
    first = select_manual_pairs(base, lora, 30)
    second = select_manual_pairs(base, lora, 30)
    first_ids = [str(pair[1]["source_index"]) for pair in first]
    assert first_ids == [str(pair[1]["source_index"]) for pair in second]
    assert first_ids == [row["source_index"] for row in rows]


def test_manual_qa_covers_required_locked_case_types():
    with (REPO_ROOT / STAGE7_DIR / "locked_manual_qa_samples.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    fabricated_flags = {
        "fabricated_contact_details",
        "fabricated_24_7_availability",
        "fabricated_fees_or_timelines",
        "unsupported_guarantee",
    }
    combined_flags = [set(filter(None, (row["base_risk_flags"] + ";" + row["lora_risk_flags"]).split(";"))) for row in rows]
    assert any(not row["lora_error_tags"] for row in rows)
    assert any(row["lora_error_tags"] for row in rows)
    assert {row["ground_truth_needs_human"] for row in rows} == {"True", "False"}
    assert any(row["base_error_tags"] and not row["lora_error_tags"] for row in rows)
    assert any(flags for flags in combined_flags)
    assert any("unsupported_action_completion" in flags for flags in combined_flags)
    assert any(flags & fabricated_flags for flags in combined_flags)


def test_risk_screening_has_all_300_paired_rows_and_matches_frozen_screening():
    base = read_jsonl(REPO_ROOT / STAGE7_DIR / "base_locked_predictions.jsonl")
    lora = read_jsonl(REPO_ROOT / STAGE7_DIR / "lora_locked_predictions.jsonl")
    base_by_id = {row["source_index"]: row for row in base}
    with (REPO_ROOT / STAGE7_DIR / "locked_response_risk_qa.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 300
    assert len({row["source_index"] for row in rows}) == 300
    for row, lora_row in zip(rows, lora):
        expected_base, expected_lora = pair_flags(base_by_id[lora_row["source_index"]], lora_row)
        assert row["base_risk_flags"] == ";".join(expected_base)
        assert row["lora_risk_flags"] == ";".join(expected_lora)


def test_promotion_inputs_have_frozen_thresholds_but_no_decision():
    inputs = artifact("promotion_gate_inputs.json")
    assert inputs["formal_promotion_decision_performed"] is False
    assert inputs["quantitative_inputs"]["intent_accuracy"]["threshold"] == "delta >= +3.0 pp"
    assert inputs["quantitative_inputs"]["json_valid_rate"]["threshold"] == "delta >= -1.0 pp"
    text = json.dumps(inputs)
    assert '"PROMOTE"' not in text
    assert '"DO_NOT_PROMOTE"' not in text


def test_integrity_no_training_or_modification_and_no_stage8():
    manifest = artifact("stage7_manifest.json")
    integrity = artifact("locked_evaluation_integrity.json")
    assert manifest["status"] == "EVALUATION_COMPLETE"
    assert integrity["status"] == "PASS"
    assert integrity["fail_count"] == 0
    assert manifest["training_performed"] is False
    assert manifest["prompt_modified"] is False
    assert manifest["evaluator_modified"] is False
    assert manifest["thresholds_modified"] is False
    assert manifest["candidate_modified"] is False
    assert manifest["stage8_performed"] is False
    assert manifest["formal_promotion_decision_performed"] is False
    stage8_manifest_path = REPO_ROOT / "artifacts/stage8/stage8_manifest.json"
    if stage8_manifest_path.exists():
        stage8_manifest = json.loads(stage8_manifest_path.read_text(encoding="utf-8"))
        assert stage8_manifest["status"] == "COMPLETE"
        assert stage8_manifest["locked_rerun_performed"] is False
        assert stage8_manifest["stage9_performed"] is False


def test_frozen_evaluator_sources_remain_unmodified_by_stage7():
    assert all(sha256_file(REPO_ROOT / path) == expected for path, expected in EXPECTED_EVALUATOR_HASHES.items())
    source = (REPO_ROOT / "src/evaluation/locked_evaluation.py").read_text(encoding="utf-8")
    assert "stream_generate" in source
    assert "Stage C8" in source
