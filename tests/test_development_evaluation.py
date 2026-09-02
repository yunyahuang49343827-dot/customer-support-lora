import csv
import json
from pathlib import Path

from src.evaluation.base_baseline import evaluate_prediction
from src.evaluation.development_evaluation import (
    ADAPTER_PATH,
    BASE_CONFIG_PATH,
    LORA_CONFIG_PATH,
    PROMPT_SHA256,
    STAGE6_DIR,
    aggregate_metrics,
    augment_metrics,
    compare_metrics,
    escalation_metrics,
    metric_delta,
    screen_response,
    select_paired_qa,
    sha256_file,
    validate_inputs,
    write_risk_qa,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def prediction(truth=True, predicted=True, source_index=1, error_tags=None):
    truth_row = {"intent": "get_refund" if truth else "track_order", "category": "REFUND" if truth else "ORDER", "needs_human": truth}
    raw = json.dumps({"intent": truth_row["intent"], "category": truth_row["category"], "needs_human": predicted, "response": "Use the secure account page."})
    row = evaluate_prediction(raw, truth_row, 10.0)
    row.update({"source_index": source_index, "stable_id": f"g{source_index}", "instruction": f"message {source_index}"})
    if error_tags is not None:
        row["error_tags"] = error_tags
    return row


def test_stage6_contract_uses_exact_dev_300_and_frozen_assets():
    contract = validate_inputs(REPO_ROOT)
    assert len(contract["dev_records"]) == 300
    assert len(contract["base_predictions"]) == 300
    assert sha256_file(REPO_ROOT / "prompts/base_system_prompt.txt") == PROMPT_SHA256
    assert contract["lora_config"]["adapter_path"] == ADAPTER_PATH


def test_base_and_lora_configs_have_identical_inference_contract():
    base = json.loads((REPO_ROOT / BASE_CONFIG_PATH).read_text())
    lora = json.loads((REPO_ROOT / LORA_CONFIG_PATH).read_text())
    for key in set(base) - {"config_version", "adapter_path"}:
        assert lora[key] == base[key]
    assert base["adapter_path"] is None
    assert lora["adapter_path"] == ADAPTER_PATH


def test_stage6_inference_source_has_no_locked_dataset_path_literal():
    source = (REPO_ROOT / "src/evaluation/development_evaluation.py").read_text(encoding="utf-8")
    assert "data/processed/" + "locked_test.jsonl" not in source


def test_escalation_precision_recall_f1_and_counts():
    rows = [prediction(True, True, 1), prediction(True, False, 2), prediction(False, True, 3), prediction(False, False, 4)]
    result = escalation_metrics(rows)
    assert (result["true_positive"], result["false_positive"], result["false_negative"], result["true_negative"]) == (1, 1, 1, 1)
    assert result["precision_percent"] == 50.0
    assert result["recall_percent"] == 50.0
    assert result["f1_percent"] == 50.0


def test_metric_delta_is_lora_minus_base():
    assert metric_delta(31.333333, 40.0)["absolute_delta"] == 8.666667


def test_same_c4_evaluator_used_for_candidate_rows():
    row = prediction(False, False)
    assert row["json_valid"] and row["schema_compliant"] and row["intent_correct"]


def test_manual_qa_selection_is_seed_42_deterministic_and_30_rows():
    base = [prediction(i % 5 == 0, i % 5 == 0, i) for i in range(40)]
    lora = [prediction(i % 5 == 0, i % 7 == 0, i) for i in range(40)]
    first = select_paired_qa(base, lora, 42, 30)
    second = select_paired_qa(base, lora, 42, 30)
    assert len(first) == 30
    assert [p[1]["source_index"] for p in first] == [p[1]["source_index"] for p in second]


def test_risk_screening_flags_required_categories_and_writes_valid_csv(tmp_path):
    flags = screen_response("Call +1 555 123 4567. We are available 24/7 and guarantee completion in 2 days. Please share your password.", False, True)
    assert {"fabricated_contact_details", "fabricated_24_7_availability", "fabricated_fees_or_timelines", "asks_for_sensitive_secret", "unsupported_guarantee", "unnecessary_escalation"} <= set(flags)
    base = [prediction(False, False, 1)]
    lora = [prediction(False, False, 1)]
    path = tmp_path / "risk.csv"
    summary = write_risk_qa(path, base, lora)
    assert summary["screened_rows"] == 1
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert "base_unsupported_update_claim" in rows[0]
    assert "lora_unnecessary_escalation" in rows[0]
    assert "unsupported_update_claim" in summary["lora_flag_counts"]


def test_comparison_contains_all_required_metrics_and_error_deltas():
    rows = [prediction(False, False, 1)]
    metrics = {
        "primary": {"intent_accuracy_percent": 100.0, "json_valid_rate_percent": 100.0, "schema_compliance_percent": 100.0},
        "secondary": {"category_accuracy_percent": 100.0, "escalation_accuracy_percent": 100.0, "escalation_positive_class": escalation_metrics(rows)},
        "operational": {"mean_latency_ms": 10.0, "p95_latency_ms": 10.0},
    }
    comparison = compare_metrics(metrics, metrics, rows, rows)
    assert set(comparison["metrics"]) == {"intent_accuracy", "category_accuracy", "json_valid_rate", "schema_compliance", "escalation_accuracy", "escalation_precision", "escalation_recall", "escalation_f1", "mean_latency", "p95_latency"}
    assert all(value["delta"] == 0 for value in comparison["error_tag_comparison"].values())


def test_generated_stage6_artifacts_cover_same_300_rows_and_reproduce_metrics():
    artifact_dir = REPO_ROOT / STAGE6_DIR
    base = [json.loads(line) for line in (REPO_ROOT / "artifacts/stage4/base_dev_predictions.jsonl").read_text().splitlines()]
    lora = [json.loads(line) for line in (artifact_dir / "lora_dev_predictions.jsonl").read_text().splitlines()]
    stored_metrics = json.loads((artifact_dir / "lora_metrics.json").read_text())
    assert len(lora) == 300
    assert len({row["source_index"] for row in lora}) == 300
    assert {row["source_index"] for row in lora} == {row["source_index"] for row in base}
    assert augment_metrics(aggregate_metrics(lora), lora) == stored_metrics
    assert stored_metrics["primary"]["json_valid_rate_percent"] == 100.0
    assert stored_metrics["primary"]["schema_compliance_percent"] == 100.0


def test_generated_manual_qa_and_risk_artifacts_are_deterministic_and_valid():
    artifact_dir = REPO_ROOT / STAGE6_DIR
    base = [json.loads(line) for line in (REPO_ROOT / "artifacts/stage4/base_dev_predictions.jsonl").read_text().splitlines()]
    lora = [json.loads(line) for line in (artifact_dir / "lora_dev_predictions.jsonl").read_text().splitlines()]
    with (artifact_dir / "manual_qa_samples.csv").open() as handle:
        qa = list(csv.DictReader(handle))
    expected = select_paired_qa(base, lora, 42, 30)
    assert len(qa) == 30
    assert [int(row["source_index"]) for row in qa] == [pair[1]["source_index"] for pair in expected]
    with (artifact_dir / "response_risk_qa.csv").open() as handle:
        risk = list(csv.DictReader(handle))
    assert len(risk) == 300
    assert all("base_unsupported_action_completion" in row for row in risk)
    assert all("lora_asks_for_sensitive_secret" in row for row in risk)
    manifest = json.loads((artifact_dir / "stage6_manifest.json").read_text())
    assert manifest["same_evaluator"] is True
    assert manifest["locked_content_accessed"] is False
    assert manifest["locked_inference_performed"] is False
    STAGE6_DIR,
    aggregate_metrics,
    augment_metrics,
