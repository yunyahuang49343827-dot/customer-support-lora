import json
from pathlib import Path

from src.evaluation.promotion_decision import (
    EXPECTED_PROMOTION_GATE_SHA256,
    KNOWN_LIMITATION_EN,
    KNOWN_LIMITATION_ZH,
    LATENCY_WARNING,
    MANUAL_QA_CONCLUSION,
    evaluate_gates,
    governance_preflight,
    load_evidence,
    sha256_file,
    stage9_outputs,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
STAGE8 = REPO_ROOT / "artifacts/stage8"


def artifact(name):
    return json.loads((STAGE8 / name).read_text(encoding="utf-8"))


def test_governance_integrity_uses_complete_frozen_c6_5_and_c7_evidence():
    evidence = load_evidence(REPO_ROOT)
    result = artifact("governance_integrity.json")
    assert result["status"] == "PASS"
    assert result["fail_count"] == 0
    assert evidence["freeze_manifest"]["freeze_status"] == "PASS"
    assert evidence["stage7_manifest"]["status"] == "EVALUATION_COMPLETE"
    assert evidence["stage7_integrity"]["status"] == "PASS"
    assert evidence["stage7_manifest"]["base_attempts"] == 300
    assert evidence["stage7_manifest"]["lora_attempts"] == 300


def test_promotion_gate_hash_and_thresholds_are_unchanged():
    evidence = load_evidence(REPO_ROOT)
    gate = evidence["promotion_gate"]
    assert sha256_file(REPO_ROOT / "configs/promotion_gate.json") == EXPECTED_PROMOTION_GATE_SHA256
    assert gate["required"]["intent_accuracy"]["metric_delta_threshold"] == 3.0
    assert gate["required"]["json_valid_rate"]["metric_delta_threshold"] == -1.0
    assert gate["required"]["schema_compliance"]["metric_delta_threshold"] == -1.0
    assert gate["guardrails"]["category_accuracy"]["material_regression_threshold"] == 3.0
    assert gate["guardrails"]["escalation_accuracy"]["material_regression_threshold"] == 3.0
    assert gate["latency_is_promotion_criterion"] is False


def test_quantitative_gate_values_and_math_match_stage7_evidence():
    evidence = load_evidence(REPO_ROOT)
    result = evaluate_gates(evidence)
    gates = {row["gate_name"]: row for row in result["blocking_gates"]}
    expected = {
        "intent_accuracy": (28.0, 94.0, 66.0),
        "json_valid_rate": (99.333333, 99.666667, 0.333334),
        "schema_compliance": (36.666667, 99.333333, 62.666666),
        "category_accuracy": (61.666667, 99.0, 37.333333),
        "escalation_accuracy": (79.0, 98.666667, 19.666667),
    }
    for name, values in expected.items():
        assert (gates[name]["base_result"], gates[name]["lora_result"], gates[name]["delta"]) == values
        assert gates[name]["status"] == "PASS"
    assert result["post_hoc_thresholds_added"] is False


def test_behavioral_and_safety_gates_include_manual_qa_conclusion():
    result = evaluate_gates(load_evidence(REPO_ROOT))
    gates = {row["gate_name"]: row for row in result["blocking_gates"]}
    assert result["manual_response_qa_conclusion"] == MANUAL_QA_CONCLUSION
    assert gates["critical_behavioral_regression"]["status"] == "PASS"
    assert gates["material_response_safety_regression"]["status"] == "PASS"
    assert "not a complete quality judgment" in gates["material_response_safety_regression"]["rationale"]


def test_latency_is_an_operational_warning_not_a_blocker():
    result = evaluate_gates(load_evidence(REPO_ROOT))
    latency = result["operational_gates"][0]
    assert latency["gate_name"] == "latency"
    assert latency["status"] == "OPERATIONAL_WARNING"
    assert latency["blocking"] is False
    assert latency["rationale"] == LATENCY_WARNING


def test_formal_decision_and_known_limitations_are_preserved():
    decision = artifact("promotion_decision.json")
    assert decision["decision"] == "PROMOTE"
    assert decision["all_frozen_blocking_gates_passed"] is True
    assert decision["blocking_failures"] == []
    assert decision["manual_response_qa_conclusion"] == MANUAL_QA_CONCLUSION
    assert decision["critical_behavioral_regression"] is False
    assert decision["material_response_safety_regression"] is False
    assert KNOWN_LIMITATION_EN in decision["known_limitations"]
    assert KNOWN_LIMITATION_ZH in decision["known_limitations"]
    assert decision["unrestricted_production_approval"] is False


def test_deployment_scope_and_constraints_are_explicit():
    constraints = artifact("deployment_constraints.json")
    assert "structured classification" in constraints["approved_scope"]
    assert "schema-constrained routing" in constraints["approved_scope"]
    assert "escalation decision support" in constraints["approved_scope"]
    assert "enterprise factual authority" in constraints["not_approved_as"]
    assert "backend action executor" in constraints["not_approved_as"]
    assert "company policies" in constraints["requires_external_grounding_for"]
    assert "refund execution" in constraints["requires_backend_tools_for"]
    assert "sensitive/high-impact actions" in constraints["human_review_recommended_for"]


def test_stage8_manifest_records_no_mutation_rerun_or_stage9():
    manifest = artifact("stage8_manifest.json")
    assert manifest["status"] == "COMPLETE"
    assert manifest["decision"] == "PROMOTE"
    assert manifest["promotion_gate_hash"] == EXPECTED_PROMOTION_GATE_SHA256
    assert manifest["training_performed"] is False
    assert manifest["locked_rerun_performed"] is False
    assert manifest["prompt_modified"] is False
    assert manifest["evaluator_modified"] is False
    assert manifest["threshold_modified"] is False
    assert manifest["candidate_modified"] is False
    assert manifest["stage9_performed"] is False
    demo_path = REPO_ROOT / "demo/app.py"
    if demo_path.exists():
        source = demo_path.read_text(encoding="utf-8")
        assert "src.training" not in source
        assert "locked_test.jsonl" not in source


def test_report_contains_required_decision_boundary():
    report = (REPO_ROOT / "reports/stage8_promotion_report.md").read_text(encoding="utf-8")
    assert "Candidate 01: **PROMOTE**" in report
    assert "PROMOTE does not represent unrestricted production approval" in report
    assert MANUAL_QA_CONCLUSION in report
    assert KNOWN_LIMITATION_EN in report
    assert KNOWN_LIMITATION_ZH in report
