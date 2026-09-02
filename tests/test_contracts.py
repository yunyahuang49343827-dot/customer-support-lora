import json
from pathlib import Path

from jsonschema import Draft202012Validator

from src.evaluation.build_contracts import ESCALATION_POLICY, build_contracts
from src.evaluation.contracts import validate_output


REPO_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_TRUE_INTENTS = {
    "complaint",
    "contact_customer_service",
    "contact_human_agent",
    "delete_account",
    "get_refund",
    "payment_issue",
}
EXPLICIT_FALSE_INTENTS = {
    "cancel_order",
    "change_order",
    "change_shipping_address",
    "recover_password",
}


def valid_payload():
    return {
        "intent": "track_order",
        "category": "ORDER",
        "needs_human": False,
        "response": "You can check the latest status from your order history.",
    }


def test_valid_output_passes_contract():
    result = validate_output(json.dumps(valid_payload()))
    assert result.valid
    assert result.errors == ()


def test_valid_output_passes_machine_readable_json_schema():
    schema = json.loads((REPO_ROOT / "configs/output_schema.json").read_text())
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(valid_payload())


def test_malformed_json_fails():
    result = validate_output('{"intent": "track_order"')
    assert not result.valid
    assert result.errors == ("malformed_json",)


def test_missing_key_fails():
    payload = valid_payload()
    del payload["response"]
    result = validate_output(json.dumps(payload))
    assert not result.valid
    assert "missing_keys:response" in result.errors


def test_extra_key_fails():
    payload = valid_payload()
    payload["confidence"] = 0.9
    result = validate_output(json.dumps(payload))
    assert not result.valid
    assert "extra_keys:confidence" in result.errors


def test_invalid_intent_fails():
    payload = valid_payload()
    payload["intent"] = "where_is_my_stuff"
    result = validate_output(json.dumps(payload))
    assert not result.valid
    assert "invalid_intent" in result.errors


def test_invalid_category_fails():
    payload = valid_payload()
    payload["category"] = "LOGISTICS"
    result = validate_output(json.dumps(payload))
    assert not result.valid
    assert "invalid_category" in result.errors


def test_non_boolean_needs_human_fails():
    payload = valid_payload()
    payload["needs_human"] = "false"
    result = validate_output(json.dumps(payload))
    assert not result.valid
    assert "needs_human_not_boolean" in result.errors


def test_empty_or_whitespace_response_fails():
    payload = valid_payload()
    payload["response"] = "   "
    result = validate_output(json.dumps(payload))
    assert not result.valid
    assert "response_empty" in result.errors


def test_surrounding_text_or_markdown_fence_fails_json_contract():
    raw = "```json\n" + json.dumps(valid_payload()) + "\n```"
    assert validate_output(raw).errors == ("malformed_json",)


def test_valid_labels_with_wrong_mapping_fail():
    payload = valid_payload()
    payload["category"] = "REFUND"
    result = validate_output(json.dumps(payload))
    assert not result.valid
    assert "intent_category_mismatch" in result.errors


def test_generated_taxonomies_schema_and_policy_are_consistent():
    intent_taxonomy = json.loads((REPO_ROOT / "configs/intent_taxonomy.json").read_text())
    category_taxonomy = json.loads((REPO_ROOT / "configs/category_taxonomy.json").read_text())
    policy = json.loads((REPO_ROOT / "configs/escalation_policy.json").read_text())
    schema = json.loads((REPO_ROOT / "configs/output_schema.json").read_text())

    intents = {entry["intent"] for entry in intent_taxonomy["intents"]}
    categories = {entry["category"] for entry in category_taxonomy["categories"]}
    policy_intents = {entry["intent"] for entry in policy["intents"]}
    assert len(intents) == intent_taxonomy["intent_count"] == 27
    assert len(categories) == category_taxonomy["category_count"] == 11
    assert policy_intents == intents == set(schema["properties"]["intent"]["enum"])
    assert categories == set(schema["properties"]["category"]["enum"])
    policy_by_intent = {entry["intent"]: entry["needs_human"] for entry in policy["intents"]}
    computed_true = {intent for intent, needs_human in policy_by_intent.items() if needs_human}
    computed_false = {intent for intent, needs_human in policy_by_intent.items() if not needs_human}
    assert computed_true == OFFICIAL_TRUE_INTENTS
    assert computed_false == intents - OFFICIAL_TRUE_INTENTS
    assert policy["true_intent_count"] == len(computed_true) == 6
    assert policy["false_intent_count"] == len(computed_false) == 21
    assert all(policy_by_intent[intent] is False for intent in EXPLICIT_FALSE_INTENTS)
    assert policy["manual_confirmation_required"] is False


def test_builder_source_matches_official_escalation_policy():
    builder_true = {intent for intent, (needs_human, _) in ESCALATION_POLICY.items() if needs_human}
    builder_false = {intent for intent, (needs_human, _) in ESCALATION_POLICY.items() if not needs_human}
    assert builder_true == OFFICIAL_TRUE_INTENTS
    assert len(builder_true) == 6
    assert len(builder_false) == 21
    assert all(ESCALATION_POLICY[intent][0] is False for intent in EXPLICIT_FALSE_INTENTS)


def test_builder_rebuild_preserves_policy_entry_and_summary_consistency(tmp_path):
    config_dir = tmp_path / "configs"
    report_dir = tmp_path / "reports"
    result = build_contracts(REPO_ROOT / "artifacts/stage1", config_dir, report_dir)
    rebuilt = json.loads((config_dir / "escalation_policy.json").read_text())
    rebuilt_true = {entry["intent"] for entry in rebuilt["intents"] if entry["needs_human"]}
    rebuilt_false = {entry["intent"] for entry in rebuilt["intents"] if not entry["needs_human"]}
    assert rebuilt_true == OFFICIAL_TRUE_INTENTS
    assert rebuilt["true_intent_count"] == len(rebuilt_true) == result["escalation_true"] == 6
    assert rebuilt["false_intent_count"] == len(rebuilt_false) == result["escalation_false"] == 21


def test_promotion_gate_encodes_pretraining_thresholds():
    gate = json.loads((REPO_ROOT / "configs/promotion_gate.json").read_text())
    assert gate["defined_before_training"] is True
    assert gate["required"]["intent_accuracy"]["metric_delta_threshold"] == 3.0
    assert gate["required"]["json_valid_rate"]["metric_delta_threshold"] == -1.0
    assert gate["required"]["schema_compliance"]["metric_delta_threshold"] == -1.0
    assert gate["guardrails"]["category_accuracy"]["material_regression_threshold"] == 3.0
    assert gate["guardrails"]["escalation_accuracy"]["material_regression_threshold"] == 3.0
