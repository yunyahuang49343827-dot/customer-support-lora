import json
from pathlib import Path

import pytest

from src.evaluation.base_baseline import (
    DEV_DATASET_RELATIVE_PATH,
    aggregate_metrics,
    evaluate_prediction,
    load_dev_records,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def truth():
    return {"intent": "track_order", "category": "ORDER", "needs_human": False}


def valid_output():
    return json.dumps({
        "intent": "track_order", "category": "ORDER", "needs_human": False,
        "response": "You can check the order status from your account.",
    })


def test_evaluator_accepts_valid_json():
    row = evaluate_prediction(valid_output(), truth(), 10.0)
    assert row["json_valid"] is True
    assert row["schema_compliant"] is True
    assert row["error_tags"] == []


def test_evaluator_rejects_malformed_json():
    row = evaluate_prediction('{"intent":"track_order"', truth(), 10.0)
    assert row["json_valid"] is False
    assert "invalid_json" in row["error_tags"]


@pytest.mark.parametrize(
    ("raw", "tag"),
    [
        ("prefix " + valid_output(), "extra_text_before_json"),
        (valid_output() + " suffix", "extra_text_after_json"),
    ],
)
def test_evaluator_tags_surrounding_text(raw, tag):
    row = evaluate_prediction(raw, truth(), 10.0)
    assert row["json_valid"] is False
    assert tag in row["error_tags"]


def test_evaluator_tags_missing_and_extra_keys():
    missing = json.loads(valid_output())
    del missing["response"]
    extra = json.loads(valid_output())
    extra["confidence"] = 1.0
    assert "missing_key" in evaluate_prediction(json.dumps(missing), truth(), 1.0)["error_tags"]
    assert "extra_key" in evaluate_prediction(json.dumps(extra), truth(), 1.0)["error_tags"]


def test_evaluator_tags_invalid_enums_and_mapping():
    invalid = json.loads(valid_output())
    invalid["intent"] = "unknown"
    mismatch = json.loads(valid_output())
    mismatch["category"] = "REFUND"
    assert "invalid_enum" in evaluate_prediction(json.dumps(invalid), truth(), 1.0)["error_tags"]
    assert "intent_category_mismatch" in evaluate_prediction(json.dumps(mismatch), truth(), 1.0)["error_tags"]


def test_evaluator_assigns_behavior_and_truncation_tags():
    payload = json.loads(valid_output())
    payload.update({"intent": "get_refund", "category": "REFUND", "needs_human": True, "response": ""})
    tags = evaluate_prediction(json.dumps(payload), truth(), 1.0, generation_truncated=True)["error_tags"]
    assert {"wrong_intent", "wrong_category", "wrong_needs_human", "empty_response", "generation_truncated"} <= set(tags)


def test_latency_aggregation_uses_mean_median_and_linear_p95():
    rows = []
    for latency in (10.0, 20.0, 30.0, 40.0):
        rows.append(evaluate_prediction(valid_output(), truth(), latency))
    operational = aggregate_metrics(rows)["operational"]
    assert operational["mean_latency_ms"] == 25.0
    assert operational["median_latency_ms"] == 25.0
    assert operational["p95_latency_ms"] == 38.5


def test_inference_loader_accepts_only_hard_coded_dev_path():
    assert len(load_dev_records(REPO_ROOT, DEV_DATASET_RELATIVE_PATH)) == 300
    with pytest.raises(ValueError, match="permits only"):
        load_dev_records(REPO_ROOT, "data/processed/train.jsonl")


def test_stage4_code_has_no_final_evaluation_dataset_path_literal():
    source = (REPO_ROOT / "src/evaluation/base_baseline.py").read_text(encoding="utf-8")
    forbidden = "data/processed/" + "locked_test.jsonl"
    assert forbidden not in source


def test_frozen_prompt_and_config_have_complete_contract():
    prompt = (REPO_ROOT / "prompts/base_system_prompt.txt").read_text(encoding="utf-8")
    config = json.loads((REPO_ROOT / "configs/base_inference.json").read_text(encoding="utf-8"))
    intents = json.loads((REPO_ROOT / "configs/intent_taxonomy.json").read_text())["intents"]
    categories = json.loads((REPO_ROOT / "configs/category_taxonomy.json").read_text())["categories"]
    assert all(entry["intent"] in prompt for entry in intents)
    assert all(entry["category"] in prompt for entry in categories)
    assert config["temperature"] == 0.0
    assert config["adapter_path"] is None
    assert config["dev_dataset_path"] == DEV_DATASET_RELATIVE_PATH
