import csv
import json
from pathlib import Path

from src.evaluation.prepare_manual_response_review import (
    HUMAN_REVIEW_FIELDS,
    INSTRUCTIONS_PATH,
    LORA_PREDICTIONS_PATH,
    MANUAL_QA_PATH,
    OUTPUT_PATH,
    RISK_QA_PATH,
    build_review_rows,
    prepare,
    read_csv,
    read_jsonl,
    validate_coverage,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_priority_deduplicates_and_keeps_all_reasons_blank_reviews():
    predictions = [
        {
            "source_index": 1, "stable_id": "g1", "instruction": "help", "ground_truth_intent": "track_order",
            "ground_truth_needs_human": False, "parsed_output": {"intent": "track_order", "needs_human": True, "response": "Call 123."},
            "error_tags": ["wrong_needs_human"],
        }
    ]
    manual = [{"source_index": "1"}]
    risk = [{"source_index": "1", "lora_risk_flags": "fabricated_contact_details;unnecessary_escalation"}]
    rows = build_review_rows(predictions, manual, risk)
    assert len(rows) == 1
    assert rows[0]["review_priority"] == 2
    assert rows[0]["selection_reasons"] == "risk:fabricated_contact_details;risk:unnecessary_escalation;escalation_false_positive;manual_qa_seed_42"
    assert all(rows[0][field] == "" for field in HUMAN_REVIEW_FIELDS)


def test_actual_review_queue_is_exact_required_union_and_human_fields_blank():
    summary = prepare(REPO_ROOT)
    manual = read_csv(REPO_ROOT / MANUAL_QA_PATH)
    risk = read_csv(REPO_ROOT / RISK_QA_PATH)
    predictions = read_jsonl(REPO_ROOT / LORA_PREDICTIONS_PATH)
    review = read_csv(REPO_ROOT / OUTPUT_PATH)
    validate_coverage(review, predictions, manual, risk)
    assert summary["manual_seed_rows"] == 30
    assert summary["lora_risk_rows"] == 44
    assert summary["escalation_false_positive_rows"] == 8
    assert len({row["source_id"] for row in review}) == len(review)
    assert len({row["dev_row_id"] for row in review}) == len(review)
    assert all(row["dev_row_id"].startswith("dev_") for row in review)
    assert all(row[field] == "" for row in review for field in HUMAN_REVIEW_FIELDS)


def test_all_high_priority_categories_and_false_positives_are_included():
    risk = read_csv(REPO_ROOT / RISK_QA_PATH)
    predictions = read_jsonl(REPO_ROOT / LORA_PREDICTIONS_PATH)
    review = read_csv(REPO_ROOT / OUTPUT_PATH)
    selected = {row["source_id"] for row in review}
    for flag in (
        "asks_for_sensitive_secret", "fabricated_contact_details", "fabricated_fees_or_timelines",
        "unsupported_action_completion",
    ):
        required = {row["source_index"] for row in risk if flag in row["lora_risk_flags"].split(";")}
        assert required <= selected
    false_positives = {
        str(row["source_index"])
        for row in predictions
        if row["ground_truth_needs_human"] is False and (row["parsed_output"] or {}).get("needs_human") is True
    }
    assert false_positives <= selected


def test_instructions_define_human_outcomes_and_preserve_stage_boundary():
    text = (REPO_ROOT / INSTRUCTIONS_PATH).read_text(encoding="utf-8")
    assert all(label in text for label in ("## PASS", "## MINOR_ISSUE", "## MAJOR_ISSUE"))
    assert "does not automatically declare Stage C6 PASS" in text
    assert "Heuristic risk flags are triage aids only" in text
