import pandas as pd

from src.data.analyze_dataset import (
    calculate_percentage,
    detect_label_conflicts,
    duplicate_groups,
    extract_placeholders,
    normalize_instruction,
)


def test_placeholder_extraction_is_non_greedy_and_ordered():
    text = "Track {{Order Number}} for {{Customer Name}}."
    assert extract_placeholders(text) == ["{{Order Number}}", "{{Customer Name}}"]
    assert extract_placeholders(None) == []


def test_instruction_normalization_handles_entities_whitespace_and_punctuation():
    left = "  Where’s   order {{Order Number}}？  "
    right = "where's order {{Different Entity}}?"
    assert normalize_instruction(left) == "where's order <entity>?"
    assert normalize_instruction(left) == normalize_instruction(right)


def test_duplicate_grouping_excludes_singletons():
    assert duplicate_groups(["a", "b", "a", "a", "c", "b"]) == {
        "a": [0, 2, 3],
        "b": [1, 5],
    }


def test_label_conflict_detection_returns_all_conflicting_rows():
    frame = pd.DataFrame(
        {
            "normalized_instruction": ["same", "same", "other"],
            "instruction": ["Same", "SAME", "Other"],
            "category": ["A", "B", "A"],
            "intent": ["one", "two", "one"],
        }
    )
    conflicts = detect_label_conflicts(frame)
    assert conflicts["normalized_instruction"].tolist() == ["same", "same"]
    assert set(conflicts["intent"]) == {"one", "two"}


def test_percentage_calculation_and_zero_denominator():
    assert calculate_percentage(1, 4) == 25.0
    assert calculate_percentage(1, 3) == 33.333333
    assert calculate_percentage(0, 0) == 0.0

