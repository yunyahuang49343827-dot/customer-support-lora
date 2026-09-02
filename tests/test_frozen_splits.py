import csv
import hashlib
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from src.data.build_frozen_splits import (
    SEED,
    SPLIT_ORDER,
    TARGET_SIZES,
    apply_source_quality_gate,
    assign_groups,
    build_records,
    build_source_groups,
    compact_response,
    discover_source_arrow,
    load_c2_contracts,
    load_source_frame,
    overlap_summary,
    validate_compaction_completeness,
    validate_source_response_quality,
)
from src.evaluation.contracts import validate_output


REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = REPO_ROOT / "data/processed"
MANIFEST_DIR = REPO_ROOT / "data/manifests"
ARTIFACT_DIR = REPO_ROOT / "artifacts/stage3"
PRE_QUALITY_GATE_LOCKED_TEST_SHA256 = "b7f7af8c5e366c743fafd68c8c8f3e7a2b101dfce53e63bf1f7a8ead0bce1fac"


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


@pytest.fixture(scope="module")
def split_records():
    return {split: read_jsonl(PROCESSED_DIR / f"{split}.jsonl") for split in SPLIT_ORDER}


@pytest.fixture(scope="module")
def contracts():
    return load_c2_contracts(REPO_ROOT / "configs")


def test_all_four_split_files_exist_with_manifest_sizes(split_records):
    manifest = json.loads((MANIFEST_DIR / "split_manifest.json").read_text())
    assert set(split_records) == set(SPLIT_ORDER)
    assert {split: len(rows) for split, rows in split_records.items()} == TARGET_SIZES
    assert manifest["actual_split_sizes"] == TARGET_SIZES
    assert manifest["group_aware"] is True


def test_same_normalized_group_and_exact_instruction_never_cross_split(split_records):
    group_owner = {}
    instruction_owner = {}
    for split, records in split_records.items():
        for record in records:
            group_id = record["metadata"]["group_id"]
            normalized = record["metadata"]["normalized_instruction"]
            instruction = record["instruction"]
            assert group_owner.setdefault(group_id, split) == split
            assert group_owner.setdefault(normalized, split) == split
            assert instruction_owner.setdefault(instruction, split) == split


def test_no_source_index_overlap(split_records):
    owners = {}
    for split, records in split_records.items():
        for record in records:
            source_index = record["metadata"]["source_index"]
            assert source_index not in owners, f"source index {source_index} in {owners[source_index]} and {split}"
            owners[source_index] = split


def test_all_records_validate_schema_taxonomies_and_policy(split_records, contracts):
    schema = json.loads((REPO_ROOT / "configs/output_schema.json").read_text())
    schema_validator = Draft202012Validator(schema)
    for split, records in split_records.items():
        assert {record["target"]["intent"] for record in records} == set(contracts["intent_to_category"])
        assert {record["target"]["category"] for record in records} == contracts["categories"]
        for record in records:
            target = record["target"]
            assert validate_output(json.dumps(target, ensure_ascii=False)).valid
            schema_validator.validate(target)
            assert target["category"] == contracts["intent_to_category"][target["intent"]]
            assert target["needs_human"] is contracts["escalation"][target["intent"]]


def test_all_intent_quotas_are_preserved(split_records):
    for split, records in split_records.items():
        counts = {}
        for record in records:
            intent = record["target"]["intent"]
            counts[intent] = counts.get(intent, 0) + 1
        if split == "train":
            assert set(counts.values()) == {100}
        else:
            assert set(counts.values()) <= {11, 12}
            assert sum(counts.values()) == 300


def test_cross_split_overlap_artifact_is_all_zero():
    overlap = json.loads((ARTIFACT_DIR / "cross_split_overlap.json").read_text())
    assert overlap["all_overlap_counts_zero"] is True
    assert all(check["overlap_count"] == 0 for check in overlap["checks"].values())


def test_locked_test_hash_exists_and_matches_file():
    hashes = json.loads((MANIFEST_DIR / "dataset_hashes.json").read_text())
    for split in SPLIT_ORDER:
        digest = hashlib.sha256((PROCESSED_DIR / f"{split}.jsonl").read_bytes()).hexdigest()
        assert hashes["files"][split]["sha256"] == digest
    locked = hashes["locked_test"]
    digest = hashlib.sha256((PROCESSED_DIR / "locked_test.jsonl").read_bytes()).hexdigest()
    assert locked["frozen"] is True
    assert locked["sha256"] == digest == hashes["files"]["locked_test"]["sha256"]
    assert locked["supersedes_sha256"] == PRE_QUALITY_GATE_LOCKED_TEST_SHA256
    assert locked["revision_id"] == "stage3_source_response_quality_gate_v1"
    assert locked["sha256"] == locked["supersedes_sha256"]


def test_group_assignment_manifest_keeps_multrow_groups_atomic():
    with (MANIFEST_DIR / "group_assignment.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected = [row for row in rows if row["assigned_split"] != "not_selected"]
    assert selected
    assert any(int(row["group_size"]) > 1 for row in selected)
    assert len({row["group_id"] for row in selected}) == len(selected)
    assert sum(int(row["selected_row_count"]) for row in selected) == sum(TARGET_SIZES.values())


def test_assignment_is_deterministic_on_actual_source():
    source = discover_source_arrow(REPO_ROOT / ".cache/huggingface")
    frame = load_source_frame(source)
    groups = build_source_groups(frame)
    intents = sorted(load_c2_contracts(REPO_ROOT / "configs")["intent_to_category"])
    first, first_quotas = assign_groups(groups, intents, TARGET_SIZES, SEED)
    second, second_quotas = assign_groups(groups, intents, TARGET_SIZES, SEED)
    assert first == second
    assert first_quotas == second_quotas


def test_response_compaction_never_hard_truncates_mid_sentence():
    short = "A short complete response."
    assert compact_response(short) == (short, "preserved_within_limit", ())

    sentence = "This complete sentence provides useful customer guidance and enough context for a safe next step."
    long = " ".join([sentence] * 20)
    compacted, strategy, errors = compact_response(long)
    assert strategy == "compacted_complete_sentence_prefix"
    assert errors == ()
    assert compacted.endswith(".")
    assert len(compacted) <= 650

    no_boundary = "x" * 700
    fallback, strategy, errors = compact_response(no_boundary)
    assert fallback == no_boundary
    assert strategy == "conservative_full_fallback"
    assert errors == ()


def test_numbered_list_is_never_compacted_to_a_marker():
    intro = "Please follow these instructions to complete the requested account task safely."
    items = " ".join(
        f"{number}. Complete step {number} using the information shown in your account settings."
        for number in range(1, 12)
    )
    original = f"{intro} {items}"
    response, strategy, errors = compact_response(original)
    assert response == original
    assert strategy == "completeness_full_fallback"
    assert "list_block_would_be_truncated" in errors
    assert not response.endswith(": 1.")


def test_bullet_list_is_never_partially_compacted():
    intro = "Here are the available methods:"
    items = " ".join(
        f"- Method {number} includes complete guidance and a safe next step for the customer."
        for number in range(1, 12)
    )
    original = f"{intro} {items}"
    response, strategy, errors = compact_response(original)
    assert response == original
    assert strategy == "completeness_full_fallback"
    assert "list_block_would_be_truncated" in errors


def test_incomplete_list_introduction_falls_back_to_original():
    first_sentence = (
        "This customer-facing explanation provides enough preliminary context to make the candidate prefix long enough "
        "for normal compaction while remaining a complete sentence with no invented content."
    )
    original = (
        f"{first_sentence} Please follow these steps: 1. Start with the first complete action shown in your settings. "
        + " ".join(f"{number}. Complete the next documented action safely." for number in range(2, 15))
    )
    response, strategy, errors = compact_response(original)
    assert response == original
    assert strategy == "completeness_full_fallback"
    assert errors
    assert validate_compaction_completeness(original, f"{first_sentence} Please follow these steps: 1.")


@pytest.mark.parametrize(
    ("response", "reason"),
    [
        ("To finish the request, please follow these steps: 1.", "incomplete_numbered_list"),
        ("Available methods:\n-", "incomplete_bullet_list"),
        ("For this request, follow these instructions:", "incomplete_list_introduction"),
    ],
)
def test_incomplete_source_lists_are_rejected(response, reason):
    assert reason in validate_source_response_quality(response)


def test_clean_numbered_list_and_clean_prose_pass_source_quality():
    clean_list = "Please follow these steps: 1. Open settings. 2. Confirm the change."
    clean_prose = "You can review the current status from the orders page in your account."
    assert validate_source_response_quality(clean_list) == ()
    assert validate_source_response_quality(clean_prose) == ()


def test_source_quality_replacements_preserve_sizes_and_label_distributions():
    manifest = json.loads((MANIFEST_DIR / "split_manifest.json").read_text())
    preservation = manifest["membership_preservation"]
    quality = manifest["source_response_quality"]
    assert preservation["all_sizes_and_label_distributions_unchanged"] is True
    assert quality["selected_rows_scanned"] == sum(TARGET_SIZES.values())
    assert quality["failed_rows"] == quality["replacements_made"] > 0
    assert quality["final_selected_quality_failures"] == 0
    for split, details in preservation["per_split"].items():
        expected = quality["per_split_replacement_counts"][split]
        assert details["removed_source_row_count"] == expected
        assert details["added_source_row_count"] == expected
        assert details["size_unchanged"] is True
        assert details["intent_distribution_unchanged"] is True
        assert details["category_distribution_unchanged"] is True
        assert details["needs_human_distribution_unchanged"] is True


def test_replacements_are_same_intent_clean_unused_groups_without_leakage():
    source = discover_source_arrow(REPO_ROOT / ".cache/huggingface")
    frame = load_source_frame(source)
    groups = build_source_groups(frame)
    contracts = load_c2_contracts(REPO_ROOT / "configs")
    assignments, _ = assign_groups(groups, sorted(contracts["intent_to_category"]), TARGET_SIZES, SEED)
    original_selected = {
        source_index
        for group in groups
        if group.group_id in assignments
        for source_index in group.source_indices
    }
    source_splits, audit, _ = apply_source_quality_gate(frame, groups, assignments, SEED)
    indexed = frame.set_index("source_index")
    group_by_source = {index: group for group in groups for index in group.source_indices}
    assert audit["failed_rows"] == audit["replacements_made"] > 0
    for replacement in audit["replacements"]:
        removed = replacement["removed_source_index"]
        added = replacement["replacement_source_index"]
        assert added not in original_selected
        assert indexed.loc[removed, "intent"] == indexed.loc[added, "intent"] == replacement["intent"]
        assert group_by_source[added].size == 1
        assert validate_source_response_quality(indexed.loc[added, "response"]) == ()
    records, _ = build_records(frame, groups, source_splits, contracts, SEED)
    assert overlap_summary(records)["all_overlap_counts_zero"] is True


def test_seed_42_quality_gate_rerun_is_identical():
    source = discover_source_arrow(REPO_ROOT / ".cache/huggingface")
    frame = load_source_frame(source)
    groups = build_source_groups(frame)
    contracts = load_c2_contracts(REPO_ROOT / "configs")
    assignments, _ = assign_groups(groups, sorted(contracts["intent_to_category"]), TARGET_SIZES, SEED)
    first_splits, first_audit, _ = apply_source_quality_gate(frame, groups, assignments, SEED)
    second_splits, second_audit, _ = apply_source_quality_gate(frame, groups, assignments, SEED)
    first_records, _ = build_records(frame, groups, first_splits, contracts, SEED)
    second_records, _ = build_records(frame, groups, second_splits, contracts, SEED)
    assert first_splits == second_splits
    assert first_audit == second_audit
    assert first_records == second_records


def test_source_quality_artifacts_are_bounded_and_complete():
    audit = json.loads((ARTIFACT_DIR / "source_response_quality.json").read_text())
    with (ARTIFACT_DIR / "source_response_quality_examples.csv").open(newline="", encoding="utf-8") as handle:
        examples = list(csv.DictReader(handle))
    assert audit["selected_rows_scanned"] == 3600
    assert audit["failed_rows"] == audit["replacements_made"]
    assert audit["final_selected_quality_failures"] == 0
    assert len(examples) <= 50
    assert sum(row["split"] == "locked_test" for row in examples) <= 2


def test_compaction_manifest_records_completeness_fallbacks():
    manifest = json.loads((MANIFEST_DIR / "split_manifest.json").read_text())
    compaction = manifest["response_compaction"]
    assert compaction["completeness_failure_count"] > 0
    assert compaction["strategy_counts"]["completeness_full_fallback"] == compaction["completeness_failure_count"]
    assert compaction["completeness_failure_reasons"]["list_block_would_be_truncated"] > 0


def test_generated_compacted_prefixes_are_structurally_complete(split_records):
    checked = 0
    for records in split_records.values():
        for record in records:
            if record["metadata"]["response_compaction"] != "compacted_complete_sentence_prefix":
                continue
            response = record["target"]["response"]
            assert validate_compaction_completeness(response, response) == ()
            assert response.endswith((".", "!", "?"))
            checked += 1
    assert checked > 0


def test_distribution_and_manual_sample_artifacts_cover_required_rows():
    with (ARTIFACT_DIR / "split_distribution.csv").open(newline="", encoding="utf-8") as handle:
        distribution = list(csv.DictReader(handle))
    with (ARTIFACT_DIR / "split_samples.csv").open(newline="", encoding="utf-8") as handle:
        samples = list(csv.DictReader(handle))
    assert len(distribution) == 4 * 27
    assert all(sum(1 for row in distribution if row["split"] == split) == 27 for split in SPLIT_ORDER)
    assert len(samples) == 4 * 5
    assert all(sum(1 for row in samples if row["split"] == split) == 5 for split in SPLIT_ORDER)
