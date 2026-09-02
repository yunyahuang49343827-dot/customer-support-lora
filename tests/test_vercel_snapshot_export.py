import json
from pathlib import Path

from src.demo.comparison import CURATED_EXAMPLES, sha256_file
from src.demo.export_vercel_snapshots import (
    C7_COMPARISON_PATH,
    C8_CONSTRAINTS_PATH,
    C8_PROMOTION_PATH,
    EXPECTED_IMMUTABLE_HASHES,
    OUTPUT_DIR,
    build_benchmark_snapshot,
    build_project_status,
    curated_cases,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_exactly_eight_existing_curated_examples_with_expected_labels():
    cases = curated_cases()
    assert len(cases) == len(CURATED_EXAMPLES) == 8
    assert len({case["id"] for case in cases}) == 8
    assert all(case["label_zh"] and case["message"] for case in cases)
    assert all(set(case["expected"]) == {"intent", "category", "needs_human"} for case in cases)
    assert [case["message"] for case in cases] == [example["message"] for example in CURATED_EXAMPLES]


def test_demo_snapshots_have_complete_real_base_and_lora_results():
    rows = read_json(REPO_ROOT / OUTPUT_DIR / "demo_cases.json")
    required = {
        "intent", "category", "needs_human", "json_valid", "schema_compliant",
        "response", "raw_output", "generation_truncated", "latency_ms", "generation_error",
    }
    assert len(rows) == 8
    for row in rows:
        assert row["provenance"]["source"] == "frozen_local_inference"
        assert row["provenance"]["candidate"] == "candidate_01"
        assert row["provenance"]["response_modified"] is False
        for role in ("base", "lora"):
            assert set(row[role]) == required
            if row[role]["generation_error"] is None:
                assert row[role]["raw_output"]
                assert row[role]["response"]


def test_benchmark_snapshot_exactly_matches_selected_stage7_metrics():
    source = read_json(REPO_ROOT / C7_COMPARISON_PATH)
    exported = read_json(REPO_ROOT / OUTPUT_DIR / "benchmark.json")
    assert exported == build_benchmark_snapshot(source)
    assert set(exported["metrics"]) == {
        "intent_accuracy", "category_accuracy", "json_valid_rate", "schema_compliance",
        "escalation_accuracy", "escalation_f1",
    }
    assert exported["locked_test_rerun_performed"] is False


def test_project_status_matches_stage8_promotion_and_constraints():
    promotion = read_json(REPO_ROOT / C8_PROMOTION_PATH)
    constraints = read_json(REPO_ROOT / C8_CONSTRAINTS_PATH)
    exported = read_json(REPO_ROOT / OUTPUT_DIR / "project_status.json")
    assert exported == build_project_status(promotion, constraints)
    assert exported["candidate"] == "candidate_01"
    assert exported["decision"] == "PROMOTE"
    assert exported["unrestricted_production_approval"] is False


def test_manifest_records_success_and_no_training_locked_rerun_or_c9_change():
    manifest = read_json(REPO_ROOT / OUTPUT_DIR / "v1_export_manifest.json")
    assert manifest["status"] == "COMPLETE"
    assert manifest["curated_examples"] == 8
    assert manifest["base_successful_inferences"] == 8
    assert manifest["lora_successful_inferences"] == 8
    assert manifest["training_performed"] is False
    assert manifest["locked_test_rerun_performed"] is False
    assert manifest["stage_c9_modified"] is False
    assert manifest["nextjs_started"] is False
    assert manifest["frozen_hashes_before"] == manifest["frozen_hashes_after"]


def test_all_frozen_c7_c8_and_c9_hashes_remain_exact():
    assert all(sha256_file(REPO_ROOT / path) == expected for path, expected in EXPECTED_IMMUTABLE_HASHES.items())


def test_exporter_has_no_training_or_locked_dataset_inference_path():
    source = (REPO_ROOT / "src/demo/export_vercel_snapshots.py").read_text(encoding="utf-8")
    assert "src.training" not in source
    assert "locked_test.jsonl" not in source
    assert "locked_evaluation" not in source
    assert '"nextjs_started": False' in source
    assert "package.json" not in source
    assert "next.config" not in source
    assert "from next" not in source.lower()
