import json
from pathlib import Path

import pytest

from src.evaluation.contracts import validate_output
from src.training.qlora_formal import (
    ADAPTER_PATH,
    CONFIG_PATH,
    MODEL_ID,
    MODEL_REVISION,
    PROMPT_SHA256,
    TARGET_KEYS,
    TRAINING_TRAIN,
    TRAINING_VALID,
    load_config,
    parse_training_log,
    prepare,
    read_jsonl,
    summarize_lengths,
    validate_training_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_all_2700_train_and_300_validation_rows_are_converted():
    assert len(read_jsonl(REPO_ROOT / TRAINING_TRAIN)) == 2700
    assert len(read_jsonl(REPO_ROOT / TRAINING_VALID)) == 300


def test_formal_chat_rows_and_target_json_are_strictly_valid():
    rows = read_jsonl(REPO_ROOT / TRAINING_TRAIN) + read_jsonl(REPO_ROOT / TRAINING_VALID)
    frozen_prompt = (REPO_ROOT / "prompts/base_system_prompt.txt").read_text().strip()
    for row in rows:
        assert set(row) == {"messages"}
        assert [message["role"] for message in row["messages"]] == ["system", "user", "assistant"]
        assert row["messages"][0]["content"] == frozen_prompt
        assistant = row["messages"][2]["content"]
        assert isinstance(json.loads(assistant), dict)
        assert validate_output(assistant).valid
        assert not assistant.startswith("```")


def test_formal_config_is_exact_candidate_01_contract():
    config = load_config(REPO_ROOT)
    assert config["model"] == MODEL_ID
    assert config["mask_prompt"] is True
    assert config["num_layers"] == 16
    assert config["lora_parameters"]["rank"] == 8
    assert config["lora_parameters"]["scale"] == 16.0
    assert tuple(config["lora_parameters"]["keys"]) == TARGET_KEYS
    assert config["batch_size"] * config["grad_accumulation_steps"] == 2
    assert config["iters"] == 1350
    assert config["adapter_path"] == ADAPTER_PATH
    assert config["adapter_path"] != "artifacts/stage5a/adapter"


def test_sequence_audit_is_deterministic_and_selects_1024():
    before = json.loads((REPO_ROOT / "artifacts/stage5/sequence_length_summary.json").read_text())
    prepare(REPO_ROOT)
    after = json.loads((REPO_ROOT / "artifacts/stage5/sequence_length_summary.json").read_text())
    assert before == after
    assert after["train"]["count"] == 2700
    assert after["validation"]["count"] == 300
    assert after["train"]["count_over_1024"] == 0
    assert after["recommended_max_seq_length"] == 1024
    assert load_config(REPO_ROOT)["max_seq_length"] == 1024


def test_sequence_summary_threshold_counts_are_independently_correct():
    summary = summarize_lengths([100, 1024, 1025, 1536, 1537])
    assert summary["count_over_1024"] == 3
    assert summary["percentage_over_1024"] == 60.0
    assert summary["count_over_1536"] == 1
    assert summary["percentage_over_1536"] == 20.0


def test_formal_preflight_records_only_allowed_source_content():
    audit = json.loads((REPO_ROOT / "artifacts/stage5/dataset_preflight.json").read_text())
    assert audit["source_files_opened"] == [
        "data/processed/train.jsonl",
        "data/processed/validation.jsonl",
    ]
    assert audit["disallowed_dataset_content_accessed"] is False
    assert audit["source_index_overlap"] == 0
    assert audit["train"]["schema_invalid_count"] == 0
    assert audit["validation"]["schema_invalid_count"] == 0
    assert audit["train"]["policy_mismatch_count"] == 0
    assert audit["validation"]["policy_mismatch_count"] == 0


def test_formal_source_has_no_disallowed_dataset_path_literals():
    source = (REPO_ROOT / "src/training/qlora_formal.py").read_text(encoding="utf-8")
    assert "data/processed/" + "dev.jsonl" not in source
    assert "data/processed/" + "locked_test.jsonl" not in source


def test_frozen_prompt_hash_and_model_revision_are_fixed():
    import hashlib

    digest = hashlib.sha256((REPO_ROOT / "prompts/base_system_prompt.txt").read_bytes()).hexdigest()
    assert digest == PROMPT_SHA256
    assert MODEL_REVISION == "8b403126fc14f14cfc99bb4cfa72ecbc129ea677"


def test_formal_training_log_parser_requires_1350_finite_iterations():
    lines = ["Iter 1: Val loss 3.000, Val took 1.000s"]
    for iteration in range(10, 1351, 10):
        if iteration % 150 == 0:
            lines.append(f"Iter {iteration}: Val loss 2.000, Val took 1.000s")
        lines.append(
            f"Iter {iteration}: Train loss 1.500, Learning Rate 1.000e-05, "
            "It/sec 1.000, Tokens/sec 100.000, Trained Tokens 100, Peak mem 4.000 GB"
        )
    parsed = parse_training_log("\n".join(lines))
    assert parsed["iterations_completed"] == 1350
    assert len(parsed["training_reports"]) == 135
    assert parsed["final_validation_loss"] == 2.0
    broken = "\n".join(lines).replace("Train loss 1.500", "Train loss inf", 1)
    with pytest.raises(ValueError, match="NaN or Inf"):
        parse_training_log(broken)


def test_formal_training_manifest_contract_accepts_only_fixed_candidate():
    payload = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "quantization": {"bits": 4, "group_size": 64},
        "train_rows": 2700,
        "validation_rows": 300,
        "prompt_sha256": PROMPT_SHA256,
        "config_sha256": "hash",
        "sequence_length_audit": {},
        "max_seq_length": 1024,
        "num_layers": 16,
        "target_modules": list(TARGET_KEYS),
        "rank": 8,
        "scale": 16.0,
        "dropout": 0.0,
        "mask_prompt": True,
        "learning_rate": 1e-5,
        "batch_size": 1,
        "gradient_accumulation_steps": 2,
        "effective_batch_size": 2,
        "iterations": 1350,
        "adapter_path": ADAPTER_PATH,
        "adapter_files": ["adapter_config.json", "adapters.safetensors"],
        "checkpoint_files": [],
        "adapter_size_bytes": 1,
        "training_success": True,
        "reload_success": True,
        "dataset_boundary": {"dev_content_accessed": False, "locked_test_content_accessed": False},
        "behavioral_evaluation_performed": False,
    }
    validate_training_manifest(payload)
    payload["mask_prompt"] = False
    with pytest.raises(ValueError, match="invariants"):
        validate_training_manifest(payload)
    payload["mask_prompt"] = True
    payload["learning_rate"] = 2e-5
    with pytest.raises(ValueError, match="invariants"):
        validate_training_manifest(payload)
