import json
from pathlib import Path

import pytest
import yaml

from src.evaluation.contracts import validate_output
from src.training.qlora_smoke import (
    CONFIG_PATH,
    REQUIRED_ADAPTER_FILES,
    SMOKE_TRAIN,
    SMOKE_VALID,
    TARGET_KEYS,
    load_config,
    parse_training_log,
    prepare,
    read_jsonl,
    validate_adapter_artifacts,
    validate_chat_rows,
    validate_reload_artifact,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_smoke_sampling_is_deterministic_and_has_80_20_counts():
    first = prepare(REPO_ROOT)
    first_train = (REPO_ROOT / SMOKE_TRAIN).read_bytes()
    first_valid = (REPO_ROOT / SMOKE_VALID).read_bytes()
    second = prepare(REPO_ROOT)
    assert (REPO_ROOT / SMOKE_TRAIN).read_bytes() == first_train
    assert (REPO_ROOT / SMOKE_VALID).read_bytes() == first_valid
    assert first == second
    assert first["train"]["row_count"] == 80
    assert first["valid"]["row_count"] == 20


def test_smoke_chat_roles_targets_and_schema_are_valid():
    train = read_jsonl(REPO_ROOT / SMOKE_TRAIN)
    valid = read_jsonl(REPO_ROOT / SMOKE_VALID)
    assert validate_chat_rows(train, 80)["schema_invalid_count"] == 0
    assert validate_chat_rows(valid, 20)["schema_invalid_count"] == 0
    for row in train + valid:
        assert [message["role"] for message in row["messages"]] == ["system", "user", "assistant"]
        assistant = row["messages"][2]["content"]
        assert isinstance(json.loads(assistant), dict)
        assert validate_output(assistant).valid
        assert not assistant.startswith("```")


def test_smoke_preflight_records_only_allowed_source_content_and_disjoint_rows():
    audit = json.loads((REPO_ROOT / "artifacts/stage5a/dataset_preflight.json").read_text())
    assert audit["source_files_opened"] == [
        "data/processed/train.jsonl",
        "data/processed/validation.jsonl",
    ]
    assert audit["disallowed_dataset_content_accessed"] is False
    assert audit["source_index_overlap"] == 0
    assert len(set(audit["train"]["source_indices"])) == 80
    assert len(set(audit["valid"]["source_indices"])) == 20


def test_stage5a_source_has_no_disallowed_dataset_path_literals():
    source = (REPO_ROOT / "src/training/qlora_smoke.py").read_text(encoding="utf-8")
    assert "data/processed/" + "dev.jsonl" not in source
    assert "data/processed/" + "locked_test.jsonl" not in source


def test_config_uses_scale_and_exact_target_keys():
    config = load_config(REPO_ROOT)
    raw = yaml.safe_load((REPO_ROOT / CONFIG_PATH).read_text())
    lora = config["lora_parameters"]
    assert raw == config
    assert lora["scale"] == 16.0
    assert "alpha" not in lora
    assert tuple(lora["keys"]) == TARGET_KEYS
    assert config["fine_tune_type"] == "lora"
    assert config["num_layers"] == 8
    assert config["iters"] == 20
    assert config["test"] is False


def test_adapter_artifact_validation_requires_config_and_weights(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    for name in REQUIRED_ADAPTER_FILES:
        (adapter / name).write_bytes(b"x")
    assert validate_adapter_artifacts(adapter) == sorted(REQUIRED_ADAPTER_FILES)
    (adapter / "adapters.safetensors").write_bytes(b"")
    with pytest.raises(ValueError, match="empty"):
        validate_adapter_artifacts(adapter)


def test_reload_test_artifact_schema_accepts_successful_nonempty_samples():
    payload = {
        "adapter_load_success": True,
        "generation_success": True,
        "sample_count": 3,
        "samples": [{"instruction": "x", "raw_output": "non-empty"}] * 3,
        "new_process": True,
    }
    validate_reload_artifact(payload)
    payload["samples"][0] = {"instruction": "x", "raw_output": ""}
    with pytest.raises(ValueError, match="empty"):
        validate_reload_artifact(payload)


def test_training_log_parser_requires_finite_20_step_run():
    lines = []
    for iteration in range(1, 21):
        if iteration in (1, 10, 20):
            lines.append(f"Iter {iteration}: Val loss 2.000, Val took 0.100s")
        lines.append(
            f"Iter {iteration}: Train loss 1.500, Learning Rate 1.000e-05, "
            "It/sec 1.000, Tokens/sec 100.000, Trained Tokens 100, Peak mem 1.250 GB"
        )
    parsed = parse_training_log("\n".join(lines))
    assert parsed["finite_losses"] is True
    assert len(parsed["training_reports"]) == 20
    assert parsed["peak_memory_gb"] == 1.25
    broken = "\n".join(lines).replace("Train loss 1.500", "Train loss nan", 1)
    with pytest.raises(ValueError, match="NaN or Inf"):
        parse_training_log(broken)
