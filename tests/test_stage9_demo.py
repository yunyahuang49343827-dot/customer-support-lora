import importlib.util
import json
import sys
import types
from pathlib import Path

from streamlit.testing.v1 import AppTest

from src.demo import comparison


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_frozen_prompt_adapter_and_contract_integrity():
    result = comparison.verify_frozen_integrity(REPO_ROOT, check_model_availability=False)
    assert result["status"] == "PASS"
    assert result["fail_count"] == 0
    assert comparison.sha256_file(REPO_ROOT / comparison.PROMPT_PATH) == comparison.PROMPT_SHA256
    assert comparison.sha256_file(REPO_ROOT / comparison.ADAPTER_PATH / "adapters.safetensors") == comparison.ADAPTER_SHA256


def test_base_adapter_absent_and_candidate_adapter_present(monkeypatch):
    calls = []
    monkeypatch.setattr(comparison, "verify_frozen_integrity", lambda *args, **kwargs: {"status": "PASS"})
    monkeypatch.setattr(comparison, "resolve_model_snapshot", lambda check_availability=True: Path("/tmp/frozen-model"))
    fake_mlx_lm = types.SimpleNamespace(
        load=lambda model, adapter_path=None: (calls.append((model, adapter_path)) or (object(), object()))
    )
    monkeypatch.setitem(sys.modules, "mlx_lm", fake_mlx_lm)

    base = comparison.load_model_bundle(REPO_ROOT, adapter_path=None)
    candidate = comparison.load_model_bundle(REPO_ROOT, adapter_path=comparison.ADAPTER_PATH)
    assert base["adapter_path"] is None
    assert base["role"] == "base"
    assert candidate["adapter_path"] == comparison.ADAPTER_PATH
    assert candidate["role"] == "lora"
    assert calls == [
        ("/tmp/frozen-model", None),
        ("/tmp/frozen-model", str(REPO_ROOT / comparison.ADAPTER_PATH)),
    ]


def test_parser_and_schema_validator_accept_valid_contract_output():
    raw = json.dumps({
        "intent": "get_refund",
        "category": "REFUND",
        "needs_human": True,
        "response": "I can explain the refund request process.",
    })
    result = comparison.parse_model_output(raw, REPO_ROOT)
    assert result["json_valid"] is True
    assert result["schema_compliant"] is True
    assert result["intent"] == "get_refund"
    assert result["category"] == "REFUND"
    assert result["needs_human"] is True


def test_invalid_json_is_handled_without_exception():
    result = comparison.parse_model_output("not valid json", REPO_ROOT)
    assert result["json_valid"] is False
    assert result["schema_compliant"] is False
    assert result["parsed_output"] is None
    assert result["intent"] is None


def test_benchmark_promotion_and_deployment_artifacts_load():
    evidence = comparison.load_project_evidence(REPO_ROOT)
    assert evidence["benchmark"]["evaluated_rows"] == 300
    assert evidence["benchmark"]["metrics"]["intent_accuracy"]["base"] == 28.0
    assert evidence["benchmark"]["metrics"]["intent_accuracy"]["lora"] == 94.0
    assert evidence["promotion"]["candidate"] == "candidate_01"
    assert evidence["promotion"]["decision"] == "PROMOTE"
    assert "structured classification" in evidence["constraints"]["approved_scope"]
    assert "enterprise factual authority" in evidence["constraints"]["not_approved_as"]


def test_curated_examples_are_new_and_not_frozen_dataset_rows():
    curated = [item["message"] for item in comparison.CURATED_EXAMPLES]
    assert len(curated) == 8
    assert len(set(curated)) == len(curated)
    frozen_instructions = set()
    for name in ("train", "validation", "dev", "locked_test"):
        with (REPO_ROOT / f"data/processed/{name}.jsonl").open(encoding="utf-8") as handle:
            frozen_instructions.update(json.loads(line)["instruction"] for line in handle if line.strip())
    assert not (set(curated) & frozen_instructions)


def test_benchmark_table_contains_all_required_metrics():
    rows = comparison.benchmark_rows(comparison.load_project_evidence(REPO_ROOT)["benchmark"])
    assert [row["Metric"] for row in rows] == [
        "Intent Accuracy", "Category Accuracy", "JSON Valid", "Schema Compliance",
        "Escalation Accuracy", "Escalation F1",
    ]


def test_importing_streamlit_app_does_not_run_inference_or_training(monkeypatch):
    called = []
    monkeypatch.setattr(comparison, "run_inference", lambda *args, **kwargs: called.append(True))
    app_path = REPO_ROOT / "demo/app.py"
    spec = importlib.util.spec_from_file_location("stage9_demo_import_check", app_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert called == []
    source = app_path.read_text(encoding="utf-8")
    assert "src.training" not in source
    assert "locked_test.jsonl" not in source


def test_streamlit_app_starts_without_inference_and_shows_required_evidence():
    app = AppTest.from_file(str(REPO_ROOT / "demo/app.py")).run(timeout=15)
    assert not app.exception
    assert not app.error
    assert [item.value for item in app.title] == ["客服分類 LoRA 微調效果比較"]
    assert [item.label for item in app.button] == ["▶ 比較 Base 與 LoRA"]
    assert [item.label for item in app.selectbox] == ["範例問題"]
    assert [item.label for item in app.text_area] == ["客服訊息"]
    markdown = "\n".join(item.value for item in app.markdown)
    assert "28% → 94%" in markdown
    assert "36.7% → 99.3%" in markdown
    assert "79% → 98.7%" in markdown
    assert "✅ Candidate 01 已通過 Promotion Gate" in markdown
    assert [item.label for item in app.expander] == [
        "▶ 查看 Locked Test 完整指標結果",
        "▶ 查看模型限制與部署邊界",
    ]
    assert all(not item.proto.expanded for item in app.expander)


def test_curated_mode_shows_expected_and_free_text_has_no_match_claim():
    app = AppTest.from_file(str(REPO_ROOT / "demo/app.py")).run(timeout=15)
    app.selectbox[0].select("Request a refund").run(timeout=15)
    curated_markdown = "\n".join(item.value for item in app.markdown)
    assert "預期結果（Expected）" in curated_markdown
    assert "get_refund" in curated_markdown
    assert "REFUND" in curated_markdown
    assert "True" in curated_markdown

    app.selectbox[0].select("自訂問題").run(timeout=15)
    app.text_area[0].set_value("A new free-form support message.").run(timeout=15)
    free_text_markdown = "\n".join(item.value for item in app.markdown)
    assert "預期結果（Expected）" not in free_text_markdown
    assert "✅ 符合" not in free_text_markdown
    assert "❌ 不符合" not in free_text_markdown


def test_curated_comparison_click_renders_cards_statuses_and_responses(monkeypatch):
    monkeypatch.setattr(comparison, "verify_frozen_integrity", lambda *args, **kwargs: {"status": "PASS"})
    monkeypatch.setattr(
        comparison,
        "load_model_bundle",
        lambda root, adapter_path=None: {"role": "base" if adapter_path is None else "lora"},
    )

    def fake_inference(bundle, message, root):
        is_lora = bundle["role"] == "lora"
        return {
            "intent": "get_refund" if is_lora else "refund",
            "category": "REFUND",
            "needs_human": is_lora,
            "response": "This English response remains unchanged.",
            "json_valid": True,
            "schema_compliant": is_lora,
            "raw_output": '{"response":"This English response remains unchanged."}',
            "generation_truncated": False,
            "latency_ms": 1200.0 if is_lora else 800.0,
        }

    monkeypatch.setattr(comparison, "run_inference", fake_inference)
    app = AppTest.from_file(str(REPO_ROOT / "demo/app.py")).run(timeout=15)
    app.selectbox[0].select("Request a refund").run(timeout=15)
    app.button[0].click().run(timeout=15)
    assert not app.exception
    assert not app.error
    markdown = "\n".join(item.value for item in app.markdown)
    assert "Base Model" in markdown
    assert "LoRA Candidate 01" in markdown
    assert "✅ 符合" in markdown
    assert "❌ 不符合" in markdown
    assert "This English response remains unchanged." in markdown
    assert [item.label for item in app.expander].count("▶ 查看完整回覆與 JSON") == 2


def test_frozen_sources_and_promotion_gate_remain_exact():
    expected = {
        "prompts/base_system_prompt.txt": comparison.PROMPT_SHA256,
        "src/evaluation/base_baseline.py": "184ca998f1a29dcf99cc4bc48788d09ad0c177314a16ac7eb4f21c0caf64fb52",
        "src/evaluation/contracts.py": "e2f8bb620a3b7d44f98c5ca0a96e985d98b944fe7d68b0a265cd26aa425a31a3",
        "src/evaluation/development_evaluation.py": "2cb2f7f37f4b5b03837eb8b2cec17355a3df77cdc1df07939f990bfb38ba9a37",
        "configs/output_schema.json": "6a3d0900b3485e5a24205ea5f7ae42360d598a6c7a7fc6d97cde2d8fde88daa2",
        "configs/intent_taxonomy.json": "8e99fdfcdd90a2bcc2dd733503e936d5f0785ef4548468fa5923b4d965e3422f",
        "configs/category_taxonomy.json": "694f2c4a56fe662d795a1315781ed7c86f68114012ea0012ead43cefc4a5ba79",
        "configs/escalation_policy.json": "c07898c29254bc584c944007bc2fd2785c9db1e70fedda0aeb7c0ec7c2ef0f2d",
        "configs/promotion_gate.json": "8e756705625c7bc61cb136d0672b785a76d21b8443f10c9f1903c87c3d2af377",
    }
    assert all(comparison.sha256_file(REPO_ROOT / path) == digest for path, digest in expected.items())
