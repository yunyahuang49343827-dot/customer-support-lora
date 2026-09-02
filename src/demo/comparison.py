"""Frozen Base-versus-QLoRA inference helpers for the Stage C9 demo."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from src.evaluation.base_baseline import _strict_json_object
from src.evaluation.contracts import load_vocabulary, validate_output


BASE_MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
BASE_REVISION = "8b403126fc14f14cfc99bb4cfa72ecbc129ea677"
CANDIDATE = "candidate_01"
ADAPTER_PATH = "artifacts/stage5/candidate_01/adapter"
ADAPTER_SHA256 = "da763e47f3c6051defb605345e9aaccd989a8768b804c802606a7f8317fc2c16"
PROMPT_PATH = "prompts/base_system_prompt.txt"
PROMPT_SHA256 = "6b84135769b7348758e8cc21a3cb168465e00de5efaf59ff8a8459087db3dc3b"
FROZEN_INFERENCE_PATH = "artifacts/stage6_5/frozen_inference_contract.json"
BENCHMARK_PATH = "artifacts/stage7/base_vs_lora_locked_comparison.json"
PROMOTION_PATH = "artifacts/stage8/promotion_decision.json"
CONSTRAINTS_PATH = "artifacts/stage8/deployment_constraints.json"
TEMPERATURE = 0.0
SEED = 42
MAX_GENERATED_TOKENS = 512

CURATED_EXAMPLES = (
    {
        "label": "Request a refund",
        "message": "Please help me request a refund for my recent purchase.",
        "expected_intent": "get_refund",
        "expected_category": "REFUND",
        "expected_needs_human": True,
    },
    {
        "label": "Track a refund",
        "message": "Can you tell me where my pending refund is right now?",
        "expected_intent": "track_refund",
        "expected_category": "REFUND",
        "expected_needs_human": False,
    },
    {
        "label": "Payment issue",
        "message": "My payment keeps failing and I need assistance resolving it.",
        "expected_intent": "payment_issue",
        "expected_category": "PAYMENT",
        "expected_needs_human": True,
    },
    {
        "label": "Cancel an order",
        "message": "I would like to cancel the order I submitted this morning.",
        "expected_intent": "cancel_order",
        "expected_category": "ORDER",
        "expected_needs_human": False,
    },
    {
        "label": "Contact a human agent",
        "message": "Could you connect me with a human support agent?",
        "expected_intent": "contact_human_agent",
        "expected_category": "CONTACT",
        "expected_needs_human": True,
    },
    {
        "label": "Check an invoice",
        "message": "I want to review the invoice details for my latest order.",
        "expected_intent": "check_invoice",
        "expected_category": "INVOICE",
        "expected_needs_human": False,
    },
    {
        "label": "Create an account",
        "message": "What do I need to do to open a new customer account?",
        "expected_intent": "create_account",
        "expected_category": "ACCOUNT",
        "expected_needs_human": False,
    },
    {
        "label": "Change shipping address",
        "message": "Please explain how to change the delivery address on my order.",
        "expected_intent": "change_shipping_address",
        "expected_category": "SHIPPING",
        "expected_needs_human": False,
    },
)

KNOWN_LIMITATIONS = (
    "Generated responses may still contain unsupported policy or capability claims.",
    "External grounding is required for company facts and policies.",
    "Backend actions require real tools or APIs.",
    "One isolated generation degeneration or truncation was observed.",
    "Candidate 01 has higher inference latency on the tested environment.",
    "Semantically similar intents can still be confused.",
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_model_snapshot(check_availability: bool = True) -> Optional[Path]:
    if not check_availability:
        return None
    from huggingface_hub import snapshot_download

    resolved = Path(snapshot_download(BASE_MODEL, revision=BASE_REVISION, local_files_only=True)).resolve()
    if resolved.name != BASE_REVISION:
        raise RuntimeError("The locally resolved Base model revision does not match the frozen revision.")
    return resolved


def verify_frozen_integrity(root: Optional[Path] = None, check_model_availability: bool = True) -> Dict[str, Any]:
    root = root or repo_root()
    checks = []

    def add(name: str, expected: Any, actual: Any) -> None:
        checks.append({"name": name, "expected": expected, "actual": actual, "status": "PASS" if expected == actual else "FAIL"})

    try:
        freeze = read_json(root / "artifacts/stage6_5/freeze_manifest.json")
        contract = read_json(root / FROZEN_INFERENCE_PATH)
        add("freeze_status", "PASS", freeze["freeze_status"])
        add("base_model", BASE_MODEL, contract["base_model"])
        add("base_revision", BASE_REVISION, contract["base_revision"])
        add("candidate", CANDIDATE, freeze["candidate"])
        add("adapter_path", ADAPTER_PATH, contract["candidate_adapter"])
        add("adapter_sha256", ADAPTER_SHA256, sha256_file(root / ADAPTER_PATH / "adapters.safetensors"))
        add("prompt_sha256", PROMPT_SHA256, sha256_file(root / PROMPT_PATH))
        add("temperature", TEMPERATURE, contract["decoding"]["temperature"])
        add("seed", SEED, contract["decoding"]["seed"])
        add("max_generated_tokens", MAX_GENERATED_TOKENS, contract["decoding"]["max_generated_tokens"])
        add("parser", "src.evaluation.base_baseline._strict_json_object", contract["parser"])
        if check_model_availability:
            snapshot = resolve_model_snapshot(True)
            add("local_model_revision", BASE_REVISION, snapshot.name if snapshot else None)
    except Exception as error:
        checks.append({"name": "integrity_check_execution", "expected": "success", "actual": type(error).__name__, "status": "FAIL"})
    failures = [item for item in checks if item["status"] == "FAIL"]
    return {
        "status": "PASS" if not failures else "FAIL",
        "checks": checks,
        "fail_count": len(failures),
        "message": "Frozen artifact integrity check failed." if failures else "Frozen artifact integrity verified.",
    }


def load_project_evidence(root: Optional[Path] = None) -> Dict[str, Any]:
    root = root or repo_root()
    benchmark = read_json(root / BENCHMARK_PATH)
    promotion = read_json(root / PROMOTION_PATH)
    constraints = read_json(root / CONSTRAINTS_PATH)
    if benchmark.get("evaluated_rows") != 300:
        raise ValueError("Locked benchmark artifact is incomplete.")
    if promotion.get("candidate") != CANDIDATE or promotion.get("decision") != "PROMOTE":
        raise ValueError("Frozen promotion decision is unavailable or inconsistent.")
    if constraints.get("candidate") != CANDIDATE:
        raise ValueError("Deployment constraints do not match Candidate 01.")
    return {"benchmark": benchmark, "promotion": promotion, "constraints": constraints}


def load_model_bundle(root: Optional[Path] = None, adapter_path: Optional[str] = None) -> Dict[str, Any]:
    root = root or repo_root()
    integrity = verify_frozen_integrity(root, check_model_availability=True)
    if integrity["status"] != "PASS":
        raise RuntimeError("Frozen artifact integrity check failed.")
    if adapter_path not in (None, ADAPTER_PATH):
        raise ValueError("Only the frozen Candidate 01 adapter is permitted.")
    from mlx_lm import load

    snapshot = resolve_model_snapshot(True)
    resolved_adapter = None if adapter_path is None else str(root / ADAPTER_PATH)
    model, tokenizer = load(str(snapshot), adapter_path=resolved_adapter)
    return {
        "model": model,
        "tokenizer": tokenizer,
        "adapter_path": adapter_path,
        "role": "base" if adapter_path is None else "lora",
    }


def parse_model_output(raw_output: Any, root: Optional[Path] = None, generation_truncated: bool = False) -> Dict[str, Any]:
    root = root or repo_root()
    json_valid, parsed = _strict_json_object(raw_output)
    validation = validate_output(raw_output, load_vocabulary(root / "configs"))
    fields = parsed if parsed is not None else {}
    return {
        "raw_output": raw_output if isinstance(raw_output, str) else "",
        "parsed_output": parsed,
        "intent": fields.get("intent"),
        "category": fields.get("category"),
        "needs_human": fields.get("needs_human"),
        "response": fields.get("response"),
        "json_valid": json_valid,
        "schema_compliant": validation.valid,
        "validation_errors": list(validation.errors),
        "generation_truncated": generation_truncated,
    }


def run_inference(bundle: Mapping[str, Any], customer_message: str, root: Optional[Path] = None) -> Dict[str, Any]:
    root = root or repo_root()
    message = customer_message.strip()
    if not message:
        raise ValueError("Please enter a customer message.")
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler
    import mlx.core as mx

    prompt_text = (root / PROMPT_PATH).read_text(encoding="utf-8").strip()
    messages = [{"role": "system", "content": prompt_text}, {"role": "user", "content": message}]
    model_prompt = bundle["tokenizer"].apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    mx.random.seed(SEED)
    sampler = make_sampler(temp=TEMPERATURE)
    raw_output = ""
    finish_reason = None
    started = time.perf_counter()
    for chunk in stream_generate(
        bundle["model"], bundle["tokenizer"], model_prompt,
        max_tokens=MAX_GENERATED_TOKENS, sampler=sampler,
    ):
        raw_output += chunk.text
        finish_reason = chunk.finish_reason or finish_reason
    latency_ms = (time.perf_counter() - started) * 1000.0
    result = parse_model_output(raw_output, root, generation_truncated=finish_reason == "length")
    result.update({"latency_ms": round(latency_ms, 3), "model_role": bundle["role"]})
    return result


def expected_for_message(message: str) -> Optional[Mapping[str, Any]]:
    return next((item for item in CURATED_EXAMPLES if item["message"] == message), None)


def comparison_markers(base: Mapping[str, Any], lora: Mapping[str, Any]) -> Dict[str, str]:
    markers = {}
    for field in ("intent", "category", "needs_human"):
        if base.get(field) is None or lora.get(field) is None:
            markers[field] = "❌ unavailable"
        elif base.get(field) == lora.get(field):
            markers[field] = "✅ same"
        else:
            markers[field] = "⚠️ different"
    return markers


def benchmark_rows(benchmark: Mapping[str, Any]) -> Tuple[Dict[str, Any], ...]:
    labels = (
        ("Intent Accuracy", "intent_accuracy"),
        ("Category Accuracy", "category_accuracy"),
        ("JSON Valid", "json_valid_rate"),
        ("Schema Compliance", "schema_compliance"),
        ("Escalation Accuracy", "escalation_accuracy"),
        ("Escalation F1", "escalation_f1"),
    )
    metrics = benchmark["metrics"]
    return tuple({"Metric": label, "Base": row["base"], "QLoRA": row["lora"], "Delta": row["absolute_delta"]} for label, key in labels for row in (metrics[key],))
