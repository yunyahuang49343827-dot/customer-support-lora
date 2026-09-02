"""Export one-time frozen inference snapshots for the static Vercel portfolio demo."""

from __future__ import annotations

import gc
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from src.demo.comparison import (
    ADAPTER_PATH,
    ADAPTER_SHA256,
    BASE_MODEL,
    BASE_REVISION,
    CANDIDATE,
    CURATED_EXAMPLES,
    MAX_GENERATED_TOKENS,
    PROMPT_PATH,
    PROMPT_SHA256,
    SEED,
    TEMPERATURE,
    load_model_bundle,
    repo_root,
    run_inference,
    sha256_file,
    verify_frozen_integrity,
)


OUTPUT_DIR = "web/data"
C7_COMPARISON_PATH = "artifacts/stage7/base_vs_lora_locked_comparison.json"
C8_PROMOTION_PATH = "artifacts/stage8/promotion_decision.json"
C8_CONSTRAINTS_PATH = "artifacts/stage8/deployment_constraints.json"
EXPECTED_IMMUTABLE_HASHES = {
    C7_COMPARISON_PATH: "094d36866bdd8a58f6ea823e81e3bb385c68c3c9f10b76ee1b47923682c8d174",
    C8_PROMOTION_PATH: "a603883c5cbecc7f2dd3fa605211af9f742a44e62ec0888dc5c28e90fd4d39c2",
    C8_CONSTRAINTS_PATH: "ecd5b98b48c795eff6204d98242e61401de77d2aed4b56c33cc8ee5dd17d4e5a",
    "demo/app.py": "8251f68bdbad51301a04599537eb9e73c191585c3bbf80f3c181956458a5f197",
    "src/demo/comparison.py": "c64758b0313a5bec47e3f191a51bb9c0a269855331496ee3e15666759ecf9a70",
    "reports/stage9_demo.md": "0036f0e3fc537769ada9cf110bacc84d8034b82751d033544f62947dfe91baa1",
}
EXAMPLE_METADATA = {
    "Request a refund": ("refund-request", "退款申請"),
    "Track a refund": ("refund-tracking", "退款進度查詢"),
    "Payment issue": ("payment-issue", "付款問題"),
    "Cancel an order": ("cancel-order", "取消訂單"),
    "Contact a human agent": ("human-agent", "聯絡真人客服"),
    "Check an invoice": ("check-invoice", "查看發票"),
    "Create an account": ("create-account", "建立帳戶"),
    "Change shipping address": ("shipping-address", "變更配送地址"),
}
SNAPSHOT_FIELDS = (
    "intent", "category", "needs_human", "json_valid", "schema_compliant",
    "response", "raw_output", "generation_truncated", "latency_ms",
)


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def current_immutable_hashes(root: Path) -> Dict[str, str]:
    return {path: sha256_file(root / path) for path in EXPECTED_IMMUTABLE_HASHES}


def verify_export_integrity(root: Path) -> Dict[str, Any]:
    frozen = verify_frozen_integrity(root, check_model_availability=True)
    current = current_immutable_hashes(root)
    mismatches = {
        path: {"expected": expected, "actual": current[path]}
        for path, expected in EXPECTED_IMMUTABLE_HASHES.items()
        if current[path] != expected
    }
    return {
        "status": "PASS" if frozen["status"] == "PASS" and not mismatches else "FAIL",
        "frozen_integrity": frozen,
        "immutable_hashes": current,
        "hash_mismatches": mismatches,
    }


def curated_cases() -> Sequence[Dict[str, Any]]:
    if len(CURATED_EXAMPLES) != 8:
        raise ValueError(f"Expected exactly 8 Stage C9 curated examples; found {len(CURATED_EXAMPLES)}")
    cases = []
    for example in CURATED_EXAMPLES:
        if example["label"] not in EXAMPLE_METADATA:
            raise ValueError(f"Missing V1 metadata for curated example: {example['label']}")
        case_id, label_zh = EXAMPLE_METADATA[example["label"]]
        cases.append({
            "id": case_id,
            "label_zh": label_zh,
            "message": example["message"],
            "expected": {
                "intent": example["expected_intent"],
                "category": example["expected_category"],
                "needs_human": example["expected_needs_human"],
            },
        })
    return cases


def inference_snapshot(bundle: Mapping[str, Any], message: str, root: Path) -> Dict[str, Any]:
    try:
        result = run_inference(bundle, message, root)
        snapshot = {field: result.get(field) for field in SNAPSHOT_FIELDS}
        snapshot["generation_error"] = None
        return snapshot
    except Exception as error:
        return {
            "intent": None,
            "category": None,
            "needs_human": None,
            "json_valid": False,
            "schema_compliant": False,
            "response": "",
            "raw_output": "",
            "generation_truncated": False,
            "latency_ms": None,
            "generation_error": f"{type(error).__name__}: {error}",
        }


def run_all(bundle: Mapping[str, Any], cases: Sequence[Mapping[str, Any]], root: Path, role: str) -> Dict[str, Dict[str, Any]]:
    results = {}
    for index, case in enumerate(cases, 1):
        results[case["id"]] = inference_snapshot(bundle, case["message"], root)
        print(f"Stage V1 {role} curated inference: {index}/{len(cases)}", flush=True)
    return results


def build_benchmark_snapshot(source: Mapping[str, Any]) -> Dict[str, Any]:
    metric_names = (
        "intent_accuracy", "category_accuracy", "json_valid_rate", "schema_compliance",
        "escalation_accuracy", "escalation_f1",
    )
    return {
        "source": C7_COMPARISON_PATH,
        "evaluated_rows": source["evaluated_rows"],
        "metrics": {name: source["metrics"][name] for name in metric_names},
        "locked_test_rerun_performed": False,
    }


def build_project_status(promotion: Mapping[str, Any], constraints: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "candidate": promotion["candidate"],
        "decision": promotion["decision"],
        "approved_scope": constraints["approved_scope"],
        "known_limitations": promotion["known_limitations"],
        "deployment_constraints": {
            "not_approved_as": constraints["not_approved_as"],
            "requires_external_grounding_for": constraints["requires_external_grounding_for"],
            "requires_backend_tools_for": constraints["requires_backend_tools_for"],
            "human_review_recommended_for": constraints["human_review_recommended_for"],
        },
        "unrestricted_production_approval": promotion["unrestricted_production_approval"],
        "sources": [C8_PROMOTION_PATH, C8_CONSTRAINTS_PATH],
    }


def run(root: Optional[Path] = None) -> Dict[str, Any]:
    root = root or repo_root()
    output_dir = root / OUTPUT_DIR
    output_paths = [
        output_dir / "demo_cases.json",
        output_dir / "benchmark.json",
        output_dir / "project_status.json",
        output_dir / "v1_export_manifest.json",
    ]
    if any(path.exists() for path in output_paths):
        raise RuntimeError("Refusing to rerun or overwrite an existing Stage V1 export")

    preflight = verify_export_integrity(root)
    if preflight["status"] != "PASS":
        raise RuntimeError("Stage V1 aborted: frozen or Stage C9 integrity check failed")
    cases = curated_cases()

    base_bundle = load_model_bundle(root, adapter_path=None)
    base_results = run_all(base_bundle, cases, root, "base")
    del base_bundle
    gc.collect()
    try:
        import mlx.core as mx
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
    except ImportError:
        pass

    lora_bundle = load_model_bundle(root, adapter_path=ADAPTER_PATH)
    lora_results = run_all(lora_bundle, cases, root, "lora")
    del lora_bundle
    gc.collect()

    exported = []
    for case in cases:
        exported.append({
            **case,
            "base": base_results[case["id"]],
            "lora": lora_results[case["id"]],
            "provenance": {
                "source": "frozen_local_inference",
                "base_model": BASE_MODEL,
                "base_revision": BASE_REVISION,
                "candidate": CANDIDATE,
                "adapter_sha256": ADAPTER_SHA256,
                "prompt_sha256": PROMPT_SHA256,
                "temperature": TEMPERATURE,
                "seed": SEED,
                "max_generated_tokens": MAX_GENERATED_TOKENS,
                "response_modified": False,
            },
        })

    postflight = verify_export_integrity(root)
    if postflight["status"] != "PASS" or postflight["immutable_hashes"] != preflight["immutable_hashes"]:
        raise RuntimeError("Stage V1 aborted: immutable artifacts changed during export")
    benchmark = build_benchmark_snapshot(read_json(root / C7_COMPARISON_PATH))
    project_status = build_project_status(
        read_json(root / C8_PROMOTION_PATH), read_json(root / C8_CONSTRAINTS_PATH)
    )
    write_json(output_dir / "demo_cases.json", exported)
    write_json(output_dir / "benchmark.json", benchmark)
    write_json(output_dir / "project_status.json", project_status)
    manifest = {
        "stage": "V1",
        "status": "COMPLETE",
        "export_timestamp": datetime.now(timezone.utc).isoformat(),
        "curated_examples": len(exported),
        "base_successful_inferences": sum(row["base"]["generation_error"] is None for row in exported),
        "lora_successful_inferences": sum(row["lora"]["generation_error"] is None for row in exported),
        "frozen_hashes_before": preflight["immutable_hashes"],
        "frozen_hashes_after": postflight["immutable_hashes"],
        "adapter_hash": ADAPTER_SHA256,
        "prompt_hash": PROMPT_SHA256,
        "training_performed": False,
        "locked_test_rerun_performed": False,
        "stage_c9_modified": False,
        "nextjs_started": False,
    }
    write_json(output_dir / "v1_export_manifest.json", manifest)
    return {"manifest": manifest, "demo_cases": exported, "benchmark": benchmark, "project_status": project_status}


def main() -> None:
    result = run()
    print(json.dumps(result["manifest"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
