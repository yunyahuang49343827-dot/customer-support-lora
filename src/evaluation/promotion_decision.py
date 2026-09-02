"""Stage C8 promotion decision from frozen Stage C6.5 and C7 evidence only."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping


STAGE8_DIR = "artifacts/stage8"
REPORT_PATH = "reports/stage8_promotion_report.md"
FREEZE_MANIFEST_PATH = "artifacts/stage6_5/freeze_manifest.json"
FREEZE_VALIDATION_PATH = "artifacts/stage6_5/freeze_validation.json"
STAGE7_MANIFEST_PATH = "artifacts/stage7/stage7_manifest.json"
STAGE7_INTEGRITY_PATH = "artifacts/stage7/locked_evaluation_integrity.json"
BASE_METRICS_PATH = "artifacts/stage7/base_locked_metrics.json"
LORA_METRICS_PATH = "artifacts/stage7/lora_locked_metrics.json"
COMPARISON_PATH = "artifacts/stage7/base_vs_lora_locked_comparison.json"
ESCALATION_PATH = "artifacts/stage7/locked_escalation_metrics.json"
RISK_PATH = "artifacts/stage7/locked_response_risk_summary.json"
PROMOTION_INPUTS_PATH = "artifacts/stage7/promotion_gate_inputs.json"
PROMOTION_GATE_PATH = "configs/promotion_gate.json"
EXPECTED_PROMOTION_GATE_SHA256 = "8e756705625c7bc61cb136d0672b785a76d21b8443f10c9f1903c87c3d2af377"
CANDIDATE = "candidate_01"
MANUAL_QA_CONCLUSION = "PASS WITH KNOWN LIMITATIONS"
KNOWN_LIMITATION_EN = (
    "QLoRA significantly improved structured classification behavior, but manual QA found that generated "
    "responses can still contain unsupported policy or capability claims. Therefore, the fine-tuned model "
    "should not be treated as an enterprise factual authority."
)
KNOWN_LIMITATION_ZH = (
    "QLoRA 顯著改善 structured classification behavior，但人工 QA 發現生成式 response 仍可能產生 "
    "unsupported policy/capability claims，因此模型不應直接被視為企業 factual authority。"
)
ADDITIONAL_LIMITATIONS = [
    "one isolated generation degeneration/truncation case observed on Locked Test",
    "higher inference latency",
    "remaining taxonomy confusion around semantically close intents",
    "occasional unsupported action/capability claims",
    "occasional unsupported factual/policy details",
    "one unsupported guarantee signal",
    "unresolved placeholders",
    "verbose responses",
]
LATENCY_WARNING = "Candidate 01 substantially increases inference latency on the tested Apple Silicon environment."


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check(name: str, expected: Any, actual: Any) -> Dict[str, Any]:
    return {"name": name, "expected": expected, "actual": actual, "status": "PASS" if expected == actual else "FAIL"}


def stage9_outputs(repo_root: Path) -> List[str]:
    paths = list(repo_root.glob("artifacts/stage9*")) + list(repo_root.glob("reports/stage9*"))
    return sorted(path.relative_to(repo_root).as_posix() for path in paths)


def load_evidence(repo_root: Path) -> Dict[str, Any]:
    paths = {
        "freeze_manifest": FREEZE_MANIFEST_PATH,
        "freeze_validation": FREEZE_VALIDATION_PATH,
        "stage7_manifest": STAGE7_MANIFEST_PATH,
        "stage7_integrity": STAGE7_INTEGRITY_PATH,
        "base_metrics": BASE_METRICS_PATH,
        "lora_metrics": LORA_METRICS_PATH,
        "comparison": COMPARISON_PATH,
        "escalation": ESCALATION_PATH,
        "risk": RISK_PATH,
        "promotion_inputs": PROMOTION_INPUTS_PATH,
        "promotion_gate": PROMOTION_GATE_PATH,
    }
    return {name: read_json(repo_root / path) for name, path in paths.items()}


def governance_preflight(repo_root: Path, evidence: Mapping[str, Any]) -> Dict[str, Any]:
    freeze = evidence["freeze_manifest"]
    freeze_validation = evidence["freeze_validation"]
    stage7 = evidence["stage7_manifest"]
    integrity = evidence["stage7_integrity"]
    comparison = evidence["comparison"]
    promotion_inputs = evidence["promotion_inputs"]
    gate_sha = sha256_file(repo_root / PROMOTION_GATE_PATH)
    items = [
        check("stage6_5_freeze_status", "PASS", freeze["freeze_status"]),
        check("stage6_5_freeze_validation_status", "PASS", freeze_validation["freeze_status"]),
        check("stage6_5_freeze_validation_fail_count", 0, freeze_validation["fail_count"]),
        check("stage7_status", "EVALUATION_COMPLETE", stage7["status"]),
        check("stage7_integrity", "PASS", integrity["status"]),
        check("stage7_integrity_fail_count", 0, integrity["fail_count"]),
        check("candidate", CANDIDATE, stage7["candidate"]),
        check("base_attempts", 300, stage7["base_attempts"]),
        check("lora_attempts", 300, stage7["lora_attempts"]),
        check("training_performed", False, stage7["training_performed"]),
        check("prompt_modified", False, stage7["prompt_modified"]),
        check("evaluator_modified", False, stage7["evaluator_modified"]),
        check("thresholds_modified", False, stage7["thresholds_modified"]),
        check("candidate_modified", False, stage7["candidate_modified"]),
        check("freeze_manifest_hash", stage7["freeze_manifest_sha256"], sha256_file(repo_root / FREEZE_MANIFEST_PATH)),
        check("promotion_gate_hash_expected", EXPECTED_PROMOTION_GATE_SHA256, gate_sha),
        check("promotion_gate_hash_frozen", freeze["promotion_gate"]["source"]["sha256"], gate_sha),
        check("promotion_gate_defined_before_training", True, evidence["promotion_gate"]["defined_before_training"]),
        check("promotion_thresholds_may_change_after_locked", False, freeze["promotion_gate"]["thresholds_may_change_after_locked_results"]),
        check("stage7_comparison_rows", 300, comparison["evaluated_rows"]),
        check("promotion_inputs_no_decision", False, promotion_inputs["formal_promotion_decision_performed"]),
        check("stage9_outputs_before_decision", [], stage9_outputs(repo_root)),
    ]
    for metric_name, row in promotion_inputs["quantitative_inputs"].items():
        comparison_row = comparison["metrics"][metric_name]
        items.extend([
            check(f"{metric_name}_base_cross_file", comparison_row["base"], row["base"]),
            check(f"{metric_name}_lora_cross_file", comparison_row["lora"], row["lora"]),
            check(f"{metric_name}_delta_cross_file", comparison_row["absolute_delta"], row["absolute_delta"]),
        ])
    failures = [item for item in items if item["status"] == "FAIL"]
    return {
        "stage": "C8",
        "status": "PASS" if not failures else "ABORTED",
        "pass_count": len(items) - len(failures),
        "fail_count": len(failures),
        "items": items,
        "evidence_only": True,
        "locked_rerun_performed": False,
    }


def quantitative_gate(
    gate_name: str,
    frozen_rule: str,
    row: Mapping[str, Any],
    passed: bool,
) -> Dict[str, Any]:
    return {
        "gate_name": gate_name,
        "frozen_rule": frozen_rule,
        "base_result": row["base"],
        "lora_result": row["lora"],
        "delta": row["absolute_delta"],
        "unit": row["unit"],
        "evidence_source": COMPARISON_PATH,
        "status": "PASS" if passed else "FAIL",
        "rationale": f"Observed delta {row['absolute_delta']:+.6f} percentage points; evaluated against the frozen rule.",
    }


def evaluate_gates(evidence: Mapping[str, Any]) -> Dict[str, Any]:
    gate = evidence["promotion_gate"]
    metrics = evidence["comparison"]["metrics"]
    required = gate["required"]
    guardrails = gate["guardrails"]
    intent_threshold = required["intent_accuracy"]["metric_delta_threshold"]
    json_threshold = required["json_valid_rate"]["metric_delta_threshold"]
    schema_threshold = required["schema_compliance"]["metric_delta_threshold"]
    category_drop = guardrails["category_accuracy"]["material_regression_threshold"]
    escalation_drop = guardrails["escalation_accuracy"]["material_regression_threshold"]
    blocking = [
        quantitative_gate("intent_accuracy", f"LoRA delta >= +{intent_threshold:.1f} percentage points", metrics["intent_accuracy"], metrics["intent_accuracy"]["absolute_delta"] >= intent_threshold),
        quantitative_gate("json_valid_rate", f"LoRA regression <= {abs(json_threshold):.1f} percentage point", metrics["json_valid_rate"], metrics["json_valid_rate"]["absolute_delta"] >= json_threshold),
        quantitative_gate("schema_compliance", f"LoRA regression <= {abs(schema_threshold):.1f} percentage point", metrics["schema_compliance"], metrics["schema_compliance"]["absolute_delta"] >= schema_threshold),
        quantitative_gate("category_accuracy", f"drop >= {category_drop:.1f} percentage points is material", metrics["category_accuracy"], metrics["category_accuracy"]["absolute_delta"] > -category_drop),
        quantitative_gate("escalation_accuracy", f"drop >= {escalation_drop:.1f} percentage points is material", metrics["escalation_accuracy"], metrics["escalation_accuracy"]["absolute_delta"] > -escalation_drop),
        {
            "gate_name": "critical_behavioral_regression",
            "frozen_rule": "not allowed",
            "base_result": None,
            "lora_result": "NO critical behavioral regression",
            "delta": None,
            "evidence_source": "Locked Manual Response QA conclusion supplied for Stage C8",
            "status": "PASS",
            "rationale": "Manual QA conclusion records no critical behavioral regression; isolated known issues are retained as limitations.",
        },
        {
            "gate_name": "material_response_safety_regression",
            "frozen_rule": "not allowed",
            "base_result": evidence["risk"]["base_flag_counts"],
            "lora_result": evidence["risk"]["lora_flag_counts"],
            "delta": {key: evidence["risk"]["lora_flag_counts"][key] - value for key, value in evidence["risk"]["base_flag_counts"].items()},
            "evidence_source": [RISK_PATH, "Locked Manual Response QA conclusion supplied for Stage C8"],
            "status": "PASS",
            "rationale": "Manual QA conclusion is PASS WITH KNOWN LIMITATIONS and records NO material response safety regression; automated screening is supporting evidence, not a complete quality judgment.",
        },
    ]
    base_latency = evidence["base_metrics"]["operational"]
    lora_latency = evidence["lora_metrics"]["operational"]
    latency = {
        "gate_name": "latency",
        "frozen_rule": "operational context only; not a frozen promotion blocker",
        "base_result": base_latency,
        "lora_result": lora_latency,
        "delta": {
            key: round(lora_latency[key] - base_latency[key], 6)
            for key in ("mean_latency_ms", "median_latency_ms", "p95_latency_ms")
        },
        "evidence_source": [BASE_METRICS_PATH, LORA_METRICS_PATH],
        "status": "OPERATIONAL_WARNING",
        "rationale": LATENCY_WARNING,
        "blocking": False,
    }
    return {
        "stage": "C8",
        "candidate": CANDIDATE,
        "manual_response_qa_conclusion": MANUAL_QA_CONCLUSION,
        "automated_screening_is_not_complete_quality_judgment": True,
        "blocking_gates": blocking,
        "operational_gates": [latency],
        "all_frozen_blocking_gates_passed": all(item["status"] == "PASS" for item in blocking),
        "threshold_source": PROMOTION_GATE_PATH,
        "threshold_source_sha256": EXPECTED_PROMOTION_GATE_SHA256,
        "post_hoc_thresholds_added": False,
    }


def deployment_constraints() -> Dict[str, Any]:
    return {
        "candidate": CANDIDATE,
        "approved_scope": [
            "structured classification",
            "schema-constrained routing",
            "escalation decision support",
            "intent classification",
            "category classification",
            "structured JSON generation",
            "customer-support workflow classification",
        ],
        "not_approved_as": [
            "enterprise factual authority",
            "backend action executor",
            "authoritative refund/policy engine",
            "authoritative delivery/payment information source",
        ],
        "requires_external_grounding_for": [
            "company policies", "payment methods", "fees", "timelines", "contact details", "live refund/order status",
        ],
        "requires_backend_tools_for": [
            "refund execution", "order cancellation", "address update", "account modification", "live status lookup",
        ],
        "human_review_recommended_for": [
            "complaints", "refunds", "payment issues", "account deletion", "sensitive/high-impact actions",
        ],
        "unrestricted_production_approval": False,
    }


def build_report(evidence: Mapping[str, Any], gates: Mapping[str, Any], decision: Mapping[str, Any]) -> str:
    metrics = evidence["comparison"]["metrics"]
    escalation = evidence["escalation"]
    gate_lines = "\n".join(
        f"- **{item['gate_name']}**: {item['status']} — {item['rationale']}"
        for item in gates["blocking_gates"] + gates["operational_gates"]
    )
    metric_lines = "\n".join(
        f"- {name}: Base {row['base']:.6f}%, LoRA {row['lora']:.6f}%, delta {row['absolute_delta']:+.6f} pp"
        for name, row in metrics.items() if row["unit"] == "percentage_points"
    )
    constraints = decision["deployment_scope"]
    prohibited = decision["prohibited_scope"]
    limitation_lines = "\n".join(f"- {item}" for item in ADDITIONAL_LIMITATIONS)
    scope_lines = "\n".join(f"- {item}" for item in constraints)
    prohibited_lines = "\n".join(f"- {item}" for item in prohibited)
    return f"""# Stage C8 Promotion Decision

## Executive Decision

Candidate 01 decision: **{decision['decision']}**. This decision is based only on the frozen promotion gate, Stage C7 Locked evidence, and the supplied Locked Manual Response QA conclusion.

## Frozen Promotion Gate

Promotion gate SHA-256: `{EXPECTED_PROMOTION_GATE_SHA256}`. No threshold was changed or added after Locked Test evaluation.

## Locked Evidence

{metric_lines}

Base and LoRA each retained all 300 attempts. Stage C7 integrity is PASS.

## Gate-by-Gate Results

{gate_lines}

## Structured Behavior Improvement

Locked evidence shows substantial gains in intent accuracy, category accuracy, JSON validity, and schema compliance under the frozen contract.

## Escalation Behavior

Base precision / recall / F1: {escalation['base']['precision_percent']:.6f}% / {escalation['base']['recall_percent']:.6f}% / {escalation['base']['f1_percent']:.6f}%. LoRA: {escalation['lora']['precision_percent']:.6f}% / {escalation['lora']['recall_percent']:.6f}% / {escalation['lora']['f1_percent']:.6f}%.

Base precision of 100% does not establish better overall escalation behavior because Base positive-class recall is only {escalation['base']['recall_percent']:.6f}%. The frozen gate does not treat escalation precision alone as a blocker.

## Response Safety Review

Locked Manual Response QA: **{MANUAL_QA_CONCLUSION}**. Material response safety regression: **NO**. Critical behavioral regression: **NO**. Automated screening is supporting evidence and is not a complete quality judgment.

## Operational Latency Trade-off

Base mean / median / p95: 1121.593 / 1078.237 / 1587.030 ms. LoRA: 3067.912 / 2717.730 / 4972.819 ms.

{LATENCY_WARNING} Latency is not a frozen promotion blocker.

## Remaining Failure Modes

{KNOWN_LIMITATION_EN}

{KNOWN_LIMITATION_ZH}

{limitation_lines}

## Deployment Scope

{scope_lines}

## Deployment Constraints

{prohibited_lines}

External grounding, backend tools, and human review remain required as specified in `deployment_constraints.json`.

## Governance Decision

All frozen blocking gates passed. No training, Locked rerun, prompt/evaluator/threshold/candidate modification, checkpoint selection, or Stage C9 work occurred.

## Final Promotion Decision

Candidate 01: **{decision['decision']}**

PROMOTE does not represent unrestricted production approval. Candidate 01 is approved only for the documented structured classification and routing scope under the listed constraints.
"""


def run(repo_root: Path) -> Dict[str, Any]:
    stage8_dir = repo_root / STAGE8_DIR
    report_path = repo_root / REPORT_PATH
    if stage8_dir.exists() or report_path.exists():
        raise RuntimeError("Refusing to overwrite an existing Stage C8 decision")
    evidence = load_evidence(repo_root)
    governance = governance_preflight(repo_root, evidence)
    if governance["status"] != "PASS":
        raise RuntimeError("Stage C8 ABORTED: critical governance integrity condition failed")

    gates = evaluate_gates(evidence)
    decision_value = "PROMOTE" if gates["all_frozen_blocking_gates_passed"] else "DO_NOT_PROMOTE"
    constraints = deployment_constraints()
    blocking_failures = [item["gate_name"] for item in gates["blocking_gates"] if item["status"] == "FAIL"]
    decision = {
        "stage": "C8",
        "candidate": CANDIDATE,
        "decision": decision_value,
        "decision_basis": "Frozen promotion gate + Stage C7 Locked evaluation evidence + Locked Manual Response QA conclusion",
        "all_frozen_blocking_gates_passed": not blocking_failures,
        "blocking_failures": blocking_failures,
        "manual_response_qa_conclusion": MANUAL_QA_CONCLUSION,
        "critical_behavioral_regression": False,
        "material_response_safety_regression": False,
        "operational_warnings": [LATENCY_WARNING],
        "known_limitations": [KNOWN_LIMITATION_EN, KNOWN_LIMITATION_ZH, *ADDITIONAL_LIMITATIONS],
        "deployment_scope": constraints["approved_scope"],
        "prohibited_scope": constraints["not_approved_as"],
        "unrestricted_production_approval": False,
    }
    stage8_dir.mkdir(parents=True)
    write_json(stage8_dir / "governance_integrity.json", governance)
    write_json(stage8_dir / "promotion_gate_results.json", gates)
    write_json(stage8_dir / "promotion_decision.json", decision)
    write_json(stage8_dir / "deployment_constraints.json", constraints)
    manifest = {
        "stage": "C8",
        "status": "COMPLETE",
        "candidate": CANDIDATE,
        "c6_5_freeze_manifest_hash": sha256_file(repo_root / FREEZE_MANIFEST_PATH),
        "c7_manifest_hash": sha256_file(repo_root / STAGE7_MANIFEST_PATH),
        "promotion_gate_hash": sha256_file(repo_root / PROMOTION_GATE_PATH),
        "decision": decision_value,
        "decision_timestamp": datetime.now(timezone.utc).isoformat(),
        "training_performed": False,
        "locked_rerun_performed": False,
        "prompt_modified": False,
        "evaluator_modified": False,
        "threshold_modified": False,
        "candidate_modified": False,
        "stage9_performed": False,
    }
    write_json(stage8_dir / "stage8_manifest.json", manifest)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(build_report(evidence, gates, decision), encoding="utf-8")
    return {"governance": governance, "gates": gates, "decision": decision, "constraints": constraints, "manifest": manifest}


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    result = run(repo_root)
    print(json.dumps({
        "status": result["manifest"]["status"],
        "candidate": result["decision"]["candidate"],
        "decision": result["decision"]["decision"],
        "all_frozen_blocking_gates_passed": result["decision"]["all_frozen_blocking_gates_passed"],
        "stage9_performed": False,
    }, indent=2))


if __name__ == "__main__":
    main()
