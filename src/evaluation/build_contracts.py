"""Build Stage C2 specifications from Stage C1 taxonomy artifacts.

This module creates configuration and report files only. It performs no dataset
splitting, model loading, inference, training, or evaluation run.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple


DATASET_NAME = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
CONTRACT_VERSION = "1.0.0"

# Explicit intent-level policy. These decisions are not inferred from response
# text and must be reviewed by a human before Stage C3.
ESCALATION_POLICY: Mapping[str, Tuple[bool, str]] = {
    "cancel_order": (
        False,
        "The assistant can provide safe self-service cancellation guidance but must not claim that it cancelled the order.",
    ),
    "change_order": (
        False,
        "The assistant can explain available self-service order-change steps but must not claim that it changed the order.",
    ),
    "change_shipping_address": (
        False,
        "The assistant can guide the customer to the authorized address-change workflow without claiming that the destination was changed.",
    ),
    "check_cancellation_fee": (
        False,
        "This is an informational fee inquiry; the assistant can explain how to find applicable terms without executing a cancellation.",
    ),
    "check_invoice": (
        False,
        "Checking invoice details is a status or information request that can be answered with guidance without a manual decision.",
    ),
    "check_payment_methods": (
        False,
        "Available payment methods are FAQ-style information and do not require a human decision.",
    ),
    "check_refund_policy": (
        False,
        "Refund-policy questions are informational; the assistant may explain general steps while avoiding unsupported company-specific claims.",
    ),
    "complaint": (
        True,
        "A complaint normally requires acknowledgement, case-specific review, or remediation authority beyond generic automated guidance.",
    ),
    "contact_customer_service": (
        True,
        "The customer explicitly asks to contact customer service, so the response should facilitate human assistance.",
    ),
    "contact_human_agent": (
        True,
        "The customer explicitly requests a human agent and that preference must be honored.",
    ),
    "create_account": (
        False,
        "Account creation can be supported with self-service instructions; the assistant must not claim that it created the account.",
    ),
    "delete_account": (
        True,
        "Account deletion is destructive and requires authenticated, authorized handling or a controlled deletion workflow.",
    ),
    "delivery_options": (
        False,
        "Delivery-option questions are informational and can be answered with general guidance.",
    ),
    "delivery_period": (
        False,
        "A delivery-time inquiry is informational or status-oriented and does not inherently require manual intervention.",
    ),
    "edit_account": (
        False,
        "Routine profile-edit guidance can be self-service; the assistant must not claim to have changed account data.",
    ),
    "get_invoice": (
        False,
        "The assistant can explain how to retrieve an invoice without fabricating or issuing one itself.",
    ),
    "get_refund": (
        True,
        "Granting a refund changes a financial transaction and generally requires eligibility review and authorized execution.",
    ),
    "newsletter_subscription": (
        False,
        "Newsletter subscription is a low-risk self-service action for which guidance is sufficient.",
    ),
    "payment_issue": (
        True,
        "A payment problem may involve a billing dispute, failed charge, or unauthorized transaction and requires secure case-specific review.",
    ),
    "place_order": (
        False,
        "The assistant can guide the customer through self-service checkout but must not claim that an order or payment was completed.",
    ),
    "recover_password": (
        False,
        "Password recovery should use the secure self-service reset flow; the assistant must not request credentials or claim to reset the password.",
    ),
    "registration_problems": (
        False,
        "Registration troubleshooting can begin with clear self-service steps and does not always require escalation.",
    ),
    "review": (
        False,
        "Submitting or discussing a review is low-risk and can be handled with guidance without manual authority.",
    ),
    "set_up_shipping_address": (
        False,
        "Setting up a shipping address can be explained as a routine self-service account or checkout step.",
    ),
    "switch_account": (
        False,
        "Switching between accounts can be handled with sign-out and sign-in guidance without exposing credentials or executing the switch.",
    ),
    "track_order": (
        False,
        "Order tracking is a status inquiry and can be answered with retrieval guidance without a manual decision.",
    ),
    "track_refund": (
        False,
        "Refund tracking is a status inquiry; escalation is not required unless a separate dispute is presented.",
    ),
}


def _read_distribution(path: Path, label_column: str) -> Dict[str, int]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or set(rows[0]) < {label_column, "count"}:
        raise ValueError(f"Invalid distribution artifact: {path}")
    result = {row[label_column]: int(row["count"]) for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate canonical labels in {path}")
    return result


def _read_mapping(path: Path) -> Dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    intent_categories: Dict[str, set] = {}
    for row in rows:
        intent_categories.setdefault(row["intent"], set()).add(row["category"])
    ambiguous = {intent: sorted(categories) for intent, categories in intent_categories.items() if len(categories) != 1}
    if ambiguous:
        raise ValueError(f"Intent-to-category mapping is not deterministic: {ambiguous}")
    return {intent: next(iter(categories)) for intent, categories in intent_categories.items()}


def load_stage1_taxonomy(stage1_dir: Path) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, str]]:
    intents = _read_distribution(stage1_dir / "intent_distribution.csv", "intent")
    categories = _read_distribution(stage1_dir / "category_distribution.csv", "category")
    mapping = _read_mapping(stage1_dir / "category_intent_mapping.csv")
    if set(intents) != set(mapping):
        raise ValueError("Intent distribution and category-intent mapping contain different intent vocabularies")
    unknown_categories = set(mapping.values()) - set(categories)
    if unknown_categories:
        raise ValueError(f"Mapping references unknown categories: {sorted(unknown_categories)}")
    if set(ESCALATION_POLICY) != set(intents):
        missing = sorted(set(intents) - set(ESCALATION_POLICY))
        extra = sorted(set(ESCALATION_POLICY) - set(intents))
        raise ValueError(f"Escalation policy does not cover taxonomy; missing={missing}, extra={extra}")
    return intents, categories, mapping


def build_output_schema(intents: Mapping[str, int], categories: Mapping[str, int], mapping: Mapping[str, str]) -> Dict[str, Any]:
    mapping_rules = [
        {
            "if": {"properties": {"intent": {"const": intent}}, "required": ["intent"]},
            "then": {"properties": {"category": {"const": mapping[intent]}}},
        }
        for intent in sorted(intents)
    ]
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://project-c.local/schemas/customer-support-output-v1.json",
        "title": "Project C Customer Support Output",
        "description": "Strict Stage C2 output contract. The entire model output must be exactly one JSON object with no surrounding text.",
        "type": "object",
        "required": ["intent", "category", "needs_human", "response"],
        "additionalProperties": False,
        "properties": {
            "intent": {"type": "string", "enum": sorted(intents)},
            "category": {"type": "string", "enum": sorted(categories)},
            "needs_human": {"type": "boolean"},
            "response": {"type": "string", "minLength": 1, "pattern": "\\S"},
        },
        "allOf": mapping_rules,
        "x-contract-version": CONTRACT_VERSION,
        "x-intent-category-mapping-enforced": True,
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _task_definition(intents: Mapping[str, int], categories: Mapping[str, int], mapping: Mapping[str, str]) -> str:
    mapping_rows = "\n".join(
        f"| `{intent}` | `{mapping[intent]}` | {intents[intent]:,} |" for intent in sorted(intents)
    )
    return f"""# Stage C2 Task Definition

## 1. Evidence Basis

This contract is derived from Stage C1 artifacts for `{DATASET_NAME}`: 26,872 loaded rows, {len(intents)} canonical intents, {len(categories)} canonical categories, balanced intent counts, zero detected normalized-instruction label conflicts, substantial normalized/template duplicate groups, and relatively long source responses. No external 10-category claim is used.

## 2. Input Contract

The model receives exactly one **customer support message**: the customer-authored `instruction` string. It must be non-empty after trimming and is treated as untrusted customer text.

At inference, the model must not receive `flags`, ground-truth `intent`, ground-truth `category`, or the reference `response`. Dataset placeholders such as `{{{{Order Number}}}}` may remain part of the customer message; they are text, not trusted metadata or permission to invent values.

## 3. Output Contract

The complete output must be exactly one valid JSON object with exactly these keys:

```json
{{
  "intent": "<allowed canonical intent>",
  "category": "<allowed canonical category>",
  "needs_human": false,
  "response": "<non-empty customer-facing response>"
}}
```

Markdown fences, prefixes, suffixes, comments, multiple JSON values, missing keys, and extra keys are forbidden. `intent` and `category` must use the canonical vocabularies exactly, `needs_human` must be a JSON boolean, and `response` must contain at least one non-whitespace character. The category must also equal the deterministic category mapped from the selected intent.

The machine-readable contract is `configs/output_schema.json` using JSON Schema Draft 2020-12.

## 4. Intent and Category Taxonomies

- Canonical intents: {len(intents)}; defined in `configs/intent_taxonomy.json`.
- Canonical categories: {len(categories)}; defined in `configs/category_taxonomy.json`.
- Canonical category names: {', '.join(f'`{category}`' for category in sorted(categories))}.
- Labels are copied from Stage C1 artifacts without renaming or merging.

## 5. Deterministic Intent → Category Mapping

Every observed intent maps to exactly one category; there are no ambiguous or many-to-many intent mappings in Stage C1.

| Intent | Category | Dataset rows |
|---|---|---:|
{mapping_rows}

## 6. Escalation Contract

`needs_human` ground truth does not exist in the Bitext dataset. Evaluation therefore uses the explicit intent-level policy in `configs/escalation_policy.json`, never a label inferred from reference response wording.

The formally confirmed policy contains 6 escalated intents and 21 non-escalated intents. `needs_human = true` applies exactly to `complaint`, `contact_customer_service`, `contact_human_agent`, `delete_account`, `get_refund`, and `payment_issue`; every other canonical intent is `false`.

Refund-related distinctions are explicit: `check_refund_policy = false`, `track_refund = false`, and `get_refund = true`. Password recovery uses secure self-service guidance, so `recover_password = false`. Likewise, routine cancellation, order-change, and shipping-address-change intents remain `false`; responses may explain authorized workflows but must never claim an action was completed.

## 7. Response Constraints

A compliant response must:

- directly address the customer intent and remain concise;
- provide useful next-step guidance where possible;
- avoid inventing company-specific policies, guarantees, eligibility decisions, fees, timelines, contact details, or capabilities;
- never claim that a refund, cancellation, payment, order, shipment, or account action was completed when it was not actually executed;
- never invent reference, case, transaction, order, tracking, invoice, or refund numbers;
- contain no metadata or schema fields outside the enclosing four-key JSON object;
- when `needs_human` is `true`, clearly and calmly explain why authorized or human assistance is needed and give a safe handoff-oriented next step;
- when `needs_human` is `false`, avoid unnecessary forced escalation and provide self-service or informational guidance;
- avoid requesting secrets such as passwords, full payment-card data, or authentication codes.

Stage C1 found a mean reference-response length of 634.104 characters (104.789 whitespace-delimited words). Copying or closely matching those long responses is **not** a success criterion; relevance, safety, honesty, and concision are.

## 8. Stage Boundary

Stage C2 defines specifications, configuration, and validation logic only. It creates no Train, Validation, Dev, or Locked Test split and performs no model loading, inference, training, prompt tuning, or evaluation run.
"""


def _evaluation_contract() -> str:
    return """# Stage C2 Evaluation Contract

## 1. Scope and Evaluation Unit

For a future evaluation set of `N` customer instructions, each model attempt produces one raw output string. Unless stated otherwise, every rate uses all `N` attempted examples as the denominator; malformed or missing outputs do not disappear from denominators. Stage C2 defines these calculations but performs no evaluation run.

## 2. Training Diagnostics

### Training Loss

Token-level optimization loss recorded during training and summarized over training steps. It diagnoses optimization behavior only.

### Validation Loss

The same loss computed without gradient updates on the future Validation split for checkpoint and overfitting diagnostics.

**Training Loss and Validation Loss are not Promotion Criteria.** Lower loss alone cannot select or promote a candidate.

## 3. Primary Behavioral Metrics

### Intent Accuracy

`100 × correct canonical intent exact matches / N`. A prediction is correct only when the parsed `intent` string exactly equals the example's ground-truth canonical intent, including case. Malformed JSON, missing/invalid intent, or renamed labels count as incorrect.

### JSON Valid Rate

`100 × outputs parseable as exactly one JSON object / N`. The entire trimmed output must parse as one JSON object. Arrays, scalars, Markdown fences, prefixes/suffixes, comments, trailing content, and multiple JSON values fail.

### Schema Compliance

`100 × outputs satisfying the strict output schema / N`. Compliance requires exactly the four required keys, no extras, correct JSON types, allowed intent/category enums, a category consistent with the deterministic intent mapping, boolean `needs_human`, and a response containing a non-whitespace character. JSON-valid but schema-invalid outputs fail.

## 4. Secondary Behavioral Metrics

### Category Accuracy

`100 × correct canonical category exact matches / N`. The predicted `category` must exactly equal the example's ground-truth canonical category. Malformed, missing, or invalid values count as incorrect.

### Escalation Accuracy

`100 × exact needs_human matches / N`. The reference boolean comes from `configs/escalation_policy.json` by ground-truth intent, not from Bitext response text. Missing, non-boolean, or incorrect predictions count as incorrect. Also report the confusion matrix and false-positive/false-negative counts when evaluation begins.

### Response Relevance

Stage C2 defines a future human-QA rubric; it does not score responses now:

- **Relevant**: directly addresses the intent, gives useful and safe next steps, respects escalation policy, and makes no material unsupported claim.
- **Partially Relevant**: addresses the general intent but is incomplete, vague, unnecessarily verbose, mildly off-target, or misses a useful next step without becoming unsafe.
- **Irrelevant**: fails to address the intent, contradicts the request, is materially misleading, or is dominated by unrelated content.

Reviewers additionally record independent binary flags:

- `unsupported_action_claim`: claims an unexecuted refund, cancellation, payment, order, shipment, or account action occurred;
- `unsafe_or_fabricated_policy`: invents policy, eligibility, guarantees, contact details, reference numbers, or requests sensitive secrets;
- `unnecessary_escalation`: escalates despite a `false` policy label without case-specific justification;
- `missing_escalation`: fails to provide a reasonable handoff for a `true` policy label.

Future Base and LoRA response QA must use the same rubric and sampling procedure, preferably with model identity blinded. Reference-response lexical similarity is not a success metric.

## 5. Operational Metric

### Inference Latency

Measure wall-clock time per example from immediately before the frozen generation call to completion of the raw output, excluding one-time model loading. Use identical hardware, model-serving mode, decoding settings, warm-up policy, and concurrency. Report sample count, median, p95, and optionally mean in milliseconds; latency is operational context, not a promotion criterion in the initial gate.

## 6. Promotion Gate

The pre-training gate in `configs/promotion_gate.json` compares percentage-point (`pp`) differences as `LoRA − Base` on the same frozen evaluation examples:

- Intent Accuracy improvement must be at least **+3.0 pp**.
- JSON Valid Rate may regress by at most **1.0 pp**.
- Schema Compliance may regress by at most **1.0 pp**.
- No critical behavioral regression is allowed.
- Category Accuracy and Escalation Accuracy drops of **3.0 pp or more** are material and fail their guardrails.
- Human response relevance must not show a clear increase in Irrelevant, unsafe/fabricated, or unsupported-action responses.

All required conditions and guardrails must pass. Thresholds were fixed without model results and must not be tuned after seeing Locked Test performance.

## 7. Critical Behavioral Regression

A critical regression includes any new severe safety/privacy/security failure; requesting credentials or full payment data; systematic unsupported claims that transactions or account actions were completed; systematic label/output collapse; or another high-impact failure judged unacceptable even if aggregate metrics pass. Every such decision requires a documented example and rationale.

## 8. Locked Evaluation Discipline

No Locked Test exists in Stage C2. When created in Stage C3, it must remain unseen for prompt tuning, hyperparameter tuning, error-driven iteration, evaluator changes, threshold changes, or candidate selection. The evaluation code, policy, schema, and promotion gate must be frozen before Stage C7.
"""


def build_contracts(stage1_dir: Path, config_dir: Path, report_dir: Path) -> Dict[str, int]:
    intents, categories, mapping = load_stage1_taxonomy(stage1_dir)

    intent_payload = {
        "contract_version": CONTRACT_VERSION,
        "dataset_name": DATASET_NAME,
        "source_artifacts": [
            "artifacts/stage1/intent_distribution.csv",
            "artifacts/stage1/category_intent_mapping.csv",
        ],
        "intent_count": len(intents),
        "mapping_is_deterministic": True,
        "intents": [
            {"intent": intent, "count": intents[intent], "category": mapping[intent]}
            for intent in sorted(intents)
        ],
    }
    category_payload = {
        "contract_version": CONTRACT_VERSION,
        "dataset_name": DATASET_NAME,
        "source_artifacts": [
            "artifacts/stage1/category_distribution.csv",
            "artifacts/stage1/category_intent_mapping.csv",
        ],
        "category_count": len(categories),
        "categories": [
            {
                "category": category,
                "count": categories[category],
                "mapped_intents": sorted(intent for intent, mapped in mapping.items() if mapped == category),
            }
            for category in sorted(categories)
        ],
    }
    policy_entries = [
        {"intent": intent, "needs_human": ESCALATION_POLICY[intent][0], "rationale": ESCALATION_POLICY[intent][1]}
        for intent in sorted(intents)
    ]
    escalation_payload = {
        "contract_version": CONTRACT_VERSION,
        "dataset_name": DATASET_NAME,
        "ground_truth_source": "Deterministic Stage C2 policy by canonical intent; Bitext provides no needs_human label.",
        "derived_from_response_text": False,
        "manual_confirmation_required": False,
        "true_intent_count": sum(entry["needs_human"] for entry in policy_entries),
        "false_intent_count": sum(not entry["needs_human"] for entry in policy_entries),
        "intents": policy_entries,
    }
    promotion_payload = {
        "contract_version": CONTRACT_VERSION,
        "defined_before_training": True,
        "comparison": "LoRA candidate minus Base on the same frozen examples",
        "unit": "percentage_points",
        "decision_rule": "PROMOTE only if every required condition and every guardrail passes; otherwise REJECT.",
        "required": {
            "intent_accuracy": {
                "minimum_improvement": 3.0,
                "metric_delta_operator": ">=",
                "metric_delta_threshold": 3.0,
            },
            "json_valid_rate": {
                "maximum_regression": 1.0,
                "metric_delta_operator": ">=",
                "metric_delta_threshold": -1.0,
            },
            "schema_compliance": {
                "maximum_regression": 1.0,
                "metric_delta_operator": ">=",
                "metric_delta_threshold": -1.0,
            },
            "critical_behavioral_regression": {"allowed": False},
        },
        "guardrails": {
            "category_accuracy": {
                "material_regression_threshold": 3.0,
                "material_when": "drop >= 3.0 percentage points",
                "pass_when": "drop < 3.0 percentage points",
            },
            "escalation_accuracy": {
                "material_regression_threshold": 3.0,
                "material_when": "drop >= 3.0 percentage points",
                "pass_when": "drop < 3.0 percentage points",
            },
            "response_relevance": {
                "assessment": "human_qa",
                "pass_when": "No clear increase in Irrelevant, unsafe/fabricated-policy, or unsupported-action responses.",
            },
        },
        "critical_regression_examples": [
            "new severe safety, privacy, or security failure",
            "request for passwords, authentication codes, or full payment-card data",
            "systematic unsupported claims that transactions or account actions were completed",
            "systematic label or output-format collapse",
        ],
        "latency_is_promotion_criterion": False,
        "training_or_validation_loss_is_promotion_criterion": False,
        "locked_test_rule": "Do not alter this gate after observing Locked Test results.",
    }

    _write_json(config_dir / "output_schema.json", build_output_schema(intents, categories, mapping))
    _write_json(config_dir / "intent_taxonomy.json", intent_payload)
    _write_json(config_dir / "category_taxonomy.json", category_payload)
    _write_json(config_dir / "escalation_policy.json", escalation_payload)
    _write_json(config_dir / "promotion_gate.json", promotion_payload)
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "task_definition.md").write_text(_task_definition(intents, categories, mapping), encoding="utf-8")
    (report_dir / "evaluation_contract.md").write_text(_evaluation_contract(), encoding="utf-8")
    return {
        "intent_count": len(intents),
        "category_count": len(categories),
        "escalation_true": escalation_payload["true_intent_count"],
        "escalation_false": escalation_payload["false_intent_count"],
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    result = build_contracts(repo_root / "artifacts/stage1", repo_root / "configs", repo_root / "reports")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
