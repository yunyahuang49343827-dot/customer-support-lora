# Stage C6 Development Evaluation

## Goal

Fairly compare the frozen C4 Base artifacts with QLoRA Candidate 01 on all 300 Frozen Dev rows.

## Fair Comparison Contract

Base and LoRA use the same model revision, frozen system prompt (SHA-256 `6b84135769b7348758e8cc21a3cb168465e00de5efaf59ff8a8459087db3dc3b`), customer instruction, greedy temperature 0 decoding, 512-token limit, evaluator, parser, schema, taxonomies, and escalation policy. The only inference difference is the Candidate 01 adapter.

## Dataset Boundary

Dev SHA-256: `a0859497b5fe23ca1adf1ab1e6a9b7da5dfca1bbcd6519c89ab7ea4f21a5b4d6`. Dev rows: 300. Locked Test content accessed: no.

## Base Metrics

The immutable C4 predictions and metrics were reused and reproduced from all 300 stored predictions; Base inference was not rerun.

## LoRA Metrics

- Intent accuracy: 92.000000%
- Category accuracy: 98.333333%
- JSON valid rate: 100.000000%
- Schema compliance: 100.000000%
- Escalation accuracy: 97.333333%

## Metric Delta

- intent_accuracy: Base 31.333333, LoRA 92.000000, delta +60.666667 pp
- category_accuracy: Base 61.000000, LoRA 98.333333, delta +37.333333 pp
- json_valid_rate: Base 99.000000, LoRA 100.000000, delta +1.000000 pp
- schema_compliance: Base 33.000000, LoRA 100.000000, delta +67.000000 pp
- escalation_accuracy: Base 78.333333, LoRA 97.333333, delta +19.000000 pp
- escalation_precision: Base 100.000000, LoRA 89.189189, delta -10.810811 pp
- escalation_recall: Base 3.030303, LoRA 100.000000, delta +96.969697 pp
- escalation_f1: Base 5.882353, LoRA 94.285714, delta +88.403361 pp
- mean_latency: Base 1235.668082, LoRA 2191.734293, delta +956.066211 ms
- p95_latency: Base 1802.524533, LoRA 3459.854888, delta +1657.330355 ms

## Escalation Precision / Recall / F1

- Precision: 89.189189%
- Recall: 100.000000%
- F1: 94.285714%
- TP / FP / FN / TN: 66 / 8 / 0 / 226
- Invalid or missing boolean: 0

## Error Reduction

- `wrong_intent`: Base 206, LoRA 24, delta -182
- `invalid_enum`: Base 152, LoRA 0, delta -152
- `wrong_category`: Base 117, LoRA 5, delta -112
- `wrong_needs_human`: Base 65, LoRA 8, delta -57
- `intent_category_mismatch`: Base 46, LoRA 0, delta -46
- `invalid_json`: Base 3, LoRA 0, delta -3
- `generation_truncated`: Base 1, LoRA 0, delta -1
- `empty_response`: Base 0, LoRA 0, delta +0
- `extra_key`: Base 0, LoRA 0, delta +0
- `extra_text_after_json`: Base 0, LoRA 0, delta +0
- `extra_text_before_json`: Base 0, LoRA 0, delta +0
- `missing_key`: Base 0, LoRA 0, delta +0

## Intent Confusions

Top Base confusions:

- `create_account` → `registration_problems`: 11
- `get_invoice` → `check_invoice`: 10
- `delete_account` → `registration_problems`: 8
- `switch_account` → `registration_problems`: 8
- `track_refund` → `check_refund_policy`: 6
- `newsletter_subscription` → `complaint`: 4
- `change_shipping_address` → `set_up_shipping_address`: 3
- `edit_account` → `set_up_shipping_address`: 3
- `recover_password` → `complaint`: 3
- `edit_account` → `registration_problems`: 2

Top LoRA confusions:

- `set_up_shipping_address` → `change_shipping_address`: 6
- `check_refund_policy` → `get_refund`: 4
- `get_invoice` → `check_invoice`: 3
- `track_refund` → `check_cancellation_fee`: 2
- `check_invoice` → `get_invoice`: 1
- `check_refund_policy` → `track_refund`: 1
- `contact_human_agent` → `contact_customer_service`: 1
- `registration_problems` → `create_account`: 1
- `review` → `contact_customer_service`: 1
- `review` → `contact_human_agent`: 1

## Response QA

`artifacts/stage6/manual_qa_samples.csv` contains 30 deterministic seed-42 paired Base/LoRA cases covering correct cases, wrong intents, both escalation labels, and schema failures when present. Manual review remains required.

## Response Risk QA

`artifacts/stage6/response_risk_qa.csv` screens all paired rows. Base flags: `{"asks_for_sensitive_secret": 2, "fabricated_24_7_availability": 2, "fabricated_contact_details": 7, "fabricated_fees_or_timelines": 3, "unnecessary_escalation": 70, "unsupported_action_completion": 3, "unsupported_guarantee": 1, "unsupported_update_claim": 0}`. LoRA flags: `{"asks_for_sensitive_secret": 1, "fabricated_24_7_availability": 0, "fabricated_contact_details": 1, "fabricated_fees_or_timelines": 2, "unnecessary_escalation": 38, "unsupported_action_completion": 7, "unsupported_guarantee": 0, "unsupported_update_claim": 0}`. These conservative rules are only risk screening and cannot establish complete response quality.

## Latency

- Base mean / p95: 1235.668 / 1802.525 ms
- LoRA mean / p95: 2191.734 / 3459.855 ms

## Development Decision

`candidate_01_strong_enough_to_freeze_pending_manual_response_qa`. This is development-oriented and is not a Locked Promotion Gate decision.

## Controlled Iteration Needed?

`false`. No retraining was started.

## Limitations

Automated exact-match metrics and heuristic risk flags do not replace blinded human response review. Latency is machine-specific. No Locked Test inspection or inference occurred.

## Stage C6 Conclusion

Candidate 01 evaluation is complete on the full Dev set. Stage C6.5 and Stage C7 were not entered.
