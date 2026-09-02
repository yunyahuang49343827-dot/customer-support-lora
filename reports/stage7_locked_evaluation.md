# Stage C7 Locked Evaluation

## Goal

Evaluate the frozen Base model and frozen Candidate 01 once on all 300 Locked Test rows under the Stage C6.5 contract.

## Locked Evaluation Boundary

This stage produced evaluation evidence only. No training, tuning, checkpoint selection, configuration change, or formal promotion decision occurred.

## Freeze Integrity

Freeze preflight and post-run integrity: **PASS**. All frozen hashes matched before Locked JSON parsing.

## Locked Dataset

Rows: 300. SHA-256: `b7f7af8c5e366c743fafd68c8c8f3e7a2b101dfce53e63bf1f7a8ead0bce1fac`. Base attempts: 300. LoRA attempts: 300. No attempted row was excluded.

## Base Locked Metrics

- Intent: 28.000000%
- Category: 61.666667%
- JSON valid: 99.333333%
- Schema compliant: 36.666667%
- Escalation: 79.000000%

## LoRA Locked Metrics

- Intent: 94.000000%
- Category: 99.000000%
- JSON valid: 99.666667%
- Schema compliant: 99.333333%
- Escalation: 98.666667%

## Base vs LoRA Delta

- intent_accuracy: Base 28.000000, LoRA 94.000000, delta +66.000000 percentage_points
- category_accuracy: Base 61.666667, LoRA 99.000000, delta +37.333333 percentage_points
- json_valid_rate: Base 99.333333, LoRA 99.666667, delta +0.333334 percentage_points
- schema_compliance: Base 36.666667, LoRA 99.333333, delta +62.666666 percentage_points
- escalation_accuracy: Base 79.000000, LoRA 98.666667, delta +19.666667 percentage_points
- escalation_precision: Base 100.000000, LoRA 95.652174, delta -4.347826 percentage_points
- escalation_recall: Base 7.575758, LoRA 100.000000, delta +92.424242 percentage_points
- escalation_f1: Base 14.084507, LoRA 97.777778, delta +83.693271 percentage_points
- mean_latency: Base 1121.592696, LoRA 3067.912041, delta +1946.319345 milliseconds
- p95_latency: Base 1587.030367, LoRA 4972.819301, delta +3385.788934 milliseconds
- median_latency: Base 1078.237270, LoRA 2717.730438, delta +1639.493168 milliseconds

## Escalation Precision / Recall / F1

- Base: precision 100.000000%, recall 7.575758%, F1 14.084507%
- LoRA: precision 95.652174%, recall 100.000000%, F1 97.777778%

## Error Reduction

- `wrong_intent`: Base 216, LoRA 18, delta -198
- `invalid_enum`: Base 143, LoRA 1, delta -142
- `wrong_category`: Base 115, LoRA 3, delta -112
- `wrong_needs_human`: Base 63, LoRA 4, delta -59
- `intent_category_mismatch`: Base 45, LoRA 0, delta -45
- `extra_text_after_json`: Base 1, LoRA 0, delta -1
- `extra_text_before_json`: Base 1, LoRA 0, delta -1
- `invalid_json`: Base 2, LoRA 1, delta -1
- `empty_response`: Base 0, LoRA 0, delta +0
- `extra_key`: Base 0, LoRA 0, delta +0
- `missing_key`: Base 0, LoRA 0, delta +0
- `generation_truncated`: Base 0, LoRA 1, delta +1

## Remaining Intent Confusions

- `get_invoice` → `check_invoice`: 4
- `set_up_shipping_address` → `change_shipping_address`: 4
- `create_account` → `edit_account`: 3
- `cancel_order` → `place_order`: 1
- `change_order` → `cancel_order`: 1
- `change_shipping_address` → `edit_account`: 1
- `place_order` → `get_refund`: 1
- `switch_account` → `edit_account`: 1

Error analysis is descriptive only and must not be used to modify Candidate 01.

## Response Risk Screening

Base flags: `{"asks_for_sensitive_secret": 0, "fabricated_24_7_availability": 1, "fabricated_contact_details": 5, "fabricated_fees_or_timelines": 5, "unnecessary_escalation": 75, "unsupported_action_completion": 2, "unsupported_guarantee": 0, "unsupported_update_claim": 0}`. LoRA flags: `{"asks_for_sensitive_secret": 0, "fabricated_24_7_availability": 0, "fabricated_contact_details": 0, "fabricated_fees_or_timelines": 1, "unnecessary_escalation": 33, "unsupported_action_completion": 6, "unsupported_guarantee": 1, "unsupported_update_claim": 0}`.

Automated risk screening is not a complete quality judgment.

## Manual QA Preparation

`locked_manual_qa_samples.csv` contains 30 deterministic paired samples. `locked_manual_response_review.csv` contains 127 deduplicated sampled/risk-flagged rows with every human field blank.

## Latency Trade-off

- Base mean / median / p95: 1121.593 / 1078.237 / 1587.030 ms
- LoRA mean / median / p95: 3067.912 / 2717.730 / 4972.819 ms

## Frozen Promotion Gate Inputs

`promotion_gate_inputs.json` records thresholds, Base results, LoRA results, and deltas only. Stage C7 makes no formal promotion decision.

## Known Response Limitations

QLoRA significantly improved structured classification behavior, but manual QA found that generated responses can still contain unsupported policy or capability claims. Therefore, the fine-tuned model should not be treated as an enterprise factual authority.

QLoRA 顯著改善 structured classification behavior，但人工 QA 發現生成式 response 仍可能產生 unsupported policy/capability claims，因此模型不應直接被視為企業 factual authority。

## Locked-Test Governance

Locked results are final-evaluation evidence and must not be used for training, tuning, evaluator changes, threshold changes, checkpoint selection, or candidate modification.

## Limitations

Exact-match metrics do not establish complete response quality. Risk screening is heuristic, manual QA remains required, and latency is specific to this machine.

## Stage C7 Conclusion

Stage C7 evaluation is complete on all 300 Locked rows. The metrics provide held-out generalization evidence under the frozen contract. Formal promotion decision is reserved for Stage C8.
