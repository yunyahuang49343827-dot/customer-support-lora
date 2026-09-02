# Stage C2 Evaluation Contract

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
