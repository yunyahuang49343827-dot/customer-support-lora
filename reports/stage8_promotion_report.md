# Stage C8 Promotion Decision

## Executive Decision

Candidate 01 decision: **PROMOTE**. This decision is based only on the frozen promotion gate, Stage C7 Locked evidence, and the supplied Locked Manual Response QA conclusion.

## Frozen Promotion Gate

Promotion gate SHA-256: `8e756705625c7bc61cb136d0672b785a76d21b8443f10c9f1903c87c3d2af377`. No threshold was changed or added after Locked Test evaluation.

## Locked Evidence

- category_accuracy: Base 61.666667%, LoRA 99.000000%, delta +37.333333 pp
- escalation_accuracy: Base 79.000000%, LoRA 98.666667%, delta +19.666667 pp
- escalation_f1: Base 14.084507%, LoRA 97.777778%, delta +83.693271 pp
- escalation_precision: Base 100.000000%, LoRA 95.652174%, delta -4.347826 pp
- escalation_recall: Base 7.575758%, LoRA 100.000000%, delta +92.424242 pp
- intent_accuracy: Base 28.000000%, LoRA 94.000000%, delta +66.000000 pp
- json_valid_rate: Base 99.333333%, LoRA 99.666667%, delta +0.333334 pp
- schema_compliance: Base 36.666667%, LoRA 99.333333%, delta +62.666666 pp

Base and LoRA each retained all 300 attempts. Stage C7 integrity is PASS.

## Gate-by-Gate Results

- **intent_accuracy**: PASS — Observed delta +66.000000 percentage points; evaluated against the frozen rule.
- **json_valid_rate**: PASS — Observed delta +0.333334 percentage points; evaluated against the frozen rule.
- **schema_compliance**: PASS — Observed delta +62.666666 percentage points; evaluated against the frozen rule.
- **category_accuracy**: PASS — Observed delta +37.333333 percentage points; evaluated against the frozen rule.
- **escalation_accuracy**: PASS — Observed delta +19.666667 percentage points; evaluated against the frozen rule.
- **critical_behavioral_regression**: PASS — Manual QA conclusion records no critical behavioral regression; isolated known issues are retained as limitations.
- **material_response_safety_regression**: PASS — Manual QA conclusion is PASS WITH KNOWN LIMITATIONS and records NO material response safety regression; automated screening is supporting evidence, not a complete quality judgment.
- **latency**: OPERATIONAL_WARNING — Candidate 01 substantially increases inference latency on the tested Apple Silicon environment.

## Structured Behavior Improvement

Locked evidence shows substantial gains in intent accuracy, category accuracy, JSON validity, and schema compliance under the frozen contract.

## Escalation Behavior

Base precision / recall / F1: 100.000000% / 7.575758% / 14.084507%. LoRA: 95.652174% / 100.000000% / 97.777778%.

Base precision of 100% does not establish better overall escalation behavior because Base positive-class recall is only 7.575758%. The frozen gate does not treat escalation precision alone as a blocker.

## Response Safety Review

Locked Manual Response QA: **PASS WITH KNOWN LIMITATIONS**. Material response safety regression: **NO**. Critical behavioral regression: **NO**. Automated screening is supporting evidence and is not a complete quality judgment.

## Operational Latency Trade-off

Base mean / median / p95: 1121.593 / 1078.237 / 1587.030 ms. LoRA: 3067.912 / 2717.730 / 4972.819 ms.

Candidate 01 substantially increases inference latency on the tested Apple Silicon environment. Latency is not a frozen promotion blocker.

## Remaining Failure Modes

QLoRA significantly improved structured classification behavior, but manual QA found that generated responses can still contain unsupported policy or capability claims. Therefore, the fine-tuned model should not be treated as an enterprise factual authority.

QLoRA 顯著改善 structured classification behavior，但人工 QA 發現生成式 response 仍可能產生 unsupported policy/capability claims，因此模型不應直接被視為企業 factual authority。

- one isolated generation degeneration/truncation case observed on Locked Test
- higher inference latency
- remaining taxonomy confusion around semantically close intents
- occasional unsupported action/capability claims
- occasional unsupported factual/policy details
- one unsupported guarantee signal
- unresolved placeholders
- verbose responses

## Deployment Scope

- structured classification
- schema-constrained routing
- escalation decision support
- intent classification
- category classification
- structured JSON generation
- customer-support workflow classification

## Deployment Constraints

- enterprise factual authority
- backend action executor
- authoritative refund/policy engine
- authoritative delivery/payment information source

External grounding, backend tools, and human review remain required as specified in `deployment_constraints.json`.

## Governance Decision

All frozen blocking gates passed. No training, Locked rerun, prompt/evaluator/threshold/candidate modification, checkpoint selection, or Stage C9 work occurred.

## Final Promotion Decision

Candidate 01: **PROMOTE**

PROMOTE does not represent unrestricted production approval. Candidate 01 is approved only for the documented structured classification and routing scope under the listed constraints.
