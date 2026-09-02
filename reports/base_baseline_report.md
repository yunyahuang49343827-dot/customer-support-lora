# Stage C4 Base Model Development Baseline

## Model

- Repository: `mlx-community/Qwen2.5-1.5B-Instruct-4bit`
- Revision: `8b403126fc14f14cfc99bb4cfa72ecbc129ea677`
- Load success: `true`
- Architecture: `Qwen2ForCausalLM`
- Quantization: 4-bit, group size 64
- Parameter size metadata: 1.5B
- Adapter: none

## Environment

- Python: `3.9.6`
- MLX: `0.29.3`
- MLX-LM: `0.29.1`
- Platform: `macOS-26.6.2-arm64-arm-64bit`

## Frozen Prompt

The system prompt is frozen at `prompts/base_system_prompt.txt` with SHA-256 `6b84135769b7348758e8cc21a3cb168465e00de5efaf59ff8a8459087db3dc3b`. It contains the complete 27-intent and 11-category vocabularies, the four-key JSON contract, and response safety constraints. Ground-truth labels are not included in model inputs. The escalation mapping is not enumerated.

## Inference Configuration

Greedy deterministic decoding (`temperature=0`), seed 42, maximum 512 generated tokens, concurrency 1, no adapters, and no per-example decoding changes.

## Dev Dataset

Evaluated all 300 frozen Dev rows. Dev SHA-256: `a0859497b5fe23ca1adf1ab1e6a9b7da5dfca1bbcd6519c89ab7ea4f21a5b4d6`. Model input contains only the frozen system prompt and customer instruction. The final-evaluation dataset content was not opened or evaluated.

## Primary Metrics

- Intent Accuracy: 31.333333%
- JSON Valid Rate: 99.000000%
- Schema Compliance: 33.000000%

## Secondary Metrics

- Category Accuracy: 61.000000%
- Escalation Accuracy: 78.333333%
- Escalation false negatives: 62
- Escalation false positives: 0

## Latency

- Mean: 1235.668 ms
- Median: 1190.411 ms
- p95: 1802.525 ms
- Samples: 300

## Error Breakdown

- `wrong_intent`: 206
- `invalid_enum`: 152
- `wrong_category`: 117
- `wrong_needs_human`: 65
- `intent_category_mismatch`: 46
- `invalid_json`: 3
- `generation_truncated`: 1

## Intent Confusions

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
- `recover_password` → `set_up_shipping_address`: 2
- `track_refund` → `check_cancellation_fee`: 2

The full canonical confusion table is `artifacts/stage4/intent_confusion.csv`.

## Manual QA Required

`artifacts/stage4/manual_qa_samples.csv` contains 30 deterministic seed-42 cases sampled across correct outputs, wrong intents, schema failures, and both escalation labels.

> **這一步需要你手動做**
>
> Review `artifacts/stage4/manual_qa_samples.csv` for relevance, unsupported action claims, fabricated policy, safety, unnecessary escalation, and missing escalation. Do not alter the frozen prompt or rerun this baseline to improve its score.

## Limitations

Automated metrics measure exact labels and output-contract behavior, not response relevance. Latency is specific to this machine and sequential MLX execution. This is a Base development baseline only; no LoRA comparison or improvement claim is made.

## Stage C4 Conclusion

The frozen Base Model baseline completed on all 300 Dev rows with a fixed prompt and deterministic decoding. No adapter, training, model modification, prompt optimization, or final-evaluation behavioral use occurred. Stage C5 was not started.
