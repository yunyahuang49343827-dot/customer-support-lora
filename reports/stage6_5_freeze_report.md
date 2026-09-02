# Stage C6.5 Freeze Report

## Goal

Create the immutable boundary that Stage C7 must use for Base versus Candidate 01 Locked Evaluation.

## Freeze Boundary

Freeze status: **PASS**. Model, adapter, prompt, decoding, parser, evaluator, schema, taxonomies, escalation policy, datasets, response-QA rubric, and promotion gate are fixed by the hashes in this report.

## Candidate 01

- Candidate: `candidate_01`
- Adapter: `artifacts/stage5/candidate_01/adapter`
- Fresh load success: `true`
- Generation performed during load validation: `false`

## Base Model Revision

- Model: `mlx-community/Qwen2.5-1.5B-Instruct-4bit`
- Revision: `8b403126fc14f14cfc99bb4cfa72ecbc129ea677`
- Architecture: `Qwen2ForCausalLM`

## Adapter Integrity

- Files: 11
- Total bytes: 211267682
- `adapter_config.json`: `ad7d87f1e16288d89a80b151a84524475297ebbaf320f73a0d5201dfdc10a91b`
- `adapters.safetensors`: `da763e47f3c6051defb605345e9aaccd989a8768b804c802606a7f8317fc2c16`
- Final checkpoint: `da763e47f3c6051defb605345e9aaccd989a8768b804c802606a7f8317fc2c16`

## Prompt Integrity

- Path: `prompts/base_system_prompt.txt`
- SHA-256: `6b84135769b7348758e8cc21a3cb168465e00de5efaf59ff8a8459087db3dc3b`
- Size: 1852 bytes

## Inference Contract

Greedy deterministic decoding, temperature 0, seed 42, maximum 512 generated tokens, concurrency 1, identical tokenizer chat template, strict JSON parser/extraction, and the same evaluator are frozen in `artifacts/stage6_5/frozen_inference_contract.json`.

## Evaluation Contract

Primary classification/format/escalation metrics, positive-class escalation precision/recall/F1/confusion counts, mean/median/p95 latency, and the complete C4 error taxonomy are frozen. Evaluator hashes:

- `src/evaluation/base_baseline.py`: `184ca998f1a29dcf99cc4bc48788d09ad0c177314a16ac7eb4f21c0caf64fb52`
- `src/evaluation/contracts.py`: `e2f8bb620a3b7d44f98c5ca0a96e985d98b944fe7d68b0a265cd26aa425a31a3`
- `src/evaluation/development_evaluation.py`: `2cb2f7f37f4b5b03837eb8b2cec17355a3df77cdc1df07939f990bfb38ba9a37`

## Schema / Taxonomy

The strict JSON Schema Draft 2020-12 contract, exactly 27 intents, exactly 11 categories, and deterministic intent-to-category mapping are frozen:

- `configs/output_schema.json`: `6a3d0900b3485e5a24205ea5f7ae42360d598a6c7a7fc6d97cde2d8fde88daa2`
- `configs/intent_taxonomy.json`: `8e99fdfcdd90a2bcc2dd733503e936d5f0785ef4548468fa5923b4d965e3422f`
- `configs/category_taxonomy.json`: `694f2c4a56fe662d795a1315781ed7c86f68114012ea0012ead43cefc4a5ba79`
- `configs/escalation_policy.json`: `c07898c29254bc584c944007bc2fd2785c9db1e70fedda0aeb7c0ec7c2ef0f2d`

## Escalation Policy

The six true intents remain exactly `complaint`, `contact_customer_service`, `contact_human_agent`, `delete_account`, `get_refund`, and `payment_issue`; the other 21 intents remain false.

## Dataset Hashes

- train: `ce35cd9ff927521a9ff5c2454b16a0012b22aa232c5c33c9b6a857f6cc57bf28` (2700 rows)
- validation: `d9a2035ebebb2eb739ecb7e5bc6d589927a0359416b0a05dde5e793a39410175` (300 rows)
- dev: `a0859497b5fe23ca1adf1ab1e6a9b7da5dfca1bbcd6519c89ab7ea4f21a5b4d6` (300 rows)
- locked_test: `b7f7af8c5e366c743fafd68c8c8f3e7a2b101dfce53e63bf1f7a8ead0bce1fac` (300 rows)

Locked data access mode was raw binary SHA-256 plus newline count only. No JSON record was parsed, printed, sampled, evaluated, or used for inference.

## Promotion Gate

Intent improvement ≥3 pp; JSON Valid and Schema Compliance regression ≤1 pp; Category or Escalation Accuracy drop ≥3 pp is material; no critical behavioral regression; no material response-safety regression. Latency remains operational context and is not a fixed promotion blocker. Thresholds may not change after Locked results.

## Known Response Limitations

QLoRA significantly improved structured classification behavior, but manual QA found that generated responses can still contain unsupported policy or capability claims. Therefore, the fine-tuned model should not be treated as an enterprise factual authority.

QLoRA 顯著改善 structured classification behavior，但人工 QA 發現生成式 response 仍可能產生 unsupported policy/capability claims，因此模型不應直接被視為企業 factual authority。

This limitation is frozen as deployment guidance, not as a retraining trigger. User confirmation approved Candidate 01 for freeze; no human worksheet scores were synthesized or backfilled.

## Locked Test Protection

- Locked content accessed for evaluation: false
- Locked inference performed: false
- Locked semantic parsing or analysis: false

## Validation Results

- `dataset_train_sha256`: **PASS**
- `dataset_train_row_count`: **PASS**
- `dataset_validation_sha256`: **PASS**
- `dataset_validation_row_count`: **PASS**
- `dataset_dev_sha256`: **PASS**
- `dataset_dev_row_count`: **PASS**
- `dataset_locked_test_sha256`: **PASS**
- `dataset_locked_test_row_count`: **PASS**
- `candidate_adapter_exists`: **PASS**
- `adapter_file_count`: **PASS**
- `adapter_hash_generated`: **PASS**
- `final_checkpoint_matches_final_adapter`: **PASS**
- `adapter_fresh_reload_success`: **PASS**
- `adapter_unchanged_after_reload`: **PASS**
- `adapter_lora_tensor_count`: **PASS**
- `model_revision`: **PASS**
- `model_architecture`: **PASS**
- `model_quantization`: **PASS**
- `stage5_model_revision`: **PASS**
- `stage6_model_revision`: **PASS**
- `qlora_model`: **PASS**
- `qlora_adapter_path`: **PASS**
- `qlora_num_layers`: **PASS**
- `qlora_rank`: **PASS**
- `prompt_sha256`: **PASS**
- `inference_temperature`: **PASS**
- `inference_seed`: **PASS**
- `inference_max_generated_tokens`: **PASS**
- `inference_concurrency`: **PASS**
- `base_lora_inference_contract_consistency`: **PASS**
- `intent_count`: **PASS**
- `category_count`: **PASS**
- `schema_intent_count`: **PASS**
- `schema_category_count`: **PASS**
- `schema_draft`: **PASS**
- `taxonomy_schema_intents_equal`: **PASS**
- `taxonomy_schema_categories_equal`: **PASS**
- `escalation_true_intents`: **PASS**
- `escalation_true_count`: **PASS**
- `escalation_false_count`: **PASS**
- `promotion_intent_threshold`: **PASS**
- `promotion_json_regression`: **PASS**
- `promotion_schema_regression`: **PASS**
- `promotion_category_material_drop`: **PASS**
- `promotion_escalation_material_drop`: **PASS**
- `promotion_critical_regression_allowed`: **PASS**
- `promotion_latency_blocker`: **PASS**
- `evaluator_hashes_recorded`: **PASS**
- `manual_qa_approval`: **PASS**
- `locked_content_parsed`: **PASS**
- `locked_inference_performed`: **PASS**
- `stage7_outputs`: **PASS**
- `candidate02_outputs`: **PASS**
- `training_performed_during_freeze`: **PASS**

## Freeze Decision

Stage C6.5 = **PASS**.

C7 may proceed only if Stage C6.5 = PASS.
