# Customer Support QLoRA Fine-tuning & Evaluation

An end-to-end, governance-first fine-tuning project that adapts **Qwen2.5-1.5B-Instruct** with **QLoRA** for customer-support intent classification, category classification, structured JSON generation, and human-escalation decisions.

The promoted candidate improves Locked Test intent accuracy from **28.0% to 94.0%** while adapting only about **0.34%** of the model parameters. The project deliberately separates structured behavior adaptation from enterprise factual knowledge: the model learns the required taxonomy, schema, and routing policy, but it is not treated as a source of company facts or as an action-execution system.

## 1. Project Summary

The Base model can write plausible customer-support language, but it does not reliably follow a business-defined taxonomy, intent-to-category mapping, JSON contract, or escalation policy. Candidate 01 uses QLoRA to improve those structured behaviors under a controlled train/development/frozen-test lifecycle.

| Locked Test metric | Base | QLoRA | Delta |
|---|---:|---:|---:|
| Intent Accuracy | 28.0% | 94.0% | +66.0 pp |
| Category Accuracy | 61.7% | 99.0% | +37.3 pp |
| Schema Compliance | 36.7% | 99.3% | +62.7 pp |
| Escalation Accuracy | 79.0% | 98.7% | +19.7 pp |

**Candidate 01: PROMOTED for structured classification and routing, with documented deployment constraints.** Promotion does not mean unrestricted production approval.

## 2. Live Demo

### Public Vercel Demo

**[Open the public portfolio demo](https://customer-support-lora.vercel.app/)**

The public site presents eight curated examples using **pre-generated, real inference snapshots** from the frozen Base model and Candidate 01. Vercel reads static JSON; it does not run MLX or perform live model inference. Customer messages and model outputs are preserved as generated.

### Local Streamlit Live Demo

The local Streamlit application accepts new, free-form English customer-support messages and runs real Base-versus-QLoRA inference on Apple Silicon. See [Local Live Inference](#16-local-live-inference) for the required model download and launch steps.

## 3. Problem

General instruction-tuned models can generate fluent support responses without consistently respecting an enterprise contract. In this project, the Base model was unreliable on:

- the 27-value canonical intent taxonomy;
- deterministic intent-to-category mapping;
- exact JSON schema and enum constraints;
- the six-intent human-escalation policy.

The fine-tuning objective is therefore **structured behavior adaptation**: teach the model how to classify and route within a fixed contract. It is not an attempt to encode changing company policies, fees, delivery timelines, contact details, or live account state into model weights.

## 4. What I Built

- A deterministic dataset analysis and source-response quality gate.
- Leakage-resistant, group-aware train/validation/dev/Locked Test splits.
- A 4-bit Qwen2.5 Base benchmark and a parameter-efficient QLoRA training pipeline in MLX-LM.
- Strict JSON parsing, schema validation, taxonomy checks, escalation metrics, and error tagging.
- A freeze boundary covering the candidate, prompt, data, evaluator, policy, and promotion gate before Locked Test access.
- Automated response-risk screening plus human-review artifacts, with screening explicitly treated as heuristic evidence.
- A formal promotion decision with approved and prohibited deployment scopes.
- Two demos: local Streamlit live inference and a static Next.js/Vercel portfolio experience.

## 5. Locked Test Results

These are the existing Stage C7 results on all 300 frozen Locked Test rows. They were not recomputed for this README.

| Metric | Base | QLoRA | Delta |
|---|---:|---:|---:|
| Intent Accuracy | 28.0% | 94.0% | +66.0 pp |
| Category Accuracy | 61.7% | 99.0% | +37.3 pp |
| JSON Valid | 99.3% | 99.7% | +0.3 pp |
| Schema Compliance | 36.7% | 99.3% | +62.7 pp |
| Escalation Accuracy | 79.0% | 98.7% | +19.7 pp |
| Escalation F1 | 14.1% | 97.8% | +83.7 pp |

The Base model's high escalation precision did not indicate strong routing: its positive-class recall was only 7.6%. Candidate 01 reached 100.0% recall and 97.8% F1 under the frozen escalation policy. Full evidence is in the [Stage C7 report](reports/stage7_locked_evaluation.md) and [comparison artifact](artifacts/stage7/base_vs_lora_locked_comparison.json).

## 6. Base vs QLoRA Example

**Customer message**

> Please help me request a refund for my recent purchase.

**Expected:** `intent=get_refund`, `category=REFUND`, `needs_human=true`

| Model | Intent | Category | Needs human | JSON valid | Schema compliant |
|---|---|---|---:|---:|---:|
| Base | `refund` | `REFUND` | `false` | Yes | No |
| QLoRA Candidate 01 | `get_refund` | `REFUND` | `true` | Yes | Yes |

The QLoRA result follows the canonical taxonomy and the escalation contract; the Base output uses an invalid intent value and misses the required escalation. This example demonstrates structured behavior improvement only. It does **not** establish that every generated response is factually grounded.

The unedited response and raw JSON snapshots are available in [demo_cases.json](web/data/demo_cases.json).

## 7. System Architecture

This repository implements a fine-tuning and evaluation lifecycle rather than an application backend architecture:

```text
Customer Support Dataset
        ↓
Data Analysis → Source Response Quality Gate
        ↓
Group-aware Split
        ↓
Base Benchmark
        ↓
QLoRA Training
        ↓
Development Evaluation
        ↓
Freeze Candidate + Contract + Evaluator + Thresholds
        ↓
Locked Test (one post-freeze evaluation)
        ↓
Risk Screening + Manual QA
        ↓
Promotion Gate
        ↓
Streamlit Live Demo / Vercel Snapshot Demo
```

## 8. QLoRA Configuration

The following settings come from the frozen training configuration and execution artifacts:

| Setting | Value |
|---|---|
| Base model | `mlx-community/Qwen2.5-1.5B-Instruct-4bit` |
| Base revision | `8b403126fc14f14cfc99bb4cfa72ecbc129ea677` |
| Quantization | 4-bit, group size 64 |
| LoRA rank / scale | 8 / 16 |
| Adapted transformer layers | 16 of 28, layers 12–27 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Batch / gradient accumulation | 1 / 2 (effective batch 2) |
| Iterations | 1,350 |
| Learning rate | `1e-5` |
| Maximum sequence length | 1,024 |
| Prompt masking | `true` |
| Trainable parameters | 5,275,648 / 1,543,714,304 (0.34175%) |
| Peak memory | 4.742 GB |
| Training runtime | 2,088 seconds (about 34.8 minutes) |

QLoRA was chosen because it makes targeted behavior adaptation practical on Apple Silicon: only about 5.28M parameters are trainable while the 1.54B-parameter Base remains quantized. See the [training report](reports/stage5_formal_qlora_training.md) and [training manifest](artifacts/stage5/training_manifest.json).

## 9. Evaluation & Governance

| Split | Rows | Role |
|---|---:|---|
| Train | 2,700 | QLoRA parameter updates |
| Validation | 300 | Training-time validation |
| Dev | 300 | Candidate development evaluation and QA |
| Locked Test | 300 | Final post-freeze evidence only |

Splitting is group-aware using normalized instructions. Source-row, exact-instruction, normalized-instruction, and group overlap are all zero across splits. The source-response quality gate excludes incomplete source targets and deterministically replaces them with clean, same-intent unused rows.

Before Locked Test evaluation, Stage C6.5 froze:

- Candidate 01 and the exact Base revision;
- system prompt, tokenizer chat template, parser, and schema validator;
- taxonomy, category mapping, and escalation policy;
- evaluator implementation and metric aggregation;
- promotion thresholds and deterministic inference settings.

Locked Test behavioral evaluation occurred only after this freeze. Training and validation loss were monitored for training health; **loss was not a promotion criterion**. Promotion used the predeclared structured-behavior gates plus manual response QA. See the [freeze report](reports/stage6_5_freeze_report.md), [evaluation contract](reports/evaluation_contract.md), and [promotion report](reports/stage8_promotion_report.md).

## 10. Error Analysis

Candidate 01 reduced Locked Test wrong-intent errors from 216 to 18. Its largest remaining confusions were:

- `get_invoice` → `check_invoice` (4 cases)
- `set_up_shipping_address` → `change_shipping_address` (4 cases)
- `create_account` → `edit_account` (3 cases)

These are close semantic boundaries within the canonical taxonomy rather than arbitrary out-of-domain labels. The Locked run also observed one generation truncation/degeneration case and higher Candidate latency.

Automated screening flagged patterns such as unsupported action-completion or factual/policy phrasing and was used to prepare manual review. It is a conservative heuristic aid, not a comprehensive factual-correctness evaluator. Exact-match classification and schema metrics likewise do not replace human response-quality review.

## 11. Deployment Scope

Stage C8 promoted Candidate 01 for:

- structured classification;
- schema-constrained routing;
- escalation decision support;
- intent classification;
- category classification;
- structured JSON generation.

It is **not approved as**:

- an enterprise factual authority;
- a backend action executor;
- an authoritative refund or policy engine;
- an authoritative delivery or payment information source.

The formal scope is machine-readable in [promotion_decision.json](artifacts/stage8/promotion_decision.json) and [deployment_constraints.json](artifacts/stage8/deployment_constraints.json).

## 12. Limitations

> 「QLoRA 顯著改善 structured classification behavior，但人工 QA 發現生成式 response 仍可能產生 unsupported policy/capability claims，因此模型不應直接被視為企業 factual authority。」

External grounding is required for:

- company policies, payment methods, fees, and timelines;
- contact details;
- live refund or order status.

Real backend tools or APIs are required for:

- refund execution and order cancellation;
- address updates and account modification;
- live status lookup.

Additional known limitations include higher inference latency, residual confusion between semantically adjacent intents, occasional unsupported action/capability claims, unresolved placeholders, verbose responses, and one isolated Locked Test truncation case. Candidate 01 does not have unrestricted production approval.

## 13. Demo Options

### Public Vercel Demo

- URL: **https://customer-support-lora.vercel.app/**
- Eight curated customer-support cases.
- Pre-generated frozen Base and Candidate 01 inference snapshots.
- No model, GPU, backend inference, or free-text inference endpoint on Vercel.

### Local Streamlit Demo

- Accepts arbitrary English customer-support messages.
- Runs real free-form Base and QLoRA inference locally.
- Displays structured fields, schema status, response text, raw output, and latency side by side.
- Requires Apple Silicon, the Python environment, the exact cached Base model revision, and the repository adapter.

## 14. Tech Stack

| Area | Technology |
|---|---|
| AI / ML | Python, Qwen2.5, QLoRA, MLX, MLX-LM |
| Evaluation | JSON Schema, custom evaluator, automated risk screening, manual QA |
| Demo | Streamlit, Next.js, TypeScript, Tailwind CSS, Vercel |
| Engineering | Pytest, Git, GitHub |

## 15. Repository Structure

```text
.
├── artifacts/       # Training, evaluation, freeze, QA, and promotion artifacts
├── configs/         # Taxonomy, schema, escalation, inference, QLoRA, and gate configs
├── data/
│   ├── manifests/   # Split membership, hashes, and provenance
│   ├── processed/   # Frozen train/validation/dev/Locked Test splits
│   └── training/    # MLX-LM formatted training inputs
├── demo/            # Local Streamlit Live Demo
├── prompts/         # Frozen system prompt
├── reports/         # Human-readable reports for each lifecycle boundary
├── src/
│   ├── data/        # Analysis and deterministic split construction
│   ├── evaluation/  # Parsers, validators, evaluation, freeze, and promotion logic
│   ├── training/    # Smoke and formal QLoRA entry points
│   └── demo/        # Frozen local inference and snapshot export helpers
├── tests/           # Pytest coverage for contracts and pipeline invariants
└── web/             # Public Vercel Portfolio Demo (static Next.js application)
```

The Candidate 01 adapter is committed under `artifacts/stage5/candidate_01/adapter/`. Downloaded Base weights, virtual environments, build output, caches, environment files, and OS metadata are excluded from version control.

## 16. Local Live Inference

### Requirements

- macOS on **Apple Silicon** (`arm64`); MLX dependencies are conditionally installed only on Darwin/arm64.
- Python **3.9 or newer**. The frozen run used Python 3.9.6, MLX 0.29.3, and MLX-LM 0.29.1.
- Enough disk space for the external 4-bit Base model snapshot.
- Network access for the one-time Base model download.

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The application intentionally resolves the exact Base revision with `local_files_only=True`; it will **not** silently download a different or newer model at launch. Pre-download the frozen revision into the Hugging Face cache:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('mlx-community/Qwen2.5-1.5B-Instruct-4bit', revision='8b403126fc14f14cfc99bb4cfa72ecbc129ea677')"
```

Then start the live comparison:

```bash
python -m streamlit run demo/app.py
```

The Candidate 01 adapter is already stored in this repository and requires no separate download. Git LFS is not currently required for the adapter files. The Base model weights are not committed and must remain in the external Hugging Face cache. A fresh clone is therefore **not** immediately inference-ready until dependencies and the exact Base snapshot are installed.

## 17. Reproducibility

- Base revision: `8b403126fc14f14cfc99bb4cfa72ecbc129ea677`
- Final adapter SHA-256: `da763e47f3c6051defb605345e9aaccd989a8768b804c802606a7f8317fc2c16`
- Prompt SHA-256: `6b84135769b7348758e8cc21a3cb168465e00de5efaf59ff8a8459087db3dc3b`
- Train split: `ce35cd9ff927521a9ff5c2454b16a0012b22aa232c5c33c9b6a857f6cc57bf28`
- Validation split: `d9a2035ebebb2eb739ecb7e5bc6d589927a0359416b0a05dde5e793a39410175`
- Dev split: `a0859497b5fe23ca1adf1ab1e6a9b7da5dfca1bbcd6519c89ab7ea4f21a5b4d6`
- Locked Test split: `b7f7af8c5e366c743fafd68c8c8f3e7a2b101dfce53e63bf1f7a8ead0bce1fac`
- Inference: temperature 0, greedy decoding, seed 42, 512 generated-token limit, concurrency 1, same tokenizer/chat template/parser/schema for Base and Candidate.

Detailed provenance is retained in [dataset_hashes.json](data/manifests/dataset_hashes.json), [frozen_component_hashes.json](artifacts/stage6_5/frozen_component_hashes.json), and [frozen_inference_contract.json](artifacts/stage6_5/frozen_inference_contract.json).

## 18. Future Work

- Add Chinese zero-shot evaluation before making multilingual claims.
- Train a Chinese QLoRA variant if evaluation shows that adaptation is needed.
- Add external grounding or RAG for company facts and policy-backed responses.
- Integrate authenticated backend tools for real customer actions.
- Optimize inference latency while preserving the frozen behavioral contract.
