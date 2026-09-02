# Stage C9 Simple Base vs QLoRA Comparison Demo

## Demo Goal

Provide a small local portfolio demonstration of the structured-behavior difference between the frozen Base model and promoted QLoRA Candidate 01. It is not a complete customer-support product and does not execute backend actions.

## Architecture

- Streamlit user interface: `demo/app.py`
- Frozen inference and validation helpers: `src/demo/comparison.py`
- Base: `mlx-community/Qwen2.5-1.5B-Instruct-4bit` at revision `8b403126fc14f14cfc99bb4cfa72ecbc129ea677`, with no adapter
- Candidate: the same Base snapshot plus `artifacts/stage5/candidate_01/adapter`
- Both sides use the same frozen system prompt, tokenizer, chat template, deterministic temperature-0 decoding, 512-token limit, parser, and schema validator
- Streamlit keeps Base and Candidate in separate resource caches so adapter state cannot be confused

The UI reads the existing Stage C7 benchmark and Stage C8 promotion/deployment artifacts. It does not automatically run Locked Test inference.

## How to Run

From the repository root:

```bash
.venv/bin/streamlit run demo/app.py
```

The exact frozen model revision must already be present in the local Hugging Face cache. If the model, prompt, or adapter is unavailable or fails integrity validation, the app displays a short error instead of a traceback.

## What the Viewer Should Observe

1. Select one of eight newly written examples or enter a free-form customer message.
2. Click **Compare Base vs QLoRA**.
3. Compare Intent, Category, `needs_human`, JSON validity, schema compliance, response text, raw output, and latency side by side.
4. For free-form messages, same/different markers compare the model outputs without claiming either is correct.
5. Curated examples show their explicitly defined expected labels.
6. Review the frozen Locked benchmark, promotion status, deployment constraints, and known limitations below the live comparison.

## Limitations

- Generated responses may still contain unsupported policy or capability claims.
- Company facts and policies require external grounding.
- Refunds, cancellations, address updates, account changes, and live lookups require real backend tools or APIs.
- One isolated generation degeneration/truncation was observed during Locked evaluation.
- Candidate 01 has higher inference latency on the tested Apple Silicon environment.
- Semantically similar intents can still be confused.
- `PROMOTE` does not mean unrestricted production approval, and Candidate 01 is not approved as an enterprise factual authority.

## Frozen Artifact References

- `artifacts/stage6_5/frozen_inference_contract.json`
- `prompts/base_system_prompt.txt`
- `artifacts/stage5/candidate_01/adapter/adapters.safetensors`
- `artifacts/stage7/base_vs_lora_locked_comparison.json`
- `artifacts/stage8/promotion_decision.json`
- `artifacts/stage8/deployment_constraints.json`

Stage C9 performs no training and modifies no frozen artifact.
