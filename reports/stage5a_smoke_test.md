# Stage C5A QLoRA Smoke Test

## Goal

Verify that the fixed MLX-LM QLoRA pipeline can load chat data, tokenize it, create and update LoRA layers, report finite losses, save an adapter, reload it in a new process, and generate non-empty output. This is not a behavioral evaluation.

## Model

- Base: `mlx-community/Qwen2.5-1.5B-Instruct-4bit`
- Revision: `8b403126fc14f14cfc99bb4cfa72ecbc129ea677`
- Architecture: `Qwen2ForCausalLM`
- Quantization: 4-bit, group size 64
- Fine-tune type: `lora` over a quantized base (QLoRA); no BF16 conversion or full fine-tuning

## Dataset

- Train smoke rows: 80 from frozen Train only
- Validation smoke rows: 20 from frozen Validation only
- Seed: 42
- Frozen source hashes and selected source indices are recorded in `artifacts/stage5a/dataset_preflight.json`.

## Chat Training Format

Each JSONL row contains exactly `messages` with roles `system`, `user`, and `assistant`. The assistant content is strict JSON serialization conforming to the C2 schema. The system content is the frozen C4 prompt; no per-row label is injected into the user message.

## QLoRA Configuration

- Layers: 8; rank: 8; scale: 16.0; dropout: 0.0
- Batch size: 1; gradient accumulation: 2; effective batch size: 2
- Iterations: 20; learning rate: 1e-5; maximum sequence length: 1024
- Config: `configs/qlora_smoke.yaml`

## Target Modules

Matched all seven requested module keys in each of the eight adapted transformer layers (56 module instances): `mlp.down_proj`, `mlp.gate_proj`, `mlp.up_proj`, `self_attn.k_proj`, `self_attn.o_proj`, `self_attn.q_proj`, `self_attn.v_proj`.

## Trainable Parameters

- Trainable: 2,637,824
- Total parameter count (quantization-aware MLX-LM calculation): 1,543,714,304
- Trainable percentage: 0.170875%

## Training Result

- Success: true
- Initial reported train loss: 2.514
- Final reported train loss: 1.651
- Finite losses: true
- Runtime: 53.249 seconds

Loss values are pipeline diagnostics only and are not evidence of model improvement.

## Validation Result

- Initial validation loss: 3.031
- Final validation loss: 1.646
- Validation batches: all 20 rows (`val_batches=-1`)

## Memory / Runtime

- Peak MLX memory: 3.470 GB
- Total training command duration: 53.249 seconds

## Adapter Save

- Path: `artifacts/stage5a/adapter`
- Files: `0000020_adapters.safetensors`, `adapter_config.json`, `adapters.safetensors`

## Adapter Reload

- New process: true
- Adapter load success: true

## Inference Reload Test

- Samples: 3
- Non-empty generation success: true
- Accuracy evaluated: false

## Dataset Boundary Validation

- Opened source content: `data/processed/train.jsonl`, `data/processed/validation.jsonl`
- Dev content accessed: no
- Locked Test content accessed: no
- Dev/Locked behavioral evaluation: none

## Limitations

Only 80/20 rows and 20 iterations were used. Loss direction and reload outputs do not establish response quality, accuracy, generalization, or candidate fitness. This smoke adapter is not a formal model candidate and must not be used for promotion.

## Stage C5A Conclusion

Stage C5A passed the mechanical QLoRA pipeline checks. Smoke Test loss and outputs do not represent behavioral improvement. Formal Stage C5 training was not started.
