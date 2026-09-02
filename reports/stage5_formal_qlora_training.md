# Stage C5 Formal QLoRA Training

## Goal

Train the first reproducible formal QLoRA Candidate 01 on the complete frozen Train set, use frozen Validation only for optimization diagnostics, save the adapter, and verify reload in a fresh process. Stage C5 does not evaluate behavioral improvement.

## Base Model

- `mlx-community/Qwen2.5-1.5B-Instruct-4bit` at revision `8b403126fc14f14cfc99bb4cfa72ecbc129ea677`
- `Qwen2ForCausalLM`, 4-bit quantization, group size 64
- Quantized Base + LoRA; no BF16 conversion, full fine-tuning, or alternative implementation

## Dataset

- Train: 2,700 rows; SHA-256 `ce35cd9ff927521a9ff5c2454b16a0012b22aa232c5c33c9b6a857f6cc57bf28`
- Validation: 300 rows; SHA-256 `d9a2035ebebb2eb739ecb7e5bc6d589927a0359416b0a05dde5e793a39410175`
- Converted data: `data/training/train.jsonl`, `data/training/valid.jsonl`

## Sequence Length Audit

- Train: min 422, mean 530.748, median 506.000, p90 632.100, p95 674.050, p99 782.000, max 851
- Train >1024: 0 (0.000000%)
- Train >1536: 0 (0.000000%)
- Validation >1024: 0 (0.000000%)
- Selected `max_seq_length`: 1024 according to the fixed >1% Train rule

## Training Format

MLX-LM chat JSONL with exact system/user/assistant roles. The assistant target is strict C2 JSON. The C4 frozen prompt is reused unchanged with SHA-256 `6b84135769b7348758e8cc21a3cb168465e00de5efaf59ff8a8459087db3dc3b`.

## QLoRA Configuration

- Candidate: 01; `mask_prompt = true`
- Rank 8, scale 16.0, dropout 0.0; learning rate 1e-5
- Physical batch 1, gradient accumulation 2, effective batch 2
- 1,350 iterations; reports every 10; 50 validation batches every 150; checkpoints every 150

## Difference from Smoke Test

Formal training uses all 2,700/300 rows, `mask_prompt=true`, 16 adapted layers, and 1,350 iterations. The C5A smoke adapter remains separate and is not overwritten or treated as a candidate.

## Target Modules

All seven requested keys matched with no fallback: `mlp.down_proj`, `mlp.gate_proj`, `mlp.up_proj`, `self_attn.k_proj`, `self_attn.o_proj`, `self_attn.q_proj`, `self_attn.v_proj`.

## Adapted Layers

Last 16 of 28 transformer layers, indices 12–27; 112 matched module instances.

## Trainable Parameters

- Trainable: 5,275,648
- Total (MLX-LM quantization-aware): 1,543,714,304
- Trainable percentage: 0.341750%

## Training Loss

- Initial reported: 1.515
- Final reported: 0.657
- Minimum reported: 0.525
- All recorded losses finite: true

## Validation Loss

iter 1: 1.536, iter 150: 0.951, iter 300: 0.853, iter 450: 0.812, iter 600: 0.809, iter 750: 0.772, iter 900: 0.759, iter 1050: 0.750, iter 1200: 0.757, iter 1350: 0.709

Final reported validation loss: 0.709. Lower validation loss does not prove behavioral improvement.

## Runtime

- Total official training command: 2088.066 seconds
- Mean throughput: 108.335 tokens/sec; 0.779 iterations/sec

## Peak Memory

- 4.742 GB

## Checkpoints

`0000150_adapters.safetensors`, `0000300_adapters.safetensors`, `0000450_adapters.safetensors`, `0000600_adapters.safetensors`, `0000750_adapters.safetensors`, `0000900_adapters.safetensors`, `0001050_adapters.safetensors`, `0001200_adapters.safetensors`, `0001350_adapters.safetensors`

No checkpoint was behaviorally selected in Stage C5.

## Final Adapter

- `artifacts/stage5/candidate_01/adapter/adapters.safetensors`
- Adapter bytes: 21,126,646

## Adapter Reload

- New process: true
- Load success: true
- Non-empty generation: true (3 Validation-derived inputs)
- Accuracy evaluation: false

## Dataset Boundary Validation

- Dev content accessed: no
- Locked Test content accessed: no
- Dev/Locked inference or behavioral metrics: none

## Limitations

Loss curves are optimization diagnostics only. Stage C5 does not establish intent accuracy, JSON/schema improvement, response quality, superiority to Base, checkpoint preference, or promotion readiness.

## Stage C5 Conclusion

Stage C5 only proves that formal QLoRA training completed successfully and its adapter can be reloaded. Behavioral improvement remains unknown until Stage C6. Stage C6 was not started.
