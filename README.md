# Customer Support QLoRA Fine-tuning & Evaluation

**客服分類 QLoRA 微調與評估**

本專案以 **Qwen2.5-1.5B-Instruct + QLoRA** 改善 Customer Support 的 Intent Classification、Category Classification、Structured JSON 與 Human Escalation。重點不是讓模型背誦企業知識，而是讓它穩定遵守可驗證的分類、Schema 與轉介規則。

## 1. 專案簡介

Base Model 能生成流暢的客服文字，但不穩定遵守企業定義的 taxonomy 與輸出契約。Candidate 01 經過 group-aware data split、QLoRA training、Dev Evaluation、Freeze、Locked Test 與 Promotion Gate 後，獲准用於 structured classification & routing。

Locked Test 主要結果：

- Intent Accuracy：**28.0% → 94.0%**
- Category Accuracy：**61.7% → 99.0%**
- Schema Compliance：**36.7% → 99.3%**
- Escalation Accuracy：**79.0% → 98.7%**

**Candidate 01：PROMOTED for structured classification & routing.** Promotion 不代表 unrestricted production approval。

## 2. Live Demo

### Public Vercel Demo

**[開啟公開 Demo](https://customer-support-lora.vercel.app/)**

公開網站展示 8 個 curated examples，內容為 Frozen Base / Candidate 01 預先產生的真實 inference snapshots。Vercel 只讀取靜態 JSON，**不是 live model inference**，也不接受自由輸入。

### Local Streamlit Demo

本機版支援自由輸入新的英文 Customer Support 問題，並實際執行 Base 與 QLoRA Candidate 01 inference。安裝與啟動方式請見 [Local Live Inference](#local-live-inference)。

## 3. 問題與目標

Base Model 雖具備一般客服語言能力，但無法穩定遵守企業自訂的：

- 27-value Intent Taxonomy
- Category Mapping
- JSON Schema
- Human Escalation Policy

本專案 Fine-tuning 的目標是 **structured behavior adaptation**，不是把公司政策、fees、delivery timelines、contact details 或 live account state 寫入 model weights。會變動的企業事實仍需 external grounding；真實操作仍需 backend tools。

## 4. QLoRA 微調與評估流程

<!-- TODO: Add QLoRA workflow diagram here -->

### ① Data

Dataset Analysis → Source Quality Check → Group-aware Split

### ② Development

Base Benchmark → QLoRA Training → Dev Evaluation

### ③ Frozen Evaluation

Freeze → Locked Test → Risk Screening + Manual QA

### ④ Decision

Promotion Gate → Candidate 01 PROMOTED → Demo

## 5. 資料處理與 Leakage Prevention

客服資料含有大量高度相似、只替換少數詞彙的 customer instructions。若直接使用一般 Random Split，近似問句可能同時進入 Train 與 Test，造成 evaluation leakage 與虛高結果。

本專案以 **normalized instruction + group-aware split** 將相似問句綁定於同一 group，確保不跨 split。最終 source row、exact instruction、normalized instruction 與 group overlap 均為 0；不完整的 source response 也會先經 deterministic quality gate 排除。

| Split | Rows | 用途 | 是否可用於調整 |
|---|---:|---|---|
| Train | 2,700 | QLoRA training | Yes |
| Validation | 300 | Training diagnostics | Yes |
| Dev | 300 | Development evaluation / QA | Yes |
| Locked Test | 300 | Final post-freeze evaluation | No |

詳細資料請見 [Split Validation Report](reports/split_validation_report.md) 與 [dataset manifests](data/manifests/)。

## 6. QLoRA 訓練設定

| 設定 | 值 |
|---|---|
| Base Model | `Qwen2.5-1.5B-Instruct-4bit` |
| LoRA Rank / Scale | 8 / 16 |
| Adapted Layers | 16 |
| Target Modules | q/k/v/o + gate/up/down |
| Learning Rate | `1e-5` |
| Iterations | 1,350 |
| Trainable Parameters | 5.28M / 1.54B（約 0.34%） |
| Peak Memory | 4.74 GB |
| Training Time | 約 34.8 分鐘 |

QLoRA 讓本專案在 Apple Silicon 上，只調整約 0.34% 參數即可完成 targeted behavior adaptation。Base revision、sequence length、prompt masking 與完整 provenance 請見 [Stage C5 Training Report](reports/stage5_formal_qlora_training.md)。

## 7. Locked Test Results

以下數值來自既有 Stage C7 artifact；本次 README 更新未重新執行 Locked Test。

| Metric | Base | QLoRA | Delta |
|---|---:|---:|---:|
| Intent Accuracy | 28.0% | 94.0% | +66.0 pp |
| Category Accuracy | 61.7% | 99.0% | +37.3 pp |
| JSON Valid | 99.3% | 99.7% | +0.3 pp |
| Schema Compliance | 36.7% | 99.3% | +62.7 pp |
| Escalation Accuracy | 79.0% | 98.7% | +19.7 pp |
| Escalation F1 | 14.1% | 97.8% | +83.7 pp |

Base 的 Escalation Precision 雖為 100%，Recall 卻只有 7.6%；Candidate 01 的 Recall 為 100%，F1 為 97.8%。這說明 Human Escalation 不能只看單一 Precision 指標。

詳細資料請見 [Stage C7 Locked Evaluation](reports/stage7_locked_evaluation.md) 與 [Base vs QLoRA comparison artifact](artifacts/stage7/base_vs_lora_locked_comparison.json)。

## 8. Base vs QLoRA Example

**Customer Message**

> Please help me request a refund for my recent purchase.

| 項目 | Expected | Base | QLoRA |
|---|---|---|---|
| Intent | `get_refund` | `refund` | `get_refund` |
| Category | `REFUND` | `REFUND` | `REFUND` |
| `needs_human` | `true` | `false` | `true` |
| JSON Valid | — | Yes | Yes |
| Schema Compliance | — | No | Yes |

QLoRA 在此案例中修正 canonical Intent、Schema Compliance 與 Human Escalation。此案例證明 structured behavior 改善，不代表生成式 response factuality 已完全解決。

未修改的 response 與 raw JSON 請見 [demo_cases.json](web/data/demo_cases.json)。

## 9. Evaluation / Freeze / Promotion

### Dev Evaluation

Dev 用於 behavioral metrics、QA、error analysis 與 controlled iteration，不作為最終泛化結果。

### Freeze

進入 Locked Test 前固定以下項目：

- Candidate 與 Base revision
- Prompt、Parser 與 Schema
- Taxonomy 與 Escalation Policy
- Evaluator 與 Promotion Thresholds

### Locked Test / Promotion

Locked Test 共 300 rows，僅在 Freeze 後執行；結果不用於 retraining、prompt tuning 或 checkpoint selection。Promotion 不以 training / validation loss 作為 criterion，而是依預先固定的 behavioral gates 與 Manual QA 結論判定。

詳細資料請見 [Stage C6.5 Freeze Report](reports/stage6_5_freeze_report.md)、[Evaluation Contract](reports/evaluation_contract.md) 與 [Stage C8 Promotion Report](reports/stage8_promotion_report.md)。

## 10. Error Analysis

Candidate 01 在 Locked Test 最主要的 Intent confusion 為：

- `get_invoice` → `check_invoice`：4
- `set_up_shipping_address` → `change_shipping_address`：4
- `create_account` → `edit_account`：3

錯誤主要集中於語義高度相近的 taxonomy boundaries。Manual QA 與 automated Risk Screening 另發現 unsupported action/capability claims、偶發 unsupported factual/policy details、單一 truncation case，以及較高 inference latency。Risk Screening 是輔助人工檢查的 heuristic，不代表全面 factual correctness 判定。

## 11. Deployment Scope & Limitations

### Approved Scope

- Intent Classification
- Category Classification
- Schema-constrained Routing
- Structured JSON
- Escalation Decision Support

### Not Approved As

- Enterprise Factual Authority
- Backend Action Executor
- Authoritative Policy Engine
- Authoritative Delivery / Payment Source

### Important Limitation

> 「QLoRA 顯著改善 structured classification behavior，但人工 QA 發現生成式 response 仍可能產生 unsupported policy/capability claims，因此模型不應直接被視為企業 factual authority。」

External grounding required for：

- company policies、payment methods、fees、timelines
- contact details
- live refund/order status

Backend tools required for：

- refund execution、order cancellation
- address update、account modification
- live status lookup

完整核准範圍與限制請見 [promotion_decision.json](artifacts/stage8/promotion_decision.json) 與 [deployment_constraints.json](artifacts/stage8/deployment_constraints.json)。Candidate 01 不具 unrestricted production approval。

## 12. Demo Preview

<!-- TODO: Add Vercel Demo screenshot here -->

Public Demo：**https://customer-support-lora.vercel.app/**

## 13. Tech Stack

| 領域 | Technology |
|---|---|
| AI / ML | Python, Qwen2.5, QLoRA, MLX, MLX-LM |
| Evaluation | JSON Schema, Custom Evaluator, Risk Screening, Manual QA |
| Demo | Streamlit, Next.js, TypeScript, Tailwind CSS, Vercel |
| Engineering | Pytest, Git, GitHub |

<a id="local-live-inference"></a>

## 14. Local Live Inference

### 環境需求

- macOS on **Apple Silicon** (`arm64`)
- Python **3.9 以上**；frozen run 使用 Python 3.9.6、MLX 0.29.3、MLX-LM 0.29.1
- 足夠空間存放外部 4-bit Base Model snapshot
- 首次下載 Base Model 時需要網路

在 repository root 執行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

程式以 `local_files_only=True` 解析 exact Base revision，因此啟動時不會自動下載不同或較新的模型。先將 frozen revision 下載至 Hugging Face cache：

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('mlx-community/Qwen2.5-1.5B-Instruct-4bit', revision='8b403126fc14f14cfc99bb4cfa72ecbc129ea677')"
```

啟動 Local Streamlit Demo：

```bash
python -m streamlit run demo/app.py
```

Candidate 01 adapter 已存放於 repository，不需另外下載，也未使用 Git LFS。Base Model weights 不在 repository 內；完成 dependencies 與 exact Base snapshot 安裝前，fresh clone 無法直接執行 inference。

## 15. Repository Structure

```text
artifacts/  # Training / evaluation / freeze / QA / promotion artifacts
configs/    # Taxonomy, Schema, escalation, inference 與 QLoRA configs
data/       # Frozen splits、training data 與 manifests
demo/       # Local Streamlit Live Demo
prompts/    # Frozen system prompt
reports/    # 各階段 evidence 與決策報告
src/        # Data、training、evaluation 與 demo implementation
tests/      # Pytest contract 與 pipeline tests
web/        # Public Vercel Portfolio Demo
```

## 16. Reproducibility

- Base revision：`8b403126fc14f14cfc99bb4cfa72ecbc129ea677`
- Adapter SHA-256：`da763e47f3c6051defb605345e9aaccd989a8768b804c802606a7f8317fc2c16`
- Prompt SHA-256：`6b84135769b7348758e8cc21a3cb168465e00de5efaf59ff8a8459087db3dc3b`
- Deterministic inference：temperature 0、greedy decoding、seed 42、max 512 generated tokens、concurrency 1；Base 與 Candidate 使用相同 tokenizer、chat template、Parser 與 Schema。

完整 dataset / frozen component hashes 請見 [data/manifests/](data/manifests/) 與 [artifacts/stage6_5/](artifacts/stage6_5/)。

## 17. Future Work

- Chinese zero-shot evaluation
- Chinese QLoRA（如評估顯示確有需要）
- RAG / external grounding for factual responses
- Backend tools for real customer actions
- Inference latency optimization
