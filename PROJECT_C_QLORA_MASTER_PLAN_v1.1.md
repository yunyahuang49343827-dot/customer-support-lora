# Project C｜Customer Support QLoRA Fine-tuning & Evaluation

Version: **v1.1**

---

## 1. Project Goal

建立一個小型、可重現的 QLoRA Fine-tuning 實驗，回答：

> QLoRA 是否能讓 Base Model 在 Customer Support 任務上，更準確辨識 Intent、更穩定遵循指定 JSON Schema，並維持合理的客服回應品質？

工程目標是：

- 真的把 Fine-tuning 做好
- 透過 Development Evaluation 找問題並有限度迭代
- 最後使用 Frozen Locked Test 驗證改善是否成立

核心原則：

> Lower training / validation loss 不等於 Fine-tuning 成功。  
> 最終結論只能由 Locked Base vs LoRA Evaluation + Promotion Gate 決定。

---

## 2. Dataset

Hugging Face Dataset：

`bitext/Bitext-customer-support-llm-chatbot-training-dataset`

主要欄位：

- `instruction`
- `category`
- `intent`
- `response`
- `flags`

第一版不要直接使用全部資料。

預計規模：

- Train：約 2,000–3,000
- Validation：約 250–350
- Dev：約 250–350
- Locked Test：約 250–350

實際數量依 Stage C1 分析結果決定。

---

## 3. Fine-tuning Task

Input：

```text
Customer message
```

Target Output：

```json
{
  "intent": "<allowed intent>",
  "category": "<allowed category>",
  "needs_human": true,
  "response": "<customer-facing response>"
}
```

Fine-tuning 主要改善：

1. Intent Classification
2. Category Classification
3. JSON Validity
4. Schema Compliance
5. Escalation Behavior
6. Customer Response Relevance

---

## 4. Evaluation Metrics

### Training Diagnostics

只用來判斷 optimization 是否正常：

- Training Loss
- Validation Loss

### Behavioral Metrics

正式比較 Base vs LoRA：

- Intent Accuracy
- Category Accuracy
- JSON Valid Rate
- Schema Compliance
- Escalation Accuracy
- Response Relevance

### Operational Metrics

- Inference Latency
- Training Time
- Adapter Size
- Memory Usage（若可取得）

---

## 5. Project Stages

### Stage C1｜Dataset Analysis

目標：

了解資料品質與 leakage risk。

分析：

- schema
- row count
- intent / category distribution
- class balance
- missing values
- exact duplicates
- normalized duplicates
- placeholders
- flags
- label conflicts
- text length
- template-like variations

產出：

- `dataset_summary.json`
- `intent_distribution.csv`
- `category_distribution.csv`
- `duplicate_summary.json`
- `placeholder_distribution.csv`
- `manual_qa_samples.csv`
- `stage1_dataset_analysis.md`

完成後停止，不進入 C2。

---

### Stage C2｜Task Definition & Evaluation Contract

目標：

正式定義模型要學的行為與成功標準。

建立：

- Input Contract
- Output JSON Schema
- Intent Vocabulary
- Category Vocabulary
- Escalation Policy
- Response Constraints
- Evaluation Metrics
- Promotion Gate

產出：

- `task_definition.md`
- `output_schema.json`
- `intent_taxonomy.json`
- `category_taxonomy.json`
- `escalation_policy.json`
- `evaluation_contract.md`
- `promotion_gate.json`

完成後停止，不進入 C3。

---

### Stage C3｜Frozen Dataset Construction

建立四個 split：

- `train.jsonl`
- `validation.jsonl`
- `dev.jsonl`
- `locked_test.jsonl`

用途：

- Train：gradient update
- Validation：loss / checkpoint monitoring
- Dev：反覆 behavioral evaluation 與 error analysis
- Locked Test：最終一次正式 evaluation

要求：

- stratified / group-aware split 視 C1 結果決定
- 避免 duplicate / template leakage
- Locked Test 建立後不得用於 prompt tuning、hyperparameter tuning 或 candidate selection

產出：

- 四個 JSONL
- `split_manifest.json`
- `dataset_hashes.json`
- `split_validation_report.md`

---

### Stage C4｜Base Model Development Baseline

目標：

在 Fine-tuning 前建立公平 baseline。

Base Model 必須使用：

- 固定 System Prompt
- 固定 JSON Schema
- 固定 Allowed Labels
- 固定 Decoding Parameters

只使用 Dev Set。

產出：

- `base_dev_predictions.jsonl`
- `base_metrics.json`
- `base_error_cases.csv`
- `base_baseline_report.md`

不得查看 Locked Test。

---

### Stage C5A｜Training Smoke Test

目標：

在正式 QLoRA Training 前，以極小規模資料驗證整條 training pipeline 是否正常。

建議：

- 50–100 training samples
- 10–30 training steps

只確認：

- Dataset formatting 正確
- Tokenization 正確
- Loss 可正常計算
- Gradient 有更新
- Adapter 可儲存
- Adapter 可重新載入
- Inference 可正常執行

Smoke Test：

- 不作為模型成效評估
- 不納入正式 experiment comparison
- 不使用 Locked Test
- 不因 Smoke Test loss 下降就宣告 Fine-tuning 成功

完成後才進入正式 C5。

---

### Stage C5｜QLoRA Training

目標：

使用小型 Base Model 完成可實際執行的 QLoRA Fine-tuning。

Apple Silicon 優先：

- MLX / MLX-LM
- 小型 Qwen Instruct Model
- 不追求 7B / 14B

第一輪只使用一套 conservative config。

#### LoRA Rank

LoRA rank 以 moderate rank 為起點，例如：

- `r = 8`
- 或 `r = 16`

原則：

- 不預設 rank 越高越好
- 不一開始使用 64 / 128 / 256
- 只有在 Development Evaluation 顯示 adapter capacity 可能不足時，才考慮增加 rank

#### LoRA Target Modules

不要預設只使用 Attention layers。

選定 Base Model 後，必須先確認模型 architecture。

優先評估：

MLP layers：

- `gate_proj`
- `up_proj`
- `down_proj`

Attention layers：

- `q_proj`
- `k_proj`
- `v_proj`
- `o_proj`

依 MLX-LM 支援與硬體資源決定使用：

- MLP-only
- 或 MLP + Attention / all-linear

不要只因 tutorial 習慣而固定使用 `q_proj / v_proj`。

#### Training Config

記錄：

- rank
- alpha
- target modules
- learning rate
- epochs
- batch size
- gradient accumulation
- effective batch size
- sequence length
- training loss
- validation loss
- training time
- adapter size

產出：

- LoRA Adapter
- `training_config.json`
- `training_metrics.json`
- `loss_curve.png`
- `training_summary.md`

Training 完成不等於 Project 成功。

---

### Stage C6｜Development Evaluation & Controlled Iteration

使用同一個 Dev Set 比較：

```text
Base Model
vs
LoRA Model
```

固定：

- Prompt
- Schema
- Decoding
- Evaluator

進行 Error Analysis：

- Wrong Intent
- Wrong Category
- Invalid JSON
- Missing Field
- Extra Text
- Escalation FP / FN
- Irrelevant Response
- Hallucinated Policy

如果 LoRA 改善不足：

先依序檢查：

1. Data Quality
2. Task Format
3. Label Ambiguity
4. Class Imbalance
5. Sequence Truncation
6. Underfitting
7. Overfitting
8. LoRA Config
9. Base Model Capacity

如果確認問題主要來自 LoRA training config，優先檢查與調整：

1. Learning Rate
2. Target Modules / Target Layers
3. Epochs / Training Steps
4. LoRA Rank
5. LoRA Alpha
6. Effective Batch Size

原則：

- 最多做約 2–3 個 controlled experiments
- 每次 experiment 必須有明確 hypothesis
- 每次只修改少量關鍵變數
- 不做大規模 hyperparameter sweep
- 不因 validation loss 更低就自動選為最佳 candidate

產出：

- `experiment_registry.csv`
- 各 experiment artifacts
- `dev_comparison.csv`
- `error_analysis.csv`
- `candidate_selection.md`

只有 Dev behavior 明顯改善，才能進入 Locked Evaluation。

---

### Stage C6.5｜Freeze

在 Locked Test 前 freeze：

- Base Model Version
- LoRA Adapter
- System Prompt
- JSON Schema
- Intent Taxonomy
- Escalation Policy
- Decoding Parameters
- Evaluation Code
- Promotion Gate

產出：

- `freeze_manifest.json`
- SHA-256 hashes

Freeze 後禁止再修改。

---

### Stage C7｜Locked Base vs LoRA Evaluation

第一次正式使用：

`locked_test.jsonl`

比較：

```text
Base Model
vs
Frozen LoRA Candidate
```

禁止修改：

- model
- adapter
- prompt
- schema
- decoding
- evaluator
- promotion thresholds

產出：

- `locked_base_predictions.jsonl`
- `locked_lora_predictions.jsonl`
- `locked_metrics.json`
- `locked_comparison.csv`
- `locked_evaluation_report.md`

建議另外人工 QA 30–50 組 response。

Locked Test 不可為了取得漂亮結果重跑或重新調參。

---

### Stage C8｜Promotion Decision

依 C2 預先定義的 Promotion Gate 判斷：

```text
PROMOTE
或
REJECT
```

核心條件：

- Intent Accuracy 改善
- JSON Valid Rate 改善或至少不退步
- Schema Compliance 改善或至少不退步
- 無重大 behavioral regression
- Response Relevance 無重大退步

產出：

- `promotion_decision.json`
- `promotion_report.md`
- `final_metrics.csv`

禁止因作品集需求修改判定標準。

---

### Stage C9｜Simple Comparison Demo

只做簡單展示，不做大型產品。

建議：

- Python + Gradio

UI：

```text
Customer Question

Base Model                QLoRA Model
------------------------------------------------
Output                    Output
Intent                    Intent
Category                  Category
JSON Valid                JSON Valid
Schema Compliance         Schema Compliance
Needs Human               Needs Human
Latency                   Latency
```

Frontend 不是本 Project 核心。

---

## 6. Fine-tuning Strategy

Project 不追求：

- Full Fine-tuning
- 7B / 14B+ 大模型
- Multi-GPU
- DeepSpeed
- FSDP
- DPO / ORPO
- RL / GRPO
- RAG
- Agent
- Vector DB
- Kubernetes
- 大型 Backend
- 複雜 Dashboard

Project 核心：

```text
Task Definition
→ High-quality Training Data
→ Fair Base Baseline
→ Smoke Test
→ QLoRA
→ Behavioral Evaluation
→ Error Analysis
→ Controlled Iteration
→ Frozen Locked Evaluation
→ Promotion Decision
```

---

## 7. Codex Execution Rules

每次只能執行目前指定 Stage。

禁止：

- 偷跑下一 Stage
- 提前建立 Locked Evaluation 結果
- 提前 Fine-tune
- 提前調 Prompt
- 提前做 UI
- 使用 Locked Test 做任何調參

每個 Stage 完成後必須回報：

1. 做了什麼
2. 建立 / 修改哪些檔案
3. 主要結果
4. 測試結果
5. 發現的問題
6. Stage PASS / FAIL
7. 停止，不進下一 Stage

---

## 8. Manual Steps

只要涉及以下操作，必須明確標示：

> **這一步需要你手動做**

包括：

- Hugging Face 登入
- 模型下載
- Training 執行與等待
- 人工 Dataset QA
- 人工 Model Response QA
- 環境或硬體操作

並提供逐步操作方法。

---

## 9. Final Success Criteria

Project 成功不是：

```text
Training completed
```

也不是：

```text
Validation loss decreased
```

真正成功條件：

```text
QLoRA Training Successful
+
Development Behavioral Improvement
+
Locked Test Improvement
+
Promotion Gate PASS
```

最終必須能回答：

> QLoRA 到底改善了哪些客服行為、改善多少，以及是否值得 Promote。
