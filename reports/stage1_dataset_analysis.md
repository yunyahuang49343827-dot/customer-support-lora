# Stage C1 Dataset Analysis

## 1. Dataset Overview

- Dataset: `bitext/Bitext-customer-support-llm-chatbot-training-dataset`
- Loaded splits: train
- Actual total rows: 26,872
- Source metadata versus loaded rows: Published dataset metadata row counts matched the loaded data for every split.

## 2. Schema

| Split | Loaded rows | Dataset metadata rows | Features |
|---|---:|---:|---|
| train | 26,872 | 26,872 | flags: string, instruction: string, category: string, intent: string, response: string |

## 3. Missing / Empty Values

| Column | Null | Empty | Whitespace-only |
|---|---:|---:|---:|
| flags | 0 (0.000%) | 0 (0.000%) | 0 (0.000%) |
| instruction | 0 (0.000%) | 0 (0.000%) | 0 (0.000%) |
| category | 0 (0.000%) | 0 (0.000%) | 0 (0.000%) |
| intent | 0 (0.000%) | 0 (0.000%) | 0 (0.000%) |
| response | 0 (0.000%) | 0 (0.000%) | 0 (0.000%) |

Across all five fields there are 0 null values, 0 empty strings, and 0 whitespace-only values.

## 4. Intent Distribution

- Unique intents: 27
- Largest class: 1,000 rows; smallest class: 950 rows; max/min ratio: 1.053
- Most frequent intents:

| intent | Count | Percentage |
|---|---:|---:|
| edit_account | 1,000 | 3.721% |
| switch_account | 1,000 | 3.721% |
| check_invoice | 1,000 | 3.721% |
| complaint | 1,000 | 3.721% |
| contact_customer_service | 1,000 | 3.721% |

- Least frequent intents:

| intent | Count | Percentage |
|---|---:|---:|
| check_cancellation_fee | 950 | 3.535% |
| change_shipping_address | 973 | 3.621% |
| delete_account | 995 | 3.703% |
| delivery_options | 995 | 3.703% |
| recover_password | 995 | 3.703% |

The full distribution is in `artifacts/stage1/intent_distribution.csv`.

## 5. Category Distribution

- Unique categories: 11

| category | Count | Percentage |
|---|---:|---:|
| ACCOUNT | 5,986 | 22.276% |
| ORDER | 3,988 | 14.841% |
| REFUND | 2,992 | 11.134% |
| INVOICE | 1,999 | 7.439% |
| CONTACT | 1,999 | 7.439% |
| PAYMENT | 1,998 | 7.435% |
| FEEDBACK | 1,997 | 7.432% |
| DELIVERY | 1,994 | 7.420% |
| SHIPPING | 1,970 | 7.331% |
| SUBSCRIPTION | 999 | 3.718% |

## 6. Category–Intent Mapping

- Observed category–intent pairs: 27
- Every intent maps to exactly one category; no unexpected many-to-many mapping was detected.
- Intents mapped to multiple categories: none

## 7. Flags Analysis

- Unique raw flag values: 394
- Most frequent values: `BL` 5,212 (19.396%), `BLQ` 2,467 (9.181%), `BIL` 2,138 (7.956%), `BLM` 1,297 (4.827%), `BILQ` 1,057 (3.933%), `BLQZ` 970 (3.610%), `BLZ` 902 (3.357%), `BKL` 862 (3.208%), `BLMQ` 600 (2.233%), `BEL` 533 (1.983%)
- `flag_distribution.csv` includes raw flag frequency and each flag × intent count. No decision is made here about use in training.

## 8. Placeholder Analysis

- Unique placeholder types: 391
- Total occurrences across instruction and response: 42,814
- Rows containing at least one placeholder: 13,041 (48.530%)
- Most frequent placeholders: `{{Order Number}}` (8,029), `{{Account Type}}` (5,440), `{{Account Category}}` (3,900), `{{Online Order Interaction}}` (2,699), `{{Customer Support Phone Number}}` (2,635), `{{Website URL}}` (2,534), `{{Customer Support Hours}}` (2,325), `{{Invoice Number}}` (1,521), `{{Person Name}}` (1,249), `{{Refund Amount}}` (1,187)
- `placeholder_distribution.csv` records placeholder × intent relationships. Raw text was not changed.

## 9. Text Characteristics

| Field | Measure | Min | Mean | Median | P95 | Max |
|---|---|---:|---:|---:|---:|---:|
| instruction | characters | 6 | 46.890 | 48.0 | 61.0 | 92 |
| instruction | words | 1 | 8.691 | 9.0 | 13.0 | 16 |
| response | characters | 57 | 634.104 | 540.0 | 1295.0 | 2472 |
| response | words | 9 | 104.789 | 90.0 | 206.0 | 402 |

## 10. Duplicate Analysis

Counts below mean occurrences beyond the first occurrence; group sizes are reported separately.

- Exact duplicate full rows: 0 across 0 groups (0 involved rows)
- Exact duplicate instructions: 2,237 across 989 groups
- Exact duplicate responses: 2 across 2 groups
- Normalized duplicate instructions: 2,868 across 1,128 groups
- Largest normalized duplicate group: 22 rows

Normalization is analysis-only: Unicode/common punctuation normalization, lowercase, trim, whitespace collapse, and replacement of each `{{...}}` placeholder with `<ENTITY>`.

## 11. Label Conflict Analysis

- Potential label-conflict groups: 0
- Rows in potential conflict groups: 0
- 0 potential label conflicts detected

## 12. Potential Evaluation Leakage

Random row-level splitting would leak normalized/template-equivalent instructions across future splits. Stage C3 should group on normalized_instruction before stratification. Exact response duplication also creates a risk of memorized response templates, so Stage C3 should validate group isolation before freezing any split.

## 13. Manual QA Required

`artifacts/stage1/manual_qa_samples.csv` contains 30 reproducible rows sampled with `random_seed = 42`.

> **這一步需要你手動做**
>
> 1. Open `artifacts/stage1/manual_qa_samples.csv`.
> 2. Check intent/category correctness, response relevance, placeholder consistency, flags, and policy-like claims.
> 3. Record concerns separately; do not edit the raw dataset or these sampled rows.

## 14. Recommendation for Stage C3 Splitting Strategy

Recommendation only (no split was created): use a group-aware split keyed by `normalized_instruction`, then preserve intent/category balance as far as group constraints allow. Also keep exact duplicate responses/templates within a single group when practical, and run cross-split duplicate checks before freezing. Label-conflict groups should be manually resolved or kept isolated as units. This recommendation must be revisited in Stage C3.

## 15. Stage C1 Conclusion

Stage C1 analysis completed. Required schema, integrity, distributions, flags, placeholders, text lengths, duplicates, conflicts, leakage risks, and manual QA samples were analyzed. No Train, Validation, Dev, or Locked Test split was created, and no model work was performed.
