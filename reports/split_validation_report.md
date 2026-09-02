# Stage C3 Frozen Dataset Construction

## Split Strategy

The source is the complete 26,872-row Hugging Face Bitext dataset. A deterministic seed of `42` and stable SHA-256 ranking are used. The selected subset targets Train 2,700, Validation 300, Dev 300, and Locked Test 300 rows. Selection occurs at group level and is intent-aware; no row-level split followed by duplicate repair is used.

Priority order is: zero group leakage, intent coverage/balance, then target size. The source-quality gate removes only failed selected source rows and replaces each one with a clean, previously unused singleton normalized group from the same canonical intent. It never rewrites or completes response content.

## Source Response Quality Gate

Raw source response validation runs before response compaction. It rejects empty responses, incomplete numbered or bulleted markers, incomplete list introductions, and conservative suspected partial enumerations. Every failed selected row receives exactly one deterministic same-intent replacement that passes source quality validation.

- Selected source responses scanned: 3,600
- Failed selected rows: 2
- Replacements made: 2
- Final selected quality failures: 0
- Failure reasons: `incomplete_list_introduction`: 2
- Audit: `artifacts/stage3/source_response_quality.json`
- Minimal QA examples: `artifacts/stage3/source_response_quality_examples.csv`

| Split | Removed rows | Added rows | Size unchanged | Intent distribution unchanged | Category distribution unchanged | Escalation distribution unchanged |
|---|---:|---:|---|---|---|---|
| train | 2 | 2 | True | True | True | True |
| validation | 0 | 0 | True | True | True | True |
| dev | 0 | 0 | True | True | True | True |
| locked_test | 0 | 0 | True | True | True | True |

## Group-aware Method

The analysis-only Stage C1 `normalize_instruction` function is reused verbatim. Each unique normalized instruction forms one group. Initial selection is whole-group and intent-aware. Quality-gate replacements use only unused singleton groups; removing one failed row from an initially selected multi-row group is permitted, while the remaining rows retain their original split. A normalized group can never cross splits.

- Total source groups: 24,004
- Selected groups: 3,257
- Group key: `normalized_instruction`
- Assignment manifest: `data/manifests/group_assignment.csv`

## Actual Split Sizes

| Split | Target | Actual | Intent coverage | Category coverage |
|---|---:|---:|---:|---:|
| train | 2,700 | 2,700 | 27 | 11 |
| validation | 300 | 300 | 27 | 11 |
| dev | 300 | 300 | 27 | 11 |
| locked_test | 300 | 300 | 27 | 11 |

## Intent Distribution

| Split | Minimum rows per intent | Maximum rows per intent | Covered intents |
|---|---:|---:|---:|
| train | 100 | 100 | 27 |
| validation | 11 | 12 | 27 |
| dev | 11 | 12 | 27 |
| locked_test | 11 | 12 | 27 |

The complete 108-row split × intent distribution is in `artifacts/stage3/split_distribution.csv`.

## Category Distribution

| Split | Covered categories | Categories |
|---|---:|---|
| train | 11 | ACCOUNT, CANCEL, CONTACT, DELIVERY, FEEDBACK, INVOICE, ORDER, PAYMENT, REFUND, SHIPPING, SUBSCRIPTION |
| validation | 11 | ACCOUNT, CANCEL, CONTACT, DELIVERY, FEEDBACK, INVOICE, ORDER, PAYMENT, REFUND, SHIPPING, SUBSCRIPTION |
| dev | 11 | ACCOUNT, CANCEL, CONTACT, DELIVERY, FEEDBACK, INVOICE, ORDER, PAYMENT, REFUND, SHIPPING, SUBSCRIPTION |
| locked_test | 11 | ACCOUNT, CANCEL, CONTACT, DELIVERY, FEEDBACK, INVOICE, ORDER, PAYMENT, REFUND, SHIPPING, SUBSCRIPTION |

## Escalation Distribution

`needs_human` is copied deterministically from the confirmed C2 intent policy (6 true intents, 21 false intents); it is never inferred from response text.

| Split | True rows | False rows |
|---|---:|---:|
| train | 600 | 2,100 |
| validation | 66 | 234 |
| dev | 66 | 234 |
| locked_test | 66 | 234 |

## Response Compaction Policy

No LLM, synthetic generation, or paraphrasing is used. Leading/trailing whitespace is removed and internal whitespace is collapsed. Responses at or below 650 characters are retained. Longer normal prose is shortened only to a prefix ending at a complete sentence boundary: the first 2 sentences are preferred, with up to 4 when needed to retain at least 250 characters of context.

For numbered or bulleted lists (`1.`, `2.`, `-`, `•`), compacting is rejected whenever it would omit any part of the list block. Candidates ending with a colon, a standalone list marker, or an incomplete list introduction are also rejected. Every rejected candidate falls back to the complete whitespace-normalized source response. Therefore neither mid-sentence nor mid-list truncation is allowed.

- Strategies: `compacted_complete_sentence_prefix`: 322, `completeness_full_fallback`: 846, `conservative_full_fallback`: 12, `preserved_within_limit`: 2,420
- Responses rejected by completeness validation and restored in full: 846
- Completeness failure reasons: `list_block_would_be_truncated`: 846
- Mean normalized original length: 636.941 characters
- Mean final length: 595.759 characters
- Maximum final length (conservative fallbacks may exceed the compacting limit): 2,321 characters

## Schema Validation

Every selected target was validated by the existing strict C2 helper for exact keys, canonical labels, intent-category consistency, boolean `needs_human`, and a non-empty response.

- Invalid generated targets: 0
- Silently discarded invalid rows: 0

## Leakage Validation

| Check | Cross-split overlap count |
|---|---:|
| source_row_index_overlap | 0 |
| exact_instruction_overlap | 0 |
| normalized_instruction_overlap | 0 |
| group_overlap | 0 |

All required cross-split overlap counts are zero: **true**.

## Locked Test Freeze

The previous Stage C3 Locked Test hash record `b7f7af8c5e366c743fafd68c8c8f3e7a2b101dfce53e63bf1f7a8ead0bce1fac` is superseded by this quality-gate revision. The recalculated `data/processed/locked_test.jsonl` SHA-256 is `b7f7af8c5e366c743fafd68c8c8f3e7a2b101dfce53e63bf1f7a8ead0bce1fac`. No Locked Test row required replacement, so the recalculated digest is unchanged. Split size, intent quota, category coverage, and escalation distribution remain unchanged. It must not be used for prompt tuning, hyperparameter tuning, behavioral error analysis, evaluator changes, threshold changes, or candidate selection. Its first authorized behavioral use is Stage C7 after the C6.5 freeze.

## Manual QA Required

`artifacts/stage3/split_samples.csv` contains 5 deterministic samples from each split (20 total) using seed 42.

> **這一步需要你手動做**
>
> 1. Open `artifacts/stage3/split_samples.csv`.
> 2. Review instruction/intent/category consistency, the confirmed `needs_human` value, response relevance, and whether compaction retained enough meaning.
> 3. Record concerns without editing the frozen JSONL files, especially `locked_test.jsonl`.

## Stage C3 Conclusion

Stage C3 passed construction-time validation: source-quality failures were excluded and replaced without content repair, all four files retain their required sizes and label distributions, each covers all 27 intents and 11 categories, every target conforms to the C2 contract, and every cross-split leakage check is zero. The final Locked Test hash is recorded. No model loading, inference, training, prompt tuning, development evaluation, or locked behavioral evaluation was performed.
