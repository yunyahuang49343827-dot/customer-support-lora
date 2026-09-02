"""Stage C1 analysis for the Bitext customer-support dataset.

This module performs analysis only. It deliberately does not create dataset
splits, load models, or modify the source data.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import DatasetDict, load_dataset


DATASET_NAME = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
EXPECTED_COLUMNS = ["flags", "instruction", "category", "intent", "response"]
PLACEHOLDER_RE = re.compile(r"\{\{.*?\}\}")
RANDOM_SEED = 42

PUNCTUATION_TRANSLATION = str.maketrans(
    {
        "‘": "'",
        "’": "'",
        "‚": "'",
        "‛": "'",
        "“": '"',
        "”": '"',
        "„": '"',
        "‟": '"',
        "–": "-",
        "—": "-",
        "−": "-",
        "…": "...",
        "，": ",",
        "。": ".",
        "！": "!",
        "？": "?",
        "：": ":",
        "；": ";",
    }
)


def extract_placeholders(text: Any) -> List[str]:
    """Return all non-greedy ``{{...}}`` placeholders in source order."""
    if not isinstance(text, str):
        return []
    return PLACEHOLDER_RE.findall(text)


def normalize_instruction(text: Any) -> str:
    """Normalize an instruction for duplicate analysis without mutating it."""
    if not isinstance(text, str):
        return ""
    value = unicodedata.normalize("NFKC", text)
    value = PLACEHOLDER_RE.sub("<ENTITY>", value)
    value = value.translate(PUNCTUATION_TRANSLATION).lower().strip()
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"\s+([,.;:!?])", r"\1", value)
    value = re.sub(r"([,.;:!?])(?=\S)", r"\1 ", value)
    return value.strip()


def calculate_percentage(count: int, total: int) -> float:
    """Calculate a percentage, returning zero for an empty denominator."""
    return round((count / total) * 100, 6) if total else 0.0


def duplicate_groups(values: Sequence[str]) -> Dict[str, List[int]]:
    """Map repeated values to their row positions, excluding singletons."""
    groups: Dict[str, List[int]] = {}
    for position, value in enumerate(values):
        groups.setdefault(value, []).append(position)
    return {value: positions for value, positions in groups.items() if len(positions) > 1}


def detect_label_conflicts(
    frame: pd.DataFrame,
    normalized_column: str = "normalized_instruction",
    label_column: str = "intent",
) -> pd.DataFrame:
    """Return rows whose normalized instruction maps to multiple labels."""
    label_counts = frame.groupby(normalized_column, dropna=False)[label_column].nunique(dropna=False)
    conflicting_values = label_counts[label_counts > 1].index
    columns = [normalized_column, "instruction", "category", label_column]
    return frame.loc[frame[normalized_column].isin(conflicting_values), columns].copy()


def _json_scalar(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if math.isnan(float(value)) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_scalar) + "\n", encoding="utf-8")


def _distribution(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    counts = frame[column].value_counts(dropna=False).rename_axis(column).reset_index(name="count")
    counts["percentage"] = counts["count"].map(lambda count: calculate_percentage(int(count), len(frame)))
    return counts


def _column_integrity(series: pd.Series, total: int) -> Dict[str, Any]:
    null_mask = series.isna()
    string_series = series.fillna("").astype(str)
    empty_mask = (~null_mask) & string_series.eq("")
    whitespace_mask = (~null_mask) & string_series.ne("") & string_series.str.strip().eq("")
    return {
        "null_count": int(null_mask.sum()),
        "null_percentage": calculate_percentage(int(null_mask.sum()), total),
        "empty_string_count": int(empty_mask.sum()),
        "empty_string_percentage": calculate_percentage(int(empty_mask.sum()), total),
        "whitespace_only_count": int(whitespace_mask.sum()),
        "whitespace_only_percentage": calculate_percentage(int(whitespace_mask.sum()), total),
    }


def _duplicate_metrics(series: pd.Series) -> Dict[str, int]:
    groups = duplicate_groups(series.astype(str).tolist())
    rows_in_groups = sum(len(positions) for positions in groups.values())
    return {
        "duplicate_count_beyond_first": rows_in_groups - len(groups),
        "duplicate_group_count": len(groups),
        "rows_in_duplicate_groups": rows_in_groups,
        "largest_group_size": max((len(positions) for positions in groups.values()), default=1),
    }


def _text_stats(series: pd.Series) -> Dict[str, Dict[str, float]]:
    text = series.fillna("").astype(str)
    character_lengths = text.str.len().to_numpy()
    word_counts = text.map(lambda value: len(value.split())).to_numpy()

    def summarize(values: np.ndarray) -> Dict[str, float]:
        return {
            "min": int(np.min(values)) if len(values) else 0,
            "mean": round(float(np.mean(values)), 6) if len(values) else 0.0,
            "median": float(np.median(values)) if len(values) else 0.0,
            "p95": float(np.percentile(values, 95)) if len(values) else 0.0,
            "max": int(np.max(values)) if len(values) else 0,
        }

    return {"character_length": summarize(character_lengths), "word_count": summarize(word_counts)}


def _histogram(values: Iterable[int], title: str, xlabel: str, output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.hist(list(values), bins=50, color="#376996", edgecolor="white")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Rows")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _feature_name(feature: Any) -> str:
    dtype = getattr(feature, "dtype", None)
    return str(dtype if dtype is not None else feature)


def analyze(frame: pd.DataFrame, split_metadata: Mapping[str, Any], artifact_dir: Path, report_path: Path) -> Dict[str, Any]:
    """Compute and write all Stage C1 artifacts for a combined dataset frame."""
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    total = len(frame)

    missing_columns = [column for column in EXPECTED_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    frame = frame.copy()
    frame["normalized_instruction"] = frame["instruction"].map(normalize_instruction)

    integrity = {column: _column_integrity(frame[column], total) for column in EXPECTED_COLUMNS}
    exact_row_mask = frame[EXPECTED_COLUMNS].duplicated(keep=False)
    exact_row_groups = frame.loc[exact_row_mask, EXPECTED_COLUMNS].drop_duplicates().shape[0]
    exact_row_excess = int(frame.duplicated(subset=EXPECTED_COLUMNS, keep="first").sum())
    instruction_duplicates = _duplicate_metrics(frame["instruction"])
    response_duplicates = _duplicate_metrics(frame["response"])
    normalized_duplicates = _duplicate_metrics(frame["normalized_instruction"])

    dataset_summary = {
        "dataset_name": DATASET_NAME,
        "split_names": list(split_metadata.keys()),
        "row_count": total,
        "columns": EXPECTED_COLUMNS,
        "feature_types": split_metadata,
        "column_integrity": integrity,
        "exact_duplicate_rows": {
            "duplicate_count_beyond_first": exact_row_excess,
            "duplicate_group_count": int(exact_row_groups),
            "rows_in_duplicate_groups": int(exact_row_mask.sum()),
        },
        "exact_duplicate_instructions": instruction_duplicates,
        "exact_duplicate_responses": response_duplicates,
        "count_definition": "Duplicate counts are occurrences beyond the first; group and involved-row counts are also reported.",
    }
    _write_json(artifact_dir / "dataset_summary.json", dataset_summary)

    intent_distribution = _distribution(frame, "intent")
    category_distribution = _distribution(frame, "category")
    intent_distribution.to_csv(artifact_dir / "intent_distribution.csv", index=False)
    category_distribution.to_csv(artifact_dir / "category_distribution.csv", index=False)

    category_intent = (
        frame.groupby(["category", "intent"], dropna=False).size().reset_index(name="count").sort_values(
            ["category", "count", "intent"], ascending=[True, False, True]
        )
    )
    category_intent.to_csv(artifact_dir / "category_intent_mapping.csv", index=False)

    flag_total = _distribution(frame, "flags").rename(columns={"flags": "flag"})
    flag_intent = frame.groupby(["flags", "intent"], dropna=False).size().reset_index(name="intent_count")
    flag_distribution = flag_intent.merge(flag_total, left_on="flags", right_on="flag", how="left").drop(columns="flag")
    flag_distribution["percentage_within_flag"] = flag_distribution.apply(
        lambda row: calculate_percentage(int(row["intent_count"]), int(row["count"])), axis=1
    )
    flag_distribution = flag_distribution[["flags", "count", "percentage", "intent", "intent_count", "percentage_within_flag"]]
    flag_distribution.sort_values(["count", "flags", "intent_count"], ascending=[False, True, False]).to_csv(
        artifact_dir / "flag_distribution.csv", index=False
    )

    combined_row_placeholders = [
        extract_placeholders(instruction) + extract_placeholders(response)
        for instruction, response in zip(frame["instruction"], frame["response"])
    ]
    placeholder_counter = Counter(item for items in combined_row_placeholders for item in items)
    placeholder_intent_rows: List[Dict[str, Any]] = []
    for placeholders, intent in zip(combined_row_placeholders, frame["intent"]):
        for placeholder, occurrence_count in Counter(placeholders).items():
            placeholder_intent_rows.append(
                {"placeholder": placeholder, "intent": intent, "occurrence_count": occurrence_count, "row_count": 1}
            )
    if placeholder_intent_rows:
        placeholder_distribution = (
            pd.DataFrame(placeholder_intent_rows)
            .groupby(["placeholder", "intent"], dropna=False)[["occurrence_count", "row_count"]]
            .sum()
            .reset_index()
        )
        placeholder_distribution["total_placeholder_occurrences"] = placeholder_distribution["placeholder"].map(placeholder_counter)
        placeholder_distribution["percentage_within_placeholder"] = placeholder_distribution.apply(
            lambda row: calculate_percentage(int(row["occurrence_count"]), int(row["total_placeholder_occurrences"])), axis=1
        )
        placeholder_distribution.sort_values(
            ["total_placeholder_occurrences", "placeholder", "occurrence_count"], ascending=[False, True, False]
        ).to_csv(artifact_dir / "placeholder_distribution.csv", index=False)
    else:
        pd.DataFrame(
            columns=[
                "placeholder",
                "intent",
                "occurrence_count",
                "row_count",
                "total_placeholder_occurrences",
                "percentage_within_placeholder",
            ]
        ).to_csv(artifact_dir / "placeholder_distribution.csv", index=False)

    text_length_summary = {
        "instruction": _text_stats(frame["instruction"]),
        "response": _text_stats(frame["response"]),
    }
    _write_json(artifact_dir / "text_length_summary.json", text_length_summary)
    _histogram(frame["instruction"].fillna("").astype(str).str.len(), "Instruction Character Length", "Characters", artifact_dir / "instruction_length.png")
    _histogram(frame["response"].fillna("").astype(str).str.len(), "Response Character Length", "Characters", artifact_dir / "response_length.png")

    duplicate_summary = {
        "count_definition": "Counts are duplicate occurrences beyond the first occurrence.",
        "exact_duplicate_instruction_count": instruction_duplicates["duplicate_count_beyond_first"],
        "exact_duplicate_instruction_groups": instruction_duplicates["duplicate_group_count"],
        "normalized_duplicate_instruction_count": normalized_duplicates["duplicate_count_beyond_first"],
        "normalized_duplicate_group_count": normalized_duplicates["duplicate_group_count"],
        "normalized_duplicate_rows_in_groups": normalized_duplicates["rows_in_duplicate_groups"],
        "largest_normalized_duplicate_group_size": normalized_duplicates["largest_group_size"],
    }
    _write_json(artifact_dir / "duplicate_summary.json", duplicate_summary)

    group_sizes = frame.groupby("normalized_instruction", dropna=False).size()
    duplicate_values = group_sizes[group_sizes > 1].sort_values(ascending=False)
    example_groups = duplicate_values.head(100)
    examples = frame.loc[
        frame["normalized_instruction"].isin(example_groups.index),
        ["normalized_instruction", "instruction", "category", "intent"],
    ].copy()
    examples["duplicate_group_size"] = examples["normalized_instruction"].map(example_groups)
    rank = {value: index + 1 for index, value in enumerate(example_groups.index)}
    examples["duplicate_group_rank"] = examples["normalized_instruction"].map(rank)
    examples.sort_values(["duplicate_group_rank", "intent", "instruction"]).to_csv(
        artifact_dir / "normalized_duplicate_examples.csv", index=False
    )

    conflicts = detect_label_conflicts(frame)
    conflicts.sort_values(["normalized_instruction", "intent", "instruction"]).to_csv(
        artifact_dir / "label_conflicts.csv", index=False
    )

    frame.sample(n=min(30, total), random_state=RANDOM_SEED)[EXPECTED_COLUMNS].to_csv(
        artifact_dir / "manual_qa_samples.csv", index=False
    )

    intent_categories = category_intent.groupby("intent")["category"].nunique()
    intents_multiple_categories = intent_categories[intent_categories > 1]
    category_intent_pairs = len(category_intent)
    placeholder_rows_count = sum(bool(items) for items in combined_row_placeholders)
    total_placeholder_occurrences = sum(placeholder_counter.values())
    conflict_groups = conflicts["normalized_instruction"].nunique() if not conflicts.empty else 0

    top_intents = intent_distribution.head(5)
    bottom_intents = intent_distribution.sort_values(["count", "intent"]).head(5)
    largest_intent = int(intent_distribution["count"].max())
    smallest_intent = int(intent_distribution["count"].min())
    imbalance_ratio = round(largest_intent / smallest_intent, 3) if smallest_intent else None

    missing_total = sum(stats["null_count"] for stats in integrity.values())
    empty_total = sum(stats["empty_string_count"] for stats in integrity.values())
    whitespace_total = sum(stats["whitespace_only_count"] for stats in integrity.values())
    recommend_grouping = normalized_duplicates["duplicate_group_count"] > 0
    mapping_note = (
        f"{len(intents_multiple_categories)} intent(s) map to multiple categories."
        if len(intents_multiple_categories)
        else "Every intent maps to exactly one category; no unexpected many-to-many mapping was detected."
    )
    leakage_note = (
        "Random row-level splitting would leak normalized/template-equivalent instructions across future splits. "
        "Stage C3 should group on normalized_instruction before stratification."
        if recommend_grouping
        else "No normalized duplicate groups were found; ordinary stratification appears acceptable, subject to Stage C3 validation."
    )

    def table_rows(distribution: pd.DataFrame, label: str, limit: int = 10) -> str:
        lines = [f"| {label} | Count | Percentage |", "|---|---:|---:|"]
        for _, row in distribution.head(limit).iterrows():
            lines.append(f"| {row[label]} | {int(row['count']):,} | {float(row['percentage']):.3f}% |")
        return "\n".join(lines)

    column_rows = ["| Column | Null | Empty | Whitespace-only |", "|---|---:|---:|---:|"]
    for column, stats in integrity.items():
        column_rows.append(
            f"| {column} | {stats['null_count']} ({stats['null_percentage']:.3f}%) | "
            f"{stats['empty_string_count']} ({stats['empty_string_percentage']:.3f}%) | "
            f"{stats['whitespace_only_count']} ({stats['whitespace_only_percentage']:.3f}%) |"
        )

    feature_lines = ["| Split | Loaded rows | Dataset metadata rows | Features |", "|---|---:|---:|---|"]
    for split, metadata in split_metadata.items():
        features = ", ".join(f"{name}: {dtype}" for name, dtype in metadata["features"].items())
        metadata_rows = metadata.get("metadata_row_count")
        metadata_display = f"{metadata_rows:,}" if metadata_rows is not None else "not published"
        feature_lines.append(f"| {split} | {metadata['row_count']:,} | {metadata_display} | {features} |")

    metadata_differences = [
        split
        for split, metadata in split_metadata.items()
        if metadata.get("metadata_row_count") is not None
        and metadata["metadata_row_count"] != metadata["row_count"]
    ]
    metadata_comparison = (
        "Loaded row counts differed from published dataset metadata for: " + ", ".join(metadata_differences) + "."
        if metadata_differences
        else "Published dataset metadata row counts matched the loaded data for every split."
    )

    flag_summary = flag_total.head(10)
    placeholder_top = Counter(placeholder_counter).most_common(10)
    report = f"""# Stage C1 Dataset Analysis

## 1. Dataset Overview

- Dataset: `{DATASET_NAME}`
- Loaded splits: {', '.join(split_metadata.keys())}
- Actual total rows: {total:,}
- Source metadata versus loaded rows: {metadata_comparison}

## 2. Schema

{chr(10).join(feature_lines)}

## 3. Missing / Empty Values

{chr(10).join(column_rows)}

Across all five fields there are {missing_total} null values, {empty_total} empty strings, and {whitespace_total} whitespace-only values.

## 4. Intent Distribution

- Unique intents: {len(intent_distribution)}
- Largest class: {largest_intent:,} rows; smallest class: {smallest_intent:,} rows; max/min ratio: {imbalance_ratio}
- Most frequent intents:

{table_rows(top_intents, 'intent', 5)}

- Least frequent intents:

{table_rows(bottom_intents, 'intent', 5)}

The full distribution is in `artifacts/stage1/intent_distribution.csv`.

## 5. Category Distribution

- Unique categories: {len(category_distribution)}

{table_rows(category_distribution, 'category', 10)}

## 6. Category–Intent Mapping

- Observed category–intent pairs: {category_intent_pairs}
- {mapping_note}
- Intents mapped to multiple categories: {', '.join(map(str, intents_multiple_categories.index.tolist())) if len(intents_multiple_categories) else 'none'}

## 7. Flags Analysis

- Unique raw flag values: {len(flag_total)}
- Most frequent values: {', '.join(f'`{row["flag"]}` {int(row["count"]):,} ({float(row["percentage"]):.3f}%)' for _, row in flag_summary.iterrows())}
- `flag_distribution.csv` includes raw flag frequency and each flag × intent count. No decision is made here about use in training.

## 8. Placeholder Analysis

- Unique placeholder types: {len(placeholder_counter)}
- Total occurrences across instruction and response: {total_placeholder_occurrences:,}
- Rows containing at least one placeholder: {placeholder_rows_count:,} ({calculate_percentage(placeholder_rows_count, total):.3f}%)
- Most frequent placeholders: {', '.join(f'`{name}` ({count:,})' for name, count in placeholder_top)}
- `placeholder_distribution.csv` records placeholder × intent relationships. Raw text was not changed.

## 9. Text Characteristics

| Field | Measure | Min | Mean | Median | P95 | Max |
|---|---|---:|---:|---:|---:|---:|
| instruction | characters | {text_length_summary['instruction']['character_length']['min']} | {text_length_summary['instruction']['character_length']['mean']:.3f} | {text_length_summary['instruction']['character_length']['median']:.1f} | {text_length_summary['instruction']['character_length']['p95']:.1f} | {text_length_summary['instruction']['character_length']['max']} |
| instruction | words | {text_length_summary['instruction']['word_count']['min']} | {text_length_summary['instruction']['word_count']['mean']:.3f} | {text_length_summary['instruction']['word_count']['median']:.1f} | {text_length_summary['instruction']['word_count']['p95']:.1f} | {text_length_summary['instruction']['word_count']['max']} |
| response | characters | {text_length_summary['response']['character_length']['min']} | {text_length_summary['response']['character_length']['mean']:.3f} | {text_length_summary['response']['character_length']['median']:.1f} | {text_length_summary['response']['character_length']['p95']:.1f} | {text_length_summary['response']['character_length']['max']} |
| response | words | {text_length_summary['response']['word_count']['min']} | {text_length_summary['response']['word_count']['mean']:.3f} | {text_length_summary['response']['word_count']['median']:.1f} | {text_length_summary['response']['word_count']['p95']:.1f} | {text_length_summary['response']['word_count']['max']} |

## 10. Duplicate Analysis

Counts below mean occurrences beyond the first occurrence; group sizes are reported separately.

- Exact duplicate full rows: {exact_row_excess:,} across {exact_row_groups:,} groups ({int(exact_row_mask.sum()):,} involved rows)
- Exact duplicate instructions: {instruction_duplicates['duplicate_count_beyond_first']:,} across {instruction_duplicates['duplicate_group_count']:,} groups
- Exact duplicate responses: {response_duplicates['duplicate_count_beyond_first']:,} across {response_duplicates['duplicate_group_count']:,} groups
- Normalized duplicate instructions: {normalized_duplicates['duplicate_count_beyond_first']:,} across {normalized_duplicates['duplicate_group_count']:,} groups
- Largest normalized duplicate group: {normalized_duplicates['largest_group_size']:,} rows

Normalization is analysis-only: Unicode/common punctuation normalization, lowercase, trim, whitespace collapse, and replacement of each `{{{{...}}}}` placeholder with `<ENTITY>`.

## 11. Label Conflict Analysis

- Potential label-conflict groups: {conflict_groups:,}
- Rows in potential conflict groups: {len(conflicts):,}
- {'0 potential label conflicts detected' if conflict_groups == 0 else 'Potential conflicts require manual review; labels were not changed.'}

## 12. Potential Evaluation Leakage

{leakage_note} Exact response duplication also creates a risk of memorized response templates, so Stage C3 should validate group isolation before freezing any split.

## 13. Manual QA Required

`artifacts/stage1/manual_qa_samples.csv` contains {min(30, total)} reproducible rows sampled with `random_seed = 42`.

> **這一步需要你手動做**
>
> 1. Open `artifacts/stage1/manual_qa_samples.csv`.
> 2. Check intent/category correctness, response relevance, placeholder consistency, flags, and policy-like claims.
> 3. Record concerns separately; do not edit the raw dataset or these sampled rows.

## 14. Recommendation for Stage C3 Splitting Strategy

Recommendation only (no split was created): {'use a group-aware split keyed by `normalized_instruction`, then preserve intent/category balance as far as group constraints allow' if recommend_grouping else 'an ordinary stratified split may be adequate'}. Also keep exact duplicate responses/templates within a single group when practical, and run cross-split duplicate checks before freezing. Label-conflict groups should be manually resolved or kept isolated as units. This recommendation must be revisited in Stage C3.

## 15. Stage C1 Conclusion

Stage C1 analysis completed. Required schema, integrity, distributions, flags, placeholders, text lengths, duplicates, conflicts, leakage risks, and manual QA samples were analyzed. No Train, Validation, Dev, or Locked Test split was created, and no model work was performed.
"""
    report_path.write_text(report, encoding="utf-8")

    return {
        "row_count": total,
        "category_count": len(category_distribution),
        "intent_count": len(intent_distribution),
        "potential_label_conflict_groups": int(conflict_groups),
        "placeholder_rows": placeholder_rows_count,
    }


def load_frame(dataset_name: str, cache_dir: Path) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """Load all published splits and return one analysis frame plus schema metadata."""
    dataset = load_dataset(dataset_name, cache_dir=str(cache_dir))
    if not isinstance(dataset, DatasetDict):
        raise TypeError(f"Expected DatasetDict, received {type(dataset).__name__}")
    frames: List[pd.DataFrame] = []
    split_metadata: Dict[str, Any] = {}
    for split_name, split in dataset.items():
        split_frame = split.to_pandas()
        split_frame["__source_split"] = split_name
        frames.append(split_frame)
        published_split = split.info.splits.get(split_name) if split.info.splits else None
        split_metadata[split_name] = {
            "row_count": len(split),
            "metadata_row_count": getattr(published_split, "num_examples", None),
            "features": {name: _feature_name(feature) for name, feature in split.features.items()},
        }
    return pd.concat(frames, ignore_index=True), split_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Project C Stage C1 dataset analysis only.")
    parser.add_argument("--dataset", default=DATASET_NAME)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/huggingface"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/stage1"))
    parser.add_argument("--report", type=Path, default=Path("reports/stage1_dataset_analysis.md"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame, split_metadata = load_frame(args.dataset, args.cache_dir)
    results = analyze(frame, split_metadata, args.artifact_dir, args.report)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
