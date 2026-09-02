"""Build Project C Stage C3 frozen, group-aware dataset splits.

This module only constructs and validates dataset files. It never loads a
model, performs inference, trains, tunes prompts, or evaluates model behavior.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import pandas as pd
from datasets import Dataset

from src.data.analyze_dataset import DATASET_NAME, normalize_instruction
from src.evaluation.contracts import validate_output


SEED = 42
SOURCE_QUALITY_REVISION = "stage3_source_response_quality_gate_v1"
NORMALIZATION_VERSION = "stage1_normalized_instruction_v1"
SPLIT_ORDER = ("train", "validation", "dev", "locked_test")
ASSIGNMENT_ORDER = ("locked_test", "dev", "validation", "train")
TARGET_SIZES: Mapping[str, int] = {
    "train": 2700,
    "validation": 300,
    "dev": 300,
    "locked_test": 300,
}
RESPONSE_MAX_CHARS = 650
RESPONSE_MIN_COMPACTED_CHARS = 250
RESPONSE_PREFERRED_SENTENCES = 2
RESPONSE_MAX_SENTENCES = 4
SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")
LIST_MARKER_RE = re.compile(r"(?:^|\s)(?:\d{1,2}\.|[-•])\s+(?=\S)")
TRAILING_MARKER_RE = re.compile(r"(?:^|\s)(?:\d{1,2}\.|[-•])\s*$")
TRAILING_COLON_MARKER_RE = re.compile(r":\s*(?:\d{1,2}\.|[-•])?\s*$")
INCOMPLETE_LIST_INTRO_RE = re.compile(
    r"(?:follow these (?:steps|instructions)|here are|the following|options|methods)\s*:?\s*(?:\d{1,2}\.)?\s*$",
    flags=re.IGNORECASE,
)
RAW_INCOMPLETE_NUMBERED_END_RE = re.compile(
    r"(?:(?:^|\n)\s*\d{1,2}\.|:\s*\d{1,2}\.)\.{0,2}\s*$"
)
RAW_INCOMPLETE_BULLET_END_RE = re.compile(r"(?:(?:^|\n)\s*[-•]|:\s*[-•])(?:\.{2,3}|…)?\s*$")
SOURCE_INCOMPLETE_INTRO_RE = re.compile(
    r"(?:follow(?:ing)? (?:these|a few simple) (?:steps|instructions)|steps you can follow|"
    r"here are(?: the)?(?: steps| instructions| options| methods| scenarios)?|"
    r"the following|some common scenarios|accepted payment options|available payment options)\s*:?\s*$",
    flags=re.IGNORECASE,
)
STRONG_ENUMERATION_CLAIM_RE = re.compile(
    r"(?:comprehensive (?:list|breakdown)|follow(?:ing)? (?:these|a few simple) (?:steps|instructions)|"
    r"steps you can follow|following (?:steps|instructions|options|methods|scenarios)|"
    r"breakdown of (?:the )?(?:steps|options|methods|scenarios)|some common scenarios|"
    r"here are (?:the )?(?:steps|instructions|options|methods|scenarios))",
    flags=re.IGNORECASE,
)
SOURCE_NUMBERED_ITEM_RE = re.compile(r"(?:^|\s)\d{1,2}\.\s+(?=\S)")
SOURCE_BULLET_ITEM_RE = re.compile(r"(?:^|\s)[-•]\s+(?=\S)")


@dataclass(frozen=True)
class SourceGroup:
    normalized_instruction: str
    group_id: str
    intent: str
    category: str
    source_indices: Tuple[int, ...]

    @property
    def size(self) -> int:
        return len(self.source_indices)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_rank(*parts: Any) -> str:
    material = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def discover_source_arrow(cache_dir: Path) -> Path:
    candidates = sorted(cache_dir.rglob("*-train.arrow"))
    expected = [path for path in candidates if "bitext-customer-support-llm-chatbot-training-dataset" in path.name]
    if len(expected) != 1:
        raise FileNotFoundError(
            f"Expected exactly one cached Bitext train Arrow file under {cache_dir}; found {len(expected)}. "
            "Pass --source-arrow explicitly."
        )
    return expected[0]


def load_source_frame(source_arrow: Path) -> pd.DataFrame:
    dataset = Dataset.from_file(str(source_arrow))
    frame = dataset.to_pandas()
    expected_columns = {"flags", "instruction", "category", "intent", "response"}
    missing = sorted(expected_columns - set(frame.columns))
    if missing:
        raise ValueError(f"Source dataset is missing columns: {missing}")
    frame = frame.reset_index(drop=True)
    frame["source_index"] = frame.index.astype(int)
    frame["normalized_instruction"] = frame["instruction"].map(normalize_instruction)
    return frame


def load_c2_contracts(config_dir: Path) -> Dict[str, Any]:
    intent_payload = json.loads((config_dir / "intent_taxonomy.json").read_text(encoding="utf-8"))
    category_payload = json.loads((config_dir / "category_taxonomy.json").read_text(encoding="utf-8"))
    escalation_payload = json.loads((config_dir / "escalation_policy.json").read_text(encoding="utf-8"))
    schema_payload = json.loads((config_dir / "output_schema.json").read_text(encoding="utf-8"))

    intent_to_category = {entry["intent"]: entry["category"] for entry in intent_payload["intents"]}
    categories = {entry["category"] for entry in category_payload["categories"]}
    escalation = {entry["intent"]: entry["needs_human"] for entry in escalation_payload["intents"]}
    true_count = sum(value is True for value in escalation.values())
    false_count = sum(value is False for value in escalation.values())
    errors = []
    if len(intent_to_category) != intent_payload["intent_count"]:
        errors.append("intent taxonomy count does not match unique entries")
    if len(categories) != category_payload["category_count"]:
        errors.append("category taxonomy count does not match unique entries")
    if set(intent_to_category) != set(escalation):
        errors.append("escalation policy does not cover exactly the canonical intents")
    if set(intent_to_category.values()) - categories:
        errors.append("intent mapping references a non-canonical category")
    if escalation_payload["true_intent_count"] != true_count:
        errors.append("escalation true summary count differs from entries")
    if escalation_payload["false_intent_count"] != false_count:
        errors.append("escalation false summary count differs from entries")
    if set(schema_payload["properties"]["intent"]["enum"]) != set(intent_to_category):
        errors.append("output schema intent enum differs from taxonomy")
    if set(schema_payload["properties"]["category"]["enum"]) != categories:
        errors.append("output schema category enum differs from taxonomy")
    if errors:
        raise ValueError("C2 contract inconsistency: " + "; ".join(errors))
    return {
        "intent_to_category": intent_to_category,
        "categories": categories,
        "escalation": escalation,
        "schema": schema_payload,
        "escalation_true_count": true_count,
        "escalation_false_count": false_count,
    }


def validate_source_contract(frame: pd.DataFrame, contracts: Mapping[str, Any]) -> Dict[str, Any]:
    reasons: Counter = Counter()
    mapping = contracts["intent_to_category"]
    escalation = contracts["escalation"]
    for row in frame.itertuples(index=False):
        if row.intent not in mapping:
            reasons["non_canonical_intent"] += 1
        elif row.category != mapping[row.intent]:
            reasons["intent_category_mismatch"] += 1
        if row.intent not in escalation:
            reasons["missing_escalation_policy"] += 1
        if not isinstance(row.instruction, str) or not row.instruction.strip():
            reasons["empty_instruction"] += 1
        if not isinstance(row.response, str) or not row.response.strip():
            reasons["empty_response"] += 1
    invalid_count = sum(reasons.values())
    if invalid_count:
        raise ValueError(f"Source contract validation failed; invalid observations by reason: {dict(reasons)}")
    return {"invalid_count": 0, "reasons": {}}


def build_source_groups(frame: pd.DataFrame) -> List[SourceGroup]:
    groups: List[SourceGroup] = []
    conflicts = []
    for normalized, rows in frame.groupby("normalized_instruction", sort=True, dropna=False):
        intents = rows["intent"].unique().tolist()
        categories = rows["category"].unique().tolist()
        if len(intents) != 1 or len(categories) != 1:
            conflicts.append(
                {"normalized_instruction": normalized, "intents": intents, "categories": categories, "size": len(rows)}
            )
            continue
        groups.append(
            SourceGroup(
                normalized_instruction=str(normalized),
                group_id=stable_rank("normalized_instruction", normalized),
                intent=str(intents[0]),
                category=str(categories[0]),
                source_indices=tuple(int(value) for value in rows["source_index"].tolist()),
            )
        )
    if conflicts:
        raise ValueError(f"Normalized groups contain label conflicts: {conflicts[:10]}")
    return groups


def build_intent_quotas(intents: Sequence[str], target_sizes: Mapping[str, int], seed: int) -> Dict[str, Dict[str, int]]:
    ordered_intents = sorted(intents)
    quotas: Dict[str, Dict[str, int]] = {}
    for split in SPLIT_ORDER:
        target = target_sizes[split]
        base, remainder = divmod(target, len(ordered_intents))
        extra_order = sorted(ordered_intents, key=lambda intent: stable_rank(seed, "quota", split, intent))
        extras = set(extra_order[:remainder])
        quotas[split] = {intent: base + (intent in extras) for intent in ordered_intents}
    return quotas


def assign_groups(
    groups: Sequence[SourceGroup],
    intents: Sequence[str],
    target_sizes: Mapping[str, int] = TARGET_SIZES,
    seed: int = SEED,
) -> Tuple[Dict[str, str], Dict[str, Dict[str, int]]]:
    """Assign whole normalized groups to splits with exact intent quotas.

    Selection is deterministic: candidate group order and per-split remainder
    intents are ranked by SHA-256 of the seed and canonical values.
    """
    quotas = build_intent_quotas(intents, target_sizes, seed)
    by_intent: MutableMapping[str, List[SourceGroup]] = defaultdict(list)
    for group in groups:
        by_intent[group.intent].append(group)

    assignment: Dict[str, str] = {}
    for intent in sorted(intents):
        available = list(by_intent[intent])
        for split in ASSIGNMENT_ORDER:
            remaining = quotas[split][intent]
            candidates = sorted(
                (group for group in available if group.group_id not in assignment),
                key=lambda group: stable_rank(seed, "assignment", split, intent, group.normalized_instruction),
            )
            for group in candidates:
                if group.size <= remaining:
                    assignment[group.group_id] = split
                    remaining -= group.size
                    if remaining == 0:
                        break
            if remaining:
                raise RuntimeError(
                    f"Could not satisfy group-aware quota for split={split}, intent={intent}; {remaining} rows remain"
                )
    return assignment, quotas


def apply_source_quality_gate(
    frame: pd.DataFrame,
    groups: Sequence[SourceGroup],
    group_assignments: Mapping[str, str],
    seed: int = SEED,
) -> Tuple[Dict[int, str], Dict[str, Any], List[Dict[str, Any]]]:
    """Replace failed selected rows with clean unused singleton-group rows of the same intent."""
    group_by_source = {source_index: group for group in groups for source_index in group.source_indices}
    source_split = {
        source_index: group_assignments[group.group_id]
        for group in groups
        if group.group_id in group_assignments
        for source_index in group.source_indices
    }
    indexed = frame.set_index("source_index", drop=False)
    failures: List[Dict[str, Any]] = []
    for source_index, split in sorted(
        source_split.items(), key=lambda item: (SPLIT_ORDER.index(item[1]), item[0])
    ):
        row = indexed.loc[source_index]
        reasons = validate_source_response_quality(row["response"])
        if reasons:
            failures.append(
                {
                    "split": split,
                    "source_index": int(source_index),
                    "intent": str(row["intent"]),
                    "category": str(row["category"]),
                    "failure_reasons": reasons,
                    "response": str(row["response"]),
                }
            )

    original_selected = set(source_split)
    failed_indices = {failure["source_index"] for failure in failures}
    for source_index in failed_indices:
        del source_split[source_index]

    selected_group_ids = {group_by_source[source_index].group_id for source_index in source_split}
    selected_instructions = {str(indexed.loc[source_index]["instruction"]) for source_index in source_split}
    candidates_by_intent: MutableMapping[str, List[int]] = defaultdict(list)
    for group in groups:
        if group.size != 1 or group.group_id in group_assignments:
            continue
        source_index = group.source_indices[0]
        row = indexed.loc[source_index]
        if validate_source_response_quality(row["response"]):
            continue
        candidates_by_intent[group.intent].append(source_index)

    replacements: List[Dict[str, Any]] = []
    for failure in sorted(
        failures,
        key=lambda item: (
            SPLIT_ORDER.index(item["split"]),
            item["intent"],
            item["source_index"],
        ),
    ):
        split = failure["split"]
        intent = failure["intent"]
        ranked_candidates = sorted(
            candidates_by_intent[intent],
            key=lambda source_index: stable_rank(
                seed, "source_quality_replacement", split, intent, failure["source_index"], source_index
            ),
        )
        chosen = None
        for source_index in ranked_candidates:
            group = group_by_source[source_index]
            instruction = str(indexed.loc[source_index]["instruction"])
            if source_index in source_split:
                continue
            if group.group_id in selected_group_ids:
                continue
            if instruction in selected_instructions:
                continue
            chosen = source_index
            break
        if chosen is None:
            raise RuntimeError(
                f"No clean unused singleton-group replacement for split={split}, intent={intent}, "
                f"failed_source_index={failure['source_index']}"
            )
        chosen_group = group_by_source[chosen]
        chosen_row = indexed.loc[chosen]
        source_split[chosen] = split
        selected_group_ids.add(chosen_group.group_id)
        selected_instructions.add(str(chosen_row["instruction"]))
        candidates_by_intent[intent].remove(chosen)
        replacements.append(
            {
                "split": split,
                "intent": intent,
                "removed_source_index": failure["source_index"],
                "replacement_source_index": int(chosen),
                "replacement_group_id": chosen_group.group_id,
            }
        )

    final_failures = [
        source_index
        for source_index in source_split
        if validate_source_response_quality(indexed.loc[source_index]["response"])
    ]
    if final_failures:
        raise ValueError(f"Source quality gate left failed selected rows: {final_failures[:20]}")

    reason_counts = Counter(reason for failure in failures for reason in failure["failure_reasons"])
    per_split = Counter(replacement["split"] for replacement in replacements)
    audit = {
        "validator_version": "source_response_quality_v1",
        "selected_rows_scanned": len(original_selected),
        "failed_rows": len(failures),
        "failure_reason_counts": dict(sorted(reason_counts.items())),
        "replacements_made": len(replacements),
        "per_split_replacement_counts": {split: per_split[split] for split in SPLIT_ORDER},
        "replacement_strategy": (
            "Deterministic SHA-256 ranking among unused, source-quality-PASS, singleton normalized groups "
            "with the same canonical intent; no content repair or generation."
        ),
        "final_selected_quality_failures": 0,
        "replacements": replacements,
    }
    return source_split, audit, failures


def validate_compaction_completeness(original: str, candidate: str) -> Tuple[str, ...]:
    """Return reasons a shortened response would be structurally incomplete."""
    reasons = []
    if len(candidate) < len(original) and LIST_MARKER_RE.search(original):
        reasons.append("list_block_would_be_truncated")
    if TRAILING_COLON_MARKER_RE.search(candidate):
        reasons.append("trailing_colon_or_marker")
    if TRAILING_MARKER_RE.search(candidate):
        reasons.append("standalone_trailing_list_marker")
    if INCOMPLETE_LIST_INTRO_RE.search(candidate):
        reasons.append("incomplete_list_introduction")
    return tuple(dict.fromkeys(reasons))


def validate_source_response_quality(text: Any) -> Tuple[str, ...]:
    """Conservatively detect structurally incomplete raw source responses."""
    if not isinstance(text, str) or not text.strip():
        return ("empty_source_response",)
    raw = text.strip()
    normalized = re.sub(r"\s+", " ", raw).strip()
    intro_normalized = re.sub(r"(?:\.{2,3}|…)\s*$", "", normalized).strip()
    reasons = []
    if RAW_INCOMPLETE_NUMBERED_END_RE.search(raw) or re.search(r":\s*\d{1,2}\.\s*$", normalized):
        reasons.append("incomplete_numbered_list")
    if RAW_INCOMPLETE_BULLET_END_RE.search(raw) or re.search(r":\s*[-•]\s*$", normalized):
        reasons.append("incomplete_bullet_list")
    if SOURCE_INCOMPLETE_INTRO_RE.search(intro_normalized):
        reasons.append("incomplete_list_introduction")

    enumeration_count = len(SOURCE_NUMBERED_ITEM_RE.findall(normalized)) + len(
        SOURCE_BULLET_ITEM_RE.findall(normalized)
    )
    if STRONG_ENUMERATION_CLAIM_RE.search(normalized) and enumeration_count == 1:
        reasons.append("suspected_incomplete_list")
    return tuple(dict.fromkeys(reasons))


def compact_response(text: str) -> Tuple[str, str, Tuple[str, ...]]:
    """Normalize whitespace and compact only when prose/list structure stays complete."""
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return normalized, "invalid_empty", ()
    if len(normalized) <= RESPONSE_MAX_CHARS:
        return normalized, "preserved_within_limit", ()
    if LIST_MARKER_RE.search(normalized):
        return normalized, "completeness_full_fallback", ("list_block_would_be_truncated",)

    sentences = [sentence.strip() for sentence in SENTENCE_BOUNDARY_RE.split(normalized) if sentence.strip()]
    if len(sentences) < 2:
        return normalized, "conservative_full_fallback", ()

    selected = sentences[:RESPONSE_PREFERRED_SENTENCES]
    while selected and len(" ".join(selected)) > RESPONSE_MAX_CHARS:
        selected.pop()
    next_index = len(selected)
    while (
        selected
        and len(" ".join(selected)) < RESPONSE_MIN_COMPACTED_CHARS
        and next_index < len(sentences)
        and len(selected) < RESPONSE_MAX_SENTENCES
    ):
        candidate = " ".join(selected + [sentences[next_index]])
        if len(candidate) > RESPONSE_MAX_CHARS:
            break
        selected.append(sentences[next_index])
        next_index += 1

    compacted = " ".join(selected).strip()
    if len(compacted) < RESPONSE_MIN_COMPACTED_CHARS or len(compacted) >= len(normalized):
        return normalized, "conservative_full_fallback", ()
    completeness_errors = validate_compaction_completeness(normalized, compacted)
    if completeness_errors:
        return normalized, "completeness_full_fallback", completeness_errors
    return compacted, "compacted_complete_sentence_prefix", ()


def build_records(
    frame: pd.DataFrame,
    groups: Sequence[SourceGroup],
    source_split_assignments: Mapping[int, str],
    contracts: Mapping[str, Any],
    seed: int,
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    group_by_source_index = {
        source_index: group
        for group in groups
        for source_index in group.source_indices
    }
    records: Dict[str, List[Dict[str, Any]]] = {split: [] for split in SPLIT_ORDER}
    strategy_counts: Counter = Counter()
    completeness_failure_reasons: Counter = Counter()
    original_lengths: List[int] = []
    final_lengths: List[int] = []
    invalid_reasons: Counter = Counter()

    indexed = frame.set_index("source_index", drop=False)
    for source_index, split in source_split_assignments.items():
        group = group_by_source_index[source_index]
        row = indexed.loc[source_index]
        response, strategy, completeness_errors = compact_response(str(row["response"]))
        target = {
            "intent": str(row["intent"]),
            "category": str(row["category"]),
            "needs_human": contracts["escalation"][str(row["intent"])],
            "response": response,
        }
        validation = validate_output(json.dumps(target, ensure_ascii=False))
        if not validation.valid:
            for reason in validation.errors:
                invalid_reasons[reason] += 1
            continue
        records[split].append(
            {
                "instruction": str(row["instruction"]).strip(),
                "target": target,
                "metadata": {
                    "source_index": int(source_index),
                    "normalized_instruction": group.normalized_instruction,
                    "group_id": group.group_id,
                    "response_compaction": strategy,
                },
            }
        )
        strategy_counts[strategy] += 1
        completeness_failure_reasons.update(completeness_errors)
        original_lengths.append(len(re.sub(r"\s+", " ", str(row["response"])).strip()))
        final_lengths.append(len(response))

    if invalid_reasons:
        raise ValueError(f"Generated target validation failed: {dict(invalid_reasons)}")
    for split in SPLIT_ORDER:
        records[split].sort(
            key=lambda record: stable_rank(seed, "record_order", split, record["metadata"]["source_index"])
        )
    compaction = {
        "max_characters_for_compacted_prefix": RESPONSE_MAX_CHARS,
        "minimum_compacted_prefix_characters": RESPONSE_MIN_COMPACTED_CHARS,
        "preferred_sentence_count": RESPONSE_PREFERRED_SENTENCES,
        "maximum_sentence_count_when_more_context_is_needed": RESPONSE_MAX_SENTENCES,
        "strategy_counts": dict(sorted(strategy_counts.items())),
        "completeness_failure_count": strategy_counts["completeness_full_fallback"],
        "completeness_failure_reason_observations": sum(completeness_failure_reasons.values()),
        "completeness_failure_reasons": dict(sorted(completeness_failure_reasons.items())),
        "original_mean_characters": round(sum(original_lengths) / len(original_lengths), 6),
        "final_mean_characters": round(sum(final_lengths) / len(final_lengths), 6),
        "final_max_characters": max(final_lengths),
        "invalid_count": 0,
    }
    return records, compaction


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def count_distributions(records: Mapping[str, Sequence[Mapping[str, Any]]]) -> Dict[str, Dict[str, Dict[str, int]]]:
    result: Dict[str, Dict[str, Dict[str, int]]] = {}
    for split, split_records in records.items():
        result[split] = {
            "intent": dict(sorted(Counter(record["target"]["intent"] for record in split_records).items())),
            "category": dict(sorted(Counter(record["target"]["category"] for record in split_records).items())),
            "needs_human": {
                "false": sum(record["target"]["needs_human"] is False for record in split_records),
                "true": sum(record["target"]["needs_human"] is True for record in split_records),
            },
        }
    return result


def overlap_summary(records: Mapping[str, Sequence[Mapping[str, Any]]]) -> Dict[str, Any]:
    extractors = {
        "source_row_index_overlap": lambda record: record["metadata"]["source_index"],
        "exact_instruction_overlap": lambda record: record["instruction"],
        "normalized_instruction_overlap": lambda record: record["metadata"]["normalized_instruction"],
        "group_overlap": lambda record: record["metadata"]["group_id"],
    }
    result: Dict[str, Any] = {"all_overlap_counts_zero": True, "checks": {}}
    for check, extractor in extractors.items():
        owners: MutableMapping[Any, set] = defaultdict(set)
        for split, split_records in records.items():
            for record in split_records:
                owners[extractor(record)].add(split)
        overlaps = [(value, sorted(splits)) for value, splits in owners.items() if len(splits) > 1]
        result["checks"][check] = {
            "overlap_count": len(overlaps),
            "examples": [{"value": str(value), "splits": splits} for value, splits in overlaps[:10]],
        }
        if overlaps:
            result["all_overlap_counts_zero"] = False
    return result


def write_group_assignment(
    path: Path, groups: Sequence[SourceGroup], source_split_assignments: Mapping[int, str]
) -> None:
    group_owners: MutableMapping[str, set] = defaultdict(set)
    selected_counts: Counter = Counter()
    for group in groups:
        for source_index in group.source_indices:
            split = source_split_assignments.get(source_index)
            if split is not None:
                group_owners[group.group_id].add(split)
                selected_counts[group.group_id] += 1
    leaking_groups = {group_id: owners for group_id, owners in group_owners.items() if len(owners) > 1}
    if leaking_groups:
        raise ValueError(f"Group assignment crosses splits: {leaking_groups}")
    final_group_assignments = {
        group_id: next(iter(owners)) for group_id, owners in group_owners.items() if owners
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "group_id",
                "normalized_instruction",
                "group_size",
                "selected_row_count",
                "intent",
                "category",
                "assigned_split",
            ],
        )
        writer.writeheader()
        for group in sorted(groups, key=lambda item: (final_group_assignments.get(item.group_id, "not_selected"), item.intent, item.normalized_instruction)):
            writer.writerow(
                {
                    "group_id": group.group_id,
                    "normalized_instruction": group.normalized_instruction,
                    "group_size": group.size,
                    "selected_row_count": selected_counts[group.group_id],
                    "intent": group.intent,
                    "category": group.category,
                    "assigned_split": final_group_assignments.get(group.group_id, "not_selected"),
                }
            )


def write_distribution(path: Path, records: Mapping[str, Sequence[Mapping[str, Any]]], intents: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "intent", "count", "percentage"])
        writer.writeheader()
        for split in SPLIT_ORDER:
            counts = Counter(record["target"]["intent"] for record in records[split])
            total = len(records[split])
            for intent in sorted(intents):
                count = counts[intent]
                writer.writerow(
                    {
                        "split": split,
                        "intent": intent,
                        "count": count,
                        "percentage": round(count * 100 / total, 6) if total else 0.0,
                    }
                )


def write_samples(path: Path, records: Mapping[str, Sequence[Mapping[str, Any]]], seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        fields = ["split", "instruction", "intent", "category", "needs_human", "response"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for split in SPLIT_ORDER:
            sample = sorted(
                records[split],
                key=lambda record: stable_rank(seed, "manual_qa", split, record["metadata"]["source_index"]),
            )[:5]
            for record in sample:
                writer.writerow(
                    {
                        "split": split,
                        "instruction": record["instruction"],
                        "intent": record["target"]["intent"],
                        "category": record["target"]["category"],
                        "needs_human": record["target"]["needs_human"],
                        "response": record["target"]["response"],
                    }
                )


def write_source_quality_examples(path: Path, failures: Sequence[Mapping[str, Any]], seed: int) -> None:
    """Write a small trailing-context QA sample, with at most two Locked Test examples."""
    ranked = sorted(
        failures,
        key=lambda item: stable_rank(seed, "source_quality_example", item["split"], item["source_index"]),
    )
    selected = []
    locked_count = 0
    for failure in ranked:
        if failure["split"] == "locked_test":
            if locked_count >= 2:
                continue
            locked_count += 1
        selected.append(failure)
        if len(selected) >= 50:
            break
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        fields = ["split", "source_index", "intent", "failure_reason", "response"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for failure in selected:
            response = re.sub(r"\s+", " ", str(failure["response"])).strip()
            response_excerpt = response if len(response) <= 500 else "…" + response[-499:]
            writer.writerow(
                {
                    "split": failure["split"],
                    "source_index": failure["source_index"],
                    "intent": failure["intent"],
                    "failure_reason": ";".join(failure["failure_reasons"]),
                    "response": response_excerpt,
                }
            )


def _fingerprint(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _snapshot_records(records: Mapping[str, Sequence[Mapping[str, Any]]]) -> Dict[str, Any]:
    snapshot: Dict[str, Any] = {}
    for split in SPLIT_ORDER:
        split_records = records[split]
        source_indices = sorted(record["metadata"]["source_index"] for record in split_records)
        normalized_assignments = sorted(
            (record["metadata"]["source_index"], record["metadata"]["normalized_instruction"])
            for record in split_records
        )
        invariant_rows = sorted(
            (
                record["metadata"]["source_index"],
                record["instruction"],
                record["metadata"]["normalized_instruction"],
                record["metadata"]["group_id"],
                record["target"]["intent"],
                record["target"]["category"],
                record["target"]["needs_human"],
            )
            for record in split_records
        )
        responses = {record["metadata"]["source_index"]: record["target"]["response"] for record in split_records}
        intent_counts = dict(sorted(Counter(record["target"]["intent"] for record in split_records).items()))
        category_counts = dict(sorted(Counter(record["target"]["category"] for record in split_records).items()))
        needs_human_counts = {
            "false": sum(record["target"]["needs_human"] is False for record in split_records),
            "true": sum(record["target"]["needs_human"] is True for record in split_records),
        }
        snapshot[split] = {
            "source_indices": source_indices,
            "normalized_assignments": normalized_assignments,
            "invariant_rows": invariant_rows,
            "responses": responses,
            "intent_counts": intent_counts,
            "category_counts": category_counts,
            "needs_human_counts": needs_human_counts,
            "membership_sha256": _fingerprint(source_indices),
            "normalized_assignment_sha256": _fingerprint(normalized_assignments),
            "non_response_invariants_sha256": _fingerprint(invariant_rows),
        }
    return snapshot


def capture_existing_split_snapshot(processed_dir: Path) -> Optional[Dict[str, Any]]:
    paths = {split: processed_dir / f"{split}.jsonl" for split in SPLIT_ORDER}
    if not any(path.exists() for path in paths.values()):
        return None
    missing = [split for split, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Cannot verify membership preservation; existing split files are incomplete: {missing}")
    existing: Dict[str, List[Dict[str, Any]]] = {}
    for split, path in paths.items():
        with path.open(encoding="utf-8") as handle:
            existing[split] = [json.loads(line) for line in handle if line.strip()]
    return _snapshot_records(existing)


def compare_split_snapshots(
    previous: Optional[Mapping[str, Any]],
    current_records: Mapping[str, Sequence[Mapping[str, Any]]],
    require_unchanged_membership: bool = True,
) -> Dict[str, Any]:
    current = _snapshot_records(current_records)
    if previous is None:
        return {"previous_splits_present": False, "all_membership_and_non_response_invariants_unchanged": None}
    per_split: Dict[str, Any] = {}
    all_unchanged = True
    for split in SPLIT_ORDER:
        old = previous[split]
        new = current[split]
        source_unchanged = old["source_indices"] == new["source_indices"]
        normalized_unchanged = old["normalized_assignments"] == new["normalized_assignments"]
        invariants_unchanged = old["invariant_rows"] == new["invariant_rows"]
        old_indices = set(old["source_indices"])
        new_indices = set(new["source_indices"])
        changed_responses = sum(
            old["responses"].get(source_index) != response
            for source_index, response in new["responses"].items()
        )
        split_unchanged = source_unchanged and normalized_unchanged and invariants_unchanged
        all_unchanged = all_unchanged and split_unchanged
        per_split[split] = {
            "source_index_membership_unchanged": source_unchanged,
            "normalized_instruction_assignment_unchanged": normalized_unchanged,
            "intent_category_needs_human_and_instruction_unchanged": invariants_unchanged,
            "size_unchanged": len(old_indices) == len(new_indices),
            "intent_distribution_unchanged": old["intent_counts"] == new["intent_counts"],
            "category_distribution_unchanged": old["category_counts"] == new["category_counts"],
            "needs_human_distribution_unchanged": old["needs_human_counts"] == new["needs_human_counts"],
            "removed_source_row_count": len(old_indices - new_indices),
            "added_source_row_count": len(new_indices - old_indices),
            "old_membership_sha256": old["membership_sha256"],
            "new_membership_sha256": new["membership_sha256"],
            "old_normalized_assignment_sha256": old["normalized_assignment_sha256"],
            "new_normalized_assignment_sha256": new["normalized_assignment_sha256"],
            "old_non_response_invariants_sha256": old["non_response_invariants_sha256"],
            "new_non_response_invariants_sha256": new["non_response_invariants_sha256"],
            "changed_response_count": changed_responses,
        }
    result = {
        "previous_splits_present": True,
        "all_membership_and_non_response_invariants_unchanged": all_unchanged,
        "per_split": per_split,
        "total_changed_response_count": sum(value["changed_response_count"] for value in per_split.values()),
        "all_sizes_and_label_distributions_unchanged": all(
            value["size_unchanged"]
            and value["intent_distribution_unchanged"]
            and value["category_distribution_unchanged"]
            and value["needs_human_distribution_unchanged"]
            for value in per_split.values()
        ),
    }
    if require_unchanged_membership and not all_unchanged:
        raise ValueError(f"Response-compaction rebuild changed split membership or non-response invariants: {result}")
    return result


def build_report(
    source_row_count: int,
    target_sizes: Mapping[str, int],
    actual_sizes: Mapping[str, int],
    distributions: Mapping[str, Mapping[str, Mapping[str, int]]],
    compaction: Mapping[str, Any],
    overlap: Mapping[str, Any],
    hashes: Mapping[str, Any],
    superseded_locked_hash: Optional[str],
    membership_preservation: Mapping[str, Any],
    source_quality: Mapping[str, Any],
    selected_group_count: int,
    total_group_count: int,
) -> str:
    size_rows = "\n".join(
        f"| {split} | {target_sizes[split]:,} | {actual_sizes[split]:,} | {len(distributions[split]['intent'])} | {len(distributions[split]['category'])} |"
        for split in SPLIT_ORDER
    )
    intent_rows = []
    category_rows = []
    escalation_rows = []
    for split in SPLIT_ORDER:
        intent_counts = distributions[split]["intent"]
        category_counts = distributions[split]["category"]
        escalation = distributions[split]["needs_human"]
        intent_rows.append(
            f"| {split} | {min(intent_counts.values())} | {max(intent_counts.values())} | {len(intent_counts)} |"
        )
        category_rows.append(f"| {split} | {len(category_counts)} | {', '.join(category_counts)} |")
        escalation_rows.append(
            f"| {split} | {escalation['true']:,} | {escalation['false']:,} |"
        )
    overlap_rows = "\n".join(
        f"| {name} | {details['overlap_count']} |" for name, details in overlap["checks"].items()
    )
    strategy_text = ", ".join(
        f"`{name}`: {count:,}" for name, count in compaction["strategy_counts"].items()
    )
    replacement_rows = "\n".join(
        f"| {split} | {details['removed_source_row_count']:,} | {details['added_source_row_count']:,} | {details['size_unchanged']} | {details['intent_distribution_unchanged']} | {details['category_distribution_unchanged']} | {details['needs_human_distribution_unchanged']} |"
        for split, details in membership_preservation.get("per_split", {}).items()
    )
    quality_reasons = ", ".join(
        f"`{reason}`: {count:,}" for reason, count in source_quality["failure_reason_counts"].items()
    ) or "none"
    completeness_reasons = ", ".join(
        f"`{reason}`: {count:,}" for reason, count in compaction["completeness_failure_reasons"].items()
    ) or "none"
    locked_hash_outcome = (
        "Locked Test bytes changed, so the digest changed."
        if superseded_locked_hash != hashes["locked_test"]["sha256"]
        else "No Locked Test row required replacement, so the recalculated digest is unchanged."
    )
    return f"""# Stage C3 Frozen Dataset Construction

## Split Strategy

The source is the complete {source_row_count:,}-row Hugging Face Bitext dataset. A deterministic seed of `{SEED}` and stable SHA-256 ranking are used. The selected subset targets Train {target_sizes['train']:,}, Validation {target_sizes['validation']:,}, Dev {target_sizes['dev']:,}, and Locked Test {target_sizes['locked_test']:,} rows. Selection occurs at group level and is intent-aware; no row-level split followed by duplicate repair is used.

Priority order is: zero group leakage, intent coverage/balance, then target size. The source-quality gate removes only failed selected source rows and replaces each one with a clean, previously unused singleton normalized group from the same canonical intent. It never rewrites or completes response content.

## Source Response Quality Gate

Raw source response validation runs before response compaction. It rejects empty responses, incomplete numbered or bulleted markers, incomplete list introductions, and conservative suspected partial enumerations. Every failed selected row receives exactly one deterministic same-intent replacement that passes source quality validation.

- Selected source responses scanned: {source_quality['selected_rows_scanned']:,}
- Failed selected rows: {source_quality['failed_rows']:,}
- Replacements made: {source_quality['replacements_made']:,}
- Final selected quality failures: {source_quality['final_selected_quality_failures']:,}
- Failure reasons: {quality_reasons}
- Audit: `artifacts/stage3/source_response_quality.json`
- Minimal QA examples: `artifacts/stage3/source_response_quality_examples.csv`

| Split | Removed rows | Added rows | Size unchanged | Intent distribution unchanged | Category distribution unchanged | Escalation distribution unchanged |
|---|---:|---:|---|---|---|---|
{replacement_rows}

## Group-aware Method

The analysis-only Stage C1 `normalize_instruction` function is reused verbatim. Each unique normalized instruction forms one group. Initial selection is whole-group and intent-aware. Quality-gate replacements use only unused singleton groups; removing one failed row from an initially selected multi-row group is permitted, while the remaining rows retain their original split. A normalized group can never cross splits.

- Total source groups: {total_group_count:,}
- Selected groups: {selected_group_count:,}
- Group key: `normalized_instruction`
- Assignment manifest: `data/manifests/group_assignment.csv`

## Actual Split Sizes

| Split | Target | Actual | Intent coverage | Category coverage |
|---|---:|---:|---:|---:|
{size_rows}

## Intent Distribution

| Split | Minimum rows per intent | Maximum rows per intent | Covered intents |
|---|---:|---:|---:|
{chr(10).join(intent_rows)}

The complete 108-row split × intent distribution is in `artifacts/stage3/split_distribution.csv`.

## Category Distribution

| Split | Covered categories | Categories |
|---|---:|---|
{chr(10).join(category_rows)}

## Escalation Distribution

`needs_human` is copied deterministically from the confirmed C2 intent policy (6 true intents, 21 false intents); it is never inferred from response text.

| Split | True rows | False rows |
|---|---:|---:|
{chr(10).join(escalation_rows)}

## Response Compaction Policy

No LLM, synthetic generation, or paraphrasing is used. Leading/trailing whitespace is removed and internal whitespace is collapsed. Responses at or below {RESPONSE_MAX_CHARS} characters are retained. Longer normal prose is shortened only to a prefix ending at a complete sentence boundary: the first {RESPONSE_PREFERRED_SENTENCES} sentences are preferred, with up to {RESPONSE_MAX_SENTENCES} when needed to retain at least {RESPONSE_MIN_COMPACTED_CHARS} characters of context.

For numbered or bulleted lists (`1.`, `2.`, `-`, `•`), compacting is rejected whenever it would omit any part of the list block. Candidates ending with a colon, a standalone list marker, or an incomplete list introduction are also rejected. Every rejected candidate falls back to the complete whitespace-normalized source response. Therefore neither mid-sentence nor mid-list truncation is allowed.

- Strategies: {strategy_text}
- Responses rejected by completeness validation and restored in full: {compaction['completeness_failure_count']:,}
- Completeness failure reasons: {completeness_reasons}
- Mean normalized original length: {compaction['original_mean_characters']:.3f} characters
- Mean final length: {compaction['final_mean_characters']:.3f} characters
- Maximum final length (conservative fallbacks may exceed the compacting limit): {compaction['final_max_characters']:,} characters

## Schema Validation

Every selected target was validated by the existing strict C2 helper for exact keys, canonical labels, intent-category consistency, boolean `needs_human`, and a non-empty response.

- Invalid generated targets: {compaction['invalid_count']}
- Silently discarded invalid rows: 0

## Leakage Validation

| Check | Cross-split overlap count |
|---|---:|
{overlap_rows}

All required cross-split overlap counts are zero: **{str(overlap['all_overlap_counts_zero']).lower()}**.

## Locked Test Freeze

The previous Stage C3 Locked Test hash record `{superseded_locked_hash or 'not applicable'}` is superseded by this quality-gate revision. The recalculated `data/processed/locked_test.jsonl` SHA-256 is `{hashes['locked_test']['sha256']}`. {locked_hash_outcome} Split size, intent quota, category coverage, and escalation distribution remain unchanged. It must not be used for prompt tuning, hyperparameter tuning, behavioral error analysis, evaluator changes, threshold changes, or candidate selection. Its first authorized behavioral use is Stage C7 after the C6.5 freeze.

## Manual QA Required

`artifacts/stage3/split_samples.csv` contains 5 deterministic samples from each split (20 total) using seed {SEED}.

> **這一步需要你手動做**
>
> 1. Open `artifacts/stage3/split_samples.csv`.
> 2. Review instruction/intent/category consistency, the confirmed `needs_human` value, response relevance, and whether compaction retained enough meaning.
> 3. Record concerns without editing the frozen JSONL files, especially `locked_test.jsonl`.

## Stage C3 Conclusion

Stage C3 passed construction-time validation: source-quality failures were excluded and replaced without content repair, all four files retain their required sizes and label distributions, each covers all 27 intents and 11 categories, every target conforms to the C2 contract, and every cross-split leakage check is zero. The final Locked Test hash is recorded. No model loading, inference, training, prompt tuning, development evaluation, or locked behavioral evaluation was performed.
"""


def build_stage3(
    repo_root: Path,
    source_arrow: Optional[Path] = None,
    seed: int = SEED,
    target_sizes: Mapping[str, int] = TARGET_SIZES,
) -> Dict[str, Any]:
    config_dir = repo_root / "configs"
    processed_dir = repo_root / "data/processed"
    manifest_dir = repo_root / "data/manifests"
    artifact_dir = repo_root / "artifacts/stage3"
    report_path = repo_root / "reports/split_validation_report.md"
    cache_dir = repo_root / ".cache/huggingface"
    source_arrow = source_arrow or discover_source_arrow(cache_dir)

    previous_hash_path = manifest_dir / "dataset_hashes.json"
    previous_hash_payload = (
        json.loads(previous_hash_path.read_text(encoding="utf-8")) if previous_hash_path.exists() else {}
    )
    previous_locked_file_hash = (
        sha256_file(processed_dir / "locked_test.jsonl")
        if (processed_dir / "locked_test.jsonl").exists()
        else None
    )
    previous_locked_metadata = previous_hash_payload.get("locked_test", {})
    if previous_locked_metadata.get("revision_id") == SOURCE_QUALITY_REVISION:
        superseded_locked_hash = previous_locked_metadata.get("supersedes_sha256")
    else:
        superseded_locked_hash = previous_locked_file_hash

    contracts = load_c2_contracts(config_dir)
    frame = load_source_frame(source_arrow)
    source_validation = validate_source_contract(frame, contracts)
    groups = build_source_groups(frame)
    intents = sorted(contracts["intent_to_category"])
    assignments, quotas = assign_groups(groups, intents, target_sizes, seed)
    baseline_source_assignments = {
        source_index: assignments[group.group_id]
        for group in groups
        if group.group_id in assignments
        for source_index in group.source_indices
    }
    baseline_records, _ = build_records(
        frame, groups, baseline_source_assignments, contracts, seed
    )
    source_split_assignments, source_quality, quality_failures = apply_source_quality_gate(
        frame, groups, assignments, seed
    )
    records, compaction = build_records(frame, groups, source_split_assignments, contracts, seed)
    membership_preservation = compare_split_snapshots(
        _snapshot_records(baseline_records), records, require_unchanged_membership=False
    )
    if not membership_preservation["all_sizes_and_label_distributions_unchanged"]:
        raise ValueError(f"Source-quality replacements changed split sizes or label distributions: {membership_preservation}")
    actual_sizes = {split: len(records[split]) for split in SPLIT_ORDER}
    if actual_sizes != dict(target_sizes):
        raise ValueError(f"Actual split sizes differ from targets: actual={actual_sizes}, targets={dict(target_sizes)}")

    overlap = overlap_summary(records)
    if not overlap["all_overlap_counts_zero"]:
        raise ValueError(f"Cross-split leakage detected: {overlap}")
    distributions = count_distributions(records)
    for split in SPLIT_ORDER:
        if set(distributions[split]["intent"]) != set(intents):
            raise ValueError(f"{split} does not cover all canonical intents")
        if set(distributions[split]["category"]) != contracts["categories"]:
            raise ValueError(f"{split} does not cover all canonical categories")

    for split in SPLIT_ORDER:
        write_jsonl(processed_dir / f"{split}.jsonl", records[split])
    write_group_assignment(manifest_dir / "group_assignment.csv", groups, source_split_assignments)
    write_distribution(artifact_dir / "split_distribution.csv", records, intents)
    write_samples(artifact_dir / "split_samples.csv", records, seed)
    write_json(artifact_dir / "cross_split_overlap.json", overlap)
    write_json(artifact_dir / "source_response_quality.json", source_quality)
    write_source_quality_examples(
        artifact_dir / "source_response_quality_examples.csv", quality_failures, seed
    )

    dataset_hashes = {
        "algorithm": "SHA-256",
        "files": {
            split: {
                "path": f"data/processed/{split}.jsonl",
                "sha256": sha256_file(processed_dir / f"{split}.jsonl"),
                "locked": split == "locked_test",
            }
            for split in SPLIT_ORDER
        },
        "locked_test": {
            "revision_id": SOURCE_QUALITY_REVISION,
            "path": "data/processed/locked_test.jsonl",
            "sha256": sha256_file(processed_dir / "locked_test.jsonl"),
            "supersedes_sha256": superseded_locked_hash,
            "frozen": True,
            "first_authorized_behavioral_use": "Stage C7 after Stage C6.5 freeze",
        },
    }
    write_json(manifest_dir / "dataset_hashes.json", dataset_hashes)

    group_file_hash = sha256_file(manifest_dir / "group_assignment.csv")
    group_by_source = {
        source_index: group for group in groups for source_index in group.source_indices
    }
    selected_group_count = len(
        {group_by_source[source_index].group_id for source_index in source_split_assignments}
    )
    manifest = {
        "seed": seed,
        "source_dataset": DATASET_NAME,
        "source_row_count": len(frame),
        "source_arrow_path": str(source_arrow.relative_to(repo_root)),
        "source_arrow_sha256": sha256_file(source_arrow),
        "normalization": {
            "version": NORMALIZATION_VERSION,
            "implementation": "src.data.analyze_dataset.normalize_instruction",
            "description": "Unicode NFKC/common punctuation normalization, placeholder replacement with <ENTITY>, lowercase, trim, and whitespace/punctuation spacing normalization.",
        },
        "target_sizes": dict(target_sizes),
        "actual_split_sizes": actual_sizes,
        "selected_row_count": sum(actual_sizes.values()),
        "unselected_source_row_count": len(frame) - sum(actual_sizes.values()),
        "source_group_count": len(groups),
        "selected_group_count": selected_group_count,
        "group_assignment_sha256": group_file_hash,
        "group_aware": True,
        "group_key": "normalized_instruction",
        "assignment_method": (
            "Intent quotas plus deterministic SHA-256 whole-group ranking, followed by deterministic "
            "same-intent source-quality replacements from unused singleton groups."
        ),
        "per_split_intent_targets": quotas,
        "per_split_intent_counts": {split: distributions[split]["intent"] for split in SPLIT_ORDER},
        "per_split_category_counts": {split: distributions[split]["category"] for split in SPLIT_ORDER},
        "per_split_needs_human_counts": {split: distributions[split]["needs_human"] for split in SPLIT_ORDER},
        "response_compaction": compaction,
        "source_response_quality": source_quality,
        "source_schema_validation": source_validation,
        "generated_target_schema_validation": {"invalid_count": compaction["invalid_count"], "reasons": {}},
        "membership_preservation": membership_preservation,
        "source_response_quality_gate": {
            "revision_id": SOURCE_QUALITY_REVISION,
            "old_locked_test_sha256": superseded_locked_hash,
            "new_locked_test_sha256": dataset_hashes["locked_test"]["sha256"],
            "previous_hash_record_superseded": True,
            "locked_test_content_changed": superseded_locked_hash != dataset_hashes["locked_test"]["sha256"],
            "failed_rows": source_quality["failed_rows"],
            "replacements_made": source_quality["replacements_made"],
        },
        "cross_split_overlap": overlap,
        "locked_test": dataset_hashes["locked_test"],
        "contract_file_sha256": {
            name: sha256_file(config_dir / name)
            for name in (
                "intent_taxonomy.json",
                "category_taxonomy.json",
                "escalation_policy.json",
                "output_schema.json",
                "promotion_gate.json",
            )
        },
    }
    write_json(manifest_dir / "split_manifest.json", manifest)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        build_report(
            len(frame),
            target_sizes,
            actual_sizes,
            distributions,
            compaction,
            overlap,
            dataset_hashes["files"],
            superseded_locked_hash,
            membership_preservation,
            source_quality,
            selected_group_count,
            len(groups),
        ),
        encoding="utf-8",
    )
    return {
        "actual_split_sizes": actual_sizes,
        "selected_group_count": selected_group_count,
        "source_group_count": len(groups),
        "schema_invalid_count": compaction["invalid_count"],
        "source_quality_failures": source_quality["failed_rows"],
        "source_quality_replacements": source_quality["replacements_made"],
        "all_overlap_counts_zero": overlap["all_overlap_counts_zero"],
        "locked_test_sha256": dataset_hashes["locked_test"]["sha256"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Project C Stage C3 frozen group-aware splits only.")
    parser.add_argument("--source-arrow", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    result = build_stage3(repo_root, args.source_arrow)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
