"""Lightweight validation for the strict Stage C2 model-output contract."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, FrozenSet, Mapping, Optional, Tuple


REQUIRED_KEYS: FrozenSet[str] = frozenset({"intent", "category", "needs_human", "response"})


@dataclass(frozen=True)
class ContractVocabulary:
    intents: FrozenSet[str]
    categories: FrozenSet[str]
    intent_to_category: Mapping[str, str]


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    errors: Tuple[str, ...]
    parsed: Optional[Dict[str, Any]]


def load_vocabulary(config_dir: Optional[Path] = None) -> ContractVocabulary:
    """Load canonical labels and deterministic mapping from Stage C2 configs."""
    if config_dir is None:
        config_dir = Path(__file__).resolve().parents[2] / "configs"
    intent_payload = json.loads((config_dir / "intent_taxonomy.json").read_text(encoding="utf-8"))
    category_payload = json.loads((config_dir / "category_taxonomy.json").read_text(encoding="utf-8"))
    mapping = {entry["intent"]: entry["category"] for entry in intent_payload["intents"]}
    return ContractVocabulary(
        intents=frozenset(mapping),
        categories=frozenset(entry["category"] for entry in category_payload["categories"]),
        intent_to_category=mapping,
    )


def validate_output(raw_output: Any, vocabulary: Optional[ContractVocabulary] = None) -> ValidationResult:
    """Parse and validate one complete raw model output string.

    Error values are stable machine-readable codes. Validation is intentionally
    independent of model libraries and does not execute inference.
    """
    if vocabulary is None:
        vocabulary = load_vocabulary()
    if not isinstance(raw_output, str):
        return ValidationResult(False, ("output_not_string",), None)
    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError:
        return ValidationResult(False, ("malformed_json",), None)
    if not isinstance(parsed, dict):
        return ValidationResult(False, ("root_not_object",), None)

    errors = []
    keys = frozenset(parsed)
    missing = sorted(REQUIRED_KEYS - keys)
    extra = sorted(keys - REQUIRED_KEYS)
    if missing:
        errors.append("missing_keys:" + ",".join(missing))
    if extra:
        errors.append("extra_keys:" + ",".join(extra))

    intent = parsed.get("intent")
    category = parsed.get("category")
    if not isinstance(intent, str) or intent not in vocabulary.intents:
        errors.append("invalid_intent")
    if not isinstance(category, str) or category not in vocabulary.categories:
        errors.append("invalid_category")
    if isinstance(intent, str) and intent in vocabulary.intent_to_category:
        expected_category = vocabulary.intent_to_category[intent]
        if isinstance(category, str) and category in vocabulary.categories and category != expected_category:
            errors.append("intent_category_mismatch")
    if type(parsed.get("needs_human")) is not bool:
        errors.append("needs_human_not_boolean")
    response = parsed.get("response")
    if not isinstance(response, str) or not response.strip():
        errors.append("response_empty")

    return ValidationResult(not errors, tuple(errors), parsed)

