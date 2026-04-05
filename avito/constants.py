from __future__ import annotations

import re
from typing import Final


SPLIT_MARKERS: Final[tuple[str, ...]] = (
	"отдельно",
	"также выполняем",
	"помимо",
)

COMPLEX_MARKERS: Final[tuple[str, ...]] = (
	"под ключ",
	"включая",
	"в том числе",
)

TURNKEY_SOURCE_TITLE: Final[str] = "Ремонт квартир и домов под ключ"
GT_SHOULD_SPLIT_RATIO: Final[float] = 0.371

MANDATORY_TEXT_FEATURES: Final[tuple[str, ...]] = (
	"description_word_count",
	"description_char_count",
	"split_marker_count",
	"complex_marker_count",
	"marker_ratio",
	"has_bullets",
)

MANDATORY_CATEGORICAL_FEATURES: Final[tuple[str, ...]] = (
	"source_mc_id",
	"is_turnkey",
)

_BULLET_PATTERN: Final[re.Pattern] = re.compile(
    r"(^|\n)\s*(?:[-*•]|\d+[.)])\s+\S", 
    flags=re.MULTILINE
)

_SENTENCE_SPLIT_PATTERN: Final[re.Pattern] = re.compile(r"[.!?]+")