from __future__ import annotations

import re
from typing import Final


SPLIT_MARKERS: Final[tuple[str, ...]] = (
    # Явное предложение отдельной услуги
    "отдельно",
    "отдельным заказом",
    "как отдельную услугу",
    "в том числе отдельно",
    "берём отдельно",

    # "также делаем X"
    "также выполняем",
    "также делаем",
    "также оказываем",
    "также производим",
    "также занимаемся",
    "также предлагаем",
    "также можем",

    # Помимо / кроме основного
    "помимо",
    "кроме того",
    "помимо этого",
    "дополнительно выполняем",
    "дополнительно предлагаем",
    "дополнительно оказываем",

    # "и X, и Y"
    "а также",
    "и отдельно",

    # Перечисление самостоятельных услуг
    "выполняем любые виды",
    "оказываем любые виды",
    "любые виды работ",
    "все виды работ отдельно",
    "широкий спектр услуг",

    # Прямые предложения
    "принимаем заказы на",
    "выезжаем на",
    "звоните по вопросам",
    "возможен отдельный заказ",

    # Часто встречающиеся формулировки про выделяемые подуслуги
    "отдельные виды работ",
    "частичный ремонт",
    "ремонт ванной комнаты",
    "сантехнические работы",
    "электромонтажные работы",
    "плиточные работы",
    "малярные работы",
    "штукатурные работы",
)

COMPLEX_MARKERS: Final[tuple[str, ...]] = (
    # Классика "под ключ"
    "под ключ",
    "полный цикл",
    "полный комплекс",
    "комплексный ремонт",
    "комплексный подход",
    "комплексные работы",
    "под полный контроль",

    # Включение в состав
    "включая",
    "включает в себя",
    "включает следующие",
    "в состав входит",
    "в перечень входит",
    "в работу входит",
    "входит в стоимость",
    "входит в цену",

    # "в том числе / в т.ч."
    "в том числе",
    "в т.ч.",
    "в т. ч.",

    # "а именно / то есть"
    "а именно",
    "то есть",
    "то бишь",

    # Перечисление как часть целого
    "весь спектр",
    "полный спектр",
    "все этапы",
    "все вид",          # без "отдельно" — скорее комплекс
    "любые вид",        # аналогично
    "от и до",
    "от начала до конца",
    "от демонтажа до",
    "от чернового до",
    "под надзором",

    # Субподрядная/бригадная логика
    "силами нашей бригады",
    "нашими специалистами",
    "в рамках одного договора",
    "в рамках проекта",
)

TURNKEY_SOURCE_TITLE: Final[str] = "под ключ"
GT_SHOULD_SPLIT_RATIO: Final[float] = 0.371

DRAFT_STATUS_STUB: Final[str] = "stub"
DRAFT_STATUS_SKIPPED_SHOULD_SPLIT_FALSE: Final[str] = "skipped_should_split_false"
DRAFT_STATUS_DISABLED_BY_CONFIG: Final[str] = "disabled_by_config"
DRAFT_STATUS_MISTRAL_DISABLED: Final[str] = "mistral_disabled"
DRAFT_STATUS_NO_CATEGORIES: Final[str] = "no_categories"
DRAFT_STATUS_GENERATED: Final[str] = "generated"

SHOULD_SPLIT_API_INFER_PATH: Final[str] = "/infer"
SHOULD_SPLIT_API_HEALTH_PATH: Final[str] = "/health"
SHOULD_SPLIT_API_OK_STATUS: Final[str] = "ok"
SHOULD_SPLIT_API_DEFAULT_HOST: Final[str] = "127.0.0.1"
SHOULD_SPLIT_API_DEFAULT_PORT: Final[int] = 8080
SHOULD_SPLIT_API_ENABLE_MISTRAL_LOGS_DEFAULT: Final[bool] = False

PIPELINE_API_CLIENT_DEFAULT_TIMEOUT_SEC: Final[int] = 60
HTTP_CONTENT_TYPE_JSON: Final[str] = "application/json"
HTTP_METHOD_POST: Final[str] = "POST"

RELABEL_DEFAULT_MODEL_NAME: Final[str] = "mistral-medium-latest"
RELABEL_DEFAULT_SAVE_EVERY: Final[int] = 50
RELABEL_DEFAULT_INPUT_FILENAME: Final[str] = "rnc_dataset_auto_annoted.json"
RELABEL_OUTPUT_TIMESTAMP_FORMAT: Final[str] = "%Y%m%d_%H%M%S"

GRADIO_DEFAULT_HOST: Final[str] = "127.0.0.1"
GRADIO_DEFAULT_PORT: Final[int] = 7860
GRADIO_DEFAULT_API_URL: Final[str] = "http://127.0.0.1:8080"
GRADIO_DEFAULT_ITEM_ID: Final[str] = "0"
GRADIO_DEFAULT_MC_ID: Final[int] = 0
GRADIO_DEFAULT_MC_TITLE: Final[str] = "Неизвестно"

SHOULD_SPLIT_CLEAN_TEXT_PATTERN: Final[re.Pattern] = re.compile(r"[^а-яА-ЯёЁ ]")
SHOULD_SPLIT_MULTI_SPACE_PATTERN: Final[re.Pattern] = re.compile(r" +")

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