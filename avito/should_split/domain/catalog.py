from __future__ import annotations

from pathlib import Path
from typing import *

import pandas as pd
from pydantic import BaseModel, Field


class MicrocategoryCatalog(BaseModel):
    """Словарь микрокатегорий и ключевых фраз.

    Attributes:
        mc_id_to_title: Отображение id микрокатегории в название.
        keyphrases_by_category: Ключевые фразы по id микрокатегории.
        ordered_category_ids: Порядок категорий из словаря.
    """

    mc_id_to_title: dict[int, str]
    keyphrases_by_category: dict[int, tuple[str, ...]]
    ordered_category_ids: list[int] = Field(default_factory=list)


def load_microcategory_catalog(mc_map_fpath: str | Path) -> MicrocategoryCatalog:
    """Загружает словарь микрокатегорий из CSV.

    Args:
        mc_map_fpath: Путь к rnc_mic_key_phrases.csv.

    Returns:
        Нормализованный каталог микрокатегорий.
    """
    mc_map_df = pd.read_csv(mc_map_fpath)

    mc_id_to_title: dict[int, str] = {}
    keyphrases_by_category: dict[int, tuple[str, ...]] = {}
    ordered_category_ids: list[int] = []

    for _, row in mc_map_df.iterrows():
        mc_id = int(row["mcId"])
        mc_title = str(row["mcTitle"]).strip()
        raw_keyphrases = str(row["keyPhrases"])
        keyphrases = tuple(
            phrase.strip().lower()
            for phrase in raw_keyphrases.split(";")
            if phrase.strip()
        )
        mc_id_to_title[mc_id] = mc_title
        keyphrases_by_category[mc_id] = keyphrases
        ordered_category_ids.append(mc_id)

    return MicrocategoryCatalog(
        mc_id_to_title=mc_id_to_title,
        keyphrases_by_category=keyphrases_by_category,
        ordered_category_ids=ordered_category_ids,
    )

