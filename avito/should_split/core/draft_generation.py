from __future__ import annotations

from typing import *

from avito.prompts import draftGenerationInstructionPrompt, draftGenerationPrompt
from avito.should_split.core.config import DraftGenerationConfig, MistralInferenceConfig
from avito.should_split.domain.models import DraftCandidate


def _normalize_draft_text(text: str) -> str:
    """Нормализует текст черновика без ограничения длины."""
    return " ".join(str(text).split())


def generate_draft_candidates(
    description: str,
    categorized_mc_ids: list[int],
    mc_id_to_title: dict[int, str],
    draft_config: DraftGenerationConfig,
    mistral_config: MistralInferenceConfig,
    mistral_caller: Callable[..., str],
) -> list[DraftCandidate]:
    """Генерирует черновики объявлений для списка микрокатегорий."""
    drafts: list[DraftCandidate] = []

    for mc_id in categorized_mc_ids:
        mc_title = mc_id_to_title.get(mc_id, f"mcId={mc_id}")
        prompt = draftGenerationPrompt(
            description=description,
            mc_title=mc_title,
        )
        response = mistral_caller(
            mistral_config.to_call_config(),
            messages=[
                {"role": "system", "content": draftGenerationInstructionPrompt},
                {"role": "user", "content": prompt},
            ],
            reasoning_effort=draft_config.reasoning_effort,
            temperature=draft_config.temperature,
        )
        text = _normalize_draft_text(str(response))
        drafts.append(
            DraftCandidate(
                mc_id=mc_id,
                mc_title=mc_title,
                text=text,
            )
        )

    return drafts
