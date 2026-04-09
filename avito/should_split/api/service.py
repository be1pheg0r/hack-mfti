from __future__ import annotations

from avito.should_split.api.schemas import AvitoApiRequest, AvitoApiResponse, AvitoDraftResponse
from avito.should_split.core.pipeline import ShouldSplitPipeline


def build_api_response(payload: AvitoApiRequest, pipeline: ShouldSplitPipeline) -> AvitoApiResponse:
    """Преобразует результат пайплайна в формат ответа из docs/cases/avito.md."""
    result = pipeline.invoke(payload.description)
    detected_mc_ids = list(result.categorized_mc_ids)

    drafts: list[AvitoDraftResponse] = [
        AvitoDraftResponse(
            mcId=draft.mc_id,
            mcTitle=draft.mc_title,
            text=draft.text,
        )
        for draft in result.drafts
    ]

    return AvitoApiResponse(
        detectedMcIds=detected_mc_ids,
        shouldSplit=bool(result.shouldSplit),
        drafts=drafts,
    )

