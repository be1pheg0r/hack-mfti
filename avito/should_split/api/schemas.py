from __future__ import annotations

from pydantic import BaseModel, Field


class AvitoApiRequest(BaseModel):
    """Входной контракт API по формату кейса Avito."""

    itemId: int | str
    mcId: int
    mcTitle: str
    description: str


class AvitoDraftResponse(BaseModel):
    """Черновик в выходном API-контракте."""

    mcId: int
    mcTitle: str
    text: str


class AvitoApiResponse(BaseModel):
    """Выходной контракт API по формату кейса Avito."""

    detectedMcIds: list[int] = Field(default_factory=list)
    shouldSplit: bool
    drafts: list[AvitoDraftResponse] = Field(default_factory=list)

