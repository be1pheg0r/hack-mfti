from __future__ import annotations

from typing import cast

from fastapi.testclient import TestClient
from pydantic import BaseModel

from avito.should_split.api.server import create_app
from avito.should_split.core.pipeline import ShouldSplitPipeline
from avito.should_split.domain.models import DraftCandidate
from common.logger import MISTRAL_LOGGER


class _StubResult(BaseModel):
    shouldSplit: bool = True
    categorized_mc_ids: list[int] = [101, 102]
    drafts: list[DraftCandidate] = [
        DraftCandidate(mc_id=101, mc_title="Сантехника", text="Короткий черновик по сантехнике."),
        DraftCandidate(mc_id=102, mc_title="Электрика", text="Короткий черновик по электрике."),
    ]


class _StubResultMultiCats(BaseModel):
    shouldSplit: bool = False
    categorized_mc_ids: list[int] = [101, 102]
    drafts: list[DraftCandidate] = []


class _StubCatalog(BaseModel):
    mc_id_to_title: dict[int, str] = {101: "Сантехника", 102: "Электрика"}


class _StubPipeline:
    def __init__(self) -> None:
        self.catalog = _StubCatalog()

    def invoke(self, description: str) -> _StubResult | _StubResultMultiCats:
        assert description
        if "multicats_false" in description:
            return _StubResultMultiCats()
        return _StubResult()


def test_fastapi_infer_matches_avito_contract() -> None:
    app = create_app(pipeline=cast(ShouldSplitPipeline, _StubPipeline()))
    client = TestClient(app)

    response = client.post(
        "/infer",
        json={
            "itemId": 5001,
            "mcId": 201,
            "mcTitle": "Ремонт квартир и домов под ключ",
            "description": "Делаем ремонт квартир под ключ, а также отдельно выполняем сантехнические и электромонтажные работы.",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert set(payload.keys()) == {"detectedMcIds", "shouldSplit", "drafts"}
    assert payload["shouldSplit"] is True
    assert payload["detectedMcIds"] == [101, 102]
    assert payload["drafts"][0]["mcId"] == 101
    assert payload["drafts"][0]["text"] == "Короткий черновик по сантехнике."


def test_fastapi_health() -> None:
    app = create_app(pipeline=cast(ShouldSplitPipeline, _StubPipeline()))
    client = TestClient(app)

    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_fastapi_infer_returns_categories_when_should_split_false() -> None:
    app = create_app(pipeline=cast(ShouldSplitPipeline, _StubPipeline()))
    client = TestClient(app)

    response = client.post(
        "/infer",
        json={
            "itemId": 5002,
            "mcId": 201,
            "mcTitle": "Ремонт квартир и домов под ключ",
            "description": "multicats_false",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["shouldSplit"] is False
    assert payload["detectedMcIds"] == [101, 102]
    assert payload["drafts"] == []


def test_fastapi_server_mutes_mistral_logger_during_lifespan() -> None:
    original_disabled = MISTRAL_LOGGER.disabled
    original_level = MISTRAL_LOGGER.level

    app = create_app(pipeline=cast(ShouldSplitPipeline, _StubPipeline()))
    with TestClient(app):
        assert MISTRAL_LOGGER.disabled is True

    assert MISTRAL_LOGGER.disabled == original_disabled
    assert MISTRAL_LOGGER.level == original_level


def test_fastapi_server_keeps_mistral_logger_enabled_with_flag() -> None:
    original_disabled = MISTRAL_LOGGER.disabled
    original_level = MISTRAL_LOGGER.level

    app = create_app(pipeline=cast(ShouldSplitPipeline, _StubPipeline()), enable_mistral_logs=True)
    with TestClient(app):
        assert MISTRAL_LOGGER.disabled is original_disabled
        assert MISTRAL_LOGGER.level == original_level

    assert MISTRAL_LOGGER.disabled == original_disabled
    assert MISTRAL_LOGGER.level == original_level



