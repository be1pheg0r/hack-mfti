from __future__ import annotations

from pathlib import Path

import pytest

from avito.should_split.domain.catalog import MicrocategoryCatalog
from avito.should_split.core.config import ShouldSplitGraphConfig
from avito.should_split.domain.models import RetrievedExample, RunState
from avito.should_split.core.node_names import GraphNodeNames, GraphRouteNames, StageNames
from avito.should_split.core.pipeline import ShouldSplitPipeline, resolve_rag_markup_fpath


def _build_pipeline_stub(config: ShouldSplitGraphConfig) -> ShouldSplitPipeline:
    pipeline = ShouldSplitPipeline.__new__(ShouldSplitPipeline)
    pipeline.catalog = MicrocategoryCatalog(
        mc_id_to_title={101: "Сантехника", 102: "Электрика"},
        keyphrases_by_category={
            101: ("сантехника", "трубы"),
            102: ("электрика", "проводка"),
        },
        ordered_category_ids=[101, 102],
    )
    pipeline.config = config
    pipeline.nodes = GraphNodeNames()
    pipeline.routes = GraphRouteNames()
    pipeline.stages = StageNames()
    pipeline.mistral_caller = lambda *args, **kwargs: ""
    return pipeline


def test_node_and_route_names_are_unique() -> None:
    node_values = list(GraphNodeNames().__dict__.values())
    route_values = list(GraphRouteNames().__dict__.values())

    assert len(node_values) == len(set(node_values))
    assert len(route_values) == len(set(route_values))


def test_pre_filter_and_stage_timing_are_saved() -> None:
    config = ShouldSplitGraphConfig.model_validate({"graph": {"verbose": False}})
    pipeline = _build_pipeline_stub(config)

    state = RunState(description="делаем сантехника и электрика")
    state = pipeline.clean_node(state)
    state = pipeline.pre_filter_node(state)

    assert state.pre_filter_verdict is True
    assert state.shouldSplit is False
    assert state.multiCats is False
    assert pipeline.stages.clean in state.stage_durations_sec
    assert pipeline.stages.pre_filter in state.stage_durations_sec
    assert state.stage_durations_sec[pipeline.stages.clean] >= 0.0
    assert state.stage_durations_sec[pipeline.stages.pre_filter] >= 0.0


def test_drafts_stub_writes_metadata() -> None:
    config = ShouldSplitGraphConfig.model_validate(
        {"graph": {"verbose": False, "draft_stub_mode": "metadata_only"}}
    )
    pipeline = _build_pipeline_stub(config)

    state = RunState(description="пример")
    state = pipeline.draft_generate_node(state)

    assert state.metadata["drafts_status"] == "stub"
    assert state.drafts == []


def test_drafts_stub_can_raise_not_implemented() -> None:
    config = ShouldSplitGraphConfig.model_validate(
        {"graph": {"verbose": False, "draft_stub_mode": "raise"}}
    )
    pipeline = _build_pipeline_stub(config)

    with pytest.raises(NotImplementedError):
        pipeline.draft_generate_node(RunState(description="пример"))


def test_draft_generate_skips_when_should_split_false() -> None:
    config = ShouldSplitGraphConfig.model_validate({"graph": {"verbose": False}})
    pipeline = _build_pipeline_stub(config)

    state = RunState(description="пример", shouldSplit=False, categorized_mc_ids=[101])
    state = pipeline.draft_generate_node(state)

    assert state.metadata["drafts_status"] == "skipped_should_split_false"
    assert state.drafts == []


def test_drafts_generate_creates_texts() -> None:
    config = ShouldSplitGraphConfig.model_validate(
        {
            "graph": {"verbose": False, "draft_stub_mode": "generate"},
            "mistral": {"enabled": True},
            "drafts": {"enabled": True},
        }
    )
    pipeline = _build_pipeline_stub(config)
    pipeline.mistral_caller = lambda *args, **kwargs: "Делаем аккуратно и быстро, работаем отдельно по категории."

    state = RunState(
        description="оригинальный стиль объявления",
        shouldSplit=True,
        categorized_mc_ids=[101],
    )
    state = pipeline.draft_generate_node(state)

    assert state.metadata["drafts_status"] == "generated"
    assert len(state.drafts) == 1
    assert state.drafts[0].mc_id == 101
    assert state.drafts[0].text


def test_categorize_node_parses_category_ids() -> None:
    config = ShouldSplitGraphConfig.model_validate({"graph": {"verbose": False}})
    pipeline = _build_pipeline_stub(config)
    pipeline.mistral_caller = lambda *args, **kwargs: "101, 999, 102"
    pipeline._get_balanced_top_k = lambda *args, **kwargs: [
        RetrievedExample(text="a", shouldSplit=True, categories=["Сантехника"]),
        RetrievedExample(text="b", shouldSplit=False, categories=[]),
    ]

    state = RunState(description="пример")
    state = pipeline.categorize_node(state)

    assert state.categorized_mc_ids == [101, 102]
    assert state.metadata["categories"] == [101, 102]
    assert pipeline.stages.categorize in state.stage_durations_sec


def test_rag_router_routes_to_categorize_only_when_rag_verdict_true() -> None:
    config = ShouldSplitGraphConfig.model_validate(
        {"graph": {"verbose": False, "enable_categorization": True}}
    )
    pipeline = _build_pipeline_stub(config)

    state = RunState(description="пример", rag_split_verdict=True, shouldSplit=True)
    assert pipeline.rag_router(state) == pipeline.routes.to_categorize

    state = RunState(description="пример", rag_split_verdict=False, shouldSplit=False)
    assert pipeline.rag_router(state) == pipeline.routes.to_llm_exit
    assert state.shouldSplit is False


def test_rag_split_node_sets_should_split_true_when_llm_returns_true() -> None:
    config = ShouldSplitGraphConfig.model_validate(
        {
            "mistral": {"enabled": True},
            "graph": {"verbose": False, "use_rag": False},
        }
    )
    pipeline = _build_pipeline_stub(config)
    pipeline.mistral_caller = lambda *args, **kwargs: '{"shouldSplit": true}'

    state = RunState(description="пример", pre_filter_verdict=True)
    state = pipeline.rag_split_node(state)

    assert state.rag_split_verdict is True
    assert state.shouldSplit is True


def test_pre_filter_router_routes_to_rag_only_when_prefilter_true() -> None:
    config = ShouldSplitGraphConfig.model_validate({"graph": {"verbose": False}})
    pipeline = _build_pipeline_stub(config)

    true_state = RunState(description="пример", pre_filter_verdict=True)
    false_state = RunState(description="пример", pre_filter_verdict=False)

    assert pipeline.pre_filter_router(true_state) == pipeline.routes.to_rag
    assert pipeline.pre_filter_router(false_state) == pipeline.routes.to_llm_exit


def test_should_split_node_uses_no_rag_prompt_mode() -> None:
    config = ShouldSplitGraphConfig.model_validate(
        {
            "mistral": {"enabled": True},
            "graph": {"verbose": False, "use_rag": False},
        }
    )
    pipeline = _build_pipeline_stub(config)
    pipeline.mistral_caller = lambda *args, **kwargs: "True"
    pipeline._get_balanced_top_k = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("retrieval не должен вызываться при use_rag=False")
    )

    state = RunState(description="пример", pre_filter_verdict=True)
    state = pipeline.rag_split_node(state)

    assert state.rag_split_verdict is True
    assert state.multiCats is False
    assert state.metadata["should_split_prompt_mode"] == "no_rag"


def test_categorize_node_uses_no_rag_prompt_mode() -> None:
    config = ShouldSplitGraphConfig.model_validate(
        {
            "mistral": {"enabled": True},
            "graph": {"verbose": False, "use_rag": False},
        }
    )
    pipeline = _build_pipeline_stub(config)
    pipeline.mistral_caller = lambda *args, **kwargs: "101, 102"
    pipeline._get_balanced_top_k = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("retrieval не должен вызываться при use_rag=False")
    )

    state = RunState(description="пример")
    state = pipeline.categorize_node(state)

    assert state.categorized_mc_ids == [101, 102]
    assert state.metadata["categorization_prompt_mode"] == "no_rag"


def test_categorize_node_uses_cached_top_k_examples_from_state() -> None:
    config = ShouldSplitGraphConfig.model_validate(
        {
            "mistral": {"enabled": True},
            "graph": {"verbose": False, "use_rag": True},
        }
    )
    pipeline = _build_pipeline_stub(config)
    pipeline.mistral_caller = lambda *args, **kwargs: "101"
    pipeline._get_balanced_top_k = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("retrieval не должен вызываться при заполненном state.top_k_examples")
    )

    state = RunState(
        description="пример",
        top_k_examples=[
            RetrievedExample(text="a", shouldSplit=True, categories=["101"]),
            RetrievedExample(text="b", shouldSplit=False, categories=[]),
        ],
    )
    state = pipeline.categorize_node(state)

    assert state.categorized_mc_ids == [101]
    assert state.metadata["categorization_prompt_mode"] == "rag"


def test_rag_split_then_categorize_reuses_cached_top_k_examples() -> None:
    config = ShouldSplitGraphConfig.model_validate(
        {
            "mistral": {"enabled": True},
            "graph": {"verbose": False, "use_rag": True},
        }
    )
    pipeline = _build_pipeline_stub(config)
    calls = {"count": 0}

    def fake_retrieval(*args, **kwargs) -> list[RetrievedExample]:
        calls["count"] += 1
        return [
            RetrievedExample(text="a", shouldSplit=True, categories=["101"]),
            RetrievedExample(text="b", shouldSplit=False, categories=[]),
        ]

    responses = iter([
        '{"shouldSplit": true}',
        "101",
    ])
    pipeline.mistral_caller = lambda *args, **kwargs: next(responses)
    pipeline._get_balanced_top_k = fake_retrieval

    state = RunState(description="пример")
    state = pipeline.rag_split_node(state)
    state = pipeline.categorize_node(state)

    assert calls["count"] == 1
    assert state.categorized_mc_ids == [101]


def test_rag_router_routes_to_llm_exit_when_rag_verdict_false() -> None:
    config = ShouldSplitGraphConfig.model_validate(
        {
            "mistral": {"enabled": True},
            "graph": {"verbose": False, "use_rag": False, "enable_categorization": True},
        }
    )
    pipeline = _build_pipeline_stub(config)
    pipeline.mistral_caller = lambda *args, **kwargs: '{"shouldSplit": false}'

    state = RunState(description="пример")
    state = pipeline.rag_split_node(state)

    assert state.shouldSplit is False
    assert state.multiCats is False
    assert pipeline.rag_router(state) == pipeline.routes.to_llm_exit


def test_resolve_rag_markup_fpath_uses_markup_filename_as_fallback() -> None:
    data_dpath = Path("C:/repo/avito/data")

    resolved = resolve_rag_markup_fpath(
        data_dpath=data_dpath,
        rag_markup_path=None,
        markup_filename="rnc_dataset_markup.json",
    )

    assert resolved == data_dpath / "rnc_dataset_markup.json"


def test_resolve_rag_markup_fpath_prefers_explicit_path() -> None:
    data_dpath = Path("C:/repo/avito/data")

    relative_resolved = resolve_rag_markup_fpath(
        data_dpath=data_dpath,
        rag_markup_path="custom/rag.json",
        markup_filename="rnc_dataset_markup.json",
    )
    absolute_resolved = resolve_rag_markup_fpath(
        data_dpath=data_dpath,
        rag_markup_path="C:/tmp/rag.json",
        markup_filename="rnc_dataset_markup.json",
    )

    assert relative_resolved == data_dpath / "custom/rag.json"
    assert absolute_resolved == Path("C:/tmp/rag.json")


