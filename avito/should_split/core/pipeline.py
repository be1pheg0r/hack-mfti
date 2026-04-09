from __future__ import annotations

import ast
import json
import re
import time
from pathlib import Path
from typing import *

import pandas as pd

from avito.constants import (
    DRAFT_STATUS_DISABLED_BY_CONFIG,
    DRAFT_STATUS_GENERATED,
    DRAFT_STATUS_MISTRAL_DISABLED,
    DRAFT_STATUS_NO_CATEGORIES,
    DRAFT_STATUS_SKIPPED_SHOULD_SPLIT_FALSE,
    DRAFT_STATUS_STUB,
    SPLIT_MARKERS,
)
from avito.prompts import (
    categorizationInstructionPrompt,
    categorizationPromptWithoutRAG,
    categorizationPromptWithRAG,
    shouldSplitInstructionPrompt,
    shouldSplitPromptWithoutRAG,
    shouldSplitPromptWithRAG,
)
from avito.should_split.domain.catalog import MicrocategoryCatalog, load_microcategory_catalog
from avito.should_split.core.config import ShouldSplitGraphConfig
from avito.should_split.domain.models import RunState, RetrievedExample
from avito.should_split.core.node_names import GraphNodeNames, GraphRouteNames, StageNames
from avito.should_split.core.draft_generation import generate_draft_candidates
from avito.should_split.infra.retrieval import ExampleRetriever, build_retriever
from avito.should_split.core.text import clean_description
from common.logger import AVITO_SHOULD_SPLIT_LOGGER as logger
from common.mistral import MistralCallConfig, call_mistral
from common.paths import get_avito_data_dpath


def warmup_mistral(
    mistral_caller: Callable[..., str],
    config: ShouldSplitGraphConfig,
    context_label: str,
) -> None:
    """Делает мягкий warmup-вызов Mistral и не роняет пайплайн при ошибке."""
    if not config.mistral.enabled:
        return
    try:
        _ = mistral_caller(
            MistralCallConfig(models_list=config.mistral.models_list),
            messages=[
                {"role": "system", "content": "Ты ассистент."},
                {"role": "user", "content": "привет"},
            ],
            temperature=0.0,
        )
        logger.info("[warmup] mistral ok | context=%s", context_label)
    except Exception as error:
        logger.warning("[warmup] mistral failed | context=%s | error=%s", context_label, error)


class ShouldSplitPipeline:
    """LangGraph пайплайн определения необходимости split.

    Notes:
        Draft-стадия реализована как заглушка и не генерирует тексты.
    """

    def __init__(
        self,
        catalog: MicrocategoryCatalog,
        examples_df: pd.DataFrame,
        retriever: ExampleRetriever,
        config: ShouldSplitGraphConfig,
        mistral_caller: Callable[..., str] = call_mistral,
    ) -> None:
        self.catalog = catalog
        self.examples_df = examples_df
        self.retriever = retriever
        self.config = config
        self.mistral_caller = mistral_caller

        self.nodes = GraphNodeNames()
        self.routes = GraphRouteNames()
        self.stages = StageNames()

        self._graph = self._build_graph()
        warmup_mistral(self.mistral_caller, self.config, context_label="pipeline_init")

    def _build_graph(self) -> Any:
        try:
            from langgraph.graph import END, START, StateGraph
        except ImportError as error:
            raise ImportError("Для запуска графа установите пакет langgraph") from error

        builder = StateGraph(RunState)

        # Регистрируем узлы по этапам: очистка/префильтр -> LLM-решения -> выходы.
        builder.add_node(self.nodes.clean, self.clean_node)
        builder.add_node(self.nodes.pre_filter, self.pre_filter_node)
        builder.add_node(self.nodes.rag_split, self.rag_split_node)
        builder.add_node(self.nodes.categorize, self.categorize_node)
        builder.add_node(self.nodes.llm_exit, self.llm_exit_node)

        builder.add_node(self.nodes.drafts, self.drafts_node)
        builder.add_node(self.nodes.drafts_exit, self.drafts_exit_node)

        # Базовый маршрут до развилки после pre_filter.
        builder.add_edge(START, self.nodes.clean)
        builder.add_edge(self.nodes.clean, self.nodes.pre_filter)

        # Если pre_filter=True, сразу идем в categorize; иначе в LLM shouldSplit-классификацию.
        builder.add_conditional_edges(
            self.nodes.pre_filter,
            self.pre_filter_router,
            {
                self.routes.to_categorize: self.nodes.categorize,
                self.routes.to_rag: self.nodes.rag_split,
            },
        )

        # После rag_split переходим к категоризации, если есть split или multiCats сигнал.
        builder.add_conditional_edges(
            self.nodes.rag_split,
            self.rag_router,
            {
                self.routes.to_categorize: self.nodes.categorize,
                self.routes.to_llm_exit: self.nodes.llm_exit,
            },
        )

        # После категоризации при enable_drafts=True заходим в drafts, иначе выходим.
        builder.add_conditional_edges(
            self.nodes.categorize,
            self.categorize_router,
            {
                self.routes.to_drafts: self.nodes.drafts,
                self.routes.to_llm_exit: self.nodes.llm_exit,
            },
        )

        builder.add_edge(self.nodes.drafts, self.nodes.drafts_exit)

        # Терминальные ветки завершения по типу выхода.
        builder.add_edge(self.nodes.llm_exit, END)
        builder.add_edge(self.nodes.drafts_exit, END)
        return builder.compile()

    def _run_timed(self, stage_name: str, state: RunState, fn: Callable[[RunState], RunState]) -> RunState:
        start = time.perf_counter()
        result = fn(state)
        result.stage_durations_sec[stage_name] = time.perf_counter() - start
        return result

    def clean_node(self, state: RunState) -> RunState:
        return self._run_timed(self.stages.clean, state, self._clean_node_impl)

    def _clean_node_impl(self, state: RunState) -> RunState:
        raw_text = state.description
        state.description = clean_description(state.description)
        if self.config.graph.verbose:
            logger.info("[clean] %s -> %s символов", len(raw_text), len(state.description))
        return state

    def pre_filter_node(self, state: RunState) -> RunState:
        return self._run_timed(self.stages.pre_filter, state, self._pre_filter_node_impl)

    def _pre_filter_node_impl(self, state: RunState) -> RunState:
        split_marker_count = sum(1 for marker in SPLIT_MARKERS if marker in state.description)
        triggered_mc_ids = self._get_triggered_mc_ids(state.description)

        keyphrases_flag = len(triggered_mc_ids) > 1
        # Префильтр дает только слабую эвристику multiCats и не форсит shouldSplit по одному маркеру.
        state.pre_filter_verdict = keyphrases_flag
        state.shouldSplit = state.pre_filter_verdict
        state.multiCats = keyphrases_flag
        state.metadata["triggered_mc_ids"] = triggered_mc_ids
        state.metadata["split_marker_count"] = split_marker_count

        if self.config.graph.verbose:
            logger.info(
                "[pre_filter] shouldSplit=%s, split_markers=%s, categories=%s",
                state.pre_filter_verdict,
                split_marker_count,
                len(triggered_mc_ids),
            )
        return state

    def _get_triggered_mc_ids(self, description: str) -> list[int]:
        triggered_mc_ids: list[int] = []
        for mc_id in self.catalog.ordered_category_ids:
            keyphrases = self.catalog.keyphrases_by_category.get(mc_id, ())
            if any(phrase in description for phrase in keyphrases):
                triggered_mc_ids.append(mc_id)
        return triggered_mc_ids

    def pre_filter_router(self, state: RunState) -> str:
        if state.pre_filter_verdict:
            return self.routes.to_categorize
        return self.routes.to_rag

    def pre_filter_exit_node(self, state: RunState) -> RunState:
        return self._run_timed(self.stages.pre_filter_exit, state, lambda current: current)

    def rag_split_node(self, state: RunState) -> RunState:
        return self._run_timed(self.stages.rag_split, state, self._rag_split_node_impl)

    def _rag_split_node_impl(self, state: RunState) -> RunState:
        examples: list[RetrievedExample] = []
        if self.config.graph.use_rag:
            examples = self._get_balanced_top_k(state.description)
        state.top_k_examples = examples

        if not self.config.mistral.enabled:
            state.metadata["rag_reason"] = "mistral_disabled"
            state.rag_split_verdict = state.pre_filter_verdict
            state.shouldSplit = state.pre_filter_verdict
            state.multiCats = state.pre_filter_verdict
            return state

        if self.config.graph.use_rag:
            prompt_top_k = self._build_prompt_top_k(examples)
            prompt = shouldSplitPromptWithRAG(desc=state.description, top_k=prompt_top_k)
            state.metadata["should_split_prompt_mode"] = "rag"
        else:
            prompt = shouldSplitPromptWithoutRAG(desc=state.description)
            state.metadata["should_split_prompt_mode"] = "no_rag"

        response = self.mistral_caller(
            self.config.mistral.to_call_config(),
            messages=[
                {"role": "system", "content": shouldSplitInstructionPrompt},
                {"role": "user", "content": prompt},
            ],
            reasoning_effort=self.config.mistral.reasoning_effort,
            temperature=self.config.mistral.temperature,
        )
        llm_should_split, llm_multi_cats = self._parse_should_split_response(str(response))
        state.rag_split_verdict = llm_should_split
        state.shouldSplit = llm_should_split
        state.multiCats = llm_multi_cats
        state.metadata["llm_response"] = str(response)
        state.metadata["llm_should_split"] = llm_should_split
        state.metadata["llm_multi_cats"] = llm_multi_cats
        return state

    @staticmethod
    def _parse_should_split_response(response_text: str) -> tuple[bool, bool]:
        text = response_text.strip()
        if not text:
            return False, False

        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                should_split = ShouldSplitPipeline._parse_bool_value(parsed.get("shouldSplit", False))
                multi_cats = ShouldSplitPipeline._parse_bool_value(parsed.get("multiCats", should_split))
                return should_split, multi_cats
        except json.JSONDecodeError:
            pass

        json_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group(0))
                if isinstance(parsed, dict):
                    should_split = ShouldSplitPipeline._parse_bool_value(parsed.get("shouldSplit", False))
                    multi_cats = ShouldSplitPipeline._parse_bool_value(parsed.get("multiCats", should_split))
                    return should_split, multi_cats
            except json.JSONDecodeError:
                pass

        should_match = re.search(r"should\s*split\s*[:=]\s*(true|false)", text, flags=re.IGNORECASE)
        multi_match = re.search(r"multi\s*cats\s*[:=]\s*(true|false)", text, flags=re.IGNORECASE)
        if should_match:
            should_split = should_match.group(1).lower() == "true"
            multi_cats = should_split
            if multi_match:
                multi_cats = multi_match.group(1).lower() == "true"
            return should_split, multi_cats

        normalized = text.lower()
        if normalized in {"true", "false"}:
            value = normalized == "true"
            return value, value

        return False, False

    @staticmethod
    def _parse_bool_value(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() == "true"
        return bool(value)

    @staticmethod
    def _build_prompt_top_k(examples: list[RetrievedExample]) -> list[tuple[str, bool, list[str]]]:
        return [(item.text, item.shouldSplit, item.categories) for item in examples]

    def rag_router(self, state: RunState) -> str:
        if self.config.graph.enable_categorization and (state.shouldSplit or state.multiCats):
            return self.routes.to_categorize
        return self.routes.to_llm_exit

    def categorize_node(self, state: RunState) -> RunState:
        return self._run_timed(self.stages.categorize, state, self._categorize_node_impl)

    def _categorize_node_impl(self, state: RunState) -> RunState:
        if not self.config.mistral.enabled:
            state.categorized_mc_ids = []
            state.metadata["categories"] = []
            state.metadata["categorization_reason"] = "mistral_disabled"
            return state

        if self.config.graph.use_rag:
            top_k = self._get_balanced_top_k(state.description)
            true_top_k = [item for item in top_k if item.shouldSplit]
            prompt_top_k = self._build_prompt_top_k(true_top_k)
            prompt = categorizationPromptWithRAG(desc=state.description, top_k=prompt_top_k)
            state.metadata["categorization_prompt_mode"] = "rag"
        else:
            prompt = categorizationPromptWithoutRAG(desc=state.description)
            state.metadata["categorization_prompt_mode"] = "no_rag"

        response = self.mistral_caller(
            self.config.mistral.to_call_config(),
            messages=[
                {"role": "system", "content": categorizationInstructionPrompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )

        raw = str(response).strip()
        ids = self._parse_known_category_ids(raw)

        state.categorized_mc_ids = ids
        state.metadata["categories"] = ids
        state.metadata["categorization_response"] = raw
        logger.info("[categorize] найдено категорий: %s", ids)
        return state

    def _parse_known_category_ids(self, response_text: str) -> list[int]:
        parsed_ids = [int(token) for token in re.findall(r"\d+", response_text)]
        known_ids = set(self.catalog.ordered_category_ids)
        return [mc_id for mc_id in parsed_ids if mc_id in known_ids]

    def categorize_router(self, state: RunState) -> str:
        if self.config.graph.enable_drafts:
            return self.routes.to_drafts
        return self.routes.to_llm_exit

    def llm_exit_node(self, state: RunState) -> RunState:
        return self._run_timed(self.stages.llm_exit, state, lambda current: current)

    def drafts_node(self, state: RunState) -> RunState:
        return self._run_timed(self.stages.drafts, state, self._drafts_node_impl)

    def _drafts_node_impl(self, state: RunState) -> RunState:
        if self.config.graph.draft_stub_mode == "raise":
            raise NotImplementedError("Draft generation is not implemented yet")

        if self.config.graph.draft_stub_mode == "metadata_only":
            return self._set_drafts_status(state, DRAFT_STATUS_STUB)

        if not state.shouldSplit:
            return self._set_drafts_status(state, DRAFT_STATUS_SKIPPED_SHOULD_SPLIT_FALSE)

        if not self.config.drafts.enabled:
            return self._set_drafts_status(state, DRAFT_STATUS_DISABLED_BY_CONFIG)

        if not self.config.mistral.enabled:
            return self._set_drafts_status(state, DRAFT_STATUS_MISTRAL_DISABLED)

        if not state.categorized_mc_ids:
            return self._set_drafts_status(state, DRAFT_STATUS_NO_CATEGORIES)

        drafts = generate_draft_candidates(
            description=state.description,
            categorized_mc_ids=state.categorized_mc_ids,
            mc_id_to_title=self.catalog.mc_id_to_title,
            draft_config=self.config.drafts,
            mistral_config=self.config.mistral,
            mistral_caller=self.mistral_caller,
        )

        state.drafts = drafts
        state.metadata["drafts_status"] = DRAFT_STATUS_GENERATED
        state.metadata["drafts_count"] = len(drafts)
        return state

    @staticmethod
    def _set_drafts_status(state: RunState, status: str) -> RunState:
        state.metadata["drafts_status"] = status
        state.drafts = []
        return state

    def drafts_exit_node(self, state: RunState) -> RunState:
        return self._run_timed(self.stages.drafts_exit, state, lambda current: current)

    def _get_balanced_top_k(self, description: str) -> list[RetrievedExample]:
        candidates = self.retriever.search(description, limit=self.config.vector_db.search_k)
        filtered = [item for item in candidates if item.text != description]

        true_examples = [item for item in filtered if item.shouldSplit]
        false_examples = [item for item in filtered if not item.shouldSplit]

        half = self.config.vector_db.balanced_k // 2
        balanced = true_examples[:half] + false_examples[:half]
        return balanced

    def invoke(self, description: str) -> RunState:
        initial_state = RunState(description=description)
        result = self._graph.invoke(initial_state)
        return RunState.model_validate(result)


def load_markup_dataframe(markup_fpath: str | Path, catalog: MicrocategoryCatalog) -> pd.DataFrame:
    """Загружает и нормализует разметку объявлений для retrieval."""
    data_df = pd.read_json(markup_fpath)
    data_df["targetSplitMcIds"] = data_df["targetSplitMcIds"].apply(ast.literal_eval)
    data_df["targetSplitMcTitles"] = data_df["targetSplitMcIds"].apply(
        lambda mc_ids: [catalog.mc_id_to_title[int(mc_id)] for mc_id in mc_ids]
    )
    data_df["description"] = data_df["description"].astype(str).apply(clean_description)
    return data_df


def resolve_rag_markup_fpath(
    data_dpath: Path,
    rag_markup_path: str | None,
    markup_filename: str,
) -> Path:
    """Резолвит путь к RAG-разметке с fallback на legacy-поле markup_filename."""
    if rag_markup_path:
        configured_path = Path(rag_markup_path)
        return configured_path if configured_path.is_absolute() else data_dpath / configured_path
    return data_dpath / markup_filename


def build_default_pipeline(config: ShouldSplitGraphConfig | None = None) -> ShouldSplitPipeline:
    """Собирает пайплайн should-split из конфигов и данных по умолчанию."""
    runtime_config = config or ShouldSplitGraphConfig.from_default_yaml()
    data_dpath = Path(get_avito_data_dpath())

    mc_map_fpath = data_dpath / runtime_config.data.mc_map_filename
    markup_fpath = resolve_rag_markup_fpath(
        data_dpath=data_dpath,
        rag_markup_path=runtime_config.data.rag_markup_path,
        markup_filename=runtime_config.data.markup_filename,
    )

    catalog = load_microcategory_catalog(mc_map_fpath)
    examples_df = load_markup_dataframe(markup_fpath, catalog=catalog)
    retriever = build_retriever(
        examples_df=examples_df,
        vector_db_config=runtime_config.vector_db,
        encoder_config=runtime_config.encoder,
    )
    return ShouldSplitPipeline(
        catalog=catalog,
        examples_df=examples_df,
        retriever=retriever,
        config=runtime_config,
    )

