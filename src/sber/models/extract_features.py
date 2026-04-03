from __future__ import annotations

"""Feature extraction helpers for the Sber case.

Этот модуль содержит каноническую реализацию извлечения внутренних признаков
из causal language model. Исторически эта логика жила в отдельном legacy-файле,
где использовалась для генерации handcrafted features для бустинга. Сейчас
вся актуальная реализация находится в `src/sber/models/extract_features.py`.

Внутри лежат:
- конфиг экстрактора;
- pydantic-модель входа для валидации shapes;
- контейнер фичей;
- сам экстрактор с hooks на hidden states / attention / MoE routing;
- dummy-модель для тестов и локальной отладки.
"""

from contextlib import AbstractContextManager
from typing import *

import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from torch import nn

from common.files import read_yaml
from common.logger import SBER_HOOKS_LOGGER as logger
from common.paths import PathLike
from ..constants import (
    DEFAULT_ATTENTION_EPSILON,
    DEFAULT_ENABLE_ATTENTION_ENTROPY,
    DEFAULT_ENABLE_MOE_ROUTING,
    DEFAULT_FEATURE_EXTRACTION_CONFIGS_FPATH,
    DEFAULT_LOGIT_EPSILON,
    DEFAULT_OUTPUT_ATTENTIONS,
    DEFAULT_PROBE_LAYERS,
)


class FeatureExtractorConfig(BaseModel):
    """Конфигурация извлечения фичей из языковой модели.

    Attributes:
        probe_layers: Индексы слоев, из которых снимаются скрытые состояния.
        logit_epsilon: Малое число для численной стабильности log/entropy.
        attention_epsilon: Малое число для численной стабильности attention entropy.
        enable_attention_entropy: Включает извлечение attention-энтропий.
        enable_moe_routing: Включает извлечение routing-фичей для MoE-моделей.
        use_output_attentions: Просит модель возвращать attention maps в forward.
    """

    model_config = ConfigDict(frozen=True)

    probe_layers: list[int] = Field(default_factory=lambda: list(DEFAULT_PROBE_LAYERS))
    logit_epsilon: float = DEFAULT_LOGIT_EPSILON
    attention_epsilon: float = DEFAULT_ATTENTION_EPSILON
    enable_attention_entropy: bool = DEFAULT_ENABLE_ATTENTION_ENTROPY
    enable_moe_routing: bool = DEFAULT_ENABLE_MOE_ROUTING
    use_output_attentions: bool = DEFAULT_OUTPUT_ATTENTIONS

    @field_validator("probe_layers")
    @classmethod
    def validate_probe_layers(cls, value: list[int]) -> list[int]:
        """Сортирует слои, убирает дубликаты и проверяет границы."""
        normalized_layers: list[int] = sorted(set(value))
        if not normalized_layers:
            raise ValueError("probe_layers не может быть пустым")
        if normalized_layers[0] < 0:
            raise ValueError("Индексы probe_layers должны быть неотрицательными")
        return normalized_layers

    @field_validator("logit_epsilon", "attention_epsilon")
    @classmethod
    def validate_epsilon(cls, value: float) -> float:
        """Проверяет, что epsilon положительный."""
        if value <= 0.0:
            raise ValueError("Epsilon должен быть положительным")
        return value

    @classmethod
    def from_yaml(cls, fpath: PathLike) -> FeatureExtractorConfig:
        """Создает конфиг из YAML-файла.

        Args:
            fpath: Путь до YAML-конфига.

        Returns:
            Валидированный конфиг.
        """
        raw_data: Any = read_yaml(fpath)
        if not isinstance(raw_data, dict):
            raise ValueError("YAML-конфиг должен быть словарем")

        payload: Any = raw_data.get("feature_extractor", raw_data)
        if not isinstance(payload, dict):
            raise ValueError("Секция feature_extractor должна быть словарем")

        return cls.model_validate(payload)

    @classmethod
    def from_default_yaml(cls) -> FeatureExtractorConfig:
        """Создает конфиг из штатного YAML-файла."""
        return cls.from_yaml(fpath=DEFAULT_FEATURE_EXTRACTION_CONFIGS_FPATH)


class FeatureExtractorInput(BaseModel):
    """Входные тензоры для извлечения фичей.

    Attributes:
        logits: Логиты модели формы [batch, seq_len, vocab_size].
        input_ids: Токены формы [batch, seq_len].
        answer_start: Индекс начала токенов ответа внутри последовательности.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    logits: torch.Tensor
    input_ids: torch.Tensor
    answer_start: int

    @model_validator(mode="after")
    def validate_shapes(self) -> FeatureExtractorInput:
        """Проверяет согласованность форм логитов и токенов."""
        if self.logits.ndim != 3:
            raise ValueError("Ожидается logits формы [batch, seq_len, vocab_size]")
        if self.input_ids.ndim != 2:
            raise ValueError("Ожидается input_ids формы [batch, seq_len]")
        if self.logits.shape[0] != self.input_ids.shape[0]:
            raise ValueError("batch размер logits и input_ids должен совпадать")
        if self.logits.shape[1] != self.input_ids.shape[1]:
            raise ValueError("seq_len logits и input_ids должен совпадать")
        if self.answer_start <= 0:
            raise ValueError("answer_start должен быть > 0")

        seq_len: int = int(self.input_ids.shape[1])
        if self.answer_start >= seq_len:
            raise ValueError("answer_start должен быть меньше длины последовательности")
        return self


class FeatureGroups(BaseModel):
    """Группы фичей для downstream-классификатора.

    Attributes:
        uncertainty: Статистики неопределенности по токенам ответа.
        internal_scalars: Поэлементные скаляры из внутренних слоев модели.
        probe_vec: Усредненный вектор ответа из выбранных probe-слоев.
        attention_entropy: Энтропийные attention-фичи.
        entropy_drops: Разности logit-lens энтропий между слоями.
        moe_routing: Агрегированные routing-фичи для MoE-моделей.
    """

    uncertainty: list[float]
    internal_scalars: list[float]
    probe_vec: list[float]
    attention_entropy: list[float]
    entropy_drops: list[float]
    moe_routing: list[float]

    def to_tensor_dict(self) -> dict[str, torch.Tensor]:
        """Конвертирует все группы фичей в float32 tensors."""
        ordered_fields: tuple[str, ...] = (
            "uncertainty",
            "internal_scalars",
            "probe_vec",
            "attention_entropy",
            "entropy_drops",
            "moe_routing",
        )
        return {field_name: torch.tensor(getattr(self, field_name), dtype=torch.float32) for field_name in ordered_fields}


class LLMFeatureExtractor(nn.Module, AbstractContextManager["LLMFeatureExtractor"]):
    """Извлекает признаки галлюцинаций из внутреннего состояния LLM."""

    def __init__(self, model: Any, config: FeatureExtractorConfig | None = None) -> None:
        """Инициализирует экстрактор.

        Args:
            model: Causal LM модель с `model.layers`, `model.norm` и `lm_head`.
            config: Настройки извлечения фичей.
        """
        super().__init__()
        self.model: Any = model
        self.config: FeatureExtractorConfig = config or FeatureExtractorConfig()
        self._hooks: list[Any] = []
        self._hidden: dict[str, torch.Tensor] = {}

    @classmethod
    def from_yaml(cls, model: Any, fpath: PathLike) -> LLMFeatureExtractor:
        """Создает экстрактор по YAML-конфигу."""
        return cls(model=model, config=FeatureExtractorConfig.from_yaml(fpath=fpath))

    @classmethod
    def from_default_yaml(cls, model: Any) -> LLMFeatureExtractor:
        """Создает экстрактор по штатному YAML-конфигу."""
        return cls.from_yaml(model=model, fpath=DEFAULT_FEATURE_EXTRACTION_CONFIGS_FPATH)

    def _extract_tensor(self, output: Any) -> torch.Tensor | None:
        """Извлекает первый Tensor из выхода hook-а."""
        if isinstance(output, torch.Tensor):
            return output
        if isinstance(output, tuple):
            for element in output:
                if isinstance(element, torch.Tensor):
                    return element
        return None

    def _extract_attention(self, output: Any) -> torch.Tensor | None:
        """Извлекает attention weights из выхода self-attention слоя."""
        if isinstance(output, torch.Tensor):
            return output
        if isinstance(output, tuple):
            for element in reversed(output):
                if isinstance(element, torch.Tensor):
                    return element
        return None

    def _resolve_moe_router_module(self, layer: Any) -> Any | None:
        """Находит модуль роутера внутри MLP слоя MoE.

        Исторически разные LLM по-разному называют gate/router module, поэтому
        здесь используется комбинация явных атрибутов и эвристического поиска.
        """
        mlp: Any | None = getattr(layer, "mlp", None)
        if mlp is None:
            return None

        for attr_name in ("router", "gate", "router_layer", "moe_gate", "expert_gate"):
            candidate: Any | None = getattr(mlp, attr_name, None)
            if candidate is not None and hasattr(candidate, "register_forward_hook"):
                return candidate

        named_modules: list[tuple[str, Any]] = list(getattr(mlp, "named_modules", lambda: [])())
        for module_name, module in named_modules:
            lowered: str = module_name.lower()
            if ("router" in lowered or lowered.endswith(".gate") or lowered == "gate") and hasattr(
                module,
                "register_forward_hook",
            ):
                return module
        return None

    def _get_layers(self) -> Any:
        """Возвращает список hidden layers модели и валидирует его наличие."""
        layers: Any = getattr(getattr(self.model, "model", None), "layers", None)
        if layers is None:
            raise ValueError("У модели не найден атрибут model.layers")
        return layers

    def _infer_hidden_size(self) -> int:
        """Пытается определить размер hidden-state для zero-fallback-ов."""
        lm_head: Any = getattr(self.model, "lm_head", None)
        hidden_size: Any = getattr(lm_head, "in_features", None)
        return int(hidden_size) if hidden_size is not None else 0

    def _infer_device(self) -> torch.device:
        """Определяет device модели для fallback-тензоров."""
        first_parameter: Any = next(self.parameters(), None)
        if isinstance(first_parameter, torch.Tensor):
            return first_parameter.device
        return torch.device("cpu")

    def attach(self) -> None:
        """Регистрирует forward-hooks на hidden states, attention и MoE routing."""
        logger.info("Регистрирую hooks для извлечения фичей")
        self._hidden.clear()
        layers: Any = self._get_layers()
        total_layers: int = len(layers)

        for layer_idx in self.config.probe_layers:
            if layer_idx < 0 or layer_idx >= total_layers:
                raise ValueError(f"probe layer {layer_idx} вне диапазона доступных слоев 0..{total_layers - 1}")

            layer = layers[layer_idx]
            layer_key: str = f"layer_{layer_idx}"

            # Hook для hidden states.
            def make_hidden_hook(key: str) -> Callable[[Any, Any, Any], None]:
                def hidden_hook(_module: Any, _input: Any, output: Any) -> None:
                    hidden_tensor: torch.Tensor | None = self._extract_tensor(output)
                    if hidden_tensor is not None:
                        self._hidden[key] = hidden_tensor.detach()

                return hidden_hook

            self._hooks.append(layer.register_forward_hook(make_hidden_hook(layer_key)))

            if self.config.enable_attention_entropy:
                attn_module: Any | None = getattr(layer, "self_attn", None)
                if attn_module is not None and hasattr(attn_module, "register_forward_hook"):
                    attn_key: str = f"attn_{layer_idx}"

                    # Hook для attention maps.
                    def make_attn_hook(key: str) -> Callable[[Any, Any, Any], None]:
                        def attn_hook(_module: Any, _input: Any, output: Any) -> None:
                            attention_tensor: torch.Tensor | None = self._extract_attention(output)
                            if attention_tensor is not None:
                                self._hidden[key] = attention_tensor.detach()

                        return attn_hook

                    self._hooks.append(attn_module.register_forward_hook(make_attn_hook(attn_key)))

            if self.config.enable_moe_routing:
                router_module: Any | None = self._resolve_moe_router_module(layer)
                if router_module is not None:
                    moe_key: str = f"moe_{layer_idx}"

                    # Hook для выхода router/gate.
                    def make_moe_hook(key: str) -> Callable[[Any, Any, Any], None]:
                        def moe_hook(_module: Any, _input: Any, output: Any) -> None:
                            route_tensor: torch.Tensor | None = self._extract_tensor(output)
                            if route_tensor is not None:
                                self._hidden[key] = route_tensor.detach()

                        return moe_hook

                    self._hooks.append(router_module.register_forward_hook(make_moe_hook(moe_key)))

    def detach(self) -> None:
        """Удаляет все зарегистрированные hooks."""
        logger.info("Снимаю hooks для извлечения фичей")
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def __enter__(self) -> LLMFeatureExtractor:
        """Включает hooks в контекстном менеджере."""
        self.attach()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """Снимает hooks при выходе из контекстного менеджера."""
        _ = (exc_type, exc_value, traceback)
        self.detach()

    def forward(self, token_ids: torch.Tensor) -> Any:
        """Выполняет forward-pass модели с попыткой включить attentions.

        Args:
            token_ids: Входные токены формы [batch, seq_len].

        Returns:
            Выход модели.
        """
        with torch.no_grad():
            if self.config.use_output_attentions:
                try:
                    return self.model(token_ids, output_attentions=True)
                except TypeError:
                    logger.warning("Модель не поддерживает output_attentions, продолжаю без attention-выходов")
            return self.model(token_ids)

    def _compute_uncertainty_features(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        answer_start: int,
    ) -> list[float]:
        """Считает uncertainty-фичи по токенам ответа."""
        seq_len: int = int(input_ids.shape[1])
        answer_logits: torch.Tensor = logits[0, answer_start:seq_len, :]
        answer_ids: torch.Tensor = input_ids[0, answer_start:seq_len]
        n_answer_tokens: int = int(answer_ids.shape[0])

        log_probs: torch.Tensor = torch.log_softmax(answer_logits, dim=-1)
        token_log_probs: torch.Tensor = log_probs.gather(1, answer_ids.unsqueeze(1)).squeeze(-1)

        probs: torch.Tensor = torch.softmax(answer_logits, dim=-1)
        entropy: torch.Tensor = -(probs * torch.log(probs + self.config.logit_epsilon)).sum(dim=-1)
        top1: torch.Tensor = probs.max(dim=-1).values
        top5: torch.Tensor = probs.topk(min(5, int(probs.shape[-1])), dim=-1).values.sum(dim=-1)
        token_std: float = float(token_log_probs.std(unbiased=False).item()) if n_answer_tokens > 1 else 0.0
        entropy_std: float = float(entropy.std(unbiased=False).item()) if n_answer_tokens > 1 else 0.0

        return [
            float(token_log_probs.mean().item()),
            float(token_log_probs.min().item()),
            float(token_log_probs.max().item()),
            token_std,
            float(entropy.mean().item()),
            float(entropy.min().item()),
            float(entropy.max().item()),
            entropy_std,
            float(n_answer_tokens),
            float(token_log_probs[0].item()),
            float(top1.mean().item()),
            float(top5.mean().item()),
        ]

    def _compute_internal_and_probe(
        self,
        answer_start: int,
        seq_len: int,
    ) -> tuple[list[float], list[float], list[float]]:
        """Считает internal-скаляры, probe-вектор и entropy drops."""
        internal_scalars: list[float] = []
        probe_vectors: list[torch.Tensor] = []
        logit_lens_entropies: list[float] = []
        hidden_size: int = self._infer_hidden_size()

        for layer_idx in self.config.probe_layers:
            hidden_key: str = f"layer_{layer_idx}"
            hidden_states_raw: torch.Tensor | None = self._hidden.get(hidden_key)

            if hidden_states_raw is None:
                logger.debug("Слой %s не попал в hooks, подставляю нули", layer_idx)
                internal_scalars.extend([0.0, 0.0, 0.0])
                probe_vectors.append(torch.zeros(hidden_size, dtype=torch.float32, device=self._infer_device()))
                logit_lens_entropies.append(0.0)
                continue

            hidden_states: torch.Tensor = hidden_states_raw[0]
            answer_hidden: torch.Tensor = hidden_states[answer_start:seq_len]
            if answer_hidden.numel() == 0:
                logger.debug("Слой %s вернул пустой slice ответа, подставляю нули", layer_idx)
                internal_scalars.extend([0.0, 0.0, 0.0])
                probe_vectors.append(torch.zeros(hidden_states.shape[-1], dtype=torch.float32, device=hidden_states.device))
                logit_lens_entropies.append(0.0)
                continue

            answer_mean: torch.Tensor = answer_hidden.mean(dim=0)
            probe_vectors.append(answer_mean)

            internal_scalars.append(float(hidden_states[answer_start - 1].norm().item()))
            internal_scalars.append(float(answer_hidden.norm(dim=-1).mean().item()))

            with torch.no_grad():
                layer_logits: torch.Tensor = self.model.lm_head(answer_hidden.unsqueeze(0))
                layer_probs: torch.Tensor = torch.softmax(layer_logits[0], dim=-1)
                layer_entropy: torch.Tensor = -(layer_probs * torch.log(layer_probs + self.config.logit_epsilon)).sum(
                    dim=-1,
                )
                mean_layer_entropy: float = float(layer_entropy.mean().item())
            internal_scalars.append(mean_layer_entropy)
            logit_lens_entropies.append(mean_layer_entropy)

        pooled_probe: torch.Tensor = torch.stack(probe_vectors, dim=0).mean(dim=0)

        entropy_drops: list[float] = []
        for index in range(len(logit_lens_entropies) - 1):
            drop_value: float = logit_lens_entropies[index] - logit_lens_entropies[index + 1]
            entropy_drops.append(float(drop_value))

        return internal_scalars, pooled_probe.tolist(), entropy_drops

    def _compute_attention_features(self, answer_start: int, seq_len: int) -> list[float]:
        """Считает attention entropy фичи по probe-слоям."""
        if not self.config.enable_attention_entropy:
            return []

        attention_features: list[float] = []
        for layer_idx in self.config.probe_layers:
            attention_key: str = f"attn_{layer_idx}"
            attention_tensor_raw: torch.Tensor | None = self._hidden.get(attention_key)
            if attention_tensor_raw is None:
                attention_features.extend([0.0, 0.0, 0.0])
                continue

            attention_tensor: torch.Tensor = attention_tensor_raw.float()
            if attention_tensor.ndim == 4:
                selected_attention: torch.Tensor = attention_tensor[0]
            elif attention_tensor.ndim == 3:
                selected_attention = attention_tensor
            elif attention_tensor.ndim == 2:
                selected_attention = attention_tensor.unsqueeze(0)
            else:
                attention_features.extend([0.0, 0.0, 0.0])
                continue

            if selected_attention.ndim == 3:
                answer_attention: torch.Tensor = selected_attention[:, answer_start:seq_len, :]
            else:
                answer_attention = selected_attention[answer_start:seq_len, :].unsqueeze(0)

            if answer_attention.numel() == 0:
                attention_features.extend([0.0, 0.0, 0.0])
                continue

            entropy: torch.Tensor = -(
                answer_attention
                * torch.log(answer_attention + self.config.attention_epsilon)
            ).sum(dim=-1)
            entropy_std: float = float(entropy.std(unbiased=False).item()) if entropy.numel() > 1 else 0.0
            attention_features.extend(
                [
                    float(entropy.mean().item()),
                    float(entropy.max().item()),
                    entropy_std,
                ]
            )

        return attention_features

    def _routing_tensor_to_answer_view(
        self,
        routing_tensor: torch.Tensor,
        answer_start: int,
        seq_len: int,
    ) -> torch.Tensor | None:
        """Приводит routing tensor к виду, удобному для анализа токенов ответа."""
        if routing_tensor.ndim == 3:
            if routing_tensor.shape[0] <= 0:
                return None
            return routing_tensor[0, answer_start:seq_len, :]

        if routing_tensor.ndim == 2:
            return routing_tensor[answer_start:seq_len, :]

        if routing_tensor.ndim >= 4:
            last_dim: int = int(routing_tensor.shape[-1])
            flattened: torch.Tensor = routing_tensor.reshape(-1, last_dim)
            answer_len: int = seq_len - answer_start
            if flattened.shape[0] < answer_len:
                return None
            return flattened[:answer_len, :]

        return None

    def _compute_moe_features(self, answer_start: int, seq_len: int) -> list[float]:
        """Считает агрегированные routing-фичи для MoE-моделей."""
        if not self.config.enable_moe_routing:
            return []

        mean_top_prob_by_layer: list[float] = []
        std_top_prob_by_layer: list[float] = []
        mean_entropy_by_layer: list[float] = []
        std_entropy_by_layer: list[float] = []
        active_ratio_by_layer: list[float] = []

        for layer_idx in self.config.probe_layers:
            moe_key: str = f"moe_{layer_idx}"
            routing_tensor_raw: torch.Tensor | None = self._hidden.get(moe_key)
            if routing_tensor_raw is None:
                continue

            routing_tensor: torch.Tensor = routing_tensor_raw.float()
            routing_answer: torch.Tensor | None = self._routing_tensor_to_answer_view(
                routing_tensor=routing_tensor,
                answer_start=answer_start,
                seq_len=seq_len,
            )
            if routing_answer is None or routing_answer.numel() == 0:
                continue

            routing_probs: torch.Tensor = torch.softmax(routing_answer, dim=-1)
            top_probs: torch.Tensor = routing_probs.max(dim=-1).values
            top_experts: torch.Tensor = routing_probs.argmax(dim=-1)
            routing_entropy: torch.Tensor = -(
                routing_probs * torch.log(routing_probs + self.config.logit_epsilon)
            ).sum(dim=-1)
            num_experts: int = int(routing_probs.shape[-1])
            active_ratio: float = float(top_experts.unique().numel() / max(num_experts, 1))

            mean_top_prob_by_layer.append(float(top_probs.mean().item()))
            std_top_prob_by_layer.append(float(top_probs.std(unbiased=False).item()))
            mean_entropy_by_layer.append(float(routing_entropy.mean().item()))
            std_entropy_by_layer.append(float(routing_entropy.std(unbiased=False).item()))
            active_ratio_by_layer.append(active_ratio)

        if not mean_top_prob_by_layer:
            return [0.0] * 10

        def aggregate(values: list[float]) -> tuple[float, float]:
            tensor_values: torch.Tensor = torch.tensor(values, dtype=torch.float32)
            return float(tensor_values.mean().item()), float(tensor_values.std(unbiased=False).item())

        mean_top_prob_stats: tuple[float, float] = aggregate(mean_top_prob_by_layer)
        std_top_prob_stats: tuple[float, float] = aggregate(std_top_prob_by_layer)
        mean_entropy_stats: tuple[float, float] = aggregate(mean_entropy_by_layer)
        std_entropy_stats: tuple[float, float] = aggregate(std_entropy_by_layer)
        active_ratio_stats: tuple[float, float] = aggregate(active_ratio_by_layer)

        return [
            mean_top_prob_stats[0],
            mean_top_prob_stats[1],
            std_top_prob_stats[0],
            std_top_prob_stats[1],
            mean_entropy_stats[0],
            mean_entropy_stats[1],
            std_entropy_stats[0],
            std_entropy_stats[1],
            active_ratio_stats[0],
            active_ratio_stats[1],
        ]

    def extract(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        answer_start: int,
    ) -> FeatureGroups:
        """Извлекает полный набор фичей из результатов forward pass."""
        validated_input: FeatureExtractorInput = FeatureExtractorInput(
            logits=logits,
            input_ids=input_ids,
            answer_start=answer_start,
        )
        seq_len: int = int(validated_input.input_ids.shape[1])

        try:
            uncertainty_features: list[float] = self._compute_uncertainty_features(
                logits=validated_input.logits,
                input_ids=validated_input.input_ids,
                answer_start=validated_input.answer_start,
            )
            internal_scalars, probe_vector, entropy_drops = self._compute_internal_and_probe(
                answer_start=validated_input.answer_start,
                seq_len=seq_len,
            )
            attention_features: list[float] = self._compute_attention_features(
                answer_start=validated_input.answer_start,
                seq_len=seq_len,
            )
            moe_features: list[float] = self._compute_moe_features(
                answer_start=validated_input.answer_start,
                seq_len=seq_len,
            )
            return FeatureGroups(
                uncertainty=uncertainty_features,
                internal_scalars=internal_scalars,
                probe_vec=probe_vector,
                attention_entropy=attention_features,
                entropy_drops=entropy_drops,
                moe_routing=moe_features,
            )
        finally:
            self._hidden.clear()


class DummyFeatureModelConfig(BaseModel):
    """Конфигурация dummy-модели фичей.

    Attributes:
        probe_dim: Размерность вектора `probe_vec`.
        vocab_size: Размер словаря для генерации фиктивных логитов.
        seed: Seed для детерминированной генерации случайных значений.
    """

    model_config = ConfigDict(frozen=True)

    probe_dim: int = 4096
    vocab_size: int = 32000
    seed: int = 42

    @field_validator("probe_dim", "vocab_size")
    @classmethod
    def validate_positive(cls, value: int) -> int:
        """Проверяет, что размерности положительные."""
        if value <= 0:
            raise ValueError("Размерность должна быть положительной")
        return value


class DummyFeatureModel(nn.Module):
    """Генерирует случайные фичи в контракте FeatureGroups для тестов."""

    def __init__(
        self,
        config: FeatureExtractorConfig | None = None,
        dummy_config: DummyFeatureModelConfig | None = None,
    ) -> None:
        """Инициализирует dummy-модель."""
        super().__init__()
        self.config: FeatureExtractorConfig = config or FeatureExtractorConfig()
        self.dummy_config: DummyFeatureModelConfig = dummy_config or DummyFeatureModelConfig()
        self._generator: torch.Generator = torch.Generator()
        self._generator.manual_seed(self.dummy_config.seed)

    def _sample_list(self, size: int) -> list[float]:
        """Возвращает список случайных чисел указанной длины."""
        if size <= 0:
            return []
        sampled: torch.Tensor = torch.randn(size, generator=self._generator, dtype=torch.float32)
        return [float(x) for x in sampled.tolist()]

    def forward(self, token_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        """Генерирует фиктивный выход модели с случайными логитами."""
        if token_ids.ndim != 2:
            raise ValueError("Ожидается token_ids формы [batch, seq_len]")

        batch_size: int = int(token_ids.shape[0])
        seq_len: int = int(token_ids.shape[1])
        logits: torch.Tensor = torch.randn(
            batch_size,
            seq_len,
            self.dummy_config.vocab_size,
            generator=self._generator,
            dtype=torch.float32,
            device=token_ids.device,
        )
        return {"logits": logits}

    def extract(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        answer_start: int,
    ) -> FeatureGroups:
        """Игнорирует входы и возвращает случайные фичи нужных размеров."""
        _ = (logits, input_ids, answer_start)

        uncertainty_size: int = 12
        internal_scalars_size: int = len(self.config.probe_layers) * 3
        probe_vec_size: int = self.dummy_config.probe_dim
        attention_size: int = len(self.config.probe_layers) * 3 if self.config.enable_attention_entropy else 0
        entropy_drops_size: int = max(len(self.config.probe_layers) - 1, 0)
        moe_size: int = 10 if self.config.enable_moe_routing else 0

        return FeatureGroups(
            uncertainty=self._sample_list(uncertainty_size),
            internal_scalars=self._sample_list(internal_scalars_size),
            probe_vec=self._sample_list(probe_vec_size),
            attention_entropy=self._sample_list(attention_size),
            entropy_drops=self._sample_list(entropy_drops_size),
            moe_routing=self._sample_list(moe_size),
        )


__all__ = [
    "DummyFeatureModel",
    "DummyFeatureModelConfig",
    "FeatureExtractorConfig",
    "FeatureExtractorInput",
    "FeatureGroups",
    "LLMFeatureExtractor",
]


