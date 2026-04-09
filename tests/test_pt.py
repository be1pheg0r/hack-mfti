from __future__ import annotations

from types import SimpleNamespace
from typing import *

import torch
from torch import nn

from src.sber.models.extract_features import DummyFeatureModel, DummyFeatureModelConfig, FeatureExtractorConfig, LLMFeatureExtractor


class DummySelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.num_heads = num_heads

    def forward(self, x: torch.Tensor, output_attentions: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        out: torch.Tensor = self.proj(x)
        if not output_attentions:
            return out

        batch_size: int
        seq_len: int
        _hidden: int
        batch_size, seq_len, _hidden = x.shape
        raw: torch.Tensor = torch.randn(batch_size, self.num_heads, seq_len, seq_len, device=x.device)
        attn: torch.Tensor = torch.softmax(raw, dim=-1)
        return out, attn


class DummyMlp(nn.Module):
    def __init__(self, hidden_size: int, num_experts: int) -> None:
        super().__init__()
        self.gate = nn.Linear(hidden_size, num_experts)
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _ = self.gate(x)
        return self.proj(torch.relu(x))


class DummyLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, num_experts: int) -> None:
        super().__init__()
        self.self_attn = DummySelfAttention(hidden_size=hidden_size, num_heads=num_heads)
        self.mlp = DummyMlp(hidden_size=hidden_size, num_experts=num_experts)

    def forward(self, x: torch.Tensor, output_attentions: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        attn_out: torch.Tensor | tuple[torch.Tensor, torch.Tensor] = self.self_attn(
            x,
            output_attentions=output_attentions,
        )
        if isinstance(attn_out, tuple):
            hidden: torch.Tensor = self.mlp(attn_out[0])
            return hidden, attn_out[1]

        hidden = self.mlp(attn_out)
        return hidden


class DummyBackbone(nn.Module):
    def __init__(self, hidden_size: int, n_layers: int, num_heads: int, num_experts: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [DummyLayer(hidden_size=hidden_size, num_heads=num_heads, num_experts=num_experts) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(hidden_size)


class DummyModel(nn.Module):
    def __init__(self, vocab_size: int = 64, hidden_size: int = 16, n_layers: int = 4) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.model = DummyBackbone(hidden_size=hidden_size, n_layers=n_layers, num_heads=4, num_experts=6)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids: torch.Tensor, output_attentions: bool = False) -> Any:
        hidden: torch.Tensor = self.embedding(input_ids)
        collected_attentions: list[torch.Tensor] = []

        for layer in self.model.layers:
            layer_out: torch.Tensor | tuple[torch.Tensor, torch.Tensor] = layer(hidden, output_attentions=output_attentions)
            if isinstance(layer_out, tuple):
                hidden = layer_out[0]
                collected_attentions.append(layer_out[1])
            else:
                hidden = layer_out

        logits: torch.Tensor = self.lm_head(self.model.norm(hidden))
        return SimpleNamespace(logits=logits, attentions=collected_attentions if output_attentions else None)


def test_feature_extractor_config_from_yaml(tmp_path: Any) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "feature_extractor:\n"
        "  probe_layers: [0, 1, 3]\n"
        "  logit_epsilon: 1e-9\n"
        "  attention_epsilon: 1e-9\n"
        "  enable_attention_entropy: true\n"
        "  enable_moe_routing: true\n"
        "  use_output_attentions: true\n",
        encoding="utf-8",
    )

    config: FeatureExtractorConfig = FeatureExtractorConfig.from_yaml(config_path)
    assert config.probe_layers == [0, 1, 3]
    assert config.logit_epsilon == 1e-9
    assert config.attention_epsilon == 1e-9


def test_llm_feature_extractor_produces_expected_feature_groups() -> None:
    torch.manual_seed(7)

    model = DummyModel(vocab_size=50, hidden_size=12, n_layers=4)
    config = FeatureExtractorConfig(
        probe_layers=[0, 1, 2, 3],
        enable_attention_entropy=True,
        enable_moe_routing=True,
    )
    extractor = LLMFeatureExtractor(model=model, config=config)

    input_ids: torch.Tensor = torch.randint(low=0, high=50, size=(1, 8), dtype=torch.long)
    answer_start: int = 3

    with extractor:
        outputs = extractor.forward(input_ids)

    features = extractor.extract(logits=outputs.logits, input_ids=input_ids, answer_start=answer_start)

    assert len(features.uncertainty) == 12
    assert len(features.internal_scalars) == len(config.probe_layers) * 3
    assert len(features.probe_vec) == 12
    assert len(features.attention_entropy) == len(config.probe_layers) * 3
    assert len(features.entropy_drops) == len(config.probe_layers) - 1
    assert len(features.moe_routing) == 10

    all_values: list[float] = (
        features.uncertainty
        + features.internal_scalars
        + features.probe_vec
        + features.attention_entropy
        + features.entropy_drops
        + features.moe_routing
    )
    assert all(torch.isfinite(torch.tensor(all_values)).tolist())


def test_dummy_feature_model_returns_random_features_with_expected_shapes() -> None:
    config = FeatureExtractorConfig(
        probe_layers=[0, 1, 2, 3],
        enable_attention_entropy=True,
        enable_moe_routing=True,
    )
    dummy_config = DummyFeatureModelConfig(probe_dim=128, vocab_size=100, seed=123)
    dummy_model = DummyFeatureModel(config=config, dummy_config=dummy_config)

    input_ids: torch.Tensor = torch.randint(low=0, high=100, size=(1, 9), dtype=torch.long)
    dummy_out: dict[str, torch.Tensor] = dummy_model.forward(input_ids)
    features = dummy_model.extract(
        logits=dummy_out["logits"],
        input_ids=input_ids,
        answer_start=3,
    )

    assert tuple(dummy_out["logits"].shape) == (1, 9, 100)
    assert len(features.uncertainty) == 12
    assert len(features.internal_scalars) == len(config.probe_layers) * 3
    assert len(features.probe_vec) == 128
    assert len(features.attention_entropy) == len(config.probe_layers) * 3
    assert len(features.entropy_drops) == len(config.probe_layers) - 1
    assert len(features.moe_routing) == 10

    all_values: list[float] = (
        features.uncertainty
        + features.internal_scalars
        + features.probe_vec
        + features.attention_entropy
        + features.entropy_drops
        + features.moe_routing
    )
    assert all(torch.isfinite(torch.tensor(all_values)).tolist())


def test_uncertainty_uses_causal_shifted_logits_for_answer_tokens() -> None:
    model = DummyModel(vocab_size=5, hidden_size=8, n_layers=2)
    config = FeatureExtractorConfig(
        probe_layers=[0],
        enable_attention_entropy=False,
        enable_moe_routing=False,
    )
    extractor = LLMFeatureExtractor(model=model, config=config)

    input_ids: torch.Tensor = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    logits: torch.Tensor = torch.zeros((1, 4, 5), dtype=torch.float32)
    logits[0, 1, 3] = 6.0  # Для токена input_ids[2] используется позиция 1.
    logits[0, 2, 4] = 5.0  # Для токена input_ids[3] используется позиция 2.
    logits[0, 3, 0] = 9.0  # Последняя позиция не должна участвовать в расчете answer log-prob.

    features = extractor.extract(logits=logits, input_ids=input_ids, answer_start=2)

    selected_logits: torch.Tensor = logits[0, 1:3, :]
    selected_ids: torch.Tensor = input_ids[0, 2:4]
    expected_log_probs: torch.Tensor = torch.log_softmax(selected_logits, dim=-1).gather(1, selected_ids.unsqueeze(1)).squeeze(-1)

    assert abs(features.uncertainty[0] - torch.mean(expected_log_probs).item()) < 1e-6
    assert abs(features.uncertainty[9] - expected_log_probs[0].item()) < 1e-6


