from __future__ import annotations

from avito.should_split.classifier import TrainedShouldSplitModel, train_should_split_models
from avito.features import (
    ShouldSplitFeatureConfig,
    TextEncoderLike,
    append_embedding_features,
    build_training_matrix,
    extract_should_split_features,
    resolve_keyphrases,
)
from avito.should_split.inference import (
    ShouldSplitArtifact,
    ShouldSplitInferenceResult,
    load_should_split_artifact,
    predict_should_split,
    predict_should_split_from_artifact,
)

__all__ = [
    "ShouldSplitArtifact",
    "ShouldSplitFeatureConfig",
    "ShouldSplitInferenceResult",
    "TextEncoderLike",
    "TrainedShouldSplitModel",
    "append_embedding_features",
    "build_training_matrix",
    "extract_should_split_features",
    "load_should_split_artifact",
    "predict_should_split",
    "predict_should_split_from_artifact",
    "train_should_split_models",
]
