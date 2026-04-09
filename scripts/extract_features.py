from __future__ import annotations

from src.sber.utils.extract_features_cli import *  # noqa: F401,F403
from src.sber.utils.extract_features_cli import (
    _build_feature_column_names,
    _flatten_feature_groups,
    _sample_balanced_queries_and_answers,
    main,
)


if __name__ == "__main__":
    main()
