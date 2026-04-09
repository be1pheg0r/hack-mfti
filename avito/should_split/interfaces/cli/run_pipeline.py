from __future__ import annotations

import argparse

from avito.should_split.core.pipeline import build_default_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run should-split pipeline")
    parser.add_argument("description", type=str, help="Текст объявления")
    args = parser.parse_args()

    pipeline = build_default_pipeline()
    result = pipeline.invoke(args.description)

    print(f"shouldSplit={result.shouldSplit}")
    print(f"stage_durations_sec={result.stage_durations_sec}")
    print(f"metadata={result.metadata}")


if __name__ == "__main__":
    main()

