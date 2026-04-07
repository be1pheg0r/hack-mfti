from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from common.mistral import MistralCallConfig, call_mistral

# Minimal logger to avoid pulling the training logger hierarchy
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

SYSTEM_PROMPT = """
You are generating synthetic Russian classifieds data for Avito's 'Remont i otdelka' category.
Return ONLY valid JSON. Top-level structure MUST be an object with a single key `items`, whose value is an array of objects.
Each item object must contain:
- itemId: placeholder integer (will be overwritten)
- sourceMcId: integer from allowed list
- sourceMcTitle: exact title for sourceMcId
- description: realistic Russian ad text (1-4 sentences, no markdown, no bullets unless caseType=bullets_mixed)
- targetDetectedMcIds: list of integers (services explicitly offered in the text)
- targetSplitMcIds: subset of targetDetectedMcIds that should be split into separate drafts (exclude sourceMcId unless really standalone)
- shouldSplit: boolean, true iff targetSplitMcIds is non-empty
- caseType: one of the allowed case types
- split: one of train/val/test (approx. train 0.7, val 0.15, test 0.15)
Rules:
- Use only allowed mcIds and caseTypes provided.
- targetSplitMcIds must be a subset of targetDetectedMcIds.
- Keep descriptions coherent: if multiple services are split, text must explicitly say they can be done separately.
- Avoid markdown; plain sentences only. Use bullets only for bullets_mixed.
Again: respond with a JSON object like {"items": [...]} and nothing else.
""".strip()

@dataclass
class BucketPlan:
    source_mc_id: int
    case_type: str
    target_count: int


def load_base_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required_cols = {
        "itemId",
        "sourceMcId",
        "sourceMcTitle",
        "description",
        "targetDetectedMcIds",
        "targetSplitMcIds",
        "shouldSplit",
        "caseType",
        "split",
    }
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in base dataset: {sorted(missing)}")
    return df


def allowed_values(df: pd.DataFrame) -> tuple[dict[int, str], list[str], dict[str, float]]:
    mc_map = (
        df[["sourceMcId", "sourceMcTitle"]]
        .drop_duplicates()
        .sort_values("sourceMcId")
        .set_index("sourceMcId")
        ["sourceMcTitle"]
        .to_dict()
    )
    case_types = sorted(df["caseType"].dropna().unique().tolist())
    split_dist = (
        df["split"].value_counts(normalize=True).to_dict()
        if "split" in df
        else {"train": 0.7, "val": 0.15, "test": 0.15}
    )
    return mc_map, case_types, split_dist


def compute_bucket_plan(df: pd.DataFrame, target_new_rows: int) -> list[BucketPlan]:
    bucket_counts = df.groupby(["sourceMcId", "caseType"]).size()
    weights = 1 / (bucket_counts + 1)  # rare buckets get higher weight
    weight_sum = float(weights.sum())
    plans: list[BucketPlan] = []

    for (mc_id, case_type), weight in weights.items():
        raw = weight / weight_sum * target_new_rows
        target_count = max(1, int(round(raw)))
        plans.append(BucketPlan(source_mc_id=int(mc_id), case_type=str(case_type), target_count=target_count))

    # Adjust total to match target_new_rows
    current_total = sum(plan.target_count for plan in plans)
    if current_total != target_new_rows and plans:
        diff = target_new_rows - current_total
        # distribute the difference starting from rarest buckets (highest weight)
        ordered = sorted(plans, key=lambda p: weights[(p.source_mc_id, p.case_type)], reverse=True)
        idx = 0
        while diff != 0:
            ordered[idx % len(ordered)].target_count += 1 if diff > 0 else -1
            diff += -1 if diff > 0 else 1
            idx += 1
    return plans


def sample_examples(df: pd.DataFrame, mc_id: int, case_type: str, k: int = 2) -> list[dict[str, Any]]:
    subset = df[(df["sourceMcId"] == mc_id) & (df["caseType"] == case_type)]
    if subset.empty:
        return []
    return subset.sample(n=min(k, len(subset)), random_state=42).to_dict(orient="records")


def format_examples(examples: list[dict[str, Any]]) -> str:
    if not examples:
        return "[]"
    cleaned = []
    for ex in examples:
        cleaned.append({
            "itemId": int(ex["itemId"]),
            "sourceMcId": int(ex["sourceMcId"]),
            "sourceMcTitle": str(ex["sourceMcTitle"]),
            "description": str(ex["description"]),
            "targetDetectedMcIds": json.loads(ex["targetDetectedMcIds"]) if isinstance(ex["targetDetectedMcIds"], str) else ex["targetDetectedMcIds"],
            "targetSplitMcIds": json.loads(ex["targetSplitMcIds"]) if isinstance(ex["targetSplitMcIds"], str) else ex["targetSplitMcIds"],
            "shouldSplit": bool(ex["shouldSplit"]),
            "caseType": str(ex["caseType"]),
            "split": str(ex.get("split", "train")),
        })
    return json.dumps(cleaned, ensure_ascii=False, indent=2)


def build_user_prompt(
    plan: BucketPlan,
    mc_map: dict[int, str],
    case_types: list[str],
    examples: list[dict[str, Any]],
    split_dist: dict[str, float],
) -> str:
    allowed_mc_lines = [f"{mc_id}: {title}" for mc_id, title in mc_map.items()]
    split_guidance = ", ".join(f"{k}:{v:.2f}" for k, v in split_dist.items())
    return (
        f"Generate {plan.target_count} rows for bucket sourceMcId={plan.source_mc_id}, caseType='{plan.case_type}'.\n"
        f"Allowed mcIds and titles:\n" + "\n".join(allowed_mc_lines) + "\n"
        f"Allowed caseTypes: {', '.join(case_types)}\n"
        f"Train/val/test target ratios: {split_guidance}\n"
        f"Few-shot examples (follow style, but vary wording):\n{format_examples(examples)}\n"
        "Return ONLY a JSON array of objects."
    )


def _coerce_int_list(value: Any) -> list[int]:
    if isinstance(value, list):
        return [int(v) for v in value]
    if isinstance(value, str):
        try:
            data = json.loads(value)
            if isinstance(data, list):
                return [int(v) for v in data]
        except json.JSONDecodeError:
            pass
    return []


def _normalize_split(value: Any, split_dist: dict[str, float]) -> str:
    allowed = set(split_dist.keys()) or {"train", "val", "test"}
    if isinstance(value, str) and value in allowed:
        return value
    choices = list(allowed)
    weights = [split_dist.get(c, 1.0 / len(choices)) for c in choices]
    total = sum(weights)
    probs = [w / total for w in weights]
    return random.choices(choices, probs, k=1)[0]


def validate_and_fix_record(
    record: dict[str, Any],
    mc_map: dict[int, str],
    case_types: list[str],
    split_dist: dict[str, float],
) -> dict[str, Any] | None:
    try:
        source_mc_id = int(record.get("sourceMcId"))
    except (TypeError, ValueError):
        return None
    if source_mc_id not in mc_map:
        return None
    source_title = mc_map[source_mc_id]
    case_type = str(record.get("caseType", "")).strip()
    if case_type not in case_types:
        return None

    detected = _coerce_int_list(record.get("targetDetectedMcIds"))
    detected = [mc for mc in detected if mc in mc_map]

    split_mc = _coerce_int_list(record.get("targetSplitMcIds"))
    split_mc = [mc for mc in split_mc if mc in detected]

    should_split = bool(split_mc)
    split_value = _normalize_split(record.get("split"), split_dist)

    description = str(record.get("description", "")).strip()
    if not description:
        return None

    return {
        "itemId": 0,  # placeholder, will be reassigned
        "sourceMcId": source_mc_id,
        "sourceMcTitle": source_title,
        "description": description,
        "targetDetectedMcIds": detected,
        "targetSplitMcIds": split_mc,
        "shouldSplit": should_split,
        "caseType": case_type,
        "split": split_value,
    }


def request_batch(
    plan: BucketPlan,
    mc_map: dict[int, str],
    case_types: list[str],
    examples: list[dict[str, Any]],
    split_dist: dict[str, float],
    config: MistralCallConfig,
    max_retries: int = 3,
) -> list[dict[str, Any]]:
    user_prompt = build_user_prompt(plan, mc_map, case_types, examples, split_dist)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    for attempt in range(1, max_retries + 1):
        try:
            response_text = call_mistral(
                config,
                messages=messages,
                temperature=0.8,
                top_p=0.9,
                response_format={"type": "json_object"},
            )
            data = json.loads(response_text)
            if isinstance(data, dict) and "items" in data and isinstance(data["items"], list):
                payload = data["items"]
            elif isinstance(data, list):
                payload = data
            else:
                raise ValueError("Response is not a JSON array or items object")
            normalized: list[dict[str, Any]] = []
            for item in payload:
                if not isinstance(item, dict):
                    continue
                fixed = validate_and_fix_record(item, mc_map, case_types, split_dist)
                if fixed:
                    normalized.append(fixed)
            if normalized:
                return normalized
            logger.warning("Empty or invalid batch returned, retrying...")
        except Exception as error:  # noqa: BLE001
            logger.warning(f"Batch generation failed (attempt {attempt}/{max_retries}): {error}")
    return []


def assign_item_ids(rows: list[dict[str, Any]], start_from: int) -> list[dict[str, Any]]:
    assigned = []
    current_id = start_from
    for row in rows:
        current_id += 1
        row = dict(row)
        row["itemId"] = current_id
        assigned.append(row)
    return assigned


def dedupe_by_description(rows: Iterable[dict[str, Any]], existing_texts: set[str]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        key = row["description"].strip().lower()
        if key in existing_texts:
            continue
        existing_texts.add(key)
        result.append(row)
    return result


def save_rows(rows: list[dict[str, Any]], output_path: Path, existing_rows: list[dict[str, Any]] | None = None) -> None:
    if not rows:
        raise ValueError("No rows to save")
    payload = rows if existing_rows is None else [*existing_rows, *rows]
    df = pd.DataFrame(payload)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df)} rows to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic shouldSplit data with Mistral")
    parser.add_argument("--base-path", type=Path, default=Path("avito/data/rnc_dataset.csv"), help="Path to the base CSV dataset")
    parser.add_argument("--output-path", type=Path, default=Path("avito/data/rnc_dataset_augmented.csv"), help="Where to write augmented CSV")
    parser.add_argument("--seed-path", type=Path, default=None, help="Optional existing augmented CSV to extend/dedupe against")
    parser.add_argument("--target-rows", type=int, default=5000, help="How many new rows to generate")
    parser.add_argument("--batch-size", type=int, default=6, help="Rows requested per model call")
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["mistral-large-latest", "mistral-small-latest"],
        help="Models list in priority order",
    )
    parser.add_argument("--timeout", type=int, default=120, help="Timeout per call")
    parser.add_argument("--max-attempts", type=int, default=1, help="Attempts per call inside safe_call")
    parser.add_argument("--api-key", type=str, default="void", help="Primary API key (fallback to .credentials/mistral_api_keys)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_df = load_base_dataset(args.base_path)
    existing_aug_df: pd.DataFrame | None = None
    if args.seed_path and args.seed_path.exists():
        existing_aug_df = pd.read_csv(args.seed_path)
        logger.info(f"Loaded seed dataset with {len(existing_aug_df)} rows from {args.seed_path}")
    mc_map, case_types, split_dist = allowed_values(base_df)
    plans = compute_bucket_plan(base_df, target_new_rows=args.target_rows)
    random.shuffle(plans)

    logger.info(f"Planned generation across {len(plans)} buckets for {args.target_rows} rows")
    mistral_config = MistralCallConfig(
        models_list=args.models,
        default_api_key=args.api_key,
        timeout=args.timeout,
        max_attempts_per_call=args.max_attempts,
    )

    all_rows: list[dict[str, Any]] = []
    existing_texts = {str(text).strip().lower() for text in base_df["description"].tolist()}
    next_item_id = int(base_df["itemId"].max())
    existing_payload: list[dict[str, Any]] = []
    if existing_aug_df is not None:
        for _, row in existing_aug_df.iterrows():
            existing_payload.append(row.to_dict())
            existing_texts.add(str(row.get("description", "")).strip().lower())
        if "itemId" in existing_aug_df:
            next_item_id = max(next_item_id, int(existing_aug_df["itemId"].max()))

    for plan in plans:
        remaining = plan.target_count
        examples = sample_examples(base_df, plan.source_mc_id, plan.case_type, k=2)
        while remaining > 0:
            batch_target = min(args.batch_size, remaining)
            partial_plan = BucketPlan(plan.source_mc_id, plan.case_type, batch_target)
            batch = request_batch(
                partial_plan,
                mc_map,
                case_types,
                examples,
                split_dist,
                mistral_config,
            )
            if not batch:
                logger.warning(
                    f"No data generated for bucket (mcId={plan.source_mc_id}, caseType={plan.case_type}); moving on"
                )
                break
            batch = dedupe_by_description(batch, existing_texts)
            assigned = assign_item_ids(batch, next_item_id)
            next_item_id = assigned[-1]["itemId"] if assigned else next_item_id
            all_rows.extend(assigned)
            remaining -= len(assigned)
            logger.info(
                f"Bucket (mcId={plan.source_mc_id}, caseType={plan.case_type}) +{len(assigned)} rows, remaining {remaining}"
            )

    if not all_rows:
        raise RuntimeError("Generation produced zero rows")

    output_path = args.output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_rows(all_rows, output_path, existing_rows=existing_payload if existing_payload else None)


if __name__ == "__main__":
    main()
