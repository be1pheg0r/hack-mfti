from __future__ import annotations
from typing import *

import argparse
import os

import gradio as gr

from avito.constants import (
    GRADIO_DEFAULT_API_URL,
    GRADIO_DEFAULT_HOST,
    GRADIO_DEFAULT_ITEM_ID,
    GRADIO_DEFAULT_MC_ID,
    GRADIO_DEFAULT_MC_TITLE,
    GRADIO_DEFAULT_PORT,
)
from avito.should_split.api.client import PipelineApiClient


def build_predict_fn(api_url: str) -> Callable[[str], tuple[str, str, dict[str, Any]]]:
    """Создает функцию инференса для Gradio.

    Args:
        api_url: Базовый URL API сервиса should_split.

    Returns:
        Функция, принимающая описание объявления и возвращающая
        shouldSplit, список mcId и полный JSON-ответ.
    """
    client = PipelineApiClient(base_url=api_url)

    def predict(description: str) -> tuple[str, str, dict[str, Any]]:
        if not description or not description.strip():
            raise gr.Error("Пожалуйста, введите описание объявления.")

        try:
            result = client.infer(
                item_id=GRADIO_DEFAULT_ITEM_ID,
                mc_id=GRADIO_DEFAULT_MC_ID,
                mc_title=GRADIO_DEFAULT_MC_TITLE,
                description=description.strip(),
            )
        except Exception as exc:
            raise gr.Error(f"Ошибка запроса к API: {exc}") from exc

        response_json = result.model_dump()
        detected = result.detectedMcIds or []

        return str(result.shouldSplit), ", ".join(map(str, detected)), response_json

    return predict


def parse_args() -> argparse.Namespace:
    """Парсит CLI-аргументы запуска UI."""
    parser = argparse.ArgumentParser(description="Gradio UI for Avito should_split")
    parser.add_argument("--host", default=GRADIO_DEFAULT_HOST, help="Host for Gradio server")
    parser.add_argument("--port", type=int, default=GRADIO_DEFAULT_PORT, help="Port for Gradio server")
    parser.add_argument(
        "--api-url",
        default=os.environ.get("API_URL", GRADIO_DEFAULT_API_URL),
        help="Base URL for inference API",
    )
    parser.add_argument("--share", action="store_true", help="Enable Gradio share link")
    args, _ = parser.parse_known_args()
    return args


def build_interface(api_url: str) -> gr.Blocks:
    """Создает Gradio интерфейс для ручной проверки пайплайна."""
    predict = build_predict_fn(api_url=api_url)

    with gr.Blocks(title="Кейс от Авито. Команда: RnD Свердловской синагоги") as demo:
        gr.Markdown("# Кейс от Авито. Команда: RnD Свердловской синагоги")
        gr.Markdown(f"**API URL:** `{api_url}`")

        description = gr.Textbox(
            label="Описание объявления",
            lines=8,
            placeholder="Введите текст объявления (например: выполняем ремонт под ключ и отдельно электрику)...",
        )
        run_btn = gr.Button("Запустить", variant="primary")

        should_split = gr.Textbox(label="Необходим сплит (shouldSplit)")
        detected_ids = gr.Textbox(label="Категории (mcId)")
        response_json = gr.JSON(label="Полный JSON ответ API")

        run_btn.click(
            fn=predict,
            inputs=[description],
            outputs=[should_split, detected_ids, response_json],
        )

    return demo


def main() -> None:
    args = parse_args()
    os.environ["API_URL"] = args.api_url

    demo = build_interface(api_url=args.api_url)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
