from __future__ import annotations

import argparse
from typing import *

from common.configs import load_config_from_namespace
from common.logger import GRADIO_DEMO_LOGGER as logger
from src.sber.constants import DEFAULT_GRADIO_DEMO_CONFIG_FPATH
from src.servers.utils.gradio_demo_utils import (
    GradioDemoConfig,
    QUICK_QUERY_EXAMPLES,
    build_gradio_dummy_predict_fn,
    build_gradio_predict_fn,
    preload_dummy_samples,
)


def _build_textbox(gr: Any, **kwargs: Any) -> Any:
    """Создает Textbox с fallback для версий Gradio без show_copy_button."""
    try:
        return gr.Textbox(**kwargs)
    except TypeError:
        sanitized_kwargs: dict[str, Any] = dict(kwargs)
        sanitized_kwargs.pop("show_copy_button", None)
        return gr.Textbox(**sanitized_kwargs)


def parse_args(argv: Sequence[str] | None = None) -> GradioDemoConfig:
    """Парсит CLI и возвращает итоговую конфигурацию Gradio demo."""
    parser = argparse.ArgumentParser(description="Запуск Gradio demo для пайплайна featureextractor -> tabular_classifier")
    parser.add_argument(
        "--config-path",
        type=str,
        default=str(DEFAULT_GRADIO_DEMO_CONFIG_FPATH),
        help="Путь до YAML-конфига Gradio demo",
    )
    parser.add_argument("--host", type=str, default=None, help="Адрес Gradio UI")
    parser.add_argument("--port", type=int, default=None, help="Порт Gradio UI")
    parser.add_argument("--title", type=str, default=None, help="Заголовок интерфейса")
    parser.add_argument("--pipeline-base-url", type=str, default=None, help="Базовый URL tabular pipeline сервера")
    parser.add_argument("--timeout-sec", type=float, default=None, help="Таймаут HTTP-запросов")
    parser.add_argument("--classification-threshold", type=float, default=None, help="Порог label по hallucination_score")
    parser.add_argument(
        "--dummy-mode",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Включить dummy UX (случайный train-пример без ручного ввода)",
    )
    parser.add_argument("--dummy-examples-csv", type=str, default=None, help="Путь к train CSV для dummy-примеров")
    parser.add_argument(
        "--share",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Включить публичную share-ссылку Gradio",
    )

    namespace: argparse.Namespace = parser.parse_args(argv)
    return load_config_from_namespace(
        config_cls=GradioDemoConfig,
        namespace=namespace,
        fpath=namespace.config_path,
        section_name="gradio_demo",
    )


def build_gradio_app(config: GradioDemoConfig) -> Any:
    """Создает Gradio UI для HTTP-пайплайна featureextractor -> tabular_classifier."""
    try:
        import gradio as gr
    except ImportError as error:
        raise RuntimeError("Пакет gradio не установлен. Установи зависимости проекта.") from error

    predict = build_gradio_predict_fn(config)
    predict_dummy = build_gradio_dummy_predict_fn(config)
    with gr.Blocks(title=config.title) as app:
        gr.Markdown(f"# {config.title}")
        if config.dummy_mode:
            gr.Markdown(
                "**Pipeline:** `Dummy sample from train -> tabular_classifier`  \n"
                "Нажмите кнопку запуска: система выберет случайный train-пример и покажет результат."
            )
        else:
            gr.Markdown(
                "**Pipeline:** `Query + Model answer -> featureextractor -> tabular_classifier`  \n"
                "Введите запрос и ответ модели: backend вернет оценку на галлюцинацию."
            )

        with gr.Row():
            with gr.Column(scale=3):
                query = gr.Textbox(
                    label="Запрос",
                    lines=5,
                    placeholder="Например: Кто написал роман 'Война и мир'?",
                    info="Запрос используется для извлечения фичей.",
                    interactive=not config.dummy_mode,
                )
                model_answer = gr.Textbox(
                    label="Ответ модели",
                    lines=6,
                    placeholder="Например: Роман 'Война и мир' написал Лев Толстой.",
                    info="Ответ классифицируется tabular-моделью после извлечения фичей.",
                    interactive=not config.dummy_mode,
                )
                correct_answer = _build_textbox(
                    gr,
                    label="Эталонный ответ (только dummy mode)",
                    lines=6,
                    interactive=False,
                    visible=config.dummy_mode,
                    show_copy_button=True,
                )
                with gr.Row():
                    run_button = gr.Button("Запустить" if not config.dummy_mode else "Запустить dummy пример", variant="primary")
                    clear_button = gr.Button("Очистить")

        gr.Markdown("### Быстрые примеры запросов")
        with gr.Row():
            for example_query in QUICK_QUERY_EXAMPLES:
                gr.Button(example_query).click(
                    fn=lambda value=example_query: value,
                    inputs=None,
                    outputs=[query],
                )

            with gr.Column(scale=2):
                gr.Markdown("### Параметры backend")
                gr.Markdown(f"- **Pipeline URL:** `{config.pipeline_base_url}`")

        with gr.Row():
            pipeline_result = _build_textbox(gr, label="Результат классификатора", lines=4, show_copy_button=True)
            details = _build_textbox(gr, label="Детали запроса", lines=8, show_copy_button=True)
            stage_timings = _build_textbox(gr, label="Время этапов пайплайна", lines=4, show_copy_button=True)

        if config.dummy_mode:
            run_button.click(
                fn=predict_dummy,
                inputs=None,
                outputs=[query, model_answer, correct_answer, pipeline_result, details, stage_timings],
            )
            clear_button.click(
                lambda: ("", "", "", "", "", ""),
                inputs=None,
                outputs=[query, model_answer, correct_answer, pipeline_result, details, stage_timings],
            )
        else:
            run_button.click(fn=predict, inputs=[query, model_answer], outputs=[pipeline_result, details, stage_timings])
            clear_button.click(
                lambda: ("", "", "", "", ""),
                inputs=None,
                outputs=[query, model_answer, pipeline_result, details, stage_timings],
            )

    return app


def run(config: GradioDemoConfig) -> None:
    """Запускает Gradio demo-сервер."""
    logger.info(
        "Запускаю Gradio demo: host=%s port=%s pipeline=%s",
        config.host,
        config.port,
        config.pipeline_base_url,
    )
    if config.dummy_mode:
        preload_dummy_samples(config)
    app: Any = build_gradio_app(config)
    try:
        import gradio as gr

        app.launch(
            server_name=config.host,
            server_port=config.port,
            share=config.share,
            theme=gr.themes.Soft(),
        )
    except TypeError:
        app.launch(server_name=config.host, server_port=config.port, share=config.share)


def main(argv: Sequence[str] | None = None) -> None:
    """Точка входа CLI."""
    config: GradioDemoConfig = parse_args(argv=argv)
    run(config=config)


if __name__ == "__main__":
    main()


