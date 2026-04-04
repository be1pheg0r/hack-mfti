from __future__ import annotations

import argparse
from typing import *

from common.configs import load_config_from_namespace
from common.logger import GRADIO_DEMO_LOGGER as logger
from src.sber.constants import DEFAULT_GRADIO_DEMO_CONFIG_FPATH
from src.servers.utils.gradio_demo_utils import GradioDemoConfig, build_gradio_predict_fn


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
    parser = argparse.ArgumentParser(description="Запуск Gradio demo для пайплайна vLLM -> HF NLI")
    parser.add_argument(
        "--config-path",
        type=str,
        default=str(DEFAULT_GRADIO_DEMO_CONFIG_FPATH),
        help="Путь до YAML-конфига Gradio demo",
    )
    parser.add_argument("--host", type=str, default=None, help="Адрес Gradio UI")
    parser.add_argument("--port", type=int, default=None, help="Порт Gradio UI")
    parser.add_argument("--title", type=str, default=None, help="Заголовок интерфейса")
    parser.add_argument("--vllm-base-url", type=str, default=None, help="Базовый URL vLLM-сервера")
    parser.add_argument("--vllm-model-name", type=str, default=None, help="Имя модели для vLLM endpoint")
    parser.add_argument("--hf-nli-base-url", type=str, default=None, help="Базовый URL HF NLI сервера")
    parser.add_argument("--timeout-sec", type=float, default=None, help="Таймаут HTTP-запросов")
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
    """Создает Gradio UI для HTTP-пайплайна vLLM -> HF NLI."""
    try:
        import gradio as gr
    except ImportError as error:
        raise RuntimeError("Пакет gradio не установлен. Установи зависимости проекта.") from error

    predict = build_gradio_predict_fn(config)
    llm_name: str = config.vllm_model_name.strip() if config.vllm_model_name.strip() else "LLM"
    llm_label: str = f"Ответ {llm_name}"

    with gr.Blocks(title=config.title) as app:
        gr.Markdown(f"# {config.title}")
        gr.Markdown(
            "**Pipeline:** `User query -> vLLM (/v1/chat/completions) -> HF NLI (/v1/hf-nli/predict)`  \n"
            "Введите запрос ниже: ответ модели будет автоматически проверен на галлюцинацию."
        )

        with gr.Row():
            with gr.Column(scale=3):
                query = gr.Textbox(
                    label="Запрос",
                    lines=5,
                    placeholder="Например: Кто написал роман 'Война и мир'?",
                    info="Текст отправляется в vLLM, затем результат проверяется HF NLI.",
                )
                with gr.Row():
                    run_button = gr.Button("Запустить", variant="primary")
                    clear_button = gr.Button("Очистить")

            with gr.Column(scale=2):
                gr.Markdown("### Параметры backend")
                gr.Markdown(f"- **LLM model:** `{config.vllm_model_name}`")
                gr.Markdown(f"- **vLLM URL:** `{config.vllm_base_url}`")
                gr.Markdown(f"- **HF NLI URL:** `{config.hf_nli_base_url}`")

        with gr.Row():
            vllm_answer = _build_textbox(gr, label=llm_label, lines=10, show_copy_button=True)
            hf_nli_result = _build_textbox(gr, label="Результат HF NLI", lines=4, show_copy_button=True)

        gr.Examples(
            examples=[
                ["Кто изобрел телефон?"],
                ["Москва находится в Германии."],
                ["В каком году началась Первая мировая война?"],
            ],
            inputs=[query],
            label="Примеры запросов",
        )

        run_button.click(fn=predict, inputs=[query], outputs=[vllm_answer, hf_nli_result])
        clear_button.click(lambda: ("", "", ""), inputs=None, outputs=[query, vllm_answer, hf_nli_result])

    return app


def run(config: GradioDemoConfig) -> None:
    """Запускает Gradio demo-сервер."""
    logger.info(
        "Запускаю Gradio demo: host=%s port=%s vllm=%s hf_nli=%s",
        config.host,
        config.port,
        config.vllm_base_url,
        config.hf_nli_base_url,
    )
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


