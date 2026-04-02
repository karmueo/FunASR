"""服务启动入口。

启动 FastAPI + Gradio 服务，模型在启动时预加载。
"""

import logging
import os

import uvicorn

from funasr_server.config import Settings
from funasr_server.api import create_app
from funasr_server.pipeline import TranscriptionPipeline
from funasr_server.gradio_frontend import create_gradio_app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """启动转写服务。"""
    settings = Settings.from_env()

    logger.info("初始化 Pipeline...")
    pipeline = TranscriptionPipeline(settings)
    logger.info("Pipeline 初始化完成")

    # 创建 FastAPI 应用（跳过自动模型加载，使用已初始化的 pipeline）
    os.environ["FUNASR_SKIP_MODEL_LOAD"] = "1"
    app = create_app(settings=settings)

    # 手动注入已初始化的 pipeline
    import funasr_server.api as api_module
    api_module._pipeline = pipeline

    # 创建 Gradio 前端并挂载
    gradio_app = create_gradio_app(pipeline)
    app.mount("/gradio", gradio_app)

    logger.info("服务启动: http://%s:%d", settings.host, settings.port)
    logger.info("API 文档: http://%s:%d/docs", settings.host, settings.port)
    logger.info("Gradio 界面: http://%s:%d/gradio", settings.host, settings.port)

    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        timeout_keep_alive=settings.request_timeout_s,
    )


if __name__ == "__main__":
    main()
