"""服务启动入口。

启动 FastAPI + Gradio 服务，模型在启动时预加载。
Gradio 在独立端口运行，避免 Gradio 6.x 子路径挂载的 FileData 兼容性问题。
"""

import asyncio
import logging
import os
import signal
import sys
import threading

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

    # Gradio 在独立端口运行，避免 Gradio 6.x 子路径挂载的兼容性问题
    gradio_port = settings.port + 1
    gradio_blocks = create_gradio_app(pipeline)

    # 使用 uvicorn.Server 直接控制生命周期
    config = uvicorn.Config(
        app,
        host=settings.host,
        port=settings.port,
        timeout_keep_alive=settings.request_timeout_s,
    )
    server = uvicorn.Server(config)

    # 注册信号处理，确保 Ctrl+C 能干净退出
    def _shutdown(signum, frame):
        logger.info("收到终止信号，正在关闭服务...")
        # 关闭 Gradio
        try:
            gradio_blocks.close()
        except Exception:
            pass
        # 通知 uvicorn 退出
        server.should_exit = True

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    def run_gradio():
        """在独立线程中启动 Gradio 服务。"""
        gradio_blocks.launch(
            server_name=settings.host,
            server_port=gradio_port,
            show_error=True,
            prevent_thread_lock=True,
        )

    gradio_thread = threading.Thread(target=run_gradio, daemon=True)
    gradio_thread.start()

    logger.info("FastAPI 服务: http://%s:%d", settings.host, settings.port)
    logger.info("API 文档: http://%s:%d/docs", settings.host, settings.port)
    logger.info("Gradio 界面: http://%s:%d", settings.host, gradio_port)

    server.run()


if __name__ == "__main__":
    main()
