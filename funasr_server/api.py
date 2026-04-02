"""FastAPI REST API 路由定义。

提供 /transcribe 和 /health 接口。
"""

import uuid
import os
import tempfile
import logging
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

from funasr_server.config import Settings
from funasr_server.pipeline import TranscriptionPipeline

logger = logging.getLogger(__name__)

# 全局 pipeline 实例，由 create_app 初始化
_pipeline: Optional[TranscriptionPipeline] = None


def get_pipeline() -> TranscriptionPipeline:
    """获取全局 pipeline 实例。

    Returns:
        已初始化的 TranscriptionPipeline 实例

    Raises:
        RuntimeError: pipeline 未初始化时抛出
    """
    if _pipeline is None:
        raise RuntimeError("Pipeline 未初始化")
    return _pipeline


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    """创建 FastAPI 应用实例。

    Args:
        settings: 服务配置，为 None 时从环境变量读取

    Returns:
        配置好路由的 FastAPI 实例
    """
    global _pipeline

    if settings is None:
        settings = Settings.from_env()

    app = FastAPI(
        title="FunASR 转写服务",
        description="多语言会议转写 API，支持中英日韩粤自动检测、VAD、说话人分离、自动翻译为中文",
        version="0.1.0",
    )

    # 仅在非测试环境加载模型
    if os.environ.get("FUNASR_SKIP_MODEL_LOAD") != "1":
        try:
            _pipeline = TranscriptionPipeline(settings)
            logger.info("Pipeline 初始化成功")
        except Exception as e:
            logger.error("Pipeline 初始化失败: %s", e)
            raise

    @app.get("/health")
    async def health_check():
        """健康检查接口。"""
        return {
            "status": "ok",
            "models_loaded": _pipeline is not None,
        }

    @app.post("/transcribe")
    async def transcribe(
        audio: UploadFile = File(...),
        language: str = Form(default="auto"),
        speaker_num: Optional[int] = Form(default=None),
    ):
        """音频转写接口。"""
        pipeline = get_pipeline()

        # 验证音频格式
        if not pipeline.validate_audio_format(audio.filename):
            raise HTTPException(
                status_code=400,
                detail=f"不支持的音频格式: {audio.filename}，支持: {pipeline.supported_extensions}",
            )

        # 保存上传文件到临时目录
        suffix = os.path.splitext(audio.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await audio.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            task_id = str(uuid.uuid4())
            duration_s = len(content) / 32000  # 粗略估计

            raw_result = pipeline.transcribe(
                audio_path=tmp_path,
                language=language,
                speaker_num=speaker_num,
            )

            result = pipeline.format_result(task_id, raw_result, duration_s)

            logger.info(
                "转写完成: task_id=%s, language=%s, speakers=%d",
                task_id, result["language"], len(result["speakers"]),
            )

            return JSONResponse(content=result)

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    return app
