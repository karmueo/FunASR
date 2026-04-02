# 多语言会议转写服务 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建一个基于 FunASR 的多语言会议转写服务，支持中英文自动检测、VAD 分段、说话人分离，提供 FastAPI REST API 和 Gradio 前端。

**Architecture:** 四层架构——Gradio 前端 → FastAPI REST API → 转写服务层（AutoModel pipeline: fsmn-vad + SenseVoiceSmall + ct-punc + cam++） → 基础设施层（配置、日志、模型管理）。前端和 API 共存于同一进程中，模型在启动时预加载。

**Tech Stack:** Python 3.10+, FastAPI, Gradio, FunASR (AutoModel), SenseVoiceSmall, fsmn-vad, ct-punc, cam++, uvicorn, pytest

---

## File Structure

```
funasr_server/
├── __init__.py              # 包标识
├── config.py                # 配置管理：环境变量 + 默认值
├── pipeline.py              # AutoModel pipeline 封装：初始化、转写、后处理
├── api.py                   # FastAPI 路由：/transcribe, /health
├── app.py                   # 启动入口：挂载 Gradio + FastAPI
├── gradio_frontend.py       # Gradio UI 定义
├── requirements.txt         # 依赖清单
tests/
├── __init__.py
├── test_config.py           # 配置解析测试
├── test_pipeline.py         # pipeline 单元测试
├── test_api.py              # API 接口测试
```

---

### Task 1: 项目骨架与配置模块

**Files:**
- Create: `funasr_server/__init__.py`
- Create: `funasr_server/config.py`
- Create: `funasr_server/requirements.txt`
- Create: `tests/__init__.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: 创建项目目录和文件**

```bash
mkdir -p funasr_server tests
touch funasr_server/__init__.py tests/__init__.py
```

- [ ] **Step 2: 编写 requirements.txt**

```
funasr
fastapi
uvicorn[standard]
gradio>=4.0.0
python-multipart
pytest
httpx
```

文件: `funasr_server/requirements.txt`

- [ ] **Step 3: 编写 config.py 测试**

文件: `tests/test_config.py`

```python
"""配置模块测试。"""

import os
import pytest
from funasr_server.config import Settings


def test_default_settings():
    """测试默认配置值。"""
    s = Settings()
    assert s.device == "cuda"
    assert s.asr_model == "iic/SenseVoiceSmall"
    assert s.vad_model == "fsmn-vad"
    assert s.punc_model == "ct-punc"
    assert s.spk_model == "cam++"
    assert s.max_single_segment_time == 60000
    assert s.batch_size_s == 300
    assert s.merge_thr == 0.5
    assert s.host == "0.0.0.0"
    assert s.port == 8000
    assert s.max_audio_duration_s == 7200


def test_env_override(monkeypatch):
    """测试环境变量覆盖默认配置。"""
    monkeypatch.setenv("FUNASR_DEVICE", "cpu")
    monkeypatch.setenv("FUNASR_PORT", "9000")
    monkeypatch.setenv("FUNASR_MAX_AUDIO_DURATION_S", "3600")
    s = Settings()
    assert s.device == "cpu"
    assert s.port == 9000
    assert s.max_audio_duration_s == 3600


def test_supported_extensions():
    """测试支持的音频格式列表。"""
    s = Settings()
    assert ".wav" in s.supported_extensions
    assert ".mp3" in s.supported_extensions
    assert ".flac" in s.supported_extensions
    assert ".txt" not in s.supported_extensions
```

- [ ] **Step 4: 运行测试确认失败**

Run: `cd /home/tl/work/FunASR && python -m pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'funasr_server'`

- [ ] **Step 5: 编写 config.py 实现**

文件: `funasr_server/config.py`

```python
"""转写服务配置管理。

从环境变量读取配置，提供合理的默认值。
"""

import os
from dataclasses import dataclass, field
from typing import List


# 支持的音频文件扩展名
SUPPORTED_EXTENSIONS: List[str] = [".wav", ".mp3", ".flac", ".ogg", ".m4a"]


@dataclass
class Settings:
    """转写服务全局配置。

    所有字段均可通过 FUNASR_ 前缀的环境变量覆盖，
    例如 FUNASR_DEVICE=cpu 覆盖 device 默认值。
    """

    # 模型配置
    asr_model: str = "iic/SenseVoiceSmall"
    vad_model: str = "fsmn-vad"
    punc_model: str = "ct-punc"
    spk_model: str = "cam++"
    device: str = "cuda"
    hub: str = "ms"

    # VAD 参数
    max_single_segment_time: int = 60000  # 毫秒

    # ASR 参数
    batch_size_s: int = 300  # 动态批大小（秒）
    batch_size_threshold_s: int = 60  # 单段最大时长（秒）

    # 说话人聚类参数
    merge_thr: float = 0.5

    # 服务配置
    host: str = "0.0.0.0"
    port: int = 8000
    max_audio_duration_s: int = 7200  # 最大音频时长（秒），默认2小时
    request_timeout_s: int = 600  # 请求超时（秒）

    # 支持的音频格式
    supported_extensions: List[str] = field(
        default_factory=lambda: list(SUPPORTED_EXTENSIONS)
    )

    @classmethod
    def from_env(cls) -> "Settings":
        """从环境变量构建配置，FUNASR_ 前缀。

        环境变量名规则：FUNASR_<大写字段名>，例如 FUNASR_DEVICE。
        """
        kwargs = {}
        for f in cls.__dataclass_fields__:
            env_key = f"FUNASR_{f.upper()}"
            env_val = os.environ.get(env_key)
            if env_val is not None:
                field_type = cls.__dataclass_fields__[f].type
                if field_type == "int" or field_type is int:
                    kwargs[f] = int(env_val)
                elif field_type == "float" or field_type is float:
                    kwargs[f] = float(env_val)
                else:
                    kwargs[f] = env_val
        return cls(**kwargs)
```

- [ ] **Step 6: 运行测试确认通过**

Run: `cd /home/tl/work/FunASR && python -m pytest tests/test_config.py -v`
Expected: 3 passed

- [ ] **Step 7: 提交**

```bash
git add funasr_server/ tests/test_config.py
git commit -m "feat: add project skeleton and config module"
```

---

### Task 2: Pipeline 封装模块

**Files:**
- Create: `funasr_server/pipeline.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: 编写 pipeline.py 测试**

文件: `tests/test_pipeline.py`

```python
"""Pipeline 模块测试。"""

import pytest
from unittest.mock import MagicMock, patch
from funasr_server.pipeline import TranscriptionPipeline, TranscriptionResult


def test_validate_audio_format_wav():
    """测试 wav 格式验证通过。"""
    with patch("funasr_server.pipeline.TranscriptionPipeline.__init__", return_value=None):
        pipeline = TranscriptionPipeline.__new__(TranscriptionPipeline)
        pipeline.supported_extensions = [".wav", ".mp3", ".flac"]
    assert pipeline.validate_audio_format("test.wav") is True


def test_validate_audio_format_unsupported():
    """测试不支持的格式验证失败。"""
    with patch("funasr_server.pipeline.TranscriptionPipeline.__init__", return_value=None):
        pipeline = TranscriptionPipeline.__new__(TranscriptionPipeline)
        pipeline.supported_extensions = [".wav", ".mp3", ".flac"]
    assert pipeline.validate_audio_format("test.avi") is False


def test_validate_audio_format_case_insensitive():
    """测试格式验证不区分大小写。"""
    with patch("funasr_server.pipeline.TranscriptionPipeline.__init__", return_value=None):
        pipeline = TranscriptionPipeline.__new__(TranscriptionPipeline)
        pipeline.supported_extensions = [".wav", ".mp3", ".flac"]
    assert pipeline.validate_audio_format("test.WAV") is True
    assert pipeline.validate_audio_format("test.Mp3") is True


def test_format_result_structure():
    """测试结果格式化输出结构。"""
    # 模拟 AutoModel 的原始输出
    raw_result = {
        "key": "test_audio",
        "text": "大家好，今天我们讨论项目进展。好的，我汇报一下。",
        "timestamp": [[0, 500], [500, 1200], [1200, 2000], [2000, 3500]],
        "sentence_info": [
            {
                "start": 0,
                "end": 2000,
                "sentence": "大家好，今天我们讨论项目进展。",
                "speaker": "spk0",
            },
            {
                "start": 2000,
                "end": 3500,
                "sentence": "好的，我汇报一下。",
                "speaker": "spk1",
            },
        ],
    }

    with patch("funasr_server.pipeline.TranscriptionPipeline.__init__", return_value=None):
        pipeline = TranscriptionPipeline.__new__(TranscriptionPipeline)

    result = pipeline.format_result("test-task-id", raw_result, duration_s=10.5)

    assert result["task_id"] == "test-task-id"
    assert result["duration_s"] == 10.5
    assert len(result["segments"]) == 2
    assert result["segments"][0]["speaker"] == "spk0"
    assert result["segments"][0]["start_ms"] == 0
    assert result["segments"][0]["end_ms"] == 2000
    assert result["segments"][0]["text"] == "大家好，今天我们讨论项目进展。"
    assert result["segments"][1]["speaker"] == "spk1"
    assert "spk0" in result["speakers"]
    assert "spk1" in result["speakers"]


def test_format_result_empty_segments():
    """测试空结果的格式化。"""
    raw_result = {
        "key": "silent_audio",
        "text": "",
        "timestamp": [],
    }

    with patch("funasr_server.pipeline.TranscriptionPipeline.__init__", return_value=None):
        pipeline = TranscriptionPipeline.__new__(TranscriptionPipeline)

    result = pipeline.format_result("empty-task", raw_result, duration_s=5.0)

    assert result["task_id"] == "empty-task"
    assert result["segments"] == []
    assert result["speakers"] == []


def test_pipeline_init_calls_auto_model():
    """测试 pipeline 初始化时正确调用 AutoModel。"""
    with patch("funasr_server.pipeline.AutoModel") as mock_auto_model:
        from funasr_server.config import Settings
        settings = Settings(device="cpu")

        pipeline = TranscriptionPipeline(settings)

        mock_auto_model.assert_called_once_with(
            model="iic/SenseVoiceSmall",
            vad_model="fsmn-vad",
            vad_kwargs={"max_single_segment_time": 60000},
            punc_model="ct-punc",
            spk_model="cam++",
            spk_kwargs={"cb_kwargs": {"merge_thr": 0.5}},
            device="cpu",
            hub="ms",
        )
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd /home/tl/work/FunASR && python -m pytest tests/test_pipeline.py -v`
Expected: FAIL — `ModuleNotFoundError` 或 `ImportError`

- [ ] **Step 3: 编写 pipeline.py 实现**

文件: `funasr_server/pipeline.py`

```python
"""转写 Pipeline 封装。

封装 FunASR AutoModel，提供初始化、音频格式验证、
转写执行和结果格式化功能。
"""

import uuid
import logging
import os
from typing import Dict, List, Optional

from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

from funasr_server.config import Settings

logger = logging.getLogger(__name__)


class TranscriptionResult:
    """转写结果类型别名，用于类型提示。"""
    pass


class TranscriptionPipeline:
    """转写管道，封装 AutoModel 的 VAD + ASR + 标点 + 说话人分离流程。

    Attributes:
        model: AutoModel 实例
        settings: 服务配置
        supported_extensions: 支持的音频格式列表
    """

    def __init__(self, settings: Settings) -> None:
        """初始化转写管道，加载所有模型。

        Args:
            settings: 服务配置实例
        """
        self.settings = settings
        self.supported_extensions = settings.supported_extensions

        logger.info(
            "正在加载模型: ASR=%s, VAD=%s, PUNC=%s, SPK=%s, device=%s",
            settings.asr_model, settings.vad_model,
            settings.punc_model, settings.spk_model, settings.device,
        )

        self.model = AutoModel(
            model=settings.asr_model,
            vad_model=settings.vad_model,
            vad_kwargs={"max_single_segment_time": settings.max_single_segment_time},
            punc_model=settings.punc_model,
            spk_model=settings.spk_model,
            spk_kwargs={"cb_kwargs": {"merge_thr": settings.merge_thr}},
            device=settings.device,
            hub=settings.hub,
        )

        logger.info("模型加载完成")

    def validate_audio_format(self, filename: str) -> bool:
        """检查音频文件格式是否支持。

        Args:
            filename: 文件名或路径

        Returns:
            格式支持返回 True，否则 False
        """
        _, ext = os.path.splitext(filename)
        return ext.lower() in self.supported_extensions

    def transcribe(
        self,
        audio_path: str,
        language: str = "auto",
        speaker_num: Optional[int] = None,
    ) -> Dict:
        """执行转写。

        Args:
            audio_path: 音频文件路径
            language: 语言标识，auto/zh/en
            speaker_num: 说话人数量，None 为自动检测

        Returns:
            AutoModel 原始结果字典
        """
        generate_kwargs = {
            "input": audio_path,
            "language": language,
            "use_itn": True,
            "batch_size_s": self.settings.batch_size_s,
            "batch_size_threshold_s": self.settings.batch_size_threshold_s,
            "merge_vad": True,
            "merge_length_s": 15,
            "return_spk_res": True,
            "sentence_timestamp": True,
        }

        if speaker_num is not None:
            generate_kwargs["preset_spk_num"] = speaker_num

        logger.info("开始转写: %s, language=%s, speaker_num=%s",
                     audio_path, language, speaker_num)

        results = self.model.generate(**generate_kwargs)

        if results and len(results) > 0:
            raw_text = results[0].get("text", "")
            results[0]["text"] = rich_transcription_postprocess(raw_text)

        logger.info("转写完成: %d 个结果段", len(results) if results else 0)
        return results[0] if results else {}

    def format_result(
        self,
        task_id: str,
        raw_result: Dict,
        duration_s: float,
    ) -> Dict:
        """将 AutoModel 原始结果格式化为 API 响应格式。

        Args:
            task_id: 任务唯一标识
            raw_result: AutoModel 原始输出
            duration_s: 音频总时长（秒）

        Returns:
            格式化后的响应字典
        """
        segments: List[Dict] = []
        speakers_set = set()

        sentence_info = raw_result.get("sentence_info", [])
        if sentence_info:
            for sent in sentence_info:
                speaker = sent.get("speaker", "spk0")
                speakers_set.add(speaker)
                segments.append({
                    "speaker": speaker,
                    "start_ms": sent.get("start", 0),
                    "end_ms": sent.get("end", 0),
                    "text": sent.get("sentence", ""),
                })

        return {
            "task_id": task_id,
            "language": _detect_language(raw_result),
            "duration_s": duration_s,
            "segments": segments,
            "speakers": sorted(speakers_set),
        }


def _detect_language(raw_result: Dict) -> str:
    """从转写结果中推断语言标识。

    Args:
        raw_result: AutoModel 原始输出

    Returns:
        语言代码，zh 或 en
    """
    text = raw_result.get("text", "")
    # SenseVoice 输出可能包含语言标记
    if "<|en|>" in text:
        return "en"
    if "<|zh|>" in text:
        return "zh"
    # 简单启发式：统计中文字符比例
    chinese_count = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    total = max(len(text.replace(" ", "")), 1)
    return "zh" if chinese_count / total > 0.3 else "en"
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd /home/tl/work/FunASR && python -m pytest tests/test_pipeline.py -v`
Expected: 6 passed

- [ ] **Step 5: 提交**

```bash
git add funasr_server/pipeline.py tests/test_pipeline.py
git commit -m "feat: add transcription pipeline module"
```

---

### Task 3: FastAPI REST API

**Files:**
- Create: `funasr_server/api.py`
- Test: `tests/test_api.py`

- [ ] **Step 1: 编写 API 测试**

文件: `tests/test_api.py`

```python
"""FastAPI 接口测试。"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from funasr_server.config import Settings


@pytest.fixture
def mock_pipeline():
    """创建 mock pipeline 实例。"""
    pipeline = MagicMock()
    pipeline.validate_audio_format.return_value = True
    pipeline.transcribe.return_value = {
        "key": "test",
        "text": "Hello world.",
        "timestamp": [[0, 1000], [1000, 2000]],
        "sentence_info": [
            {"start": 0, "end": 2000, "sentence": "Hello world.", "speaker": "spk0"},
        ],
    }
    pipeline.format_result.return_value = {
        "task_id": "test-uuid",
        "language": "en",
        "duration_s": 10.0,
        "segments": [
            {"speaker": "spk0", "start_ms": 0, "end_ms": 2000, "text": "Hello world."},
        ],
        "speakers": ["spk0"],
    }
    pipeline.supported_extensions = [".wav", ".mp3", ".flac"]
    return pipeline


@pytest.fixture
def client(mock_pipeline):
    """创建测试客户端，注入 mock pipeline。"""
    with patch("funasr_server.api.get_pipeline", return_value=mock_pipeline):
        from funasr_server.api import create_app
        app = create_app(settings=Settings(device="cpu"))
        # 替换实际的 pipeline 为 mock
        with patch("funasr_server.api._pipeline", mock_pipeline):
            yield TestClient(app)


def test_health_check(client):
    """测试健康检查接口。"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["models_loaded"] is True


def test_transcribe_success(client, mock_pipeline, tmp_path):
    """测试正常转写请求。"""
    # 创建临时音频文件
    audio_file = tmp_path / "test.wav"
    audio_file.write_bytes(b"fake audio data")

    with open(audio_file, "rb") as f:
        response = client.post(
            "/transcribe",
            files={"audio": ("test.wav", f, "audio/wav")},
            data={"language": "en"},
        )

    assert response.status_code == 200
    data = response.json()
    assert "task_id" in data
    assert data["language"] == "en"
    assert len(data["segments"]) == 1
    assert data["segments"][0]["speaker"] == "spk0"


def test_transcribe_unsupported_format(client, mock_pipeline, tmp_path):
    """测试不支持的音频格式返回 400。"""
    mock_pipeline.validate_audio_format.return_value = False
    text_file = tmp_path / "test.txt"
    text_file.write_text("not audio")

    with open(text_file, "rb") as f:
        response = client.post(
            "/transcribe",
            files={"audio": ("test.txt", f, "text/plain")},
        )

    assert response.status_code == 400
    assert "不支持的音频格式" in response.json()["detail"]
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd /home/tl/work/FunASR && python -m pytest tests/test_api.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'funasr_server.api'`

- [ ] **Step 3: 编写 api.py 实现**

文件: `funasr_server/api.py`

```python
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
        TranscriptionPipeline 实例

    Raises:
        RuntimeError: pipeline 未初始化
    """
    if _pipeline is None:
        raise RuntimeError("Pipeline 未初始化")
    return _pipeline


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    """创建 FastAPI 应用实例。

    Args:
        settings: 服务配置，为 None 时从环境变量读取

    Returns:
        配置好的 FastAPI 应用
    """
    global _pipeline

    if settings is None:
        settings = Settings.from_env()

    app = FastAPI(
        title="FunASR 转写服务",
        description="多语言会议转写 API，支持中英文自动检测、VAD、说话人分离",
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
        """音频转写接口。

        Args:
            audio: 上传的音频文件
            language: 语言标识，auto/zh/en
            speaker_num: 说话人数量，null 为自动检测

        Returns:
            结构化转写结果 JSON
        """
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

            # 计算音频时长（通过文件大小粗略估计，精确值由 pipeline 计算）
            duration_s = len(content) / 32000  # 粗略估计，16kHz 16bit mono

            # 执行转写
            raw_result = pipeline.transcribe(
                audio_path=tmp_path,
                language=language,
                speaker_num=speaker_num,
            )

            # 格式化结果
            result = pipeline.format_result(task_id, raw_result, duration_s)

            logger.info(
                "转写完成: task_id=%s, language=%s, speakers=%d",
                task_id, result["language"], len(result["speakers"]),
            )

            return JSONResponse(content=result)

        finally:
            # 清理临时文件
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    return app
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd /home/tl/work/FunASR && FUNASR_SKIP_MODEL_LOAD=1 python -m pytest tests/test_api.py -v`
Expected: 3 passed

- [ ] **Step 5: 提交**

```bash
git add funasr_server/api.py tests/test_api.py
git commit -m "feat: add FastAPI REST API with /transcribe and /health"
```

---

### Task 4: Gradio 前端

**Files:**
- Create: `funasr_server/gradio_frontend.py`

- [ ] **Step 1: 编写 Gradio 前端**

文件: `funasr_server/gradio_frontend.py`

```python
"""Gradio 轻量前端界面。

提供音频上传、转写、结果展示和 JSON 下载功能。
"""

import json
import gradio as gr
from funasr_server.pipeline import TranscriptionPipeline


def create_gradio_app(pipeline: TranscriptionPipeline) -> gr.Blocks:
    """创建 Gradio 前端应用。

    Args:
        pipeline: 转写管道实例

    Returns:
        Gradio Blocks 应用
    """
    def transcribe_audio(audio_path, language, speaker_num):
        """处理音频转写请求。

        Args:
            audio_path: 上传的音频文件路径
            language: 语言选择
            speaker_num: 说话人数量，0 表示自动检测

        Returns:
            results_table: 结果表格数据
            json_output: JSON 字符串输出
            status_text: 状态信息
        """
        if audio_path is None:
            return [], '{"error": "请上传音频文件"}', "未上传文件"

        try:
            import uuid
            task_id = str(uuid.uuid4())

            lang_map = {"自动检测": "auto", "中文": "zh", "英文": "en"}
            lang = lang_map.get(language, "auto")

            spk_num = None if speaker_num == 0 else speaker_num

            raw_result = pipeline.transcribe(
                audio_path=audio_path,
                language=lang,
                speaker_num=spk_num,
            )

            # 计算音频时长
            import os
            file_size = os.path.getsize(audio_path)
            duration_s = file_size / 32000

            result = pipeline.format_result(task_id, raw_result, duration_s)

            # 构建表格数据
            table_data = []
            for seg in result["segments"]:
                start_sec = seg["start_ms"] / 1000
                end_sec = seg["end_ms"] / 1000
                time_str = f"{start_sec:.1f}s - {end_sec:.1f}s"
                table_data.append([seg["speaker"], time_str, seg["text"]])

            json_str = json.dumps(result, ensure_ascii=False, indent=2)
            status = f"转写完成 | 语言: {result['language']} | 时长: {duration_s:.1f}s | 说话人: {len(result['speakers'])} 人"

            return table_data, json_str, status

        except Exception as e:
            return [], json.dumps({"error": str(e)}, ensure_ascii=False), f"错误: {str(e)}"

    with gr.Blocks(title="FunASR 会议转写", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# FunASR 会议转写服务")
        gr.Markdown("上传音频文件，自动进行语音识别、说话人分离，支持中文和英文。")

        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    label="上传音频",
                    type="filepath",
                    sources=["upload"],
                )
                language_radio = gr.Radio(
                    choices=["自动检测", "中文", "英文"],
                    value="自动检测",
                    label="语言",
                )
                speaker_num_slider = gr.Slider(
                    minimum=0, maximum=10, value=0, step=1,
                    label="说话人数量（0=自动检测）",
                )
                transcribe_btn = gr.Button("开始转写", variant="primary")
                status_text = gr.Textbox(label="状态", interactive=False)

            with gr.Column(scale=2):
                results_table = gr.Dataframe(
                    headers=["说话人", "时间", "文本"],
                    label="转写结果",
                    interactive=False,
                )
                json_output = gr.Textbox(
                    label="JSON 输出",
                    lines=10,
                    interactive=False,
                )
                download_btn = gr.DownloadButton(label="下载 JSON")

        transcribe_btn.click(
            fn=transcribe_audio,
            inputs=[audio_input, language_radio, speaker_num_slider],
            outputs=[results_table, json_output, status_text],
        )

    return demo
```

- [ ] **Step 2: 验证模块可导入**

Run: `cd /home/tl/work/FunASR && python -c "from funasr_server.gradio_frontend import create_gradio_app; print('OK')"`
Expected: OK

- [ ] **Step 3: 提交**

```bash
git add funasr_server/gradio_frontend.py
git commit -m "feat: add Gradio frontend for transcription demo"
```

---

### Task 5: 启动入口

**Files:**
- Create: `funasr_server/app.py`

- [ ] **Step 1: 编写启动入口**

文件: `funasr_server/app.py`

```python
"""服务启动入口。

启动 FastAPI + Gradio 服务，模型在启动时预加载。
"""

import logging
import uvicorn
from fastapi import FastAPI
from starlette.routing import Mount

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
    import os
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
```

- [ ] **Step 2: 验证模块可导入**

Run: `cd /home/tl/work/FunASR && python -c "from funasr_server.app import main; print('OK')"`
Expected: OK

- [ ] **Step 3: 提交**

```bash
git add funasr_server/app.py
git commit -m "feat: add service entry point with Gradio + FastAPI"
```

---

### Task 6: 端到端验证

**Files:**
- 无新文件，使用测试音频验证完整流程

- [ ] **Step 1: 启动服务**

Run: `cd /home/tl/work/FunASR && python -m funasr_server.app`

Expected: 日志输出 "模型加载完成"、"服务启动: http://0.0.0.0:8000"

- [ ] **Step 2: 健康检查**

Run: `curl http://localhost:8000/health`
Expected: `{"status":"ok","models_loaded":true}`

- [ ] **Step 3: 使用英文测试音频发送转写请求**

Run:
```bash
curl -X POST http://localhost:8000/transcribe \
  -F "audio=@/home/tl/data/tmp/From skeptic to true believer： How OpenClaw changed my life ｜ Claire Vo [DIa0MYJzM5I].mp3" \
  -F "language=en"
```

Expected: 返回 JSON，包含 `task_id`、`language: "en"`、`segments` 数组和 `speakers` 数组

- [ ] **Step 4: 验证 Gradio 界面可访问**

浏览器打开 `http://localhost:8000/gradio`，上传测试音频，点击"开始转写"，确认结果表格和 JSON 输出正常显示

- [ ] **Step 5: 运行全部单元测试**

Run: `cd /home/tl/work/FunASR && FUNASR_SKIP_MODEL_LOAD=1 python -m pytest tests/ -v`
Expected: 所有测试通过

- [ ] **Step 6: 最终提交**

```bash
git add -A
git commit -m "feat: complete multilingual transcription service"
```
