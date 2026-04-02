"""FastAPI 接口测试。"""

import os
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
            {
                "speaker": "spk0",
                "start_ms": 0,
                "end_ms": 2000,
                "text": "Hello world.",
                "text_zh": "你好，世界。",
                "language": "en",
            },
        ],
        "speakers": ["spk0"],
    }
    pipeline.supported_extensions = [".wav", ".mp3", ".flac"]
    return pipeline


@pytest.fixture
def client(mock_pipeline):
    """创建测试客户端，注入 mock pipeline 并跳过模型加载。"""
    os.environ["FUNASR_SKIP_MODEL_LOAD"] = "1"
    try:
        from funasr_server.api import create_app
        app = create_app(settings=Settings(device="cpu"))
        # 替换全局 pipeline 为 mock
        with patch("funasr_server.api._pipeline", mock_pipeline):
            yield TestClient(app)
    finally:
        os.environ.pop("FUNASR_SKIP_MODEL_LOAD", None)


def test_health_check(client, mock_pipeline):
    """测试健康检查接口。"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["models_loaded"] is True


def test_transcribe_success(client, mock_pipeline, tmp_path):
    """测试正常转写请求。"""
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
    assert data["segments"][0]["text_zh"] == "你好，世界。"
    assert data["segments"][0]["language"] == "en"

    # 验证 pipeline 方法被正确调用
    mock_pipeline.transcribe.assert_called_once()
    mock_pipeline.format_result.assert_called_once()


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
