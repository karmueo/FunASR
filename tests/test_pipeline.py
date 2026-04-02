"""Pipeline 模块测试。"""

import pytest
from unittest.mock import MagicMock, patch
from funasr_server.pipeline import TranscriptionPipeline


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
    mock_auto_model_class = MagicMock()
    with patch("funasr_server.pipeline._get_auto_model_class", return_value=mock_auto_model_class):
        from funasr_server.config import Settings
        settings = Settings(device="cpu")

        pipeline = TranscriptionPipeline(settings)

        mock_auto_model_class.assert_called_once_with(
            model="iic/SenseVoiceSmall",
            vad_model="fsmn-vad",
            vad_kwargs={"max_single_segment_time": 60000},
            punc_model="ct-punc",
            spk_model="cam++",
            spk_kwargs={"cb_kwargs": {"merge_thr": 0.5}},
            device="cpu",
            hub="ms",
        )
