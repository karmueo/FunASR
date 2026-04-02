"""Pipeline 模块测试。"""

import pytest
from unittest.mock import MagicMock, patch
from funasr_server.pipeline import TranscriptionPipeline, _detect_language


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
    pipeline.translator = None

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
    # 无翻译器时 text_zh 为 None
    assert result["segments"][0].get("text_zh") is None


def test_format_result_empty_segments():
    """测试空结果的格式化。"""
    raw_result = {
        "key": "silent_audio",
        "text": "",
        "timestamp": [],
    }

    with patch("funasr_server.pipeline.TranscriptionPipeline.__init__", return_value=None):
        pipeline = TranscriptionPipeline.__new__(TranscriptionPipeline)
    pipeline.translator = None

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
            model=settings.asr_model,
            vad_model=settings.vad_model,
            vad_kwargs={"max_single_segment_time": settings.max_single_segment_time},
            punc_model=settings.punc_model,
            spk_model=settings.spk_model,
            spk_kwargs={"cb_kwargs": {"merge_thr": settings.merge_thr}},
            spk_mode="vad_segment",
            device="cpu",
            hub=settings.hub,
        )


def test_detect_language_from_sensevoice_tokens():
    """测试从 SenseVoice 语言标记检测语言。"""
    assert _detect_language({"text": "<|en|>Hello world"}) == "en"
    assert _detect_language({"text": "<|ja|>こんにちは"}) == "ja"
    assert _detect_language({"text": "<|ko|>안녕하세요"}) == "ko"
    assert _detect_language({"text": "<|yue|>你好嗎"}) == "yue"
    assert _detect_language({"text": "<|zh|>你好"}) == "zh"


def test_detect_language_fallback_chinese_ratio():
    """测试无语言标记时的中文比例启发式。"""
    assert _detect_language({"text": "今天天气很好"}) == "zh"
    assert _detect_language({"text": "The weather is nice"}) == "en"


def test_format_result_with_translation():
    """测试格式化结果时集成翻译。"""
    raw_result = {
        "key": "test_audio",
        "text": "<|en|>Hello world.",
        "sentence_info": [
            {
                "start": 0,
                "end": 2000,
                "sentence": "Hello world.",
                "speaker": "spk0",
            },
        ],
    }

    mock_translator = MagicMock()
    mock_translator.batch_translate.side_effect = lambda segs: [
        {**s, "text_zh": "[zh]" + s["text"]} if s.get("language") != "zh" else {**s, "text_zh": None}
        for s in segs
    ]

    with patch("funasr_server.pipeline.TranscriptionPipeline.__init__", return_value=None):
        pipeline = TranscriptionPipeline.__new__(TranscriptionPipeline)
    pipeline.translator = mock_translator

    result = pipeline.format_result("task-1", raw_result, duration_s=5.0)

    assert result["language"] == "en"
    assert result["segments"][0]["text_zh"] == "[zh]Hello world."
    mock_translator.batch_translate.assert_called_once()
