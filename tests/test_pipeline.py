"""Pipeline 模块测试。"""

import os

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


def test_format_result_accepts_funasr_spk_field():
    """测试格式化结果时兼容 FunASR 原生的 spk 字段。"""
    raw_result = {
        "key": "test_audio",
        "text": "你好。再见。",
        "sentence_info": [
            {
                "start": 0,
                "end": 1000,
                "sentence": "你好。",
                "spk": 0,
            },
            {
                "start": 1000,
                "end": 2000,
                "sentence": "再见。",
                "spk": 1,
            },
        ],
    }

    with patch("funasr_server.pipeline.TranscriptionPipeline.__init__", return_value=None):
        pipeline = TranscriptionPipeline.__new__(TranscriptionPipeline)
    pipeline.translator = None

    result = pipeline.format_result("task-spk", raw_result, duration_s=2.0)

    assert result["segments"][0]["speaker"] == "spk0"
    assert result["segments"][1]["speaker"] == "spk1"
    assert result["speakers"] == ["spk0", "spk1"]


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


def test_pipeline_init_calls_auto_model_on_cpu(monkeypatch):
    """测试 CPU 模式初始化时不应强制切换到 CUDA。"""
    mock_auto_model_class = MagicMock()
    # 保留一个原始环境变量，验证 CPU 路径不会改写它。
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5")
    with patch("funasr_server.pipeline._get_auto_model_class", return_value=mock_auto_model_class):
        from funasr_server.config import Settings
        settings = Settings(device="cpu", enable_translation=False)

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
        assert pipeline.translator is None
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "4,5"


def test_pipeline_init_isolates_cuda_and_restores_env(monkeypatch):
    """测试 CUDA 模式初始化时会临时隔离 GPU 并在结束后恢复环境变量。"""
    # 模拟一个多卡环境，验证初始化期间只暴露指定 ASR 卡。
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5")
    # 记录初始化时看到的环境变量，确认 AutoModel 在隔离后的环境里被加载。
    seen_cuda_visible = {}

    def _capture_auto_model(**kwargs):
        """捕获 AutoModel 初始化参数和环境变量。"""
        seen_cuda_visible["value"] = os.environ.get("CUDA_VISIBLE_DEVICES")
        return MagicMock()

    with patch("funasr_server.pipeline._get_auto_model_class", return_value=_capture_auto_model):
        from funasr_server.config import Settings
        settings = Settings(device="cuda:3", enable_translation=False)

        TranscriptionPipeline(settings)

    assert seen_cuda_visible["value"] == "3"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "4,5"


def test_pipeline_init_skips_cuda_isolation_for_cross_gpu_translation(monkeypatch):
    """测试跨 GPU 翻译场景不应通过环境变量隔离 ASR 显卡。"""
    seen_cuda_visible = {}

    def _capture_auto_model(**kwargs):
        """捕获 AutoModel 初始化时看到的可见 GPU 列表。"""
        seen_cuda_visible["value"] = os.environ.get("CUDA_VISIBLE_DEVICES")
        return MagicMock()

    mock_translator_class = MagicMock(return_value=MagicMock())
    # 模拟当前进程在多卡环境中运行，验证不会改写可见卡列表。
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")

    with patch("funasr_server.pipeline._get_auto_model_class", return_value=_capture_auto_model):
        with patch("funasr_server.translator.Translator", mock_translator_class):
            from funasr_server.config import Settings
            settings = Settings(
                device="cuda:0",
                device_translation="cuda:1",
                enable_translation=True,
            )

            pipeline = TranscriptionPipeline(settings)

    mock_translator_class.assert_called_once_with(
        model_name=settings.translation_model,
        device="cuda:1",
        max_length=settings.translation_max_length,
    )
    assert pipeline.translator is not None
    assert seen_cuda_visible["value"] == "0,1,2,3"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"


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


def test_detect_language_from_unicode_script() -> None:
    """测试无语言标记时可根据文字脚本识别日文和韩文。"""
    assert _detect_language({"text": "今日は晴れです"}) == "ja"
    assert _detect_language({"text": "안녕하세요"}) == "ko"


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


def test_format_result_with_japanese_text_triggers_translation():
    """测试日文文本在无语言标记时仍会触发中文翻译。"""
    raw_result = {
        "key": "test_audio",
        "text": "今日は晴れです。",
        "sentence_info": [
            {
                "start": 0,
                "end": 2000,
                "sentence": "今日は晴れです。",
                "speaker": "spk0",
            },
        ],
    }

    mock_translator = MagicMock()
    mock_translator.batch_translate.side_effect = lambda segs: [
        {**segment, "text_zh": "[zh]" + segment["text"]}
        for segment in segs
    ]

    with patch("funasr_server.pipeline.TranscriptionPipeline.__init__", return_value=None):
        pipeline = TranscriptionPipeline.__new__(TranscriptionPipeline)
    pipeline.translator = mock_translator

    result = pipeline.format_result("task-ja", raw_result, duration_s=5.0)

    assert result["language"] == "ja"
    assert result["segments"][0]["language"] == "ja"
    assert result["segments"][0]["text_zh"] == "[zh]今日は晴れです。"
    mock_translator.batch_translate.assert_called_once()
