"""配置模块测试。"""

import os
import pytest
from funasr_server.config import Settings


def test_default_settings():
    """测试默认配置值。"""
    s = Settings()
    assert s.device == "cuda"
    assert "SenseVoiceSmall" in s.asr_model
    assert s.vad_model == "fsmn-vad"
    assert s.punc_model == "ct-punc"
    assert s.spk_model == "cam++"
    assert s.max_single_segment_time == 60000
    assert s.batch_size_s == 300
    assert s.merge_thr == 0.5
    assert s.host == "0.0.0.0"
    assert s.port == 8000
    assert s.max_audio_duration_s == 7200
    assert s.enable_translation is True
    assert "hunyuan-mt-7b" in s.translation_model
    assert s.translation_max_length == 512


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


def test_translation_env_override(monkeypatch):
    """测试翻译配置的环境变量覆盖。"""
    monkeypatch.setenv("FUNASR_ENABLE_TRANSLATION", "false")
    monkeypatch.setenv("FUNASR_TRANSLATION_MODEL", "custom/model")
    monkeypatch.setenv("FUNASR_TRANSLATION_MAX_LENGTH", "256")
    s = Settings()
    assert s.enable_translation is False
    assert s.translation_model == "custom/model"
    assert s.translation_max_length == 256
