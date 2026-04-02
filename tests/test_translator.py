"""Translator 模块测试。"""

import pytest
from unittest.mock import MagicMock, patch
from funasr_server.translator import Translator, LANG_MAP


def test_lang_map_contains_all_sensevoice_languages():
    """测试语言映射包含 SenseVoice 支持的所有非中文语言。"""
    for lang in ("en", "ja", "ko", "yue"):
        assert lang in LANG_MAP, f"缺少 {lang} 的 NLLB 语言映射"


def test_lang_map_excludes_chinese():
    """测试语言映射不包含中文（中文无需翻译）。"""
    assert "zh" not in LANG_MAP


def test_translate_skips_empty_text():
    """测试空文本返回空字符串。"""
    with patch("funasr_server.translator.Translator.__init__", return_value=None):
        translator = Translator.__new__(Translator)
    assert translator.translate("", "en") == ""
    assert translator.translate("   ", "en") == ""


def test_translate_skips_chinese():
    """测试中文文本不翻译，直接返回原文。"""
    with patch("funasr_server.translator.Translator.__init__", return_value=None):
        translator = Translator.__new__(Translator)
    assert translator.translate("这是中文", "zh") == "这是中文"


def test_translate_skips_unknown_language():
    """测试未知语言不翻译，直接返回原文。"""
    with patch("funasr_server.translator.Translator.__init__", return_value=None):
        translator = Translator.__new__(Translator)
    assert translator.translate("hello", "unknown") == "hello"


def test_batch_translate_groups_by_language():
    """测试批量翻译按语言分组。"""
    segments = [
        {"text": "Hello", "language": "en"},
        {"text": "こんにちは", "language": "ja"},
        {"text": "你好", "language": "zh"},
        {"text": "World", "language": "en"},
    ]

    mock_translate = MagicMock(side_effect=lambda t, l: {
        "en": f"[zh]{t}",
        "ja": f"[zh]{t}",
    }.get(l, t))

    with patch("funasr_server.translator.Translator.__init__", return_value=None):
        translator = Translator.__new__(Translator)
    translator.translate = mock_translate

    result = translator.batch_translate(segments)

    # 中文片段不翻译
    assert result[2]["text_zh"] is None
    # 英文和日文片段被翻译
    assert result[0]["text_zh"] == "[zh]Hello"
    assert result[1]["text_zh"] == "[zh]こんにちは"
    assert result[3]["text_zh"] == "[zh]World"


def test_batch_translate_handles_translation_failure():
    """测试翻译失败时 text_zh 为 None，不影响其他片段。"""
    call_count = 0

    def mock_translate_fn(text, lang):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("翻译模型出错")
        return f"[zh]{text}"

    segments = [
        {"text": "Hello", "language": "en"},
        {"text": "World", "language": "en"},
    ]

    with patch("funasr_server.translator.Translator.__init__", return_value=None):
        translator = Translator.__new__(Translator)
    translator.translate = mock_translate_fn

    result = translator.batch_translate(segments)

    # 第一个片段翻译失败
    assert result[0]["text_zh"] is None
    # 第二个片段正常翻译
    assert result[1]["text_zh"] == "[zh]World"
