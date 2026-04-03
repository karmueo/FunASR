"""Translator 模块测试。"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from funasr_server.translator import Translator, TRANSLATION_PROMPT


def test_translation_prompt_template():
    """测试翻译 prompt 模板包含关键指令。"""
    assert "中文" in TRANSLATION_PROMPT
    assert "不要额外解释" in TRANSLATION_PROMPT


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


def test_translate_calls_model_for_non_chinese():
    """测试非中文文本调用模型进行翻译。"""
    with patch("funasr_server.translator.Translator.__init__", return_value=None):
        translator = Translator.__new__(Translator)
    translator.device = "cpu"
    translator.max_length = 512

    # mock tokenizer 和 model
    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template.return_value = MagicMock(
        to=MagicMock(return_value=MagicMock())
    )
    mock_tokenizer.decode.return_value = "你好世界"
    translator.tokenizer = mock_tokenizer

    mock_model = MagicMock()
    mock_model.generate.return_value = [MagicMock()]
    mock_model.device = "cpu"
    translator.model = mock_model

    result = translator.translate("Hello world", "en")

    assert result == "你好世界"
    mock_tokenizer.apply_chat_template.assert_called_once()
    mock_model.generate.assert_called_once()


def test_translate_decodes_only_new_tokens():
    """测试翻译结果只解码新增 token，不包含输入 prompt。"""
    with patch("funasr_server.translator.Translator.__init__", return_value=None):
        translator = Translator.__new__(Translator)
    translator.device = "cpu"
    translator.max_length = 64

    mock_input_ids = MagicMock()
    mock_input_ids.shape = (1, 5)
    mock_input_ids.to.return_value = mock_input_ids

    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template.return_value = mock_input_ids
    mock_tokenizer.decode.return_value = "你好，世界。"
    translator.tokenizer = mock_tokenizer

    mock_model = MagicMock()
    mock_model.generate.return_value = [[101, 102, 103, 104, 105, 201, 202]]
    translator.model = mock_model

    result = translator.translate("Hello world", "en")

    assert result == "你好，世界。"
    mock_tokenizer.decode.assert_called_once_with([201, 202], skip_special_tokens=True)


def test_batch_translate_adds_text_zh():
    """测试批量翻译为非中文片段添加 text_zh 字段。"""
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
