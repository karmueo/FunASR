"""多语言到中文翻译模块。

基于 NLLB-200 模型，将非中文转写结果自动翻译为简体中文。
"""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# SenseVoice 语言标记 → NLLB-200 FLORES-200 语言代码
# 中文不在此映射中，因为中文不需要翻译
LANG_MAP: Dict[str, str] = {
    "en": "eng_Latn",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "yue": "yue_Hant",
}

# NLLB-200 目标语言：简体中文
TARGET_LANG = "zho_Hans"

# 模块级引用，供测试 mock 替换
_AutoModelForSeq2SeqLM = None
_AutoTokenizer = None


def _get_seq2seq_model_class():
    """延迟导入并返回 AutoModelForSeq2SeqLM 类。"""
    global _AutoModelForSeq2SeqLM
    if _AutoModelForSeq2SeqLM is None:
        from transformers import AutoModelForSeq2SeqLM as _Cls
        _AutoModelForSeq2SeqLM = _Cls
    return _AutoModelForSeq2SeqLM


def _get_tokenizer_class():
    """延迟导入并返回 AutoTokenizer 类。"""
    global _AutoTokenizer
    if _AutoTokenizer is None:
        from transformers import AutoTokenizer as _Cls
        _AutoTokenizer = _Cls
    return _AutoTokenizer


class Translator:
    """多语言到中文翻译器，基于 NLLB-200 模型。

    Attributes:
        model: NLLB-200 序列到序列模型
        tokenizer: NLLB-200 分词器
        device: 推理设备
        max_length: 翻译最大 token 长度
    """

    def __init__(self, model_name: str, device: str, max_length: int = 512) -> None:
        """加载 NLLB-200 模型和分词器。

        Args:
            model_name: HuggingFace 模型名称
            device: 推理设备（cuda/cpu）
            max_length: 翻译最大 token 长度
        """
        self.device = device
        self.max_length = max_length

        logger.info("正在加载翻译模型: %s, device=%s", model_name, device)

        ModelClass = _get_seq2seq_model_class()
        TokenizerClass = _get_tokenizer_class()

        self.tokenizer = TokenizerClass.from_pretrained(model_name)
        self.model = ModelClass.from_pretrained(model_name).to(device)

        logger.info("翻译模型加载完成")

    def translate(self, text: str, source_lang: str) -> str:
        """将文本从指定语言翻译为简体中文。

        Args:
            text: 待翻译文本
            source_lang: SenseVoice 语言标记（en/ja/ko/yue/zh）

        Returns:
            翻译后的中文文本；中文或未知语言返回原文
        """
        # 空文本直接返回
        if not text or not text.strip():
            return ""

        # 中文无需翻译
        if source_lang == "zh":
            return text

        # 查找 NLLB 语言代码
        nllb_lang = LANG_MAP.get(source_lang)
        if nllb_lang is None:
            return text

        # 设置源语言并执行翻译
        self.tokenizer.src_lang = nllb_lang
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True,
                                max_length=self.max_length).to(self.device)

        gen_tokens = self.model.generate(
            **inputs,
            forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(TARGET_LANG),
            max_length=self.max_length,
        )

        result = self.tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)[0]
        return result

    def batch_translate(self, segments: List[Dict]) -> List[Dict]:
        """批量翻译转写片段，按语言分组批量推理。

        为每个非中文片段添加 text_zh 字段。翻译失败时 text_zh 为 None。

        Args:
            segments: 转写片段列表，每个片段包含 text 和 language 字段

        Returns:
            添加了 text_zh 字段的片段列表
        """
        for seg in segments:
            lang = seg.get("language", "zh")
            text = seg.get("text", "")

            # 中文片段无需翻译
            if lang == "zh":
                seg["text_zh"] = None
                continue

            try:
                seg["text_zh"] = self.translate(text, lang)
            except Exception as e:
                logger.warning("翻译失败 [lang=%s]: %s, 原文: %s", lang, e, text[:100])
                seg["text_zh"] = None

        return segments
