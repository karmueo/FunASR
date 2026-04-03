"""多语言到中文翻译模块。

基于 Hunyuan-MT 模型，将非中文转写结果自动翻译为简体中文。
使用 chat template 构造翻译 prompt，通过 causal LM 生成翻译结果。
"""

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

# 翻译 prompt 模板：将任意语言翻译为中文
TRANSLATION_PROMPT = "把下面的文本翻译成中文，不要额外解释。\n\n"

# 模块级引用，供测试 mock 替换
_AutoModelForCausalLM = None
_AutoTokenizer = None


def _get_causal_model_class():
    """延迟导入并返回 AutoModelForCausalLM 类。"""
    global _AutoModelForCausalLM
    if _AutoModelForCausalLM is None:
        from transformers import AutoModelForCausalLM as _Cls
        _AutoModelForCausalLM = _Cls
    return _AutoModelForCausalLM


def _get_tokenizer_class():
    """延迟导入并返回 AutoTokenizer 类。"""
    global _AutoTokenizer
    if _AutoTokenizer is None:
        from transformers import AutoTokenizer as _Cls
        _AutoTokenizer = _Cls
    return _AutoTokenizer


class Translator:
    """多语言到中文翻译器，基于 Hunyuan-MT 模型。

    使用 causal LM + chat template 方式进行翻译，
    支持 38 种语言到中文的高质量翻译。

    Attributes:
        model: Hunyuan-MT causal LM 模型
        tokenizer: Hunyuan-MT 分词器
        device: 推理设备
        max_length: 翻译最大 token 长度
    """

    def __init__(self, model_name: str, device: str, max_length: int = 512) -> None:
        """加载 Hunyuan-MT 模型和分词器。

        Args:
            model_name: HuggingFace 模型名称（如 tencent/Hunyuan-MT-7B-fp8）
            device: 推理设备（cuda/cpu）
            max_length: 翻译最大生成 token 长度
        """
        self.device = device
        self.max_length = max_length

        logger.info("正在加载翻译模型: %s, device=%s", model_name, device)

        ModelClass = _get_causal_model_class()
        TokenizerClass = _get_tokenizer_class()

        self.tokenizer = TokenizerClass.from_pretrained(model_name)
        self.model = ModelClass.from_pretrained(model_name, torch_dtype="auto").to(device)

        logger.info("翻译模型加载完成")

    def translate(self, text: str, source_lang: str) -> str:
        """将文本从指定语言翻译为简体中文。

        Args:
            text: 待翻译文本
            source_lang: SenseVoice 语言标记（en/ja/ko/yue/zh）

        Returns:
            翻译后的中文文本；中文或空文本/未知语言返回原文
        """
        # 空文本直接返回
        if not text or not text.strip():
            return ""

        # 中文无需翻译
        if source_lang == "zh":
            return text

        # 构造 chat template 消息
        messages = [
            {"role": "user", "content": TRANSLATION_PROMPT + text},
        ]

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt",
        ).to(self.device)

        # 推荐推理参数
        gen_tokens = self.model.generate(
            input_ids,
            max_new_tokens=self.max_length,
            top_k=20,
            top_p=0.6,
            temperature=0.7,
            repetition_penalty=1.05,
        )

        # 仅解码新增输出，避免把输入 prompt 和原文一并返回。
        generated_tokens = gen_tokens[0][input_ids.shape[-1]:]
        result = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return result

    def batch_translate(self, segments: List[Dict]) -> List[Dict]:
        """批量翻译转写片段。

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
