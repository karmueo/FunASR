"""转写 Pipeline 封装。

封装 FunASR AutoModel，提供初始化、音频格式验证、
转写执行和结果格式化功能。

注意：funasr 的导入在方法内部延迟执行，
使测试可在不安装完整 FunASR 依赖的情况下运行。
"""

import logging
import os
from typing import Dict, List, Optional

from funasr_server.config import Settings

logger = logging.getLogger(__name__)

# 模块级引用，供测试 mock 替换
AutoModel = None


def _get_auto_model_class():
    """延迟导入并返回 AutoModel 类。

    Returns:
        AutoModel 类引用
    """
    global AutoModel
    if AutoModel is None:
        from funasr import AutoModel as _AutoModel

        AutoModel = _AutoModel
    return AutoModel


def _get_rich_transcription_postprocess():
    """延迟导入并返回 rich_transcription_postprocess 函数。

    Returns:
        rich_transcription_postprocess 函数引用
    """
    from funasr.utils.postprocess_utils import rich_transcription_postprocess

    return rich_transcription_postprocess


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
            settings.asr_model,
            settings.vad_model,
            settings.punc_model,
            settings.spk_model,
            settings.device,
        )

        AutoModelClass = _get_auto_model_class()
        self.model = AutoModelClass(
            model=settings.asr_model,
            vad_model=settings.vad_model,
            vad_kwargs={"max_single_segment_time": settings.max_single_segment_time},
            punc_model=settings.punc_model,
            spk_model=settings.spk_model,
            spk_kwargs={"cb_kwargs": {"merge_thr": settings.merge_thr}},
            spk_mode="vad_segment",  # 使用 VAD 段边界做说话人分配，不依赖 ASR 时间戳
            device=settings.device,
            hub=settings.hub,
        )

        # 初始化翻译器（翻译功能启用时）
        self.translator = None
        if settings.enable_translation:
            try:
                from funasr_server.translator import Translator
                self.translator = Translator(
                    model_name=settings.translation_model,
                    device=settings.device_translation,
                    max_length=settings.translation_max_length,
                )
            except Exception as e:
                logger.warning("翻译模型加载失败，翻译功能不可用: %s", e)

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
        # 构建生成参数
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
            "output_timestamp": True,  # 说话人分离依赖时间戳
        }

        # 指定说话人数量时启用预设模式
        if speaker_num is not None:
            generate_kwargs["preset_spk_num"] = speaker_num

        logger.info(
            "开始转写: %s, language=%s, speaker_num=%s",
            audio_path,
            language,
            speaker_num,
        )

        results = self.model.generate(**generate_kwargs)

        # 对 SenseVoice 输出进行后处理，移除特殊标记
        if results and len(results) > 0:
            raw_text = results[0].get("text", "")
            postprocess = _get_rich_transcription_postprocess()
            results[0]["text"] = postprocess(raw_text)

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
        detected_lang = _detect_language(raw_result)
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
                    "language": detected_lang,
                })

        # 翻译非中文片段
        if self.translator and segments:
            segments = self.translator.batch_translate(segments)

        return {
            "task_id": task_id,
            "language": detected_lang,
            "duration_s": duration_s,
            "segments": segments,
            "speakers": sorted(speakers_set),
        }


def _detect_language(raw_result: Dict) -> str:
    """从转写结果中推断语言标识。

    优先从 SenseVoice 语言标记检测，回退到中文字符比例启发式。

    Args:
        raw_result: AutoModel 原始输出

    Returns:
        语言代码（zh/en/ja/ko/yue）
    """
    text = raw_result.get("text", "")
    # SenseVoice 语言标记检测
    for lang_tag in ("zh", "en", "ja", "ko", "yue"):
        if f"<|{lang_tag}|>" in text:
            return lang_tag
    # 回退：中文字符比例启发式
    chinese_count = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    total = max(len(text.replace(" ", "")), 1)
    return "zh" if chinese_count / total > 0.3 else "en"
