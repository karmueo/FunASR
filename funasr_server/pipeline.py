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
# 记录 CUDA_VISIBLE_DEVICES 的历史值，保证阶段性切换后可恢复。
_CUDA_VISIBLE_DEVICES_STACK: List[Optional[str]] = []


def _extract_cuda_index(device: str) -> Optional[str]:
    """从设备字符串中提取 CUDA 卡号。

    Args:
        device: 设备字符串，例如 ``cuda:0``、``cuda``、``cpu``

    Returns:
        如果是 CUDA 设备则返回卡号字符串，否则返回 None
    """
    if not device.startswith("cuda"):
        return None
    if ":" not in device:
        return "0"
    return device.split(":", 1)[1]


def _set_cuda_visible(gpu_index: str) -> None:
    """临时设置当前进程可见的 CUDA 设备。

    Args:
        gpu_index: 需要暴露的 GPU 索引字符串
    """
    # 备份进入隔离前的环境变量，供后续恢复。
    previous_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    _CUDA_VISIBLE_DEVICES_STACK.append(previous_cuda_visible)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_index


def _restore_cuda_visible() -> None:
    """恢复最近一次隔离前的 CUDA_VISIBLE_DEVICES。"""
    if not _CUDA_VISIBLE_DEVICES_STACK:
        return

    # 取出最近一次隔离前的环境变量值。
    previous_cuda_visible = _CUDA_VISIBLE_DEVICES_STACK.pop()
    if previous_cuda_visible is None:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = previous_cuda_visible


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

        通过 CUDA_VISIBLE_DEVICES 分阶段隔离 GPU，
        防止 FunASR AutoModel 将子模型泄漏到翻译模型所在 GPU。

        Args:
            settings: 服务配置实例
        """
        self.settings = settings
        self.supported_extensions = settings.supported_extensions

        # 从 device 字符串提取 GPU 索引；非 CUDA 设备返回 None。
        asr_gpu = _extract_cuda_index(settings.device)
        # 提取翻译模型使用的 GPU 索引，用于判断是否跨卡部署。
        translation_gpu = _extract_cuda_index(settings.device_translation)
        # 跨 GPU 部署时不能在同一进程内用 CUDA_VISIBLE_DEVICES 缩卡，
        # 否则 ASR 初始化后，翻译阶段可能无法再访问其他 GPU。
        use_cross_gpu_translation = (
            settings.enable_translation
            and asr_gpu is not None
            and translation_gpu is not None
            and translation_gpu != asr_gpu
        )
        # 仅单 GPU 场景允许通过环境变量做显卡隔离。
        use_cuda_isolation = asr_gpu is not None and not use_cross_gpu_translation
        # AutoModel 真正使用的 device；隔离后单卡环境统一映射为 cuda:0。
        asr_model_device = "cuda:0" if use_cuda_isolation else settings.device

        # ---- 阶段 1：必要时仅暴露 ASR 所在 GPU，加载 ASR 全套模型 ----
        logger.info(
            "正在加载 ASR 模型: ASR=%s, VAD=%s, PUNC=%s, SPK=%s, device=%s",
            settings.asr_model, settings.vad_model,
            settings.punc_model, settings.spk_model, settings.device,
        )
        if use_cross_gpu_translation:
            logger.info(
                "检测到跨 GPU 翻译场景，跳过 CUDA_VISIBLE_DEVICES 隔离: asr=%s, translation=%s",
                settings.device,
                settings.device_translation,
            )
        AutoModelClass = _get_auto_model_class()
        if use_cuda_isolation:
            _set_cuda_visible(asr_gpu)
        try:
            self.model = AutoModelClass(
                model=settings.asr_model,
                vad_model=settings.vad_model,
                vad_kwargs={"max_single_segment_time": settings.max_single_segment_time},
                punc_model=settings.punc_model,
                spk_model=settings.spk_model,
                spk_kwargs={"cb_kwargs": {"merge_thr": settings.merge_thr}},
                spk_mode="vad_segment",
                device=asr_model_device,
                hub=settings.hub,
            )
        finally:
            if use_cuda_isolation:
                _restore_cuda_visible()

        # ---- 阶段 2：恢复全部 GPU 可见，加载翻译模型到指定 GPU ----
        self.translator = None
        if settings.enable_translation:
            try:
                from funasr_server.translator import Translator
                logger.info(
                    "正在加载翻译模型: %s, device=%s",
                    settings.translation_model,
                    settings.device_translation,
                )
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
                # 兼容 FunASR 的 spk 字段和服务内部的 speaker 字段。
                speaker = _normalize_speaker_label(sent)
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
    # 提取待检测文本，优先使用主结果文本。
    text = raw_result.get("text", "")
    # SenseVoice 语言标记检测
    for lang_tag in ("zh", "en", "ja", "ko", "yue"):
        if f"<|{lang_tag}|>" in text:
            return lang_tag
    # 日文假名命中时优先判定为日文，避免被汉字比例误伤。
    if any("\u3040" <= char <= "\u30ff" for char in text):
        return "ja"
    # 韩文 Hangul 命中时直接判定为韩文。
    if any("\uac00" <= char <= "\ud7af" for char in text):
        return "ko"
    # 回退：中文字符比例启发式
    # 统计 CJK 统一汉字数量，用于区分中文与英文等语言。
    chinese_count = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
    # 统计非空白字符总数，避免分母为 0。
    total = max(len(text.replace(" ", "")), 1)
    return "zh" if chinese_count / total > 0.3 else "en"


def _normalize_speaker_label(sentence_info: Dict) -> str:
    """将不同来源的说话人字段统一为前端标签格式。

    Args:
        sentence_info: 单句转写结果，可能包含 ``speaker`` 或 ``spk`` 字段

    Returns:
        标准化后的说话人标签，例如 ``spk0``、``spk1``
    """
    # 优先兼容当前服务内部字段命名。
    speaker_value = sentence_info.get("speaker")
    if speaker_value is None:
        # FunASR 原生输出使用 spk 整数标签。
        speaker_value = sentence_info.get("spk", 0)

    if isinstance(speaker_value, str):
        if speaker_value.startswith("spk"):
            return speaker_value
        return f"spk{speaker_value}"

    return f"spk{int(speaker_value)}"
