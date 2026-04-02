"""转写服务配置管理。

从环境变量读取配置，提供合理的默认值。
"""

import os
from dataclasses import dataclass, field
from typing import List


# 支持的音频文件扩展名
SUPPORTED_EXTENSIONS: List[str] = [".wav", ".mp3", ".flac", ".ogg", ".m4a"]


@dataclass
class Settings:
    """转写服务全局配置。

    所有字段均可通过 FUNASR_ 前缀的环境变量覆盖，
    例如 FUNASR_DEVICE=cpu 覆盖 device 默认值。
    """

    # 模型配置
    asr_model: str = "iic/SenseVoiceSmall"
    vad_model: str = "fsmn-vad"
    punc_model: str = "ct-punc"
    spk_model: str = "cam++"
    device: str = "cuda:0"       # ASR 流水线设备（VAD+ASR+Punc+Spk）
    device_translation: str = "cuda:1"  # 翻译模型设备
    hub: str = "ms"

    # VAD 参数
    max_single_segment_time: int = 60000  # 毫秒

    # ASR 参数
    batch_size_s: int = 300  # 动态批大小（秒）
    batch_size_threshold_s: int = 60  # 单段最大时长（秒）

    # 说话人聚类参数
    merge_thr: float = 0.5

    # 翻译配置
    enable_translation: bool = True
    translation_model: str = "/home/tl/models/hunyuan-mt-7b"
    translation_max_length: int = 512

    # 服务配置
    host: str = "0.0.0.0"
    port: int = 8000
    max_audio_duration_s: int = 7200  # 最大音频时长（秒），默认2小时
    request_timeout_s: int = 600  # 请求超时（秒）

    # 支持的音频格式
    supported_extensions: List[str] = field(
        default_factory=lambda: list(SUPPORTED_EXTENSIONS)
    )

    def __post_init__(self) -> None:
        """构造后自动从环境变量覆盖同名字段。

        环境变量名规则：FUNASR_<大写字段名>，例如 FUNASR_DEVICE。
        类型自动按字段注解转换（int / float / str）。
        """
        for f_name, f_info in self.__dataclass_fields__.items():
            env_key = f"FUNASR_{f_name.upper()}"
            env_val = os.environ.get(env_key)
            if env_val is None:
                continue
            # 仅处理简单类型，跳过 List 等复杂类型
            f_type = f_info.type
            if f_type is int:
                setattr(self, f_name, int(env_val))
            elif f_type is float:
                setattr(self, f_name, float(env_val))
            elif f_type is bool:
                setattr(self, f_name, env_val.lower() in ("1", "true", "yes"))
            elif f_type is str:
                setattr(self, f_name, env_val)

    @classmethod
    def from_env(cls) -> "Settings":
        """从环境变量构建配置，FUNASR_ 前缀。

        环境变量名规则：FUNASR_<大写字段名>，例如 FUNASR_DEVICE。
        """
        return cls()
