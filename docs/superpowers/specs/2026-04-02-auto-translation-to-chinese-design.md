# 自动翻译为中文功能设计

## 概述

为 FunASR 转写服务添加自动翻译功能：所有非中文的转写结果自动翻译为中文。ASR 模型从 Paraformer（仅中英文）切换为 SenseVoiceSmall（中/粤/英/日/韩 5 种语言），翻译使用 NLLB-200-distilled-600M 本地模型。

## 方案选择

**选定方案 A**：SenseVoice(5 语言) + NLLB-200 翻译

| 维度 | 决策 |
|---|---|
| ASR 模型 | SenseVoiceSmall，支持中文/粤语/英文/日文/韩文 |
| 翻译模型 | facebook/nllb-200-distilled-600M，支持 200 种翻译方向 |
| 触发方式 | 自动：非中文片段自动翻译，无需用户操作 |
| 部署方式 | 本地推理，不依赖外部 API |

## 架构

### 流水线

```
Audio → VAD(fsmn-vad) → SenseVoice ASR → ct-punc → cam++ → 语言检测 → [非中文? NLLB翻译] → 输出
```

### 新增模块

**`funasr_server/translator.py`**：封装 NLLB-200 模型加载、语言映射、批量翻译逻辑。

### 修改模块

| 文件 | 改动 |
|---|---|
| `config.py` | ASR 模型改为 SenseVoiceSmall；新增翻译相关配置字段 |
| `pipeline.py` | ASR 替换为 SenseVoice；语言检测改用 SenseVoice 语言标记；集成翻译步骤 |
| `api.py` | 响应结构扩展 `text_zh` 和 `language` 字段 |
| `gradio_frontend.py` | 结果表格增加翻译列和语言列 |
| `requirements.txt` | 新增 `transformers`、`sentencepiece` 依赖 |

### 不改动

- VAD、标点恢复、说话人分离模块不变
- FastAPI/Gradio 的请求路由不变

## 详细设计

### translator.py 核心接口

```python
class Translator:
    """多语言到中文翻译器，基于 NLLB-200 模型"""

    def __init__(self, model_name: str, device: str):
        """加载 NLLB-200 模型和 tokenizer，目标语言固定为 zho_Hans"""

    def translate(self, text: str, source_lang: str) -> str:
        """将文本从指定语言翻译为中文"""

    def batch_translate(self, segments: List[Dict]) -> List[Dict]:
        """批量翻译转写片段，按语言分组批量推理，添加 text_zh 字段"""
```

### 语言映射

SenseVoice 语言标记 → NLLB-200 FLORES-200 代码：

| SenseVoice | NLLB-200 代码 | 语言 |
|---|---|---|
| `zh` | — | 中文，跳过翻译 |
| `yue` | `yue_Hant` | 粤语 |
| `en` | `eng_Latn` | 英文 |
| `ja` | `jpn_Jpan` | 日文 |
| `ko` | `kor_Hang` | 韩文 |

### 配置新增字段

```python
# config.py
asr_model: str = "iic/SenseVoiceSmall"
enable_translation: bool = True
translation_model: str = "facebook/nllb-200-distilled-600M"
translation_max_length: int = 512
```

### API 响应结构

```json
{
    "segments": [
        {
            "speaker": "spk0",
            "start_time": 0.5,
            "end_time": 3.2,
            "text": "Hello, how are you?",
            "text_zh": "你好，你好吗？",
            "language": "en"
        },
        {
            "speaker": "spk1",
            "start_time": 3.5,
            "end_time": 5.0,
            "text": "这是中文",
            "text_zh": null,
            "language": "zh"
        }
    ]
}
```

`text_zh` 和 `language` 为新增字段，向后兼容。

### 错误处理

- 翻译失败时 `text_zh` 返回 `null`，不影响原始转写结果
- 记录警告日志，包含失败原因和原文片段
- 常见失败场景：文本为空、语言无法映射、模型推理超时

### 性能策略

- 翻译模型在服务启动时预加载，与 ASR 模型一起初始化
- 按语言分组批量翻译：同一语言的片段合并处理，减少推理次数
- 长文本分片：超过 `translation_max_length` 时按句子边界分片翻译后拼接
- 可通过 `FUNASR_ENABLE_TRANSLATION=false` 禁用翻译功能
- 预估 GPU 显存增量：约 1.5-3GB（含 SenseVoice 和 NLLB-200）

### 混合语言场景

SenseVoice 输出中可能包含语言标记，逐段检测语言，每段独立决定是否翻译。

### Gradio 界面变更

结果表格增加"翻译"列和"语言"列，非中文行显示翻译内容，中文行显示"—"。

### 依赖新增

```
transformers
sentencepiece
```

## 测试计划

1. **单元测试**：Translator 类的 translate/batch_translate 方法，覆盖各语言映射
2. **集成测试**：端到端音频转写 + 翻译，验证响应结构
3. **边界测试**：中文输入（跳过翻译）、空文本、混合语言、翻译失败场景
4. **API 测试**：验证响应字段向后兼容
