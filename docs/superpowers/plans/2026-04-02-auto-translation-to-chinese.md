# 自动翻译为中文功能实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 FunASR 转写服务添加自动翻译功能——ASR 切换为 SenseVoiceSmall，非中文转写结果自动翻译为中文。

**Architecture:** 流水线变为 VAD → SenseVoice ASR → 标点 → 说话人分离 → 语言检测 → [NLLB-200 翻译] → 输出。新增 `translator.py` 模块封装 NLLB-200 翻译逻辑，pipeline 在格式化结果时调用翻译。

**Tech Stack:** FunASR SenseVoiceSmall、HuggingFace transformers (NLLB-200-distilled-600M)、sentencepiece

---

## 文件结构

| 操作 | 文件 | 职责 |
|---|---|---|
| 创建 | `funasr_server/translator.py` | NLLB-200 翻译器封装 |
| 创建 | `tests/test_translator.py` | 翻译器单元测试 |
| 修改 | `funasr_server/config.py` | 新增翻译配置、切换 ASR 模型 |
| 修改 | `funasr_server/pipeline.py` | ASR 替换、语言检测增强、集成翻译 |
| 修改 | `funasr_server/api.py` | 响应结构扩展 |
| 修改 | `funasr_server/gradio_frontend.py` | UI 增加翻译列和语言列 |
| 修改 | `funasr_server/app.py` | 启动时初始化翻译器 |
| 修改 | `funasr_server/requirements.txt` | 新增依赖 |
| 修改 | `tests/test_config.py` | 更新配置断言 |
| 修改 | `tests/test_pipeline.py` | 更新 pipeline 测试 |
| 修改 | `tests/test_api.py` | 更新 API 响应测试 |

---

### Task 1: 配置变更

**Files:**
- Modify: `funasr_server/config.py`
- Modify: `tests/test_config.py`

- [ ] **Step 1: 更新 config.py**

在 `Settings` dataclass 中：
1. 将 `asr_model` 默认值改为 `"iic/SenseVoiceSmall"`
2. 在"说话人聚类参数"区块后新增"翻译配置"区块，包含三个字段：

```python
    # 翻译配置
    enable_translation: bool = True
    translation_model: str = "facebook/nllb-200-distilled-600M"
    translation_max_length: int = 512
```

注意：`__post_init__` 中需要处理 `bool` 类型。在 `elif f_type is str:` 之前添加：

```python
            elif f_type is bool:
                setattr(self, f_name, env_val.lower() in ("1", "true", "yes"))
```

- [ ] **Step 2: 更新 tests/test_config.py**

修改 `test_default_settings`：
1. 将 `assert "seaco_paraformer" in s.asr_model` 改为 `assert "SenseVoiceSmall" in s.asr_model`
2. 新增断言验证翻译默认值：

```python
    assert s.enable_translation is True
    assert "nllb-200" in s.translation_model
    assert s.translation_max_length == 512
```

新增测试函数验证翻译配置的环境变量覆盖：

```python
def test_translation_env_override(monkeypatch):
    """测试翻译配置的环境变量覆盖。"""
    monkeypatch.setenv("FUNASR_ENABLE_TRANSLATION", "false")
    monkeypatch.setenv("FUNASR_TRANSLATION_MODEL", "custom/model")
    monkeypatch.setenv("FUNASR_TRANSLATION_MAX_LENGTH", "256")
    s = Settings()
    assert s.enable_translation is False
    assert s.translation_model == "custom/model"
    assert s.translation_max_length == 256
```

- [ ] **Step 3: 运行测试**

Run: `cd /home/tl/work/FunASR && python -m pytest tests/test_config.py -v`
Expected: 全部通过

- [ ] **Step 4: 提交**

```bash
git add funasr_server/config.py tests/test_config.py
git commit -m "feat: add translation config and switch ASR to SenseVoiceSmall"
```

---

### Task 2: 翻译器模块

**Files:**
- Create: `funasr_server/translator.py`
- Create: `tests/test_translator.py`

- [ ] **Step 1: 编写翻译器失败测试**

创建 `tests/test_translator.py`：

```python
"""Translator 模块测试。"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd /home/tl/work/FunASR && python -m pytest tests/test_translator.py -v`
Expected: FAIL（ModuleNotFoundError: funasr_server.translator）

- [ ] **Step 3: 实现翻译器模块**

创建 `funasr_server/translator.py`：

```python
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
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd /home/tl/work/FunASR && python -m pytest tests/test_translator.py -v`
Expected: 全部通过

- [ ] **Step 5: 提交**

```bash
git add funasr_server/translator.py tests/test_translator.py
git commit -m "feat: add Translator module with NLLB-200 integration"
```

---

### Task 3: Pipeline 变更

**Files:**
- Modify: `funasr_server/pipeline.py`
- Modify: `tests/test_pipeline.py`

- [ ] **Step 1: 更新 pipeline 测试**

在 `tests/test_pipeline.py` 中：

1. 修改 `test_pipeline_init_calls_auto_model` 断言中的 `settings.asr_model` —— 因为 config 默认值已变，此测试应自动适配，无需改动。

2. 新增语言检测增强测试：

```python
def test_detect_language_from_sensevoice_tokens():
    """测试从 SenseVoice 语言标记检测语言。"""
    from funasr_server.pipeline import _detect_language
    assert _detect_language({"text": "<|en|>Hello world"}) == "en"
    assert _detect_language({"text": "<|ja|>こんにちは"}) == "ja"
    assert _detect_language({"text": "<|ko|>안녕하세요"}) == "ko"
    assert _detect_language({"text": "<|yue|>你好嗎"}) == "yue"
    assert _detect_language({"text": "<|zh|>你好"}) == "zh"


def test_detect_language_fallback_chinese_ratio():
    """测试无语言标记时的中文比例启发式。"""
    from funasr_server.pipeline import _detect_language
    assert _detect_language({"text": "今天天气很好"}) == "zh"
    assert _detect_language({"text": "The weather is nice"}) == "en"


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
```

3. 更新 `test_format_result_structure`——现有断言中 segment 不含 `text_zh`，需确认兼容：现有测试中 pipeline 没有 translator 属性，需在无翻译场景下 `text_zh` 为 None。

在 `test_format_result_structure` 的 pipeline mock 构建后添加：

```python
    pipeline.translator = None
```

并添加断言：

```python
    # 无翻译器时 text_zh 为 None
    assert result["segments"][0].get("text_zh") is None
```

同理更新 `test_format_result_empty_segments`：

```python
    pipeline.translator = None
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd /home/tl/work/FunASR && python -m pytest tests/test_pipeline.py -v`
Expected: 新测试 FAIL（_detect_language 尚未支持 ja/ko/yue，format_result 尚未集成翻译）

- [ ] **Step 3: 更新 pipeline.py**

1. 在文件顶部导入区新增：

```python
from funasr_server.translator import Translator
```

2. 修改 `TranscriptionPipeline.__init__`：在 `logger.info("模型加载完成")` 之前，新增翻译器初始化：

```python
        # 初始化翻译器（翻译功能启用时）
        self.translator = None
        if settings.enable_translation:
            try:
                self.translator = Translator(
                    model_name=settings.translation_model,
                    device=settings.device,
                    max_length=settings.translation_max_length,
                )
            except Exception as e:
                logger.warning("翻译模型加载失败，翻译功能不可用: %s", e)
```

3. 修改 `_detect_language` 函数，支持所有 SenseVoice 语言标记：

```python
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
```

4. 修改 `format_result` 方法，在构建 segments 列表后调用翻译器：

将 `format_result` 方法改为：

```python
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
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd /home/tl/work/FunASR && python -m pytest tests/test_pipeline.py -v`
Expected: 全部通过

- [ ] **Step 5: 运行所有已有测试确认无回归**

Run: `cd /home/tl/work/FunASR && python -m pytest tests/ -v`
Expected: 全部通过

- [ ] **Step 6: 提交**

```bash
git add funasr_server/pipeline.py tests/test_pipeline.py
git commit -m "feat: integrate SenseVoice ASR and NLLB translation into pipeline"
```

---

### Task 4: API 响应变更

**Files:**
- Modify: `funasr_server/api.py`
- Modify: `tests/test_api.py`

- [ ] **Step 1: 更新 API 测试**

在 `tests/test_api.py` 中，更新 `mock_pipeline` fixture 的 `format_result.return_value`：

```python
    pipeline.format_result.return_value = {
        "task_id": "test-uuid",
        "language": "en",
        "duration_s": 10.0,
        "segments": [
            {
                "speaker": "spk0",
                "start_ms": 0,
                "end_ms": 2000,
                "text": "Hello world.",
                "text_zh": "你好，世界。",
                "language": "en",
            },
        ],
        "speakers": ["spk0"],
    }
```

在 `test_transcribe_success` 中新增断言：

```python
    assert data["segments"][0]["text_zh"] == "你好，世界。"
    assert data["segments"][0]["language"] == "en"
```

- [ ] **Step 2: 运行测试确认通过**

Run: `cd /home/tl/work/FunASR && python -m pytest tests/test_api.py -v`
Expected: 全部通过（api.py 无需改动，因为它只是透传 format_result 的输出）

- [ ] **Step 3: 更新 api.py 的 API 描述**

将 FastAPI 描述从"支持中英文自动检测"改为"支持多语言自动检测与中文翻译"：

```python
        description="多语言会议转写 API，支持中英日韩粤自动检测、VAD、说话人分离、自动翻译为中文",
```

- [ ] **Step 4: 运行全部测试**

Run: `cd /home/tl/work/FunASR && python -m pytest tests/ -v`
Expected: 全部通过

- [ ] **Step 5: 提交**

```bash
git add funasr_server/api.py tests/test_api.py
git commit -m "feat: update API response to include translation fields"
```

---

### Task 5: Gradio 界面变更

**Files:**
- Modify: `funasr_server/gradio_frontend.py`

- [ ] **Step 1: 更新 Gradio 界面**

修改 `gradio_frontend.py` 中的 `transcribe_audio` 函数和界面组件：

1. 语言选项扩展（支持日/韩/粤）：

```python
            lang_map = {"自动检测": "auto", "中文": "zh", "英文": "en", "日文": "ja", "韩文": "ko", "粤语": "yue"}
```

2. 表格数据构建中添加翻译列和语言列：

```python
            table_data = []
            for seg in result["segments"]:
                start_sec = seg["start_ms"] / 1000
                end_sec = seg["end_ms"] / 1000
                time_str = f"{start_sec:.1f}s - {end_sec:.1f}s"
                translation = seg.get("text_zh") or "—"
                lang = seg.get("language", "")
                table_data.append([seg["speaker"], time_str, seg["text"], translation, lang])
```

3. 状态信息增加翻译提示：

```python
            translated_count = sum(1 for s in result["segments"] if s.get("text_zh"))
            status = (
                f"转写完成 | 语言: {result['language']} | "
                f"时长: {duration_s:.1f}s | 说话人: {len(result['speakers'])} 人 | "
                f"翻译: {translated_count} 段"
            )
```

4. 界面 Markdown 说明更新：

```python
        gr.Markdown("上传音频文件，自动进行语音识别、说话人分离，支持中英日韩粤多语言并自动翻译为中文。")
```

5. 语言选项更新：

```python
                language_radio = gr.Radio(
                    choices=["自动检测", "中文", "英文", "日文", "韩文", "粤语"],
                    value="自动检测",
                    label="语言",
                )
```

6. 表格列名更新：

```python
                results_table = gr.Dataframe(
                    headers=["说话人", "时间", "原文", "中文翻译", "语言"],
                    label="转写结果",
                    interactive=False,
                )
```

- [ ] **Step 2: 运行全部测试**

Run: `cd /home/tl/work/FunASR && python -m pytest tests/ -v`
Expected: 全部通过

- [ ] **Step 3: 提交**

```bash
git add funasr_server/gradio_frontend.py
git commit -m "feat: update Gradio UI with translation and language columns"
```

---

### Task 6: 启动入口变更

**Files:**
- Modify: `funasr_server/app.py`

- [ ] **Step 1: 更新 app.py**

无需改动 `app.py`。因为 `TranscriptionPipeline.__init__` 已内置翻译器初始化逻辑，而 `app.py` 通过 `TranscriptionPipeline(settings)` 创建 pipeline 时会自动加载翻译模型。`api_module._pipeline = pipeline` 注入的 pipeline 已包含翻译器。

仅需验证 `app.py` 的逻辑无冲突。检查确认：无需改动。

- [ ] **Step 2: 提交（无文件变更则跳过此步骤）**

---

### Task 7: 依赖更新

**Files:**
- Modify: `funasr_server/requirements.txt`

- [ ] **Step 1: 添加翻译依赖**

在 `requirements.txt` 末尾添加：

```
transformers
sentencepiece
```

最终文件内容：

```
funasr
fastapi
uvicorn[standard]
gradio>=4.0.0
python-multipart
pytest
httpx
transformers
sentencepiece
```

- [ ] **Step 2: 运行全部测试**

Run: `cd /home/tl/work/FunASR && python -m pytest tests/ -v`
Expected: 全部通过

- [ ] **Step 3: 提交**

```bash
git add funasr_server/requirements.txt
git commit -m "feat: add transformers and sentencepiece dependencies"
```

---

## 自检清单

- **Spec 覆盖**: config 变更(Task1) ✓ | translator 模块(Task2) ✓ | pipeline 集成(Task3) ✓ | API 响应(Task4) ✓ | Gradio UI(Task5) ✓ | 启动入口(Task6) ✓ | 依赖(Task7) ✓ | 错误处理(Task2/3) ✓ | 向后兼容(Task4) ✓
- **占位符扫描**: 无 TBD/TODO
- **类型一致性**: `Translator.translate(text, source_lang)` → `batch_translate(segments)` → `pipeline.format_result` 调用 `translator.batch_translate(segments)` → segments 包含 `text_zh` 和 `language` 字段。所有引用一致。
