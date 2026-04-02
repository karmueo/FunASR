# 多语言会议转写服务设计文档

## 概述

基于 FunASR 构建一个会议/访谈录音转写服务，支持中文和英文语音识别、VAD 语音活动检测、说话人分离，提供 REST API 和 Gradio 轻量前端界面。

## 需求摘要

- **场景**: 会议/访谈录音离线转写
- **说话人**: 3-6 人，数量未知，需自动检测
- **语言**: 单次会议以一种语言为主（中文或英文），需自动检测
- **输出**: 结构化 JSON（带说话人标签和时间戳）
- **部署**: 服务化（REST API + Gradio 前端）
- **音频**: 典型 1-2 小时中等长度

## 技术方案

**选型**: SenseVoiceSmall + fsmn-vad + ct-punc + cam++ (聚类)

Pipeline 组装:
```
fsmn-vad → SenseVoiceSmall(auto) → ct-punc → cam++ (说话人聚类)
```

选择理由: SenseVoice 原生支持 50+ 语言自动检测，一个模型覆盖中英文，AutoModel 原生 pipeline 直接组装，无需额外语言检测环节。

## 系统架构

```
┌─────────────────────────────────────────────────┐
│               前端 (Gradio Demo 页面)             │
│  音频上传 | 转写结果展示 | 说话人标注显示           │
└──────────────────┬──────────────────────────────┘
                   │ HTTP
┌──────────────────▼──────────────────────────────┐
│               REST API 层 (FastAPI)              │
│     POST /transcribe  (音频上传)                  │
│     GET  /health      (健康检查)                  │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│             转写服务层                            │
│  AutoModel pipeline:                             │
│  fsmn-vad → SenseVoiceSmall → ct-punc            │
│       → cam++ (说话人聚类)                        │
│                                                  │
│  后处理: 说话人标签分配 + JSON 格式化               │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│             基础设施层                            │
│  模型缓存 | 日志 | 配置管理 | 健康检查             │
└─────────────────────────────────────────────────┘
```

## 数据流

```
音频文件 (wav/mp3)
    │
    ▼
[1] VAD 分段 (fsmn-vad)
    │ 输出: [[start_ms, end_ms], ...]
    ▼
[2] ASR 识别 (SenseVoiceSmall, language="auto")
    │ 输出: 每段文本 + 词级时间戳 + 语言标识
    ▼
[3] 标点恢复 (ct-punc)
    │ 输出: 带标点的文本
    ▼
[4] 说话人分离 (cam++ 聚类)
    │ 输出: 每段语音的说话人标签 (spk0, spk1, ...)
    ▼
[5] 后处理 & JSON 格式化
    │ 输出: 结构化结果
    ▼
```

## API 设计

### POST /transcribe

请求:
```json
{
  "audio": "<文件上传>",
  "language": "auto",
  "speaker_num": null
}
```

- `language`: `auto`(自动检测) | `zh`(中文) | `en`(英文)
- `speaker_num`: `null`(自动检测) 或指定整数

响应:
```json
{
  "task_id": "uuid",
  "language": "zh",
  "duration_s": 3621.5,
  "segments": [
    {
      "speaker": "spk0",
      "start_ms": 0,
      "end_ms": 5200,
      "text": "大家好，今天我们主要讨论一下项目进展。"
    },
    {
      "speaker": "spk1",
      "start_ms": 5300,
      "end_ms": 10200,
      "text": "好的，我先汇报一下我这边的情况。"
    }
  ],
  "speakers": ["spk0", "spk1", "spk2"]
}
```

### GET /health

响应: `{"status": "ok", "models_loaded": true}`

## Gradio 前端

- 上传音频文件 → 点击"开始转写" → 显示进度 → 展示结果表格（说话人 | 时间 | 文本）→ 支持下载 JSON

## 技术选型

| 组件 | 选型 | 理由 |
|------|------|------|
| Web 框架 | FastAPI | 异步支持好，自动生成 API 文档 |
| 前端 | Gradio | ML Demo 标准工具，与 FastAPI 可共存 |
| ASR 模型 | `iic/SenseVoiceSmall` | 50+ 语言自动检测，内置 LID |
| VAD 模型 | `fsmn-vad` | 轻量 0.4M 参数，FunASR 原生集成 |
| 说话人模型 | `cam++` + ClusterBackend | 聚类方式支持未知说话人数 |
| 标点模型 | `ct-punc` | 中英文标点恢复 |

## 长音频处理策略

- VAD 分段: `max_single_segment_time=60000ms`（单段最长 60 秒）
- 批处理: `batch_size_s=300`（动态批处理，每次最多 300 秒）
- 超时保护: API 层设置请求超时上限 600 秒

## 说话人聚类参数

- 默认 `merge_thr=0.5`，自动推断说话人数
- 支持通过 `speaker_num` 参数覆盖，指定人数时聚类更准确

## 异常处理

| 场景 | 处理方式 |
|------|----------|
| 音频格式不支持 | 返回 400，提示支持格式（wav/mp3/flac） |
| 音频为空/无语音 | VAD 返回空段，返回空 segments |
| 模型加载失败 | 服务启动时预加载，失败则拒绝启动 |
| GPU 内存不足 | 回退 CPU 并记录警告 |
| 转写超时 | 返回 task_id，支持异步轮询结果 |

## 项目目录结构

```
funasr_server/
├── app.py                # FastAPI + Gradio 启动入口
├── pipeline.py           # AutoModel pipeline 封装
├── config.py             # 配置管理
├── requirements.txt      # 依赖
└── README.md
```

## 测试策略

| 层级 | 内容 | 工具 |
|------|------|------|
| 单元测试 | pipeline.py 各函数：音频预处理、后处理格式化、配置解析 | pytest |
| 集成测试 | 完整 pipeline 调用（使用短音频样本），验证 JSON 输出结构 | pytest + 短音频 fixture |
| API 测试 | HTTP 接口测试：正常上传、空文件、错误格式、超时 | FastAPI TestClient |

### 关键测试用例

- 英文音频转写 + 说话人标签正确
- 中文音频转写 + 语言检测为 zh
- 空音频/纯静音返回空 segments
- 超长音频（>1h）不 OOM，正常分段处理
- 指定 speaker_num 时聚类结果数量匹配

### 测试音频

- 英文: `/home/tl/data/tmp/From skeptic to true believer： How OpenClaw changed my life ｜ Claire Vo [DIa0MYJzM5I].mp3`
- 中文: 后续补充

## 非功能性约束

- 首次请求延迟: 模型预加载，推理时无加载开销
- 并发: 单 GPU 场景下串行处理请求，队列排队
- 日志: 每个转写任务记录耗时、语言检测结果、说话人数量
