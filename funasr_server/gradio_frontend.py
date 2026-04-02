"""Gradio 轻量前端界面。

提供音频上传、转写、结果展示和 JSON 下载功能。
"""

import json
import os
import uuid

import gradio as gr

from funasr_server.pipeline import TranscriptionPipeline


def create_gradio_app(pipeline: TranscriptionPipeline) -> gr.Blocks:
    """创建 Gradio 前端应用。

    Args:
        pipeline: 转写管道实例

    Returns:
        Gradio Blocks 应用
    """

    def transcribe_audio(audio_path, language, speaker_num):
        """处理音频转写请求。

        Args:
            audio_path: 上传的音频文件路径
            language: 语言选项（自动检测/中文/英文）
            speaker_num: 说话人数量（0=自动检测）

        Returns:
            tuple: (表格数据列表, JSON字符串, 状态消息)
        """
        if audio_path is None:
            return [], '{"error": "请上传音频文件"}', "未上传文件"

        try:
            task_id = str(uuid.uuid4())

            # 语言映射：UI 选项 -> 内部语言代码
            lang_map = {"自动检测": "auto", "中文": "zh", "英文": "en"}
            lang = lang_map.get(language, "auto")

            # 说话人数量：0 表示自动检测，传 None 给 pipeline
            spk_num = None if speaker_num == 0 else speaker_num

            raw_result = pipeline.transcribe(
                audio_path=audio_path,
                language=lang,
                speaker_num=spk_num,
            )

            # 根据文件大小估算音频时长（16kHz 16bit 单声道 -> 32000 bytes/s）
            file_size = os.path.getsize(audio_path)
            duration_s = file_size / 32000

            result = pipeline.format_result(task_id, raw_result, duration_s)

            # 将分段数据转换为表格行
            table_data = []
            for seg in result["segments"]:
                start_sec = seg["start_ms"] / 1000
                end_sec = seg["end_ms"] / 1000
                time_str = f"{start_sec:.1f}s - {end_sec:.1f}s"
                table_data.append([seg["speaker"], time_str, seg["text"]])

            json_str = json.dumps(result, ensure_ascii=False, indent=2)
            status = (
                f"转写完成 | 语言: {result['language']} | "
                f"时长: {duration_s:.1f}s | 说话人: {len(result['speakers'])} 人"
            )

            return table_data, json_str, status

        except Exception as e:
            return [], json.dumps({"error": str(e)}, ensure_ascii=False), f"错误: {str(e)}"

    with gr.Blocks(title="FunASR 会议转写", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# FunASR 会议转写服务")
        gr.Markdown("上传音频文件，自动进行语音识别、说话人分离，支持中文和英文。")

        with gr.Row():
            with gr.Column(scale=1):
                # 音频上传组件
                audio_input = gr.Audio(
                    label="上传音频",
                    type="filepath",
                    sources=["upload"],
                )
                # 语言选择
                language_radio = gr.Radio(
                    choices=["自动检测", "中文", "英文"],
                    value="自动检测",
                    label="语言",
                )
                # 说话人数量滑块
                speaker_num_slider = gr.Slider(
                    minimum=0,
                    maximum=10,
                    value=0,
                    step=1,
                    label="说话人数量（0=自动检测）",
                )
                # 转写按钮
                transcribe_btn = gr.Button("开始转写", variant="primary")
                # 状态输出
                status_text = gr.Textbox(label="状态", interactive=False)

            with gr.Column(scale=2):
                # 转写结果表格
                results_table = gr.Dataframe(
                    headers=["说话人", "时间", "文本"],
                    label="转写结果",
                    interactive=False,
                )
                # JSON 原始输出
                json_output = gr.Textbox(
                    label="JSON 输出",
                    lines=10,
                    interactive=False,
                )

        # 绑定按钮点击事件
        transcribe_btn.click(
            fn=transcribe_audio,
            inputs=[audio_input, language_radio, speaker_num_slider],
            outputs=[results_table, json_output, status_text],
        )

    return demo
