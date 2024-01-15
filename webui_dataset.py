import argparse
import os

import gradio as gr
import yaml

from common.log import logger
from common.subprocess_utils import run_script_with_log

dataset_root = ".\\raw\\"


def do_slice(
    model_name: str,
    min_sec: int,
    max_sec: int,
    min_silence_dur_ms: int,
):
    if model_name == "":
        return "Error: 角色名不能为空"
    logger.info("Start slicing...")
    output_dir = os.path.join(dataset_root, model_name, ".\\wavs")


    cmd = [
        "audio_slicer_pre.py",
        "--model_name",
        model_name,
        "--min_sec",
        str(min_sec),
        "--max_sec",
        str(max_sec),
        "--min_silence_dur_ms",
        str(min_silence_dur_ms),
    ]
    

    success, message = run_script_with_log(cmd, ignore_warning=True)
    if not success:
        return f"Error: {message}"
    return "切分完毕"


def do_transcribe(
    model_name,mytype,language
):
    if model_name == "":
        return "Error: 角色名不能为空"
    

    cmd_py = "short_audio_transcribe_ali.py"

    if mytype == "Whisper":
    
        cmd_py = "short_audio_transcribe_whisper.py"

    elif mytype == "必剪asr":

        cmd_py = "short_audio_transcribe_bcut.py"


    

    success, message = run_script_with_log(
        [
            cmd_py,
            "--model_name",
            model_name,
            "--language",
            language,

        ]
    )
    if not success:
        return f"Error: {message}"
    return "转写完毕"


initial_md = """

请把格式为 角色名.wav 的素材文件放入项目的raw目录

作者：刘悦的技术博客  https://space.bilibili.com/3031494

"""

with gr.Blocks(theme="NoCrypt/miku") as app:
    gr.Markdown(initial_md)
    model_name = gr.Textbox(label="角色名")
    
    with gr.Accordion("音频素材切割"):
        with gr.Row():
            with gr.Column():
                
                min_sec = gr.Slider(
                    minimum=0, maximum=7000, value=2500, step=100, label="最低几毫秒"
                )
                max_sec = gr.Slider(
                    minimum=0, maximum=15000, value=5000, step=100, label="最高几毫秒"
                )
                min_silence_dur_ms = gr.Slider(
                    minimum=500,
                    maximum=5000,
                    value=700,
                    step=100,
                    label="max_sil_kept长度",
                )
                slice_button = gr.Button("开始切分")
            result1 = gr.Textbox(label="結果")

    with gr.Row():
        with gr.Column():
            
            language = gr.Dropdown(["ja", "en", "zh"], value="zh", label="选择转写的语言")

            mytype = gr.Dropdown(["阿里达摩院asr", "Whisper","必剪asr"], value="阿里达摩院asr", label="选择转写的软件")
            
        transcribe_button = gr.Button("开始转写")
        result2 = gr.Textbox(label="結果")

    slice_button.click(
        do_slice,
        inputs=[model_name, min_sec, max_sec, min_silence_dur_ms],
        outputs=[result1],
    )
    transcribe_button.click(
        do_transcribe,
        inputs=[
            model_name,
            mytype,
            language,
        ],
        outputs=[result2],
    )

parser = argparse.ArgumentParser()
parser.add_argument(
    "--server-name",
    type=str,
    default=None,
    help="Server name for Gradio app",
)
parser.add_argument(
    "--no-autolaunch",
    action="store_true",
    default=False,
    help="Do not launch app automatically",
)
args = parser.parse_args()

app.launch(inbrowser=not args.no_autolaunch, server_name=args.server_name)
