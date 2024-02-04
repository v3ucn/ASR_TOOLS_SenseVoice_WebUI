import argparse
import os

import gradio as gr
import yaml

from common.log import logger
from common.subprocess_utils import run_script_with_log

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from videoclipper import VideoClipper
import librosa
import soundfile as sf
import numpy as np
import random

dataset_root = ".\\raw\\"





# 字幕语音切分
inference_pipeline = pipeline(
    task=Tasks.auto_speech_recognition,
    model='damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
    vad_model='damo/speech_fsmn_vad_zh-cn-16k-common-pytorch',
    punc_model='damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch',
    ncpu=16,
)
sd_pipeline = pipeline(
    task='speaker-diarization',
    model='damo/speech_campplus_speaker-diarization_common',
    model_revision='v1.0.0'
)
audio_clipper = VideoClipper(inference_pipeline, sd_pipeline)

def audio_change(audio):

    print(audio)

    sf.write('./output_44100.wav', audio[1], audio[0], 'PCM_24')

    y, sr = librosa.load('./output_44100.wav', sr=16000)

    # sf.write('./output_16000.wav', y, sr, 'PCM_24')

    # arr = np.array(y, dtype=np.int32)

    # y, sr = librosa.load('./output_16000.wav', sr=16000)

    audio_data = np.array(y)

    print(y, sr)

    return (16000,audio_data)

def write_list(text,audio):
    
    random_number = random.randint(10000, 99999)

    wav_name = f'./wavs/sample_{random_number}.wav'

    sf.write(wav_name, audio[1], audio[0], 'PCM_24')

    text = text.replace("#",",")

    with open("./esd.list","a",encoding="utf-8")as f:f.write(f"\n{wav_name}|sample|en|{text}")




def audio_recog(audio_input, sd_switch):
    print(audio_input)
    return audio_clipper.recog(audio_input, sd_switch)

def audio_clip(dest_text, audio_spk_input, start_ost, end_ost, state):
    return audio_clipper.clip(dest_text, start_ost, end_ost, state, dest_spk=audio_spk_input)

# 音频降噪

def reset_tts_wav(audio):

    ans = pipeline(
    Tasks.acoustic_noise_suppression,
    model='damo/speech_frcrn_ans_cirm_16k')
    ans(audio,output_path='./output_ins.wav')

    return "./output_ins.wav","./output_ins.wav"


def do_slice(
    dataset_path: str,
    min_sec: int,
    max_sec: int,
    min_silence_dur_ms: int,
):
    if dataset_path == "":
        return "Error: 数据集路径不能为空"
    logger.info("Start slicing...")
    output_dir = os.path.join(dataset_root, dataset_path, ".\\wavs")


    cmd = [
        "audio_slicer_pre.py",
        "--dataset_path",
        dataset_path,
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


def do_transcribe_fwhisper(
    model_name,mytype,language,input_file,file_pos
):
    # if model_name == "":
    #     return "Error: 角色名不能为空"
    
    
    cmd_py = "short_audio_transcribe_fwhisper.py"


    success, message = run_script_with_log(
        [
            cmd_py,
            "--model_name",
            model_name,
            "--language",
            language,
            "--mytype",
            mytype,"--input_file",
            input_file,
            "--file_pos",
            file_pos,

        ]
    )
    if not success:
        return f"Error: {message}"
    return "转写完毕"

def do_transcribe_whisper(
    model_name,mytype,language,input_file,file_pos
):
    # if model_name == "":
    #     return "Error: 角色名不能为空"
    
    
    cmd_py = "short_audio_transcribe_whisper.py"


    success, message = run_script_with_log(
        [
            cmd_py,
            "--model_name",
            model_name,
            "--language",
            language,
            "--mytype",
            mytype,"--input_file",
            input_file,
            "--file_pos",
            file_pos,

        ]
    )
    if not success:
        return f"Error: {message}"
    return "转写完毕"


def do_transcribe_all(
    model_name,mytype,language,input_file,file_pos
):
    # if model_name == "":
    #     return "Error: 角色名不能为空"
    

    cmd_py = "short_audio_transcribe_ali.py"


    if mytype == "bcut":

        cmd_py = "short_audio_transcribe_bcut.py"

    success, message = run_script_with_log(
        [
            cmd_py,
            "--model_name",
            model_name,
            "--language",
            language,
            "--input_file",
            input_file,
            "--file_pos",
            file_pos,

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
    model_name = gr.Textbox(label="角色名",placeholder="请输入角色名",visible=False)


    with gr.Accordion("干声抽离和降噪"):
        with gr.Row():
            audio_inp_path = gr.Audio(label="请上传克隆对象音频", type="filepath")
            reset_inp_button = gr.Button("针对原始素材进行降噪", variant="primary",visible=True)
            reset_dataset_path = gr.Textbox(label="降噪后音频地址",placeholder="降噪后生成的音频地址")

        
    reset_inp_button.click(reset_tts_wav,[audio_inp_path],[audio_inp_path,reset_dataset_path])
    
    with gr.Accordion("音频素材切割"):
        with gr.Row():
            ##add by hyh 添加一个数据集路径的文本框
            dataset_path = gr.Textbox(label="音频素材所在路径，默认在项目的raw文件夹,支持批量角色切分",placeholder="设置音频素材所在路径",value="./raw/")
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
                    value=500,
                    step=100,
                    label="max_sil_kept长度",
                )
                slice_button = gr.Button("开始切分")
            result1 = gr.Textbox(label="結果")

    with gr.Accordion("音频素材手动按字幕切割"):
        audio_state = gr.State()
        with gr.Row():
            with gr.Column():
                # oaudio_input = gr.Audio(label="🔊音频输入 44100hz Audio Input",type="filepath")
                # rec_audio = gr.Button("👂重新采样")
                audio_input = gr.Audio(label="🔊音频输入 16000hz Audio Input")
                audio_sd_switch = gr.Radio(["no", "yes"], label="👥是否区分说话人 Recognize Speakers", value='no')
                recog_button1 = gr.Button("👂识别 Recognize")
                audio_text_output = gr.Textbox(label="✏️识别结果 Recognition Result")
                audio_srt_output = gr.Textbox(label="📖SRT字幕内容 RST Subtitles")
            with gr.Column():
                audio_text_input = gr.Textbox(label="✏️待裁剪文本 Text to Clip (多段文本使用'#'连接)")
                audio_spk_input = gr.Textbox(label="✏️待裁剪说话人 Speaker to Clip (多个说话人使用'#'连接)")
                with gr.Row():
                    audio_start_ost = gr.Slider(minimum=-500, maximum=1000, value=0, step=50, label="⏪开始位置偏移 Start Offset (ms)")
                    audio_end_ost = gr.Slider(minimum=-500, maximum=1000, value=0, step=50, label="⏩结束位置偏移 End Offset (ms)")
                with gr.Row():
                    clip_button1 = gr.Button("✂️裁剪 Clip")
                    write_button1 = gr.Button("写入转写文件")
                audio_output = gr.Audio(label="🔊裁剪结果 Audio Clipped")
                audio_mess_output = gr.Textbox(label="ℹ️裁剪信息 Clipping Log")
                audio_srt_clip_output = gr.Textbox(label="📖裁剪部分SRT字幕内容 Clipped RST Subtitles")

            audio_input.change(inputs=audio_input, outputs=audio_input, fn=audio_change)

            write_button1.click(write_list,[audio_text_input,audio_output],[])
            
            # rec_audio.click(re_write,[oaudio_input],[rec_audio])
            recog_button1.click(audio_recog, 
                            inputs=[audio_input, audio_sd_switch],
                            outputs=[audio_text_output, audio_srt_output, audio_state])
            clip_button1.click(audio_clip, 
                            inputs=[audio_text_input, audio_spk_input, audio_start_ost, audio_end_ost, audio_state], 
                            outputs=[audio_output, audio_mess_output, audio_srt_clip_output])



    with gr.Row():
        with gr.Column():
            
            language = gr.Dropdown(["ja", "en", "zh"], value="zh", label="选择转写的语言")

            mytype = gr.Dropdown(["small","medium","large-v3","large-v2"], value="medium", label="选择Whisper模型")

            input_file = gr.Textbox(label="切片所在目录",placeholder="不填默认为./wavs目录")
            
            file_pos = gr.Textbox(label="切片名称前缀",placeholder="不填只有切片文件名")
            
        transcribe_button_whisper = gr.Button("Whisper开始转写")

        transcribe_button_fwhisper = gr.Button("Faster-Whisper开始转写")

        transcribe_button_ali = gr.Button("阿里ASR开始转写")

        transcribe_button_bcut = gr.Button("必剪ASR开始转写")


        result2 = gr.Textbox(label="結果")

    slice_button.click(
        do_slice,
        inputs=[dataset_path, min_sec, max_sec, min_silence_dur_ms],
        outputs=[result1],
    )
    transcribe_button_whisper.click(
        do_transcribe_whisper,
        inputs=[
            model_name,
            mytype,
            language,input_file,file_pos
        ],
        outputs=[result2],)


    transcribe_button_fwhisper.click(
        do_transcribe_fwhisper,
        inputs=[
            model_name,
            mytype,
            language,input_file,file_pos
        ],
        outputs=[result2],)


    ali = gr.Text(value="ali",visible=False)

    bcut = gr.Text(value="bcut",visible=False)


    transcribe_button_ali.click(
        do_transcribe_all,
        inputs=[
            model_name,
            ali,
            language,input_file,file_pos
        ],
        outputs=[result2],
    )

    transcribe_button_bcut.click(
        do_transcribe_all,
        inputs=[
            model_name,
            bcut,
            language,input_file,file_pos
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

app.launch(inbrowser=not args.no_autolaunch, server_name=args.server_name, server_port=7971)
