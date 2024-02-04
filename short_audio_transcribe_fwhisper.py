import os
import argparse

import torch

from tqdm import tqdm
import sys
import os

from common.constants import Languages
from common.log import logger
from common.stdout_wrapper import SAFE_STDOUT

import re

from transformers import pipeline

from faster_whisper import WhisperModel

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = None

lang2token = {
            'zh': "ZH|",
            'ja': "JP|",
            "en": "EN|",
        }


def transcribe_bela(audio_path):

    transcriber = pipeline(
    "automatic-speech-recognition", 
    model="BELLE-2/Belle-whisper-large-v2-zh",
    device=device
    )

    transcriber.model.config.forced_decoder_ids = (
    transcriber.tokenizer.get_decoder_prompt_ids(
        language="zh", 
        task="transcribe",
    )
    )

    transcription = transcriber(audio_path) 

    print(transcription["text"])
    return transcription["text"]


def transcribe_one(audio_path,mytype):
    
    segments, info = model.transcribe(audio_path, beam_size=5,vad_filter=True,vad_parameters=dict(min_silence_duration_ms=500),)
    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

    text_str = ""
    for segment in segments:
        text_str += f"{segment.text.lstrip()},"
    print(text_str.rstrip(","))

    return text_str.rstrip(","),info.language



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--language", type=str, default="ja", choices=["ja", "en", "zh"]
    )

    parser.add_argument(
        "--mytype", type=str, default="medium"
    )

    parser.add_argument("--model_name", type=str, required=True)

    parser.add_argument("--input_file", type=str, default="./wavs/")

    parser.add_argument("--file_pos", type=str, default="")
    

    args = parser.parse_args()

    speaker_name = args.model_name

    language = args.language

    mytype = args.mytype

    input_file = args.input_file

    if input_file == "":
        input_file = "./wavs/"

    file_pos = args.file_pos

    if device == "cuda":
        try:
            model = WhisperModel(mytype, device="cuda", compute_type="float16",download_root="./whisper_model",local_files_only=False)
        except Exception as e:
            model = WhisperModel(mytype, device="cuda", compute_type="int8_float16",download_root="./whisper_model",local_files_only=False)
    else:
        model = WhisperModel(mytype, device="cpu", compute_type="int8",download_root="./whisper_model",local_files_only=False)


    wav_files = [
        f for f in os.listdir(f"{input_file}") if f.endswith(".wav")
    ]



    with open("./esd.list", "w", encoding="utf-8") as f:
        for wav_file in tqdm(wav_files, file=SAFE_STDOUT):
            file_name = os.path.basename(wav_file)
            
            if model:
                text,lang = transcribe_one(f"{input_file}"+wav_file,mytype)
            else:
                text,lang = transcribe_bela(f"{input_file}"+wav_file)

            # 使用正则表达式提取'deedee'
            match = re.search(r'(^.*?)_.*?(\..*?$)', wav_file)
            if match:
                extracted_name = match.group(1) + match.group(2)
            else:
                print("No match found")
                extracted_name = "sample"

            if lang == "ja":
                language_id = "JA"
            elif lang == "en":
                language_id = "EN"
            elif lang == "zh":
                language_id = "ZH"

            f.write(file_pos+f"{file_name}|{extracted_name.replace('.wav','')}|{language_id}|{text}\n")

            f.flush()
    sys.exit(0)


