import os
import argparse
import whisper
import torch

from tqdm import tqdm
import sys
import os

from common.constants import Languages
from common.log import logger
from common.stdout_wrapper import SAFE_STDOUT

from transformers import pipeline

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
    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model

    if mytype == "large-v3":

        mel = whisper.log_mel_spectrogram(audio,n_mels=128).to(model.device)

    else:

        mel = whisper.log_mel_spectrogram(audio).to(model.device)


    # detect the spoken language
    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")
    lang = max(probs, key=probs.get)
    # decode the audio


    if lang == "zh":


        if torch.cuda.is_available():
            options = whisper.DecodingOptions(beam_size=5,prompt="生于忧患，死于欢乐。不亦快哉！")
        else:
            options = whisper.DecodingOptions(beam_size=5,fp16 = False,prompt="生于忧患，死于欢乐。不亦快哉！")

    else:

    

        if torch.cuda.is_available():
            options = whisper.DecodingOptions(beam_size=5)
        else:
            options = whisper.DecodingOptions(beam_size=5,fp16 = False)






    result = whisper.decode(model, mel, options)

    # print the recognized text
    print(result.text)
    return result.text


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

    try:
        model = whisper.load_model(mytype,download_root="./whisper_model/")
    except Exception as e:

        print(str(e))
        print("中文特化逻辑")


    wav_files = [
        f for f in os.listdir(f"{input_file}") if f.endswith(".wav")
    ]


    if language == "ja":
        language_id = Languages.JP
    elif language == "en":
        language_id = Languages.EN
    elif language == "zh":
        language_id = Languages.ZH
    else:
        raise ValueError(f"{language} is not supported.")

    with open("./esd.list", "w", encoding="utf-8") as f:
        for wav_file in tqdm(wav_files, file=SAFE_STDOUT):
            file_name = os.path.basename(wav_file)
            
            if model:
                text = transcribe_one(f"{input_file}"+wav_file,mytype)
            else:
                text = transcribe_bela(f"{input_file}"+wav_file)

            f.write(file_pos+f"{file_name}|{speaker_name}|{language_id}|{text}\n")

            f.flush()
    sys.exit(0)


