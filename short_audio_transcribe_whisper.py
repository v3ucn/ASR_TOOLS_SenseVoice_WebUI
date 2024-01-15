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

model = whisper.load_model("medium")


lang2token = {
            'zh': "ZH|",
            'ja': "JP|",
            "en": "EN|",
        }


def transcribe_one(audio_path):
    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # detect the spoken language
    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")
    lang = max(probs, key=probs.get)
    # decode the audio

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
    parser.add_argument("--model_name", type=str, required=True)
    

    args = parser.parse_args()

    speaker_name = args.model_name

    language = args.language


    wav_files = [
        f for f in os.listdir("./wavs/") if f.endswith(".wav")
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
            
            text = transcribe_one("./wavs/"+wav_file)

            f.write(f"{file_name}|{speaker_name}|{language_id}|{text}\n")

            f.flush()
    sys.exit(0)


