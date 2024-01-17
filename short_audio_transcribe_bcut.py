import os
import argparse


from tqdm import tqdm
import sys
import os

from common.constants import Languages
from common.log import logger
from common.stdout_wrapper import SAFE_STDOUT

from bcut_asr import BcutASR
from bcut_asr.orm import ResultStateEnum



lang2token = {
            'zh': "ZH|",
            'ja': "JP|",
            "en": "EN|",
        }


def transcribe_one(audio_path):

    asr = BcutASR(audio_path)
    asr.upload() # 上传文件
    asr.create_task() # 创建任务

    # 轮询检查结果
    while True:
        result = asr.result()
        # 判断识别成功
        if result.state == ResultStateEnum.COMPLETE:
            break

    # 解析字幕内容
    subtitle = result.parse()

    # 判断是否存在字幕
    if subtitle.has_data():

        

        text = subtitle.to_txt()
        text = repr(text)
        text = text.replace("'","")
        text = text.replace("\\n",",")
        text = text.replace("\\r",",")

        print(text)

        # 输出srt格式
        return text
    else:
        return "必剪无法识别"
    


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--language", type=str, default="ja", choices=["ja", "en", "zh"]
    )
    parser.add_argument("--model_name", type=str, required=True)

    parser.add_argument("--input_file", type=str, default="./wavs/")

    parser.add_argument("--file_pos", type=str, default="")
    

    args = parser.parse_args()

    speaker_name = args.model_name

    language = args.language

    input_file = args.input_file

    if input_file == "":
        input_file = "./wavs/"

    file_pos = args.file_pos


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
            
            text = transcribe_one(f"{input_file}"+wav_file)

            f.write(file_pos+f"{file_name}|{speaker_name}|{language_id}|{text}\n")

            f.flush()
    sys.exit(0)


