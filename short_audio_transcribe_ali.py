import os
import argparse


from tqdm import tqdm
import sys
import os
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks


from modelscope.hub.snapshot_download import snapshot_download


from common.constants import Languages
from common.log import logger
from common.stdout_wrapper import SAFE_STDOUT

# 指定本地目录
local_dir_root = "./models_from_modelscope"
model_dir = snapshot_download('damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch', cache_dir=local_dir_root)
model_dir_punc_ct = snapshot_download('damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch', cache_dir=local_dir_root)
model_dir_vad = snapshot_download('damo/speech_fsmn_vad_zh-cn-16k-common-pytorch', cache_dir=local_dir_root)

model_dir_ja = snapshot_download('damo/speech_UniASR_asr_2pass-ja-16k-common-vocab93-tensorflow1-offline', cache_dir=local_dir_root)


model_dir_en = snapshot_download('damo/speech_UniASR_asr_2pass-en-16k-common-vocab1080-tensorflow1-offline', cache_dir=local_dir_root)





inference_pipeline = pipeline(
    task=Tasks.auto_speech_recognition,
    model=model_dir,
    vad_model=model_dir_vad,
    punc_model=model_dir_punc_ct,
    #lm_model='damo/speech_transformer_lm_zh-cn-common-vocab8404-pytorch',
    #lm_weight=0.15,
    #beam_size=10,
)
param_dict = {}
param_dict['use_timestamp'] = False
# folderpath = sys.argv[1]
extensions = ['wav']



inference_pipeline_ja = pipeline(
    task=Tasks.auto_speech_recognition,
    model=model_dir_ja,
   # vad_model=model_dir_vad,
  #  punc_model=model_dir_punc_ct,
    #lm_model='damo/speech_transformer_lm_zh-cn-common-vocab8404-pytorch',
    #lm_weight=0.15,
    #beam_size=10,
)


inference_pipeline_en = pipeline(
    task=Tasks.auto_speech_recognition,
    model=model_dir_en,
   # vad_model=model_dir_vad,
  #  punc_model=model_dir_punc_ct,
    #lm_model='damo/speech_transformer_lm_zh-cn-common-vocab8404-pytorch',
    #lm_weight=0.15,
    #beam_size=10,
)



lang2token = {
            'zh': "ZH|",
            'ja': "JP|",
            "en": "EN|",
        }


def transcribe_one(audio_path,language):

    if language == "zh":
    
        rec_result = inference_pipeline(audio_in=audio_path, param_dict=param_dict)
    elif language == "ja":
        rec_result = inference_pipeline_ja(audio_in=audio_path, param_dict=param_dict)
    else:
        rec_result = inference_pipeline_en(audio_in=audio_path, param_dict=param_dict)

    print(rec_result["text"])

    return rec_result["text"]


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
            
            text = transcribe_one(f"{input_file}"+wav_file,language)

            f.write(file_pos+f"{file_name}|{speaker_name}|{language_id}|{text}\n")

            f.flush()
    sys.exit(0)


