import librosa
import soundfile
import os
import argparse
from slicer2 import Slicer

# 设置命令行参数
parser = argparse.ArgumentParser()
parser.add_argument(
    "--min_sec", "-m", type=int, default=2000, help="Minimum seconds of a slice"
)
parser.add_argument(
    "--max_sec", "-M", type=int, default=5000, help="Maximum seconds of a slice"
)
parser.add_argument(
    "--dataset_path",
    type=str,
    default="inputs",
    help="Directory of input wav files",
)
parser.add_argument(
    "--min_silence_dur_ms",
    "-s",
    type=int,
    default=700,
    help="Silence above this duration (ms) is considered as a split point.",
)
args = parser.parse_args()

# 清空输出目录
folder_path = './wavs'
if os.path.exists(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
else:
    os.makedirs(folder_path)

# 遍历指定目录下的所有.wav文件
audio_directory = f'{args.dataset_path}'
for filename in os.listdir(audio_directory):
    file_path = os.path.join(audio_directory, filename)
    if os.path.isfile(file_path) and filename.endswith('.wav'):
        # 加载音频文件
        audio, sr = librosa.load(file_path, sr=None, mono=False)

        # 创建Slicer对象
        slicer = Slicer(
            sr=sr,
            threshold=-40,
            min_length=args.min_sec,
            min_interval=300,
            hop_size=10,
            max_sil_kept=args.min_silence_dur_ms
        )

        # 切割音频
        chunks = slicer.slice(audio)
        for i, chunk in enumerate(chunks):
            if len(chunk.shape) > 1:
                chunk = chunk.T  # Swap axes if the audio is stereo.
            soundfile.write(f'./wavs/{filename[:-4]}_{i}.wav', chunk, sr)
