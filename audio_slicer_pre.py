import librosa  # Optional. Use any library you like to read audio files.
import soundfile  # Optional. Use any library you like to write audio files.
import gradio as gr
import os

import datetime
import json

import os

from slicer2 import Slicer

import argparse



parser = argparse.ArgumentParser()
parser.add_argument(
    "--min_sec", "-m", type=int, default=2000, help="Minimum seconds of a slice"
)
parser.add_argument(
    "--max_sec", "-M", type=int, default=5000, help="Maximum seconds of a slice"
)
parser.add_argument(
    "--model_name",
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



audio, sr = librosa.load(f'./raw/{args.model_name}.wav', sr=None, mono=False)  # Load an audio file with librosa.
slicer = Slicer(
    sr=sr,
    threshold=-40,
    min_length=args.min_sec,
    min_interval=300,
    hop_size=10,
    max_sil_kept=args.min_silence_dur_ms
)


folder_path = './wavs'
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    if os.path.isfile(file_path):
        os.remove(file_path)


chunks = slicer.slice(audio)
for i, chunk in enumerate(chunks):
    if len(chunk.shape) > 1:
        chunk = chunk.T  # Swap axes if the audio is stereo.
    soundfile.write(f'./wavs/{args.model_name}_{i}.wav', chunk, sr)  # Save sliced audio files with soundfile.

