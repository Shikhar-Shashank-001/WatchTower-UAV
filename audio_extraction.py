import os
import subprocess
from pydub import AudioSegment


FFMPEG_PATH = r"C:/Users/shikh/Downloads/ffmpeg-2026-01-29-git-c898ddb8fe-essentials_build/ffmpeg-2026-01-29-git-c898ddb8fe-essentials_build/bin/ffmpeg.exe"
FFPROBE_PATH = r"C:/Users/shikh/Downloads/ffmpeg-2026-01-29-git-c898ddb8fe-essentials_build/ffmpeg-2026-01-29-git-c898ddb8fe-essentials_build/bin/ffprobe.exe"

AudioSegment.converter = FFMPEG_PATH
AudioSegment.ffmpeg = FFMPEG_PATH
AudioSegment.ffprobe = FFPROBE_PATH



INPUT_DIR = r"C:/Users/shikh/Downloads/Input"
OUTPUT_DIR = r"C:/Users/shikh/Downloads/Output2"
TEMP_DIR = r"C:/Users/shikh/Downloads/temp_wav"

SAMPLE_RATE = 16000
CHUNK_MS = 60 * 1000        # 1 minute
MIN_REMAINDER_MS = 30 * 1000


os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

def extract_audio_ffmpeg(mp4_path, wav_path):
    command = [
        FFMPEG_PATH,            
        "-y",
        "-i", mp4_path,
        "-ac", "1",             
        "-ar", str(SAMPLE_RATE),
        "-vn",
        wav_path
    ]
    subprocess.run(command, check=True)

def process_mp4(mp4_path):
    base = os.path.splitext(os.path.basename(mp4_path))[0]
    temp_wav = os.path.join(TEMP_DIR, base + ".wav")

    
    extract_audio_ffmpeg(mp4_path, temp_wav)

    
    audio = AudioSegment.from_wav(temp_wav)
    total_len = len(audio)

    chunk_id = 0
    for start in range(0, total_len, CHUNK_MS):
        chunk = audio[start:start + CHUNK_MS]

        # discard if < 30s
        if len(chunk) < MIN_REMAINDER_MS:
            break

        out_name = f"{base}_chunk_{chunk_id}.wav"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        chunk.export(out_path, format="wav")

        chunk_id += 1

    os.remove(temp_wav)
    print(f"{base}: {chunk_id} chunks created")


for file in os.listdir(INPUT_DIR):
    if file.lower().endswith(".mp4"):
        process_mp4(os.path.join(INPUT_DIR, file))
