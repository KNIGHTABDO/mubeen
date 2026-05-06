"""Audio preprocessing pipeline for Mubeen"""

import numpy as np
import librosa
import noisereduce as nr
import os
import glob
from tqdm import tqdm
import sys
sys.path.insert(0, '/home/ubuntu/mubeen/src')
from config import *

def preprocess_audio(audio_path, sr=SAMPLE_RATE):
    """Load and preprocess a single audio file."""
    # Load
    y, orig_sr = librosa.load(audio_path, sr=sr, mono=True)
    
    # Noise reduction (gentle, to preserve voice quality)
    y = nr.reduce_noise(y=y, sr=sr, prop_decrease=0.5)
    
    # Normalize
    y = librosa.util.normalize(y)
    
    # Trim silence
    y_trimmed, _ = librosa.effects.trim(y, top_db=25)
    
    return y_trimmed


def create_clips(y, sr=SAMPLE_RATE, clip_duration=CLIP_DURATION, overlap=CLIP_OVERLAP):
    """Split audio into overlapping clips."""
    clip_len = int(clip_duration * sr)
    hop_len = int(clip_len * (1 - overlap))
    
    clips = []
    for start in range(0, len(y) - clip_len + 1, hop_len):
        clip = y[start:start + clip_len]
        clips.append(clip)
    
    return clips


def process_dataset(raw_dir=RAW_DIR, output_dir=PROCESSED_DIR):
    """Process all reciters' audio files."""
    for reciter_name in tqdm(os.listdir(raw_dir), desc='Processing reciters'):
        reciter_raw = os.path.join(raw_dir, reciter_name)
        if not os.path.isdir(reciter_raw):
            continue
        
        reciter_out = os.path.join(output_dir, reciter_name)
        os.makedirs(reciter_out, exist_ok=True)
        
        audio_files = glob.glob(os.path.join(reciter_raw, '*.mp3')) +                       glob.glob(os.path.join(reciter_raw, '*.wav'))
        
        clip_idx = 0
        for audio_file in audio_files:
            try:
                y = preprocess_audio(audio_file)
                clips = create_clips(y)
                for clip in clips:
                    out_path = os.path.join(reciter_out, f'clip_{clip_idx:04d}.wav')
                    import soundfile as sf
                    sf.write(out_path, clip, SAMPLE_RATE)
                    clip_idx += 1
            except Exception as e:
                print(f'Error processing {audio_file}: {e}')
        
        print(f'  {reciter_name}: {clip_idx} clips created')
    
    print(f'\nDataset processed! Output: {output_dir}')


if __name__ == '__main__':
    process_dataset()
