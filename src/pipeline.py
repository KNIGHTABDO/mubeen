import os
import sys
import json
import time
import subprocess
import glob
from tqdm import tqdm

# Add src to path
sys.path.insert(0, '/home/ubuntu/mubeen/src')
from config import *
import download_dataset
import preprocess
import embeddings

# Load dynamic library
LIB_PATH = '/home/ubuntu/mubeen/src/reciters_lib.json'
if os.path.exists(LIB_PATH):
    with open(LIB_PATH) as f:
        RECITERS_DATA = json.load(f)
else:
    RECITERS_DATA = {}

STATUS_FILE = '/home/ubuntu/mubeen/src/embedding_status.json'

def update_status(data):
    try:
        with open(STATUS_FILE, 'w') as f:
            json.dump(data, f)
    except:
        pass

def run_pipeline(selected_codes=None):
    if not selected_codes:
        selected_codes = list(RECITERS_DATA.keys())
    
    update_status({
        'status': 'running',
        'message': 'Initializing master pipeline...',
        'progress': 0,
        'total_reciters': len(selected_codes)
    })
    
    extractor = embeddings.EmbeddingExtractor()
    
    for i, code in enumerate(selected_codes):
        name = RECITERS_DATA.get(code)
        if not name: continue
        
        # 1. Download
        update_status({
            'status': 'running',
            'message': f'Downloading {name}...',
            'current_reciter': name,
            'step': 'Downloading',
            'reciter_idx': i + 1,
            'total_reciters': len(selected_codes)
        })
        download_dataset.download_reciter(code, name)
        
        # 2. Preprocess
        update_status({
            'status': 'running',
            'message': f'Preprocessing {name}...',
            'current_reciter': name,
            'step': 'Preprocessing',
            'reciter_idx': i + 1,
            'total_reciters': len(selected_codes)
        })
        raw_path = os.path.join(RAW_DIR, name)
        out_path = os.path.join(PROCESSED_DIR, name)
        os.makedirs(out_path, exist_ok=True)
        
        audio_files = glob.glob(os.path.join(raw_path, '*.mp3'))
        clip_idx = 0
        for audio_file in audio_files:
            try:
                y = preprocess.preprocess_audio(audio_file)
                clips = preprocess.create_clips(y)
                for clip in clips:
                    p = os.path.join(out_path, f'clip_{clip_idx:04d}.wav')
                    import soundfile as sf
                    sf.write(p, clip, SAMPLE_RATE)
                    clip_idx += 1
            except Exception as e:
                print(f"Error on {audio_file}: {e}")
                continue
                
        # 3. Memory-Friendly: Add directly to FAISS and Delete Audio
        update_status({
            'status': 'running',
            'message': f'Extracting Embeddings for {name}...',
            'current_reciter': name,
            'step': 'Indexing',
            'reciter_idx': i + 1,
            'total_reciters': len(selected_codes)
        })
        
        # Add to index
        extractor.add_reciter_to_index(name, out_path)
        
        # SMART DELETION CACHE CLEAR: Remove the heavy audio data entirely!
        import shutil
        update_status({
            'status': 'running',
            'message': f'Cleaning up storage for {name}...',
            'current_reciter': name,
            'step': 'Cleanup',
            'reciter_idx': i + 1,
            'total_reciters': len(selected_codes)
        })
        try:
            shutil.rmtree(raw_path)
            shutil.rmtree(out_path)
        except Exception as e:
            print(f"Cleanup error: {e}")
            
    update_status({
        'status': 'done',
        'message': 'Pipeline complete!',
        'finished_at': time.strftime('%Y-%m-%d %H:%M:%S UTC')
    })

if __name__ == '__main__':
    codes = sys.argv[1:] if len(sys.argv) > 1 else None
    run_pipeline(codes)
