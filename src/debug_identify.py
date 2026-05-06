import sys
sys.path.insert(0, '/home/ubuntu/mubeen/src')
from embeddings import EmbeddingExtractor
import librosa
import numpy as np
import faiss
import json
from collections import defaultdict

def debug_identify(audio_path):
    print(f"Loading {audio_path}...")
    ext = EmbeddingExtractor()
    index = faiss.read_index('/home/ubuntu/mubeen/data/embeddings/reciters.index')
    with open('/home/ubuntu/mubeen/data/embeddings/reciters_meta.json') as f:
        metadata = json.load(f)
        
    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    window_samples = int(5 * 16000)
    step_samples = int(2.5 * 16000)
    
    if len(y) < window_samples:
        y = np.pad(y, (0, window_samples - len(y)))
    
    num_segments = max(1, (len(y) - window_samples) // step_samples + 1)
    
    print(f"Total segments: {num_segments}")
    
    reciter_votes = defaultdict(int)
    
    for i in range(num_segments):
        start = i * step_samples
        end = start + window_samples
        segment = y[start:end]
        
        if len(segment) < window_samples:
            segment = np.pad(segment, (0, window_samples - len(segment)))
            
        query_emb = ext.extract_from_clip_array(segment)
        query_emb = query_emb.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query_emb)
        
        similarities, indices = index.search(query_emb, 5)
        
        best_sim = -1
        best_reciter = None
        for j, (sim, idx) in enumerate(zip(similarities[0], indices[0])):
            reciter = metadata['labels'][idx]
            if sim > best_sim:
                best_sim = float(sim)
                best_reciter = reciter
                
        if best_reciter and best_sim > 0.2:
            reciter_votes[best_reciter] += 1
            
    print("\n--- FINAL VOTES ---")
    for r, v in sorted(reciter_votes.items(), key=lambda item: item[1], reverse=True)[:5]:
        print(f"  {r}: {v} votes")

if __name__ == '__main__':
    audio_file = sys.argv[1] if len(sys.argv) > 1 else '/home/ubuntu/mubeen/data/debug_test.mp3'
    debug_identify(audio_file)
