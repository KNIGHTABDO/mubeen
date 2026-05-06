"""Speaker embedding extraction using ECAPA-TDNN"""

import numpy as np
import torch
import os
import glob
import json
import time
from tqdm import tqdm
import faiss
import sys
sys.path.insert(0, '/home/ubuntu/mubeen/src')
from config import *
import preprocess


class EmbeddingExtractor:
    def __init__(self, model_dir=None):
        from speechbrain.inference import SpeakerRecognition
        
        if model_dir is None:
            model_dir = os.path.join(MODELS_DIR, 'ecapa-tdnn')
        
        self.model = SpeakerRecognition.from_hparams(
            source=ECAPA_MODEL,
            savedir=model_dir,
            run_opts={'device': 'cpu'}
        )
        print(f'ECAPA-TDNN model loaded from {model_dir}')
    
    def extract_embedding(self, audio_path):
        """Extract speaker embedding from an audio file."""
        import librosa
        import soundfile as sf
        
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        
        # Ensure minimum length (at least 1 second)
        if len(y) < SAMPLE_RATE:
            y = np.pad(y, (0, SAMPLE_RATE - len(y)))
        
        waveform = torch.tensor(y).unsqueeze(0).float()
        
        with torch.no_grad():
            embedding = self.model.encode_batch(waveform)
        
        return embedding.squeeze().cpu().numpy()
    
    def extract_from_clip_array(self, clip_array, sr=SAMPLE_RATE):
        """Extract embedding from a numpy array directly."""
        waveform = torch.tensor(clip_array).unsqueeze(0).float()
        
        with torch.no_grad():
            embedding = self.model.encode_batch(waveform)
        
        return embedding.squeeze().cpu().numpy()
    
    def add_reciter_to_index(self, reciter_name, processed_dir):
        """Add embeddings from a specific reciter's processed clips to the FAISS index."""
        os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
        
        if not os.path.exists(FAISS_INDEX_PATH):
            # Create a new index if it doesn't exist
            index = faiss.IndexFlatIP(EMBEDDING_DIM)
            metadata = {'labels': [], 'files': [], 'reciters': [], 'total_vectors': 0}
        else:
            index = faiss.read_index(FAISS_INDEX_PATH)
            with open(FAISS_METADATA_PATH) as f:
                metadata = json.load(f)
        
        clip_files = sorted(glob.glob(os.path.join(processed_dir, '*.wav')))
        if not clip_files:
            print(f"No clips found in {processed_dir}")
            return
            
        all_embeddings = []
        all_labels = []
        all_files = []
        
        for i, clip_file in enumerate(clip_files):
            try:
                emb = self.extract_embedding(clip_file)
                all_embeddings.append(emb)
                all_labels.append(reciter_name)
                all_files.append(clip_file)
                
                if (i + 1) % 50 == 0 or (i + 1) == len(clip_files):
                    with open('/home/ubuntu/mubeen/src/embedding_status.json', 'w') as f:
                        json.dump({
                            'status': 'running',
                            'message': f'Indexing {reciter_name}: {i+1}/{len(clip_files)} clips...',
                            'current_reciter': reciter_name,
                            'step': 'Indexing',
                            'progress_in_reciter': f"{i+1}/{len(clip_files)}"
                        }, f)
            except Exception as e:
                print(f"Error on {clip_file}: {e}")
                
        if not all_embeddings:
            return
            
        embeddings_matrix = np.array(all_embeddings).astype('float32')
        faiss.normalize_L2(embeddings_matrix)
        index.add(embeddings_matrix)
        
        # Atomic update
        metadata['labels'].extend(all_labels)
        metadata['files'].extend(all_files)
        metadata['reciters'] = sorted(list(set(metadata['labels'])))
        metadata['total_vectors'] = index.ntotal
        metadata['updated_at'] = time.strftime('%Y-%m-%d %H:%M:%S UTC')
        
        # Sync check before saving
        if len(metadata['labels']) != index.ntotal:
            print(f"CRITICAL SYNC ERROR: Index ({index.ntotal}) != Labels ({len(metadata['labels'])}). Rolling back.")
            return
            
        with open(FAISS_METADATA_PATH, 'w') as f:
            json.dump(metadata, f)
        faiss.write_index(index, FAISS_INDEX_PATH)
        
        print(f"Added {len(all_labels)} vectors for {reciter_name}. Total index size: {index.ntotal}")

    def build_index(self, processed_dir=PROCESSED_DIR):
        """Build FAISS index from all processed clips."""
        all_embeddings = []
        all_labels = []
        all_files = []
        
        reciter_dirs = sorted([d for d in os.listdir(processed_dir) 
                               if os.path.isdir(os.path.join(processed_dir, d))])
        
        total_clips = sum(len(glob.glob(os.path.join(processed_dir, d, '*.wav'))) for d in reciter_dirs)
        clips_done = 0
        
        print(f'\nExtracting embeddings for {len(reciter_dirs)} reciters...')
        
        for reciter_name in tqdm(reciter_dirs):
            reciter_path = os.path.join(processed_dir, reciter_name)
            clip_files = sorted(glob.glob(os.path.join(reciter_path, '*.wav')))
            
            for clip_file in clip_files:
                try:
                    emb = self.extract_embedding(clip_file)
                    all_embeddings.append(emb)
                    all_labels.append(reciter_name)
                    all_files.append(clip_file)
                    
                    clips_done += 1
                    if clips_done % 10 == 0:
                        with open('/home/ubuntu/mubeen/src/embedding_status.json', 'w') as f:
                            json.dump({
                                'clips_processed': clips_done,
                                'total_clips': total_clips,
                                'current_reciter': reciter_name,
                                'status': 'running'
                            }, f)
                except Exception as e:
                    print(f'  Error on {clip_file}: {e}')
        
        if not all_embeddings:
            print('ERROR: No embeddings extracted!')
            return None, None
        
        # Final status update
        with open('/home/ubuntu/mubeen/src/embedding_status.json', 'w') as f:
            json.dump({'status': 'done', 'clips_processed': clips_done, 'total_clips': total_clips}, f)

        # Convert to numpy matrix
        embeddings_matrix = np.array(all_embeddings).astype('float32')
        
        # L2 normalize for cosine similarity
        faiss.normalize_L2(embeddings_matrix)
        
        # Build FAISS index (Inner Product = cosine similarity after normalization)
        index = faiss.IndexFlatIP(EMBEDDING_DIM)
        index.add(embeddings_matrix)
        
        print(f'\nFAISS index built: {index.ntotal} vectors, {len(reciter_dirs)} reciters')
        
        # Save index and metadata
        os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
        faiss.write_index(index, FAISS_INDEX_PATH)
        
        metadata = {
            'labels': all_labels,
            'files': all_files,
            'reciters': sorted(list(set(all_labels))),
            'total_vectors': index.ntotal,
            'updated_at': time.strftime('%Y-%m-%d %H:%M:%S UTC')
        }
        with open(FAISS_METADATA_PATH, 'w') as f:
            json.dump(metadata, f)
        
        print(f'Index saved to {FAISS_INDEX_PATH}')
        print(f'Metadata saved to {FAISS_METADATA_PATH}')
        
        return index, metadata
    
    def identify(self, audio_path, k=5, window_duration=5, step_size=2.5, min_sim_threshold=0.2):
        """Identify reciter using a Segment Voting System to ignore intros and noise."""
        if not os.path.exists(FAISS_INDEX_PATH):
            return []
        
        index = faiss.read_index(FAISS_INDEX_PATH)
        with open(FAISS_METADATA_PATH) as f:
            metadata = json.load(f)
            
        import librosa
        import numpy as np
        from collections import defaultdict
        
        try:
            # Mandatory Preprocessing: Match the training environment
            y = preprocess.preprocess_audio(audio_path, sr=SAMPLE_RATE)
            
            window_samples = int(window_duration * SAMPLE_RATE)
            step_samples = int(step_size * SAMPLE_RATE)
            
            if len(y) < window_samples:
                y = np.pad(y, (0, window_samples - len(y)))
            
            num_segments = max(1, (len(y) - window_samples) // step_samples + 1)
            
            reciter_votes = defaultdict(int)
            reciter_sim_sum = defaultdict(float)
            
            # Increase internal k to get better distribution per segment, but we only vote for the top 1
            search_k = max(k, 50) 
            
            for i in range(num_segments):
                start = i * step_samples
                end = start + window_samples
                segment = y[start:end]
                
                if len(segment) < window_samples:
                    segment = np.pad(segment, (0, window_samples - len(segment)))
                    
                query_emb = self.extract_from_clip_array(segment)
                query_emb = query_emb.reshape(1, -1).astype('float32')
                faiss.normalize_L2(query_emb)
                
                similarities, indices = index.search(query_emb, search_k)
                
                # Find the best match for this segment
                best_sim = -1
                best_reciter = None
                
                for sim, idx in zip(similarities[0], indices[0]):
                    if sim > best_sim:
                        best_sim = float(sim)
                        best_reciter = metadata['labels'][idx]
                        
                # Cast a vote if it passes the threshold
                if best_reciter and best_sim >= min_sim_threshold:
                    reciter_votes[best_reciter] += 1
                    reciter_sim_sum[best_reciter] += best_sim

            if not reciter_votes:
                return []
                
            sorted_results = []
            for reciter, votes in reciter_votes.items():
                avg_sim = reciter_sim_sum[reciter] / votes
                sorted_results.append({
                    'reciter': reciter,
                    'similarity': float(votes + (avg_sim / 10.0)),  # Composite score for backward compatibility
                    'votes': votes,
                    'avg_sim': avg_sim
                })
                
            sorted_results.sort(key=lambda x: (x['votes'], x['avg_sim']), reverse=True)
            
            return sorted_results[:k]
            
        except Exception as e:
            print(f"Error during identification: {e}")
            return []


if __name__ == '__main__':
    import time
    extractor = EmbeddingExtractor()
    index, metadata = extractor.build_index()
    print('\nDone!')
