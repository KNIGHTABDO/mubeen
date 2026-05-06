#!/usr/bin/env python3
"""Evaluate Mubeen reciter identification accuracy"""
import sys, os, random, glob, json
sys.path.insert(0, '/home/ubuntu/mubeen/src')
from embeddings import EmbeddingExtractor
from collections import Counter

def main():
    print('=== Mubeen Evaluation ===')
    
    # Load metadata
    with open('data/embeddings/reciters_meta.json') as f:
        meta = json.load(f)
    print(f'Index: {meta["total_vectors"]} vectors, {len(meta["reciters"])} reciters')
    counts = Counter(meta['labels'])
    for r, c in sorted(counts.items()):
        print(f'  {r}: {c} embeddings')
    
    print()
    extractor = EmbeddingExtractor()
    
    processed_dir = '/home/ubuntu/mubeen/data/processed'
    test_dirs = sorted([d for d in os.listdir(processed_dir) 
                        if os.path.isdir(os.path.join(processed_dir, d))])
    
    print(f'Testing {len(test_dirs)} reciters (5 random clips each)...\n')
    
    correct = 0
    total = 0
    
    for reciter_dir in test_dirs:
        clips = glob.glob(os.path.join(processed_dir, reciter_dir, '*.wav'))
        if len(clips) < 3:
            continue
        
        test_clips = random.sample(clips, min(5, len(clips)))
        
        for test_clip in test_clips:
            results = extractor.identify(test_clip, k=3)
            if not results:
                continue
            
            top_match = results[0]['reciter']
            sim = results[0]['similarity']
            is_correct = top_match == reciter_dir
            mark = 'PASS' if is_correct else 'FAIL'
            
            print(f'  [{mark}] {reciter_dir[:25]:25s} -> {top_match[:25]:25s} (sim: {sim:.3f})')
            
            if is_correct:
                correct += 1
            total += 1
    
    print(f'\n========================================')
    print(f'  ACCURACY: {correct}/{total} = {correct/total*100:.1f}%')
    print(f'========================================')

if __name__ == '__main__':
    main()
