import json
import os

meta_path = '/home/ubuntu/mubeen/data/embeddings/reciters_meta.json'
if not os.path.exists(meta_path):
    print("Metadata file not found")
    exit(1)

with open(meta_path) as f:
    meta = json.load(f)

print("Total vectors:", meta['total_vectors'])
print("Total labels:", len(meta['labels']))
print("Total files:", len(meta['files']))

print("\n--- First 10 samples ---")
for i in range(10):
    print(f"[{i}] Label: {meta['labels'][i]}, File: {meta['files'][i]}")

print("\n--- Last 10 samples ---")
for i in range(meta['total_vectors']-10, meta['total_vectors']):
    print(f"[{i}] Label: {meta['labels'][i]}, File: {meta['files'][i]}")

unique_labels = sorted(list(set(meta['labels'])))
print("\nUnique labels count:", len(unique_labels))
print("First 5 labels:", unique_labels[:5])
print("Last 5 labels:", unique_labels[-5:])
