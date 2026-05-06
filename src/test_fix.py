import sys
sys.path.insert(0, '/home/ubuntu/mubeen/src')
from embeddings import EmbeddingExtractor
ext = EmbeddingExtractor()
results = ext.identify('/home/ubuntu/mubeen/data/yasser_test.mp3')
print("\n=== STUDIO TEST RESULT ===")
for r in results:
    print(f"{r['reciter']}: {r['votes']} votes, avg_sim: {r['avg_sim']:.4f}")

user_results = ext.identify('/home/ubuntu/mubeen/data/debug_test.mp3')
print("\n=== USER AUDIO TEST RESULT ===")
for r in user_results:
    print(f"{r['reciter']}: {r['votes']} votes, avg_sim: {r['avg_sim']:.4f}")
