import sys
sys.path.insert(0, '/home/ubuntu/mubeen/src')
from embeddings import EmbeddingExtractor
print("Methods in EmbeddingExtractor:", [m for m in dir(EmbeddingExtractor) if not m.startswith('__')])
