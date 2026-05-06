from __future__ import annotations

from pathlib import Path

# Repo root (mubeen1/config.py -> repo root)
REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = REPO_ROOT / "data"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"

FAISS_INDEX_PATH = EMBEDDINGS_DIR / "reciters.index"
FAISS_METADATA_PATH = EMBEDDINGS_DIR / "reciters_meta.json"

# SpeechBrain ECAPA-TDNN model (downloaded automatically on first run)
ECAPA_MODEL = "speechbrain/spkrec-ecapa-voxceleb"
SAMPLE_RATE = 16000
EMBEDDING_DIM = 192

