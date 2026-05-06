# Mubeen-1

Mubeen-1 is an inference-ready Quran reciter identification model built on speaker embeddings + fast similarity search.

- Embedding model: ECAPA-TDNN (speechbrain/spkrec-ecapa-voxceleb)
- Index: FAISS (cosine similarity via normalized inner product)
- Output: top-k matching reciters with similarity scores

## Quickstart

1) Clone (with Git LFS)

    git lfs install
    git clone https://github.com/KNIGHTABDO/mubeen.git
    cd mubeen
    git lfs pull

2) Install

    pip install -r requirements.txt

3) Run identification

    python -m mubeen1.infer path/to/audio.wav --topk 5

Example output:

    1. Mishary_Alafasy    (similarity=0.8123)
    2. Maher_Al_Muaiqly   (similarity=0.7741)

## Repo contents

- data/embeddings/reciters.index (Git LFS)
- data/embeddings/reciters_meta.json (Git LFS)
- mubeen1/ (minimal inference package)
- MODEL_CARD.md

## Disclaimer

Similarity scores are not probabilities; validate results for your use case.

