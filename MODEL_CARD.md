# Mubeen-1 - Model Card

## Summary
Mubeen-1 is an inference-ready Quran reciter identification model. It uses ECAPA-TDNN speaker embeddings and a FAISS similarity index built from Quran recitations.

## What is included
- Prebuilt FAISS index: `data/embeddings/reciters.index`
- Labels and metadata: `data/embeddings/reciters_meta.json`
- Minimal inference code: `mubeen1/`

## Intended use
- Identify which reciter (speaker) is present in a short audio clip.

## How it works
1) Extract speaker embedding using ECAPA-TDNN (SpeechBrain).
2) Normalize and search a FAISS inner-product index (cosine similarity).
3) Return top-k matches.

## Limitations
- Only recognizes reciters present in the index.
- Similarity scores are not calibrated probabilities.
- Accuracy depends on clip length and audio quality.

## Dependencies
- The ECAPA-TDNN weights are downloaded automatically on first run via SpeechBrain.

