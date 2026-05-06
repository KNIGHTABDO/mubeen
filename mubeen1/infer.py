from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

from .config import (
    ECAPA_MODEL,
    EMBEDDING_DIM,
    FAISS_INDEX_PATH,
    FAISS_METADATA_PATH,
    SAMPLE_RATE,
)


@dataclass
class Match:
    reciter: str
    similarity: float


class Mubeen1:
    """Mubeen-1: Quran reciter identification via speaker embeddings + FAISS.

    Inference-only wrapper around a prebuilt FAISS index.
    """

    def __init__(self):
        from speechbrain.inference import SpeakerRecognition

        self.model = SpeakerRecognition.from_hparams(
            source=ECAPA_MODEL,
            savedir=str(Path.home() / ".cache" / "mubeen1" / "ecapa-tdnn"),
            run_opts={"device": "cpu"},
        )

        if not FAISS_INDEX_PATH.exists():
            raise FileNotFoundError(f"Missing FAISS index: {FAISS_INDEX_PATH}")
        if not FAISS_METADATA_PATH.exists():
            raise FileNotFoundError(f"Missing metadata: {FAISS_METADATA_PATH}")

        import faiss  # type: ignore

        self.faiss = faiss
        self.index = faiss.read_index(str(FAISS_INDEX_PATH))

        with FAISS_METADATA_PATH.open("r", encoding="utf-8") as f:
            self.meta = json.load(f)
        if "labels" not in self.meta:
            raise ValueError("Invalid metadata: missing labels")

    def _embed(self, audio_path: str | Path) -> np.ndarray:
        import librosa
        import torch

        y, _sr = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
        if len(y) < SAMPLE_RATE:
            y = np.pad(y, (0, SAMPLE_RATE - len(y)))

        wav = torch.tensor(y).unsqueeze(0).float()
        with torch.no_grad():
            emb = self.model.encode_batch(wav).squeeze().cpu().numpy()

        emb = emb.reshape(1, -1).astype("float32")
        if emb.shape[1] != EMBEDDING_DIM:
            raise ValueError(
                f"Unexpected embedding dim {emb.shape[1]} (expected {EMBEDDING_DIM})"
            )
        self.faiss.normalize_L2(emb)
        return emb

    def identify(self, audio_path: str | Path, top_k: int = 5) -> List[Match]:
        q = self._embed(audio_path)
        sims, idxs = self.index.search(q, top_k)

        labels = self.meta["labels"]
        out: List[Match] = []
        for sim, idx in zip(sims[0], idxs[0]):
            if idx < 0:
                continue
            out.append(Match(reciter=labels[idx], similarity=float(sim)))
        return out


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(
        description="Mubeen-1: identify a Quran reciter from an audio clip"
    )
    p.add_argument("audio", help="Path to an audio file (wav/mp3/m4a)")
    p.add_argument("--topk", type=int, default=5, help="Number of matches to return")
    args = p.parse_args()

    m = Mubeen1()
    matches = m.identify(args.audio, top_k=args.topk)
    for i, match in enumerate(matches, start=1):
        print(f"{i}. {match.reciter}\t(similarity={match.similarity:.4f})")


if __name__ == "__main__":
    main()

