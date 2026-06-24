from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torchaudio

import config
import manifest


def load_segment(path: str, start_sec: float, end_sec: float, target_sr: int) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    a, b = int(start_sec * sr), int(end_sec * sr)
    wav = wav[:, a:b].mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav


class LayerCache:
    def __init__(self, layer: int):
        self.path = config.EMB_CACHE / f"w2vL{layer}.npz"
        self.data = {}
        if self.path.exists():
            with np.load(self.path) as z:
                self.data = {k: z[k] for k in z.files}

    def has(self, key): return key in self.data
    def put(self, key, vec): self.data[key] = vec.astype(np.float32)
    def flush(self): np.savez_compressed(self.path, **self.data)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", nargs="+", type=int, default=[4, 8, 12, 16, 20, 24])
    args = ap.parse_args()

    from transformers import Wav2Vec2Model
    model = Wav2Vec2Model.from_pretrained(config.W2V_FR_CKPT).to(config.DEVICE).eval()

    tracks = manifest.load_tracks().set_index("track_id")
    segments = manifest.load_segments()

    n_layers = model.config.num_hidden_layers
    layers = [L for L in args.layers if 1 <= L <= n_layers]
    print(f"Modèle : {n_layers} couches transformer. Extraction de : {layers}")

    caches = {L: LayerCache(L) for L in layers}
    n_new = 0

    for _, s in segments.iterrows():
        tid = s["track_id"]
        if tid not in tracks.index:
            continue
        key = f"{s['segment_id']}::voc"
        todo = [L for L in layers if not caches[L].has(key)]
        if not todo:
            continue
        vpath = tracks.loc[tid, "vocals_path"]
        if not isinstance(vpath, str) or not vpath:
            continue
        try:
            wav = load_segment(vpath, s["start_sec"], s["end_sec"], config.SR_W2V)
            x = wav.to(config.DEVICE)
            x = (x - x.mean()) / (x.std() + 1e-7)
            with torch.no_grad():
                hs = model(x, output_hidden_states=True).hidden_states  # tuple len n_layers+1
            for L in todo:
                vec = hs[L].mean(dim=1).squeeze(0).cpu().numpy()
                caches[L].put(key, vec)
            n_new += 1
        except Exception as e:  # noqa: BLE001
            print(f"  échec {key} : {e}")
        if n_new and n_new % 50 == 0:
            for c in caches.values():
                c.flush()

    for L, c in caches.items():
        c.flush()
        print(f"[w2vL{L}] {len(c.data)} embeddings -> {c.path}")


if __name__ == "__main__":
    main()