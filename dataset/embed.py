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
    wav = wav[:, a:b]                               
    wav = wav.mean(dim=0, keepdim=True)              
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav                                     

class NpzCache:
    def __init__(self, name: str):
        self.path = config.EMB_CACHE / f"{name}.npz"
        self.data: dict[str, np.ndarray] = {}
        if self.path.exists():
            with np.load(self.path) as z:
                self.data = {k: z[k] for k in z.files}

    def has(self, key: str) -> bool:
        return key in self.data

    def put(self, key: str, vec: np.ndarray) -> None:
        self.data[key] = vec.astype(np.float32)

    def flush(self) -> None:
        np.savez_compressed(self.path, **self.data)

class ClapEncoder:
    sr = config.SR_CLAP

    def __init__(self):
        import laion_clap
        self.m = laion_clap.CLAP_Module(enable_fusion=False, device=config.DEVICE)
        self.m.load_ckpt(config.CLAP_CKPT)

    @torch.no_grad()
    def embed(self, wav_1xT: torch.Tensor) -> np.ndarray:
        x = wav_1xT.reshape(1, -1).cpu().numpy()          
        emb = self.m.get_audio_embedding_from_data(x=x, use_tensor=False)
        return np.asarray(emb).reshape(-1)


class MuqEncoder:
    sr = config.SR_MUQ

    def __init__(self):
        from muq import MuQMuLan
        self.m = MuQMuLan.from_pretrained(config.MUQ_CKPT).to(config.DEVICE).eval()

    @torch.no_grad()
    def embed(self, wav_1xT: torch.Tensor) -> np.ndarray:
        wav = wav_1xT.to(config.DEVICE)
        emb = self.m(wavs=wav)                         
        return emb.detach().cpu().numpy().reshape(-1)


class W2VFrEncoder:
    sr = config.SR_W2V

    def __init__(self):
        from transformers import Wav2Vec2Model
        self.m = Wav2Vec2Model.from_pretrained(config.W2V_FR_CKPT).to(config.DEVICE).eval()

    @torch.no_grad()
    def embed(self, wav_1xT: torch.Tensor) -> np.ndarray:
        x = wav_1xT.to(config.DEVICE)
        x = (x - x.mean()) / (x.std() + 1e-7)      
        out = self.m(x).last_hidden_state             
        return out.mean(dim=1).detach().cpu().numpy().reshape(-1)


ENCODERS = {"clap": ClapEncoder, "muq": MuqEncoder, "w2v": W2VFrEncoder}
VIEWS = {
    "clap": [("audio_path", "mix"), ("accomp_path", "accomp")],
    "muq":  [("audio_path", "mix"), ("accomp_path", "accomp")],
    "w2v":  [("vocals_path", "voc")],
}


def run_encoder(name: str, tracks, segments) -> None:
    enc = ENCODERS[name]()
    cache = NpzCache(name)
    views = VIEWS[name]
    tinfo = tracks.set_index("track_id")

    n_new = 0
    for _, s in segments.iterrows():
        tid = s["track_id"]
        if tid not in tinfo.index:
            continue
        trow = tinfo.loc[tid]
        for path_col, view in views:
            key = f"{s['segment_id']}::{view}"
            if cache.has(key):
                continue
            path = trow[path_col]
            if not isinstance(path, str) or not path:
                continue
            try:
                wav = load_segment(path, s["start_sec"], s["end_sec"], enc.sr)
                cache.put(key, enc.embed(wav))
                n_new += 1
            except Exception as e:  # noqa: BLE001
                print(f"  échec {name} {key} : {e}")
        if n_new and n_new % 50 == 0:
            cache.flush()
    cache.flush()
    print(f"[{name}] {n_new} nouveaux embeddings -> {cache.path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", nargs="+", default=["clap", "muq", "w2v"],
                    choices=list(ENCODERS))
    args = ap.parse_args()

    tracks = manifest.load_tracks()
    segments = manifest.load_segments()

    for name in args.encoders:
        run_encoder(name, tracks, segments)

    seg_tids = set(segments["track_id"])
    for i, r in tracks.iterrows():
        if r["track_id"] in seg_tids and r["status"] == "segmented":
            tracks.at[i, "status"] = "embedded"
    manifest.save_tracks(tracks)


if __name__ == "__main__":
    main()
