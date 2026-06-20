from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
import torch
import torchaudio

import config
import manifest


def _rms_envelope(wav: torch.Tensor, sr: int, win_sec: float) -> np.ndarray:
    mono = wav.mean(dim=0)                      
    win = max(1, int(win_sec * sr))
    n = mono.shape[0] // win
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    trimmed = mono[: n * win].reshape(n, win)
    rms = trimmed.pow(2).mean(dim=1).sqrt()
    return rms.cpu().numpy().astype(np.float32)


def segment_track(track_id: str, audio_path: str, vocals_path: str):
    mix, sr = torchaudio.load(audio_path)
    voc, srv = torchaudio.load(vocals_path)
    if srv != sr:                              
        voc = torchaudio.functional.resample(voc, srv, sr)

    rms_mix = _rms_envelope(mix, sr, config.SEGMENT_HOP_SEC)
    rms_voc = _rms_envelope(voc, sr, config.SEGMENT_HOP_SEC)
    n = min(len(rms_mix), len(rms_voc))
    if n == 0:
        return [], False, 0.0

    eps = 1e-8
    ratio = rms_voc[:n] / (rms_mix[:n] + eps)
    active = ratio > config.VOCAL_ENERGY_RATIO_MIN
    vocal_fraction = float(active.mean())
    has_vocals = vocal_fraction >= config.TRACK_VOCAL_FRACTION_MIN

    idx_active = np.where(active)[0]
    idx_active = idx_active[np.argsort(-ratio[idx_active])][: config.MAX_SEGMENTS_PER_TRACK]
    idx_active.sort()

    segs = []
    for j, w in enumerate(idx_active):
        start = w * config.SEGMENT_HOP_SEC
        segs.append({
            "segment_id": f"{track_id}_{j:03d}",
            "track_id": track_id,
            "start_sec": round(float(start), 3),
            "end_sec": round(float(start + config.SEGMENT_SEC), 3),
            "vocal_energy": round(float(ratio[w]), 4),
            "split": None,
        })
    return segs, has_vocals, round(vocal_fraction, 4)


def main() -> None:
    tracks = manifest.load_tracks()
    print("DEBUG ->", config.TRACKS_MANIFEST,
          "| rows:", len(tracks),
          "| separated:", int((tracks["status"] == "separated").sum()) if len(tracks) else 0)
    segments = manifest.load_segments()
    todo = tracks[tracks["status"] == "separated"]
    if not len(todo):
        print("Rien à segmenter (status='separated' vide).")
        return

    for _, r in todo.iterrows():
        segs, has_vocals, frac = segment_track(r["track_id"], r["audio_path"], r["vocals_path"])
        # maj track
        row = r.to_dict()
        row["has_vocals"] = has_vocals
        row["vocal_fraction"] = frac
        row["status"] = "segmented"
        tracks = manifest.upsert_track(tracks, row)

        if has_vocals and segs:
            segments = segments[segments["track_id"] != r["track_id"]]   
            segments = pd.concat([segments, pd.DataFrame(segs)], ignore_index=True)
            print(f"-> {r['artist']} - {r['title']} : {len(segs)} segments (voix {frac:.0%})")
        else:
            print(f"-> {r['artist']} - {r['title']} : SANS VOIX, écarté (voix {frac:.0%})")

        manifest.save_tracks(tracks)
        manifest.save_segments(segments)


if __name__ == "__main__":
    main()
