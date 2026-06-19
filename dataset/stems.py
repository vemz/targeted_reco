from __future__ import annotations

import pandas as pd
import torch
import torchaudio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
import manifest


def _separator():
    from demucs.pretrained import get_model
    model = get_model(config.DEMUCS_MODEL)     
    model.to(config.DEVICE).eval()
    return model


def separate_track(model, audio_path: str) -> tuple[str, str]:
    from demucs.apply import apply_model
    from demucs.audio import AudioFile

    wav = AudioFile(audio_path).read(
        streams=0,
        samplerate=model.samplerate,
        channels=model.audio_channels,
    )
    ref = wav.mean(0)
    wav = (wav - ref.mean()) / (ref.std() + 1e-8)   

    with torch.no_grad():
        sources = apply_model(model, wav[None], device=config.DEVICE, progress=False)[0]
    sources = sources * ref.std() + ref.mean()      

    stems = dict(zip(model.sources, sources))         
    vocals = stems["vocals"]
    accomp = sum(v for k, v in stems.items() if k != "vocals")

    base = Path(audio_path).stem
    vpath = config.STEMS / f"{base}.vocals.flac"
    apath = config.STEMS / f"{base}.accomp.flac"
    torchaudio.save(str(vpath), vocals.cpu(), model.samplerate)
    torchaudio.save(str(apath), accomp.cpu(), model.samplerate)
    return str(vpath), str(apath)


def main() -> None:
    tracks = manifest.load_tracks()
    todo = tracks[tracks["status"] == "downloaded"]
    if not len(todo):
        print("Rien à séparer (status='downloaded' vide).")
        return

    sep = _separator()
    for _, r in todo.iterrows():
        print(f"-> séparation : {r['artist']} - {r['title']}")
        try:
            vpath, apath = separate_track(sep, r["audio_path"])
        except Exception as e:  
            print(f"  échec Demucs : {e}")
            continue
        row = r.to_dict()
        row["vocals_path"] = vpath
        row["accomp_path"] = apath
        row["status"] = "separated"
        tracks = manifest.upsert_track(tracks, row)
        manifest.save_tracks(tracks)


if __name__ == "__main__":
    main()
