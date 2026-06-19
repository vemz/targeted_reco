from __future__ import annotations

import argparse
import re

import pandas as pd
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import dataset.config as config
import dataset.manifest as manifest

_PROD_RE = re.compile(
    r"prod(?:uced)?\.?\s*(?:by|par|:)?\s*([A-Za-z0-9 ._&'-]{2,40})",
    flags=re.IGNORECASE,
)

def parse_producer(*texts: str) -> str | None:
    for t in texts:
        if not t:
            continue
        m = _PROD_RE.search(t)
        if m:
            name = m.group(1).strip(" .-_&")
            name = re.split(r"[|/]|\s[-–]\s", name)[0].strip()
            if name:
                return name
    return None


def acquire_one(artist: str, title: str, source_url: str | None) -> dict | None:
    import yt_dlp

    out_tmpl = str(config.AUDIO_RAW / "%(id)s.%(ext)s")
    ydl_opts = {
        "format": config.YTDLP_AUDIO_FORMAT,
        "outtmpl": out_tmpl,
        "quiet": True,
        "no_warnings": True,
        "postprocessors": [{
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "192",
                }],
        "noplaylist": True,
        "sleep_interval": 2,
        "max_sleep_interval": 5,
    }
    target = source_url or f"ytsearch1:{artist} {title}"
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(target, download=True)
        if "entries" in info:          
            info = info["entries"][0]

    dur = float(info.get("duration") or 0)
    if not (config.MIN_TRACK_DURATION_SEC <= dur <= config.MAX_TRACK_DURATION_SEC):
        print(f"  skip (durée {dur:.0f}s hors bornes) : {artist} - {title}")
        return None

    audio_path = config.AUDIO_RAW / f"{info['id']}.mp3"
    producer = parse_producer(info.get("title", ""), info.get("description", ""))
    platform = "soundcloud" if "soundcloud" in (info.get("extractor") or "") else "youtube"

    return {
        "artist": artist,
        "title": title,
        "producer": producer,
        "source_platform": platform,
        "source_url": info.get("webpage_url"),
        "audio_path": str(audio_path),
        "duration_sec": dur,
        "status": "downloaded",
    }

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    curated = pd.read_parquet(config.CURATED_LIST)
    tracks = manifest.load_tracks()
    done = set(tracks.loc[tracks["status"] != "curated", "track_id"]) if len(tracks) else set()

    todo = curated[~curated["track_id"].isin(done)]
    if args.limit:
        todo = todo.head(args.limit)

    for _, r in todo.iterrows():
        print(f"-> {r['artist']} - {r['title']}")
        try:
            rec = acquire_one(r["artist"], r["title"], r.get("source_url"))
        except Exception as e:  
            print(f"  échec yt-dlp : {e}")
            continue
        if rec is None:
            continue
        rec["track_id"] = r["track_id"]
        rec["collective"] = r.get("collective")
        tracks = manifest.upsert_track(tracks, rec)
        manifest.save_tracks(tracks) 

    print(f"Terminé. {len(manifest.load_tracks())} morceaux dans le manifeste.")


if __name__ == "__main__":
    main()
