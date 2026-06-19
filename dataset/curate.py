from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import dataset.config as config
from dataset.manifest import make_track_id, _norm


def from_ytdlp_playlists(playlist_urls: list[str]) -> list[dict]:
    import yt_dlp
    rows: list[dict] = []
    opts = {
        "quiet": True,
        "skip_download": True,
        "extract_flat": False,  
        "ignoreerrors": True,   
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        for url in playlist_urls:
            info = ydl.extract_info(url, download=False)
            for e in (info.get("entries") or []):
                if not e:                
                    continue
                title = (e.get("title") or "").strip()
                artist, sep, t = title.partition(" - ")
                if sep:
                    artist, title = artist.strip(), t.strip()
                else:
                    artist = (e.get("uploader") or e.get("channel") or "").strip()
                if not title:
                    continue
                rows.append({
                    "artist": artist,
                    "title": title,
                    "collective": None,
                    "source_url": e.get("webpage_url") or e.get("url"),
                })
    return rows


def from_seeds_csv(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("artist") and r.get("title"):
                rows.append({
                    "artist": r["artist"].strip(),
                    "title": r["title"].strip(),
                    "collective": (r.get("collective") or None),
                })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--playlists", nargs="*", default=[], help="URLs de playlists yt / sc")
    ap.add_argument("--seeds", type=Path, default=None, help="CSV artist,title[,collective]")
    args = ap.parse_args()

    rows: list[dict] = []
    if args.playlists:
        rows += from_ytdlp_playlists(args.playlists)
    if args.seeds and args.seeds.exists():
        rows += from_seeds_csv(args.seeds)

    df = pd.DataFrame(rows, columns=["artist", "title", "collective", "source_url"])
    df["track_id"] = [make_track_id(a, t) for a, t in zip(df["artist"], df["title"])]

    df["_k"] = [f"{_norm(a)}::{_norm(t)}" for a, t in zip(df["artist"], df["title"])]
    df = df.drop_duplicates("_k").drop(columns="_k").reset_index(drop=True)

    df.to_parquet(config.CURATED_LIST, index=False)
    print(f"{len(df)} unique tracks curated -> {config.CURATED_LIST}")


if __name__ == "__main__":
    main()
