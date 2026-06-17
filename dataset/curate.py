from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import config
from manifest import make_track_id, _norm


def from_spotify_playlists(playlist_urls: list[str]) -> list[dict]:
    import spotipy
    from spotipy.oauth2 import SpotifyClientCredentials

    sp = spotipy.Spotify(
        auth_manager=SpotifyClientCredentials(),  # lit SPOTIPY_CLIENT_ID/SECRET
        requests_timeout=15,
        retries=3,
    )
    rows: list[dict] = []
    for url in playlist_urls:
        pid = url.rstrip("/").split("/")[-1].split("?")[0]
        results = sp.playlist_items(pid, additional_types=("track",))
        while results:
            for it in results["items"]:
                tr = it.get("track") or {}
                if not tr.get("name") or not tr.get("artists"):
                    continue
                rows.append({
                    "artist": tr["artists"][0]["name"],
                    "title": tr["name"],
                    "collective": None,
                })
            results = sp.next(results) if results.get("next") else None
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
    ap.add_argument("--playlists", nargs="*", default=[], help="URLs de playlists Spotify publiques")
    ap.add_argument("--seeds", type=Path, default=None, help="CSV artist,title[,collective]")
    args = ap.parse_args()

    rows: list[dict] = []
    if args.playlists:
        rows += from_spotify_playlists(args.playlists)
    if args.seeds and args.seeds.exists():
        rows += from_seeds_csv(args.seeds)

    df = pd.DataFrame(rows, columns=["artist", "title", "collective"])
    df["track_id"] = [make_track_id(a, t) for a, t in zip(df["artist"], df["title"])]

    df["_k"] = [f"{_norm(a)}::{_norm(t)}" for a, t in zip(df["artist"], df["title"])]
    df = df.drop_duplicates("_k").drop(columns="_k").reset_index(drop=True)

    df.to_parquet(config.CURATED_LIST, index=False)
    print(f"{len(df)} unique tracks curated -> {config.CURATED_LIST}")


if __name__ == "__main__":
    main()
