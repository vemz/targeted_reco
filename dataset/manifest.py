from __future__ import annotations

import pandas as pd

import config

TRACK_COLUMNS = [
    "track_id",        
    "artist",
    "title",
    "producer",       
    "collective",    
    "source_platform", 
    "source_url",
    "audio_path",    
    "vocals_path",  
    "accomp_path",     
    "duration_sec",
    "has_vocals",   
    "vocal_fraction", 
    "status",        
]

SEGMENT_COLUMNS = [
    "segment_id",      
    "track_id",
    "start_sec",
    "end_sec",
    "vocal_energy",   
    "split",          
]


def make_track_id(artist: str, title: str) -> str:
    import hashlib
    key = f"{_norm(artist)}::{_norm(title)}"
    return "trk_" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


def _norm(s: str) -> str:
    import re
    s = (s or "").lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def load_tracks() -> pd.DataFrame:
    if config.TRACKS_MANIFEST.exists():
        return pd.read_parquet(config.TRACKS_MANIFEST)
    return pd.DataFrame(columns=TRACK_COLUMNS)


def save_tracks(df: pd.DataFrame) -> None:
    df = df.reindex(columns=TRACK_COLUMNS)
    df.to_parquet(config.TRACKS_MANIFEST, index=False)


def load_segments() -> pd.DataFrame:
    if config.SEGMENTS_MANIFEST.exists():
        return pd.read_parquet(config.SEGMENTS_MANIFEST)
    return pd.DataFrame(columns=SEGMENT_COLUMNS)


def save_segments(df: pd.DataFrame) -> None:
    df = df.reindex(columns=SEGMENT_COLUMNS)
    df.to_parquet(config.SEGMENTS_MANIFEST, index=False)


def upsert_track(df: pd.DataFrame, row: dict) -> pd.DataFrame:
    tid = row["track_id"]
    df = df[df["track_id"] != tid]
    new = pd.DataFrame([row]).reindex(columns=TRACK_COLUMNS)
    return new if df.empty else pd.concat([df, new], ignore_index=True)
