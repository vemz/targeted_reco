from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

AUDIO_RAW = DATA / "audio_raw"         
STEMS = DATA / "stems"               
SEGMENTS = DATA / "segments"        
EMB_CACHE = DATA / "embeddings"         
MANIFEST_DIR = DATA / "manifests"      

for _p in (AUDIO_RAW, STEMS, SEGMENTS, EMB_CACHE, MANIFEST_DIR):
    _p.mkdir(parents=True, exist_ok=True)

TRACKS_MANIFEST = MANIFEST_DIR / "tracks.parquet"
SEGMENTS_MANIFEST = MANIFEST_DIR / "segments.parquet"
CURATED_LIST = MANIFEST_DIR / "curated.parquet"  

DEVICE = "mps"       

SEGMENT_SEC = 7.0          
SEGMENT_HOP_SEC = 7.0     
MAX_SEGMENTS_PER_TRACK = 6  

VOCAL_ENERGY_RATIO_MIN = 0.10
TRACK_VOCAL_FRACTION_MIN = 0.05

SR_CLAP = 48000
SR_MUQ = 24000
SR_W2V = 16000      

CLAP_CKPT = "630k-audioset-best.pt"         
MUQ_CKPT = "OpenMuQ/MuQ-MuLan-large"
W2V_FR_CKPT = "LeBenchmark/wav2vec2-FR-7K-large" 

DEMUCS_MODEL = "htdemucs"    

YTDLP_AUDIO_FORMAT = "bestaudio/best"
DOWNLOAD_SR = 44100   
MAX_TRACK_DURATION_SEC = 600   
MIN_TRACK_DURATION_SEC = 45 
