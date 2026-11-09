"""Environment-backed configuration for local KERNELGPT models."""
import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR / ".env")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").strip().rstrip("/")
CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen3:4b").strip()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "embeddinggemma").strip()
VISION_MODEL = os.getenv("VISION_MODEL", "qwen3-vl:2b").strip()
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
MAX_PDF_SIZE_MB = int(os.getenv("MAX_PDF_SIZE_MB", "50"))
TOP_K = int(os.getenv("TOP_K", "3"))
MAX_COSINE_DISTANCE = float(os.getenv("MAX_COSINE_DISTANCE", "0.55"))
DATA_DIR = Path(os.getenv("DATA_DIR", str(PROJECT_DIR / "data")))
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", str(PROJECT_DIR / "chroma_db")))

if CHUNK_SIZE < 100:
    raise ValueError("CHUNK_SIZE must be at least 100 characters.")
if CHUNK_OVERLAP < 0 or CHUNK_OVERLAP >= CHUNK_SIZE:
    raise ValueError("CHUNK_OVERLAP must be >= 0 and smaller than CHUNK_SIZE.")
if MAX_PDF_SIZE_MB < 1:
    raise ValueError("MAX_PDF_SIZE_MB must be at least 1.")
if TOP_K < 1:
    raise ValueError("TOP_K must be at least 1.")
if not 0.0 <= MAX_COSINE_DISTANCE <= 2.0:
    raise ValueError("MAX_COSINE_DISTANCE must be between 0 and 2.")
