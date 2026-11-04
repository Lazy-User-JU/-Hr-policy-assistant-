import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DB_PATH = Path(os.getenv("DB_PATH", "data/hr.db"))
MAX_UPLOAD_BYTES = 5 * 1024 * 1024

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.6-flash")
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "30"))

MAX_CHUNK_WORDS = 120
CANDIDATES_PER_SEARCH = 20
TOP_K = int(os.getenv("TOP_K", "6"))

SEARCH_MODE = os.getenv("SEARCH_MODE", "hybrid")

MIN_COSINE = float(os.getenv("MIN_COSINE", "0.30"))
MIN_COVERAGE = float(os.getenv("MIN_COVERAGE", "0.50"))

EXTRACTIVE_MIN_COVERAGE = 0.70
