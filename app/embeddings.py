import re
import zlib

import numpy as np

from app import config

_model = None
_loaded_name = None
HASHING_DIMENSIONS = 1024


def embed(texts: list[str]) -> np.ndarray:
    if config.EMBEDDING_MODEL == "hashing":
        return np.vstack([hashed_vector(text) for text in texts])

    global _model, _loaded_name
    if _model is None or _loaded_name != config.EMBEDDING_MODEL:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(config.EMBEDDING_MODEL)
        _loaded_name = config.EMBEDDING_MODEL
    return _model.encode(texts, normalize_embeddings=True, convert_to_numpy=True).astype("float32")


def hashed_vector(text: str) -> np.ndarray:
    vector = np.zeros(HASHING_DIMENSIONS, dtype="float32")
    for word in re.findall(r"[a-z0-9]+", text.lower()):
        features = [word] + [word[i : i + 3] for i in range(len(word) - 2)]
        for feature in features:
            vector[zlib.crc32(feature.encode()) % HASHING_DIMENSIONS] += 1.0
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector
