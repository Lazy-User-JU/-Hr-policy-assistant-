import re
from dataclasses import dataclass

import numpy as np

from app import config, store
from app.embeddings import embed

RRF_K = 60

STOPWORDS = set(
    "a about all am an and any are as at be been but by can could did do does for from get got "
    "had has have how i if in into is it its me many may much my no not of on or our please "
    "shall should so tell that the their them there this to up us was we were what when where "
    "which who why will with would you your".split()
)


@dataclass
class Retrieval:
    hits: list[dict]
    best_cosine: float
    best_coverage: float
    weak: bool
    has_documents: bool = True


def content_words(text: str) -> list[str]:
    seen = dict.fromkeys(re.findall(r"[a-z0-9]+", text.lower()))
    return [word for word in seen if word not in STOPWORDS]


def stem(word: str) -> str:
    for ending in ("ments", "ment", "ing", "ies", "ied", "ed", "es", "s"):
        if word.endswith(ending) and len(word) - len(ending) >= 3:
            word = word[: -len(ending)]
            if ending in ("ies", "ied"):
                word += "y"
            break
    return word.rstrip("e")


def stems(text: str) -> set[str]:
    return {stem(word) for word in content_words(text)}


def coverage(question: str, text: str) -> float:
    wanted = stems(question)
    return len(wanted & stems(text)) / len(wanted) if wanted else 0.0


def retrieve(question: str) -> Retrieval:
    chunks = store.active_chunks()
    if not chunks:
        return Retrieval([], 0.0, 0.0, True, has_documents=False)

    matrix = np.vstack([np.frombuffer(chunk["embedding"], dtype="float32") for chunk in chunks])
    cosines = matrix @ embed([question])[0]
    by_id = {chunk["id"]: (chunk, float(score)) for chunk, score in zip(chunks, cosines)}

    limit = config.CANDIDATES_PER_SEARCH
    vector_ids = [chunks[i]["id"] for i in np.argsort(-cosines)[:limit]]
    keyword_ids = store.keyword_search(content_words(question), limit)
    if config.SEARCH_MODE == "vector":
        keyword_ids = []
    elif config.SEARCH_MODE == "keyword":
        vector_ids = []

    fused: dict[int, float] = {}
    for ranking in (vector_ids, keyword_ids):
        for rank, chunk_id in enumerate(ranking):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1 / (RRF_K + rank + 1)

    hits = []
    for chunk_id in sorted(fused, key=fused.get, reverse=True)[: config.TOP_K]:
        chunk, cosine = by_id[chunk_id]
        hits.append({
            **chunk,
            "cosine": cosine,
            "coverage": coverage(question, chunk["search_text"]),
        })

    if not hits:
        return Retrieval([], 0.0, 0.0, True)

    best_cosine = max(hit["cosine"] for hit in hits)
    best_coverage = max(hit["coverage"] for hit in hits)
    weak = best_cosine < config.MIN_COSINE and best_coverage < config.MIN_COVERAGE
    return Retrieval(hits, best_cosine, best_coverage, weak)
