import hashlib
import logging

from app import store
from app.chunking import chunk_document
from app.embeddings import embed
from app.parsing import UnreadableDocument, read_document
from app.schemas import UploadResponse

logger = logging.getLogger(__name__)


class IndexingFailed(Exception):
    pass


def index_document(name: str, data: bytes) -> UploadResponse:
    sha256 = hashlib.sha256(data).hexdigest()
    current = store.active_version(name)
    if current and current["sha256"] == sha256:
        chunks = next(d["chunks"] for d in store.list_documents() if d["name"] == name)
        return UploadResponse(status="unchanged", document=name, version=current["version"], chunks=chunks)

    document = read_document(name, data)
    chunks = chunk_document(document)
    if not chunks:
        raise UnreadableDocument("The file has no readable text.")

    try:
        vectors = embed([chunk.search_text for chunk in chunks])
    except Exception as error:
        logger.exception("Embedding failed for %s", name)
        raise IndexingFailed("Could not index the file. The previous version is still live.") from error

    version = store.save_document(name, document.title, document.applies_to, sha256, chunks, vectors)
    return UploadResponse(status="indexed", document=name, version=version, chunks=len(chunks))
