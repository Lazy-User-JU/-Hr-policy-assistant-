from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app import config, store
from app.answer import answer_question
from app.ingest import IndexingFailed, index_document
from app.parsing import SUPPORTED_TYPES, UnreadableDocument
from app.schemas import AskRequest, AskResponse, UploadResponse

INDEX_PAGE = Path(__file__).parent / "static" / "index.html"


@asynccontextmanager
async def lifespan(app: FastAPI):
    store.init()
    yield


app = FastAPI(title="HR Policy Assistant", lifespan=lifespan)


def require_admin(x_role: str = Header(default="employee")) -> None:
    if x_role != "admin":
        raise HTTPException(403, "Only an admin can change documents.")


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(INDEX_PAGE)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/documents", response_model=UploadResponse, status_code=201, dependencies=[Depends(require_admin)])
def upload_document(file: UploadFile):
    name = Path(file.filename or "").name
    if Path(name).suffix.lower() not in SUPPORTED_TYPES:
        raise HTTPException(415, "Upload a .md, .txt or .pdf file.")

    data = file.file.read(config.MAX_UPLOAD_BYTES + 1)
    if len(data) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(413, "The file is larger than 5 MB.")
    if not data:
        raise HTTPException(422, "The file is empty.")

    try:
        return index_document(name, data)
    except UnreadableDocument as error:
        raise HTTPException(422, str(error)) from error
    except IndexingFailed as error:
        raise HTTPException(503, str(error)) from error


@app.get("/documents")
def list_documents():
    return store.list_documents()


@app.delete("/documents/{name}", status_code=204, dependencies=[Depends(require_admin)])
def delete_document(name: str):
    if not store.deactivate_document(name):
        raise HTTPException(404, "No such document.")


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    return answer_question(request.question)
