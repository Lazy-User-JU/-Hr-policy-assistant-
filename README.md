# HR Policy Assistant

A small RAG service that answers employee questions from uploaded HR policies. Every answer cites the document and section it came from. If the policies do not contain the answer, it refuses and points to HR.

The design and its trade-offs are in [DESIGN.md](DESIGN.md).

## Run it locally

You need Python 3.10 or newer. Python 3.11 or 3.12 is the safest choice for PyTorch. The only key you need is a free Groq API key from https://console.groq.com/keys.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # Windows: copy .env.example .env
```

Paste your key into `LLM_API_KEY` in `.env`, then run:

```bash
python -m app.check              # checks the database, embeddings and Groq
python -m app.seed               # loads the three sample policies
uvicorn app.main:app --reload
```

Open http://localhost:8000 and ask a question. To add a new policy, use **Admin: manage policies** on that page.

The first run downloads the embedding model (about 90 MB).

On Linux, install the CPU build of PyTorch first to avoid a large GPU download:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## LLM and embedding path

| Part | Default | Cost |
| --- | --- | --- |
| Embeddings | `all-MiniLM-L6-v2` with sentence-transformers, running locally | Free, no key |
| Answers | Groq (`openai/gpt-oss-120b`) by default, through its OpenAI-compatible API | Free tier |
| No key set | Exact quotes from the policy, no LLM call | Free |

Ollama is optional. Any OpenAI-compatible endpoint works, including Gemini, OpenAI and Ollama: change `LLM_BASE_URL` and `LLM_MODEL`. Models and quotas change, so check your provider's current list.

Check your provider's data-retention terms before uploading real confidential policies.

Rate limits are retried twice. If the LLM still fails, the service falls back to quoting. The `answered_by` field in each response shows which path was used.

## Environment variables

| Name | Default | Purpose |
| --- | --- | --- |
| `LLM_API_KEY` | empty | Turns on the LLM. Empty means quote-only mode. |
| `LLM_BASE_URL` | Groq | OpenAI-compatible endpoint |
| `LLM_MODEL` | `openai/gpt-oss-120b` | Chat model |
| `LLM_TIMEOUT_SECONDS` | `30` | Time allowed before falling back to quoting |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Any sentence-transformers model, or `hashing` for no download |
| `DB_PATH` | `data/hr.db` | SQLite file |
| `TOP_K` | `6` | Chunks passed to the answer step |
| `SEARCH_MODE` | `hybrid` | `vector` and `keyword` are for comparing results |
| `MIN_COSINE` | `0.30` | Refusal gate on vector similarity |
| `MIN_COVERAGE` | `0.50` | Refusal gate on word overlap |

## API

Interactive docs are at http://localhost:8000/docs.

```bash
# Admin upload. A new file with the same name becomes the next version.
curl -X POST localhost:8000/documents -H "X-Role: admin" -F "file=@policies/leave-policy.md"

# Ask a question
curl -X POST localhost:8000/ask -H "Content-Type: application/json" \
  -d '{"question": "What is the casual leave carry-forward limit?"}'
```

```json
{
  "status": "answered",
  "answer": "A maximum of 8 days of unused casual leave may be carried forward to the next calendar year.",
  "citations": [
    {
      "document": "leave-policy.md",
      "version": 1,
      "section": "4. Carry-forward and encashment > 4.1 Casual leave carry-forward",
      "quote": "A maximum of 8 days of unused casual leave may be carried forward to the next calendar year."
    }
  ],
  "refusal_reason": null,
  "answered_by": "extractive"
}
```

| Method | Path | Who | Purpose |
| --- | --- | --- | --- |
| POST | `/documents` | admin | Upload or replace a `.md`, `.txt` or `.pdf` file |
| GET | `/documents` | anyone | List live documents |
| DELETE | `/documents/{name}` | admin | Remove a document from search |
| POST | `/ask` | anyone | Ask a question |
| GET | `/health` | anyone | Health check |

Admin is the header `X-Role: admin`. This is a stand-in, since real auth is out of scope.

## Tests and evaluation

```bash
pip install -r requirements-dev.txt
pytest
python -m eval.run --calibrate  # suggests MIN_COSINE, no API calls
python -m eval.run --compare    # hybrid vs vector vs keyword, no API calls
python -m eval.run --show-all   # full run, prints every question
python -m eval.run --no-llm     # exact-quote mode
```

The tests need no model or API key. The eval builds a temporary index from `policies/` and `eval/heldout/`, so it does not touch your data. The held-out files are policies the code was not built around.

A full run writes [eval/results.md](eval/results.md). It waits 5 seconds between LLM calls and uses about 35 requests, so it takes a few minutes. If it warns about fallbacks, you hit a rate limit, so try `--delay 10`.

## Project layout

```text
app/
  main.py         API routes and upload checks
  ingest.py       one path from file to index
  parsing.py      file to title, sections and tables
  chunking.py     blocks to chunks
  embeddings.py   text to vectors
  store.py        SQLite: documents, chunks, vectors, keyword index
  retrieval.py    hybrid search and the weak-retrieval gate
  llm.py          prompt and model call
  answer.py       refusal logic and answer checks
  schemas.py      request and response shapes
  check.py        setup check
  seed.py         loads the sample policies
  static/         one-page UI
eval/             questions, runner, held-out policies
policies/         sample policies
tests/
```
