# Design: HR Policy Assistant

The main requirement is not "answer questions". It is "never state a policy that does not exist". Most decisions here favour a refusal over a risky answer.

## 1. Architecture

```text
UPLOAD                                    ASK
file (.md .txt .pdf)                      question
   |                                         |
parsing.py    sections, tables            embed + keyword search
   |                                         |
chunking.py   section chunks, table rows  retrieval.py  rank fusion, top 6
   |                                         |
embeddings.py MiniLM vectors              gate: is retrieval weak? --yes--> refuse
   |                                         | no
store.py      SQLite, one transaction     llm.py  Gemini, JSON schema reply
                                             |    (failure --> quote-only answer)
                                          answer.py  check quotes and numbers
                                             |                 |
                                          answer + citations   refuse
```

Upload and query share only the SQLite file. Upload runs in the request, and FastAPI runs it in a worker thread, so other requests keep working.

Failure paths:

| Failure | Result |
| --- | --- |
| Empty or binary file, scanned PDF | `422` with a clear message, nothing stored |
| Embedding fails during upload | `503`, previous version stays live |
| Rate limit | Retries twice, with a short wait |
| LLM timeout, lasting quota error or bad JSON | Falls back to a verbatim quote |
| Provider rejects the JSON schema | Retries without it, and `ModelReply` still checks the reply |
| Model answer fails a check | Refusal, because this is a grounding failure and not an outage |
| Empty question | `422` |
| No documents uploaded | Refusal with reason `no_documents` |

## 2. Chunking and retrieval

**Chunks follow the document, not a character count.** Chunks never cross a heading, because a clause and its exceptions usually sit together. For example, section 4.1 of the leave policy has the 8-day limit in one paragraph and the 31 March deadline in the next. A fixed-size splitter could separate them. Long sections split on sentences at 120 words, which fits inside MiniLM's 256-token limit, and repeat one sentence across the split.

**Each table row is one chunk and repeats the column names:**

```text
Health tier: Standard | Eligible bands: Band B, Band C | ... | Dental implants: Not covered
```

A plain splitter would separate "Not covered" from "Standard" and "Dental implants". That is how a system gives a confident wrong answer about the wrong tier.

**Metadata stored:** document name, title, version, heading path, chunk type, SHA-256 of the file, and who the policy applies to. The title and heading path are added to the text before embedding and keyword indexing, so "casual leave carry-forward" also matches through the heading. The "Applies to" line goes to the model with each excerpt, so a rule for full-time staff is not given to a contractor.

**Ranking.** Vector search finds meaning ("roll over" and "carry forward"). BM25 through SQLite FTS5 finds exact terms such as "LTA" or "Band C", which embeddings handle poorly. Each returns 20 candidates. They are merged with reciprocal rank fusion, which uses rank instead of score because cosine and BM25 scores are on different scales. The top 6 go to the answer step. Six is enough for a question that spans a table, since sibling rows usually rank close together.

**Choosing the numbers.** `python -m eval.run --calibrate` shows what each `MIN_COSINE` would block, and suggests the highest value that blocks no answerable question. `python -m eval.run --compare` measures hybrid search against vector-only and keyword-only, so hybrid has to earn its place with a number.

## 3. Grounding

Refusal is a layered check, not one prompt instruction. Each layer catches what the last one misses.

1. **Retrieval gate.** If the best cosine is under 0.30 and under half the question's words appear in the best chunk, we refuse without calling the LLM. Both must be low, so a paraphrase with no shared words can still pass on meaning. This stops "What is the capital of France?" early and saves quota.
2. **Prompt.** The model may use only the numbered excerpts. It must not calculate, must copy figures exactly, and treats excerpts as reference text, not instructions.
3. **Schema.** The provider is sent a JSON schema with `answerable`, `answer` and `evidence[]`. Pydantic checks the shape again on our side.
4. **Citation check.** Each quote must really appear in the excerpt it names. The citation's document and section come from the database, so the model cannot invent a source.
5. **Number check.** Every number in the answer must appear in a quote. This catches a wrong limit next to a real citation.
6. **Right row.** If the question names a table row, such as "Standard", a citation from a different row is refused. Only words that tell rows apart count, so "Band" is ignored and "C" is not.
7. **Right group.** If the question names a group such as contractors, and the cited text and its "Applies to" line never mention that group, it is refused.
8. **Contradiction.** An answer that starts with "Yes" and rests on a negative quote, such as "cannot be encashed", is refused.

Layers 4 to 8 are plain code, so they work the same with any model.

Without an LLM, or when one fails, the service quotes the best sentence or table cell word for word. It needs 70% of the question's words in that sentence or row, and refuses otherwise. This mode is deliberately cautious: it refuses more often, but it cannot invent text.

**The home gym case.** The brief expects a refusal for "Can I expense a personal home gym?". The sample benefits policy has a clause for this: section 4 says gym or home-fitness equipment is not a reimbursable benefit. With an LLM, I let the system say no, cite section 4 and point to HR, because hiding a real clause would not help the employee. It invents nothing and approves nothing. Quote-only mode refuses this question because "expense" is not in the policy, so both modes stay safe, and the eval accepts either. A true unknown such as a work-from-home stipend is refused.

## 4. Schema and APIs

`POST /ask` takes `{"question": "..."}` and returns:

```json
{
  "status": "answered | refused",
  "answer": "text",
  "citations": [{"document": "", "version": 1, "section": "", "quote": ""}],
  "refusal_reason": "weak_retrieval | model_refused | quote_not_found | ...",
  "answered_by": "llm | extractive"
}
```

- `status` lets a client handle a refusal without parsing text.
- `citations` is always a list, and is empty on refusal.
- `quote` lets a reader check the claim in seconds.
- `version` shows which version of a policy the answer used.
- `refusal_reason` makes refusals measurable, so we can tell a weak search from a failed check.

`POST /documents` is multipart and returns `status`, `document`, `version` and `chunks`. Uploading a file with an existing name creates a new version in one transaction. Identical content is detected by hash and skipped. Other routes are listed in the README.

## 5. Trade-offs

**Rejected: a separate vector database (Chroma, pgvector).** SQLite gives me vectors and FTS5 in one file, so one transaction can swap a document version across both. With Chroma plus a keyword index, I would have to keep two stores in sync. Brute-force search with NumPy is fine at this size. It would need replacing at a much larger scale, but a set of HR policies stays small.

**Rejected: fixed-size chunks with overlap.** Simple, but it cuts tables and clauses in the wrong places, and overlap does not fix a split table.

**Rejected: LangChain or LlamaIndex.** The whole app is about 1,000 lines, and I want the gates and checks visible. A framework would hide the parts this project is judged on. Calling the OpenAI-format API directly also means changing provider takes three settings.

**Rejected: letting the model write citations.** A model can name a section that does not exist. It returns an excerpt number and quote instead, and the server builds the citation.

**Rejected: async ingestion with a queue.** A policy file indexes quickly, and a queue adds job status and failure handling that this project does not need. It becomes worthwhile with large PDF batches.

**Rejected: a cross-encoder reranker.** It could improve ranking, but it adds a model and latency. I would add it only if the eval showed the right section missing from the top 6.

**Known limits:** the checks prove that the evidence exists, not that the reasoning is right, so a model could still misread a real clause. The system does no arithmetic, so a pro-rata leave question gets a refusal. The number check is strict, so "5 lakh" for ₹5,00,000 is refused, which is the safe direction. The group check uses a short word list. Tables split across PDF pages lose their header on the second page, and those rows are read as plain text. The search is English only.

## 6. Evaluation

`eval/questions.json` has 41 questions: direct facts, table cells, paraphrases, unanswerable questions, and held-out policies in Markdown and plain text that the code was not built around. Each run reports answers, citations, false answers, refusal reasons, right-section recall and mean reciprocal rank.

A false answer on an unanswerable question counts as the worst failure, so it has its own line. The eval also warns if rate limits pushed questions to the fallback, so a quote-only result cannot pass as an LLM result.

The latest numbers and the exact setup are in [eval/results.md](eval/results.md), which each full run rewrites.

## 7. With two more weeks

1. **Eval first.** Grow the set to about 100 questions, including real employee wording, and run it on every change. Every other improvement needs this to prove it helped.
2. **Feedback.** Add thumbs up or down, and store the question, chunks and refusal reason, so bad answers become new eval cases.
3. **Better PDF tables.** Carry headers across pages and handle merged cells.
4. **Real auth and audit.** Add SSO and record which document version produced each answer.
5. **Conflict handling.** Detect when two live policies disagree, and refuse or show both.
