import re

from app import config, llm
from app.chunking import split_sentences
from app.retrieval import coverage, retrieve, stems
from app.schemas import AskResponse, Citation, ModelReply

REFUSAL_TEXT = "I don't have enough information in the uploaded policies to answer this. Please contact HR."

POLICY_IS_SILENT = re.compile(r"\bpolicy does not\b", re.IGNORECASE)
NEGATIVE_WORD = re.compile(r"\b(?:not|cannot|never|no)\b|n't\b", re.IGNORECASE)

GROUP_WORDS = stems("contractor consultant freelancer intern trainee vendor")


def refuse(reason: str) -> AskResponse:
    return AskResponse(status="refused", answer=REFUSAL_TEXT, citations=[], refusal_reason=reason)


def answer_question(question: str) -> AskResponse:
    retrieval = retrieve(question)
    if not retrieval.has_documents:
        return refuse("no_documents")
    if retrieval.weak:
        return refuse("weak_retrieval")

    if llm.enabled():
        try:
            reply = llm.ask(question, retrieval.hits)
        except llm.LLMError:
            return extractive_answer(question, retrieval.hits)
        return check_model_reply(reply, retrieval.hits, question)
    return extractive_answer(question, retrieval.hits)


def check_model_reply(reply: ModelReply, hits: list[dict], question: str) -> AskResponse:
    if not reply.answerable:
        return refuse("model_refused")
    if not reply.answer.strip() or not reply.evidence:
        return refuse("no_evidence")

    citations = []
    for item in reply.evidence:
        if not 1 <= item.excerpt <= len(hits):
            return refuse("unknown_excerpt")
        hit = hits[item.excerpt - 1]
        if not normalise(item.quote) or normalise(item.quote) not in normalise(hit["text"]):
            return refuse("quote_not_found")
        if cites_wrong_row(question, hit, hits):
            return refuse("wrong_table_row")
        if outside_scope(question, hit):
            return refuse("outside_policy_scope")
        citations.append(make_citation(hit, item.quote))

    quoted = " ".join(item.quote for item in reply.evidence)
    if not numbers_in(reply.answer) <= numbers_in(quoted):
        return refuse("unsupported_number")
    if reply.answer.strip().lower().startswith("yes") and NEGATIVE_WORD.search(quoted):
        return refuse("contradicts_quote")

    return AskResponse(
        status="answered",
        answer=reply.answer.strip(),
        citations=unique(citations),
        answered_by="llm",
    )


def outside_scope(question: str, hit: dict) -> bool:
    asked = stems(question) & GROUP_WORDS
    return bool(asked - stems(f"{hit['applies_to']} {hit['section']} {hit['text']}"))


def row_name(hit: dict) -> str:
    return hit["text"].split(" | ")[0].partition(": ")[2]


def row_column(hit: dict) -> str:
    return hit["text"].split(":")[0]


def cites_wrong_row(question: str, cited: dict, hits: list[dict]) -> bool:
    if cited["kind"] != "table":
        return False

    rows = [
        hit for hit in hits
        if hit["kind"] == "table"
        and (hit["document"], hit["section"], row_column(hit))
        == (cited["document"], cited["section"], row_column(cited))
    ]
    wanted = stems(question)
    named = []
    for row in rows:
        others = set().union(*[stems(row_name(hit)) for hit in rows if hit is not row], set())
        if wanted & (stems(row_name(row)) - others):
            named.append(row)
    return bool(named) and cited not in named


def extractive_answer(question: str, hits: list[dict]) -> AskResponse:
    best, best_key, best_hit = None, (0.0, 0.0), None
    for hit in hits:
        parts = [hit["text"]] if hit["kind"] == "table" else split_sentences(hit["text"])
        for part in parts:
            scored_text = f"{hit['section']} {part}" if hit["kind"] == "table" else part
            key = (coverage(question, scored_text), coverage(question, f"{hit['section']} {part}"))
            if key > best_key:
                best, best_key, best_hit = part, key, hit

    if best is None or best_key[0] < config.EXTRACTIVE_MIN_COVERAGE:
        return refuse("no_exact_match")
    if POLICY_IS_SILENT.search(best):
        return refuse("policy_is_silent")
    if outside_scope(question, best_hit):
        return refuse("outside_policy_scope")

    quote = best_cells(question, best) if best_hit["kind"] == "table" else best
    return AskResponse(
        status="answered",
        answer=quote.replace(" | ", "; "),
        citations=[make_citation(best_hit, quote)],
        answered_by="extractive",
    )


def best_cells(question: str, row_text: str) -> str:
    cells = row_text.split(" | ")
    wanted = stems(question)
    overlap = [len(wanted & stems(cell.split(":")[0])) for cell in cells[1:]]
    if not overlap or max(overlap) == 0:
        return row_text
    return " | ".join(cells[:1] + [c for c, n in zip(cells[1:], overlap) if n == max(overlap)])


def make_citation(hit: dict, quote: str) -> Citation:
    return Citation(
        document=hit["document"],
        version=hit["version"],
        section=hit["section"] or None,
        quote=quote.strip(),
    )


def unique(citations: list[Citation]) -> list[Citation]:
    return list({(c.document, c.section, c.quote): c for c in citations}.values())


def normalise(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9₹]+", text.lower()))


def numbers_in(text: str) -> set[str]:
    return {match.replace(",", "") for match in re.findall(r"\d[\d,]*(?:\.\d+)?", text)}
