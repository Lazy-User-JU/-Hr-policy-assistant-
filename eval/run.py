import argparse
import json
import statistics
import sys
import tempfile
import time
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

from app import config, store
from app.answer import answer_question
from app.ingest import index_document
from app.retrieval import retrieve

ROOT = Path(__file__).parent.parent
FOLDERS = [ROOT / "policies", ROOT / "eval" / "heldout"]
RESULTS_FILE = ROOT / "eval" / "results.md"


def build_index() -> None:
    store.init()
    for folder in FOLDERS:
        for path in sorted(folder.iterdir()):
            index_document(path.name, path.read_bytes())


def matches_place(item: dict, document: str, section: str | None) -> bool:
    return document == item["document"] and item["section"].lower() in (section or "").lower()


def rank_of_right_section(item: dict, hits: list[dict]) -> int | None:
    for rank, hit in enumerate(hits, start=1):
        if matches_place(item, hit["document"], hit["section"]):
            return rank
    return None


def score(item: dict) -> dict:
    retrieval = retrieve(item["question"])
    response = answer_question(item["question"])
    answered = response.status == "answered"
    outcome = {
        "group": item["group"],
        "question": item["question"],
        "status": response.status,
        "answer": response.answer,
        "reason": response.refusal_reason,
        "answered_by": response.answered_by,
        "cosine": retrieval.best_cosine,
        "coverage": retrieval.best_coverage,
        "has_place": "document" in item,
        "rank": rank_of_right_section(item, retrieval.hits) if "document" in item else None,
        "false_answer": item["expect"] == "refuse" and answered,
    }

    text_ok = answered and all(word.lower() in response.answer.lower() for word in item.get("contains", []))
    cited_ok = "document" not in item or any(
        matches_place(item, citation.document, citation.section) for citation in response.citations
    )
    if item["expect"] == "refuse":
        outcome["correct"] = not answered
    elif item["expect"] == "answer":
        outcome["correct"] = text_ok and cited_ok
    else:
        outcome["correct"] = not answered or (text_ok and cited_ok)
    return outcome


def retrieval_summary(results: list[dict]) -> tuple[int, int, float]:
    placed = [r for r in results if r["has_place"]]
    found = sum(r["rank"] is not None for r in placed)
    mrr = sum(1 / r["rank"] for r in placed if r["rank"]) / len(placed)
    return found, len(placed), mrr


def summary_lines(results: list[dict]) -> list[str]:
    groups = defaultdict(list)
    for result in results:
        groups[result["group"]].append(result)

    found, placed, mrr = retrieval_summary(results)
    answered_by = Counter(r["answered_by"] for r in results if r["answered_by"])
    lines = ["| Group | Correct |", "| --- | --- |"]
    lines += [f"| {name} | {sum(r['correct'] for r in items)}/{len(items)} |" for name, items in groups.items()]
    lines += [
        "",
        f"- Total correct: {sum(r['correct'] for r in results)}/{len(results)}",
        f"- False answers: {sum(r['false_answer'] for r in results)}",
        f"- Right section in top {config.TOP_K}: {found}/{placed}",
        f"- Mean reciprocal rank: {mrr:.2f}",
        f"- Answered by: {dict(answered_by)}",
        f"- Refusal reasons: {dict(Counter(r['reason'] for r in results if r['reason']))}",
    ]
    return lines


def setup_name() -> str:
    answers = f"LLM {config.LLM_MODEL}" if config.LLM_API_KEY else "exact quotes, no LLM"
    return f"embeddings {config.EMBEDDING_MODEL}, {config.SEARCH_MODE} search, {answers}"


def report(results: list[dict], show_all: bool) -> None:
    for result in results:
        if show_all or not result["correct"]:
            mark = "ok  " if result["correct"] else "FAIL"
            print(f"{mark} [{result['group']}] {result['question']}")
            print(f"     {result['status']} ({result['reason'] or result['answered_by']}): {result['answer'][:110]}")
            print(f"     cosine {result['cosine']:.2f}, coverage {result['coverage']:.2f}")

    lines = summary_lines(results)
    print("\n" + "\n".join(lines))

    fallbacks = sum(r["answered_by"] == "extractive" for r in results)
    if config.LLM_API_KEY and fallbacks:
        print(f"\nWARNING: {fallbacks} questions fell back to exact quotes, so the LLM was not tested on them.")
        print("This usually means a rate limit. Try again with a larger --delay.")

    RESULTS_FILE.write_text(
        f"# Eval results\n\nLast run: {date.today()}\n\nSetup: {setup_name()}\n\n" + "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def calibrate(questions: list[dict]) -> None:
    scores = [(item["expect"], retrieve(item["question"])) for item in questions]
    answerable = [r for expect, r in scores if expect == "answer"]
    unanswerable = [r for expect, r in scores if expect == "refuse"]

    def blocked(items: list, cutoff: float) -> int:
        return sum(r.best_cosine < cutoff and r.best_coverage < config.MIN_COVERAGE for r in items)

    print("MIN_COSINE   answerable blocked   unanswerable blocked")
    best = 0.0
    for step in range(10, 75, 5):
        cutoff = step / 100
        wrong, right = blocked(answerable, cutoff), blocked(unanswerable, cutoff)
        print(f"{cutoff:<12.2f} {wrong}/{len(answerable):<18} {right}/{len(unanswerable)}")
        if wrong == 0:
            best = cutoff

    cosines = [r.best_cosine for r in answerable]
    print(f"\nAnswerable best cosine: min {min(cosines):.2f}, median {statistics.median(cosines):.2f}")
    print(f"Suggested MIN_COSINE={best:.2f}. It is the highest value that blocks no answerable question.")
    print("The gate only stops clear misses. The LLM and the answer checks handle the rest.")


def compare(questions: list[dict]) -> None:
    print(f"Mode       Right section in top {config.TOP_K}   Mean reciprocal rank")
    for mode in ("hybrid", "vector", "keyword"):
        config.SEARCH_MODE = mode
        ranks = []
        for item in questions:
            if "document" in item:
                ranks.append(rank_of_right_section(item, retrieve(item["question"]).hits))
        mrr = sum(1 / rank for rank in ranks if rank) / len(ranks)
        print(f"{mode:<10} {sum(r is not None for r in ranks)}/{len(ranks):<23} {mrr:.2f}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--show-all", action="store_true")
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--delay", type=float, default=5.0, help="seconds between LLM calls")
    args = parser.parse_args()

    if args.no_llm:
        config.LLM_API_KEY = ""
    print(f"Setup: {setup_name()}\n")

    questions = json.loads((ROOT / "eval" / "questions.json").read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory() as folder:
        config.DB_PATH = Path(folder) / "eval.db"
        build_index()
        if args.calibrate:
            calibrate(questions)
            return 0
        if args.compare:
            compare(questions)
            return 0

        results = []
        for number, item in enumerate(questions, start=1):
            results.append(score(item))
            print(f"\rQuestion {number}/{len(questions)}", end="", file=sys.stderr, flush=True)
            if config.LLM_API_KEY and results[-1]["reason"] != "weak_retrieval":
                time.sleep(args.delay)
        print(file=sys.stderr)

    report(results, args.show_all)
    return 0 if all(r["correct"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
