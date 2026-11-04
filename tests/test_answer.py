import requests

import pytest

from app import config, llm
from app.answer import answer_question, check_model_reply, extractive_answer
from app.retrieval import coverage, retrieve, stem
from app.schemas import Evidence, ModelReply
from tests.conftest import index

POLICY = """# Leave Policy

## Casual leave

A maximum of 8 days of unused casual leave may be carried forward.

## Health tiers

| Tier | Dental implants | Optical |
| --- | --- | --- |
| Standard | Not covered | ₹8,000 |
| Plus | Covered up to ₹40,000 | ₹12,000 |

## Scope

This policy does not define relocation bonuses.
"""


QUESTION = "What does the plan cover?"
STANDARD_ROW = "Tier: Standard | Dental implants: Not covered | Optical: ₹8,000"
PLUS_ROW = "Tier: Plus | Dental implants: Covered up to ₹40,000 | Optical: ₹12,000"


def hit(text=PLUS_ROW):
    return {
        "document": "leave.md",
        "version": 2,
        "section": "Health tiers",
        "text": text,
        "title": "Leave Policy",
        "applies_to": "",
        "kind": "table",
    }


def hits(text=PLUS_ROW):
    return [hit(text)]


def reply(answer, quote, excerpt=1):
    return ModelReply(answerable=True, answer=answer, evidence=[Evidence(excerpt=excerpt, quote=quote)])


def test_good_reply_gets_citation_from_stored_metadata():
    result = check_model_reply(reply("Covered up to ₹40,000.", "Covered up to ₹40,000"), hits(), QUESTION)
    assert result.status == "answered"
    citation = result.citations[0]
    assert (citation.document, citation.version, citation.section) == ("leave.md", 2, "Health tiers")


def test_model_saying_not_answerable_is_refused():
    result = check_model_reply(ModelReply(answerable=False), hits(), QUESTION)
    assert (result.status, result.refusal_reason) == ("refused", "model_refused")
    assert result.citations == []


def test_invented_quote_is_refused():
    result = check_model_reply(reply("Implants are free.", "Implants are free for everyone"), hits(), QUESTION)
    assert result.refusal_reason == "quote_not_found"


def test_number_missing_from_the_quote_is_refused():
    result = check_model_reply(reply("Covered up to ₹50,000.", "Covered up to ₹40,000"), hits(), QUESTION)
    assert result.refusal_reason == "unsupported_number"


def test_excerpt_number_that_was_never_sent_is_refused():
    result = check_model_reply(reply("Covered.", "Covered", excerpt=7), hits(), QUESTION)
    assert result.refusal_reason == "unknown_excerpt"


def test_answer_without_evidence_is_refused():
    result = check_model_reply(ModelReply(answerable=True, answer="Yes."), hits(), QUESTION)
    assert result.refusal_reason == "no_evidence"


def test_quote_check_ignores_case_and_punctuation():
    result = check_model_reply(reply("Yes.", "covered up to ₹40,000."), hits(), QUESTION)
    assert result.status == "answered"


def test_citing_the_wrong_table_row_is_refused():
    both = [hit(STANDARD_ROW), hit(PLUS_ROW)]
    wrong = reply("Covered up to ₹40,000.", "Covered up to ₹40,000", excerpt=2)
    result = check_model_reply(wrong, both, "Does the Standard tier cover dental implants?")
    assert result.refusal_reason == "wrong_table_row"


def test_citing_the_named_row_is_allowed():
    both = [hit(STANDARD_ROW), hit(PLUS_ROW)]
    good = reply("Covered up to ₹40,000.", "Covered up to ₹40,000", excerpt=2)
    assert check_model_reply(good, both, "Does the Plus tier cover dental implants?").status == "answered"


def test_row_check_ignores_words_shared_by_all_rows():
    rows = [hit("Band: Band B | LTA: ₹50,000"), hit("Band: Band C | LTA: ₹75,000")]
    assert check_model_reply(reply("₹75,000.", "LTA: ₹75,000", excerpt=2), rows, "LTA for Band C?").status == "answered"
    assert check_model_reply(reply("₹50,000.", "LTA: ₹50,000", excerpt=1), rows, "LTA for Band C?").refusal_reason == "wrong_table_row"


def test_row_check_is_skipped_when_no_row_is_named():
    both = [hit(STANDARD_ROW), hit(PLUS_ROW)]
    good = reply("Covered up to ₹40,000.", "Covered up to ₹40,000", excerpt=2)
    assert check_model_reply(good, both, "How much for implants?").status == "answered"


def test_yes_answer_on_a_negative_quote_is_refused():
    row = "Sick leave cannot be encashed."
    result = check_model_reply(reply("Yes, it can be encashed.", row), hits(row), QUESTION)
    assert result.refusal_reason == "contradicts_quote"


def test_group_the_policy_never_mentions_is_refused():
    row = "Employees receive 12 casual leave days."
    result = check_model_reply(reply("12 days.", row), hits(row), "How many days do contractors get?")
    assert result.refusal_reason == "outside_policy_scope"


def test_group_named_in_applies_to_is_allowed():
    scoped = {**hit("Passwords must be 12 characters."), "applies_to": "All employees and contractors"}
    result = check_model_reply(reply("12 characters.", "12 characters"), [scoped], "Password length for contractors?")
    assert result.status == "answered"


def test_table_answer_returns_the_asked_cell_only():
    index("leave.md", POLICY)
    result = answer_question("Does the Standard tier cover dental implants?")
    assert result.answer == "Tier: Standard; Dental implants: Not covered"
    assert result.citations[0].section == "Health tiers"


def test_extractive_refuses_when_policy_says_it_is_silent():
    index("leave.md", POLICY)
    result = extractive_answer("Does this policy define relocation bonuses?", retrieve("relocation bonuses").hits)
    assert result.refusal_reason == "policy_is_silent"


def test_no_documents_is_a_refusal_not_a_crash():
    assert answer_question("How many days?").refusal_reason == "no_documents"


def test_off_topic_question_is_refused_before_the_llm(monkeypatch):
    index("leave.md", POLICY)
    monkeypatch.setattr(config, "LLM_API_KEY", "key")

    def fail(*args, **kwargs):
        raise AssertionError("LLM should not be called")

    monkeypatch.setattr(requests, "post", fail)
    assert answer_question("Who won the football world cup?").refusal_reason == "weak_retrieval"


class FakeResponse:
    def __init__(self, content="", status_code=200, headers=None):
        self.content = content
        self.status_code = status_code
        self.text = content
        self.headers = headers or {}

    def json(self):
        return {"choices": [{"message": {"content": self.content}}]}


GOOD_JSON = '{"answerable": true, "answer": "8 days.", "evidence": [{"excerpt": 1, "quote": "A maximum of 8 days"}]}'


def test_llm_json_inside_a_code_fence_is_accepted(monkeypatch):
    index("leave.md", POLICY)
    monkeypatch.setattr(config, "LLM_API_KEY", "key")
    content = f"```json\n{GOOD_JSON}\n```"
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse(content))
    result = answer_question("How many casual leave days can be carried forward?")
    assert (result.status, result.answered_by) == ("answered", "llm")


def test_llm_outage_falls_back_to_quoting(monkeypatch):
    index("leave.md", POLICY)
    monkeypatch.setattr(config, "LLM_API_KEY", "key")

    def timeout(*args, **kwargs):
        raise requests.Timeout()

    monkeypatch.setattr(requests, "post", timeout)
    result = answer_question("How many days of casual leave may be carried forward?")
    assert (result.status, result.answered_by) == ("answered", "extractive")


def test_bad_llm_json_falls_back_to_quoting(monkeypatch):
    index("leave.md", POLICY)
    monkeypatch.setattr(config, "LLM_API_KEY", "key")
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse("Sure! You get 8 days."))
    assert answer_question("How many days of casual leave may be carried forward?").answered_by == "extractive"


def test_rate_limit_is_retried(monkeypatch):
    replies = [FakeResponse(status_code=429), FakeResponse(GOOD_JSON)]
    monkeypatch.setattr(requests, "post", lambda *a, **k: replies.pop(0))
    monkeypatch.setattr(llm.time, "sleep", lambda seconds: None)
    assert llm.ask("Days?", hits()).answer == "8 days."


def test_rate_limit_that_never_clears_raises(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse(status_code=429))
    monkeypatch.setattr(llm.time, "sleep", lambda seconds: None)
    with pytest.raises(llm.LLMError):
        llm.ask("Days?", hits())


def test_short_retry_hint_is_waited_out(monkeypatch):
    replies = [FakeResponse("Please retry in 20.5s.", status_code=429), FakeResponse(GOOD_JSON)]
    waits = []
    monkeypatch.setattr(requests, "post", lambda *a, **k: replies.pop(0))
    monkeypatch.setattr(llm.time, "sleep", waits.append)
    assert llm.ask("Days?", hits()).answer == "8 days."
    assert waits == [20.5]


def test_daily_quota_fails_fast_and_answer_falls_back(monkeypatch):
    index("leave.md", POLICY)
    monkeypatch.setattr(config, "LLM_API_KEY", "key")
    calls = []
    monkeypatch.setattr(requests, "post", lambda *a, **k: calls.append(1) or FakeResponse("Please retry in 13h12m52.7s.", status_code=429))
    monkeypatch.setattr(llm.time, "sleep", lambda seconds: pytest.fail("should not wait for a daily quota"))
    with pytest.raises(llm.QuotaExhausted):
        llm.ask("Days?", hits())
    assert len(calls) == 1
    assert answer_question("How many days of casual leave may be carried forward?").answered_by == "extractive"


def test_provider_that_rejects_schemas_is_retried_without_one(monkeypatch):
    payloads = []

    def fake_post(url, json, headers, timeout):
        payloads.append(dict(json))
        return FakeResponse(status_code=400) if "response_format" in json else FakeResponse(GOOD_JSON)

    monkeypatch.setattr(requests, "post", fake_post)
    assert llm.ask("Days?", hits()).answer == "8 days."
    assert "response_format" in payloads[0] and "response_format" not in payloads[1]


def test_empty_model_content_is_an_llm_error(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse(None))
    with pytest.raises(llm.LLMError):
        llm.ask("Days?", hits())


def test_prompt_sends_the_excerpts_and_the_question():
    prompt = llm.build_prompt("How many days?", hits())
    assert "[1] Leave Policy > Health tiers" in prompt
    assert prompt.endswith("Question: How many days?")


def test_prompt_shows_who_a_policy_applies_to():
    scoped = {**hit(), "applies_to": "Full-time employees in India"}
    assert "(applies to: Full-time employees in India)" in llm.build_prompt("Days?", [scoped])


def test_stem_matches_word_forms():
    assert stem("carried") == stem("carry")
    assert coverage("covered", "cover") == 1.0
    assert coverage("implants", "implant") == 1.0
