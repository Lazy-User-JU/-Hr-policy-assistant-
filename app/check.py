import sys

from app import config, llm, store
from app.answer import check_model_reply
from app.embeddings import embed

SAMPLE_HIT = {
    "document": "check.md",
    "version": 1,
    "title": "Check Policy",
    "section": "Leave",
    "applies_to": "",
    "kind": "text",
    "text": "Employees receive 12 casual leave days per calendar year.",
}
SAMPLE_QUESTION = "How many casual leave days do employees receive?"


def check_database() -> bool:
    store.init()
    print(f"ok    Database: {config.DB_PATH}")
    return True


def check_embeddings() -> bool:
    print(f"...   Embeddings: loading {config.EMBEDDING_MODEL} (the first run downloads it)")
    try:
        size = embed(["hello"]).shape[1]
    except Exception as error:
        print(f"FAIL  Embeddings: {type(error).__name__}: {error}")
        print("      Run `pip install -r requirements.txt`, and check your internet connection.")
        return False
    print(f"ok    Embeddings: {size} dimensions")
    return True


def check_llm() -> bool:
    if not llm.enabled():
        print("note  LLM: no LLM_API_KEY in .env, so answers will be exact quotes")
        return True

    print(f"...   LLM: calling {config.LLM_MODEL}")
    try:
        reply = llm.ask(SAMPLE_QUESTION, [SAMPLE_HIT])
    except llm.QuotaExhausted as error:
        print(f"warn  LLM: {error}")
        print("      Your key and model were accepted. Until the quota resets, answers will be exact quotes.")
        print("      To use the LLM sooner, switch to another key, a paid tier, or a model with its own quota.")
        return True
    except llm.LLMError as error:
        print(f"FAIL  LLM: {error}")
        print("      Check LLM_API_KEY, and check LLM_MODEL against your provider's current model list.")
        return False

    result = check_model_reply(reply, [SAMPLE_HIT], SAMPLE_QUESTION)
    if result.status != "answered":
        print(f"FAIL  LLM replied, but its answer failed the checks: {result.refusal_reason}")
        return False
    print(f"ok    LLM: {result.answer}")
    return True


def main() -> int:
    passed = [check_database(), check_embeddings(), check_llm()]
    print("\nReady." if all(passed) else "\nFix the FAIL lines above.")
    return 0 if all(passed) else 1


if __name__ == "__main__":
    sys.exit(main())
