from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

from app.core.types import AskResponse


# Exact refusal text required by the project + eval spec.
REFUSAL_TEXT = "I can’t find that in the provided documentation."


def _repo_root() -> Path:
    # tests/ -> project root
    return Path(__file__).resolve().parents[1]


def _load_cases() -> list[dict]:
    cases_path = _repo_root() / "evals" / "cases.json"
    return json.loads(cases_path.read_text(encoding="utf-8"))


def _has_any_citation_token(answer: str) -> bool:
    # Supports both legacy and multi-doc formats: [doc.txt#chunk-03] or [runbook.md#chunk-07]
    return re.search(r"\[[^\]#]+#chunk-\d+\]", answer or "") is not None


def _extract_chunk_ids_from_answer(answer: str) -> set[str]:
    # We only need chunk_id extraction for the "cited chunk_ids were retrieved" check.
    return set(re.findall(r"\[[^\]#]+#(chunk-\d+)\]", answer or ""))


@pytest.mark.parametrize("case", _load_cases())
def test_eval_cases_agent_mode(case: dict) -> None:
    """
    Lightweight evaluation harness (agent path).

    Checks:
    - Response schema is valid (AskResponse)
    - If expected_behavior=answer:
      - answer contains at least 1 citation token
      - all cited chunk_ids were retrieved (based on debug.retrieved)
    - If expected_behavior=refuse:
      - refusal text is exact
    """
    # Skip if tests can't run without an OpenAI key.
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is not set; skipping evals that require LLM calls.")

    from app.crew.crew import answer_question_agent

    result = answer_question_agent(case["question"], top_k=4, debug=True)
    resp = AskResponse(**result)

    if case["expected_behavior"] == "refuse":
        assert resp.answer == REFUSAL_TEXT
        return

    # expected_behavior == "answer"
    assert resp.answer != REFUSAL_TEXT
    assert _has_any_citation_token(resp.answer)

    retrieved_ids = {r.chunk_id for r in (resp.debug.retrieved if resp.debug else [])}
    assert retrieved_ids, "Expected debug.retrieved to be populated when debug=True"

    # All cited chunk_ids were retrieved.
    cited_ids_from_text = _extract_chunk_ids_from_answer(resp.answer)
    assert cited_ids_from_text, "Expected at least one cited chunk_id in answer text"
    assert cited_ids_from_text.issubset(retrieved_ids)

    # All citations objects should reference retrieved chunks as well.
    cited_ids_from_citations = {c.chunk_id for c in resp.citations}
    assert cited_ids_from_citations.issubset(retrieved_ids)


