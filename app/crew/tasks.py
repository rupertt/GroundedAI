from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Type-only import; executed by type checkers, not at runtime.
    from crewai import Task


def build_evidence_task(question: str, top_k: int, agent) -> Task:
    """
    Task 1: Retriever builds an Evidence Pack using retrieve_chunks().
    """
    # Import lazily so we can apply the sqlite3 compatibility shim before CrewAI imports Chroma.
    from crewai import Task

    return Task(
        description=(
            "User question:\n"
            f"{question}\n\n"
            "Generate 2–4 retrieval queries and call retrieve_chunks(query, top_k) for each.\n"
            f"Use top_k={top_k}.\n"
            "Return ONLY the Evidence Pack bullet list in the required format."
        ),
        expected_output="Evidence Pack bullet list only.",
        agent=agent,
    )


def build_draft_task(question: str, evidence_task: Task, agent) -> Task:
    """
    Task 2: Responder writes a draft answer using ONLY the Evidence Pack.
    """
    # Import lazily so we can apply the sqlite3 compatibility shim before CrewAI imports Chroma.
    from crewai import Task

    return Task(
        description=(
            "Write the final answer to the user question using ONLY the Evidence Pack.\n\n"
            "User question:\n"
            f"{question}\n"
        ),
        expected_output="Customer-ready answer with inline citations [<filename>#chunk-XX].",
        agent=agent,
        context=[evidence_task],
    )


def build_verify_task(evidence_task: Task, draft_task: Task, agent) -> Task:
    """
    Task 3: Verifier checks the draft and requests follow-up queries if needed.
    """
    # Import lazily so we can apply the sqlite3 compatibility shim before CrewAI imports Chroma.
    from crewai import Task

    return Task(
        description=(
            "Verify the Draft Answer against the Evidence Pack.\n"
            "Return OK if grounded and properly cited; otherwise return FOLLOWUP_QUERIES list."
        ),
        expected_output="OK or FOLLOWUP_QUERIES list (strict format).",
        agent=agent,
        context=[evidence_task, draft_task],
    )


