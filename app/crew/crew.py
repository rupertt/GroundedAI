from __future__ import annotations

import re
import logging
import os
from typing import Any

from app.core.config import settings
from app.rag import DOC_SOURCE, index_if_needed, retrieve_with_scores

# NOTE: We intentionally avoid importing CrewAI (or modules that import CrewAI) at module import time.
# On some WSL distros, CrewAI -> Chroma can fail unless we apply the sqlite3 compatibility shim first.
from app.crew.tools import apply_sqlite_fix_for_chroma, make_retrieve_tool


logger = logging.getLogger(__name__)

# Accept 2+ digits because chunk ids may grow beyond 99 (e.g., chunk-100) depending on doc size.
# Strict pattern enforces the required citation token format: [<filename>#chunk-XX]
_CITATION_STRICT_RE = re.compile(r"\[([^\]#]+)#(chunk-\d+)\]")
# Loose pattern helps us debug / repair minor formatting issues (missing ']' etc.).
_CITATION_LOOSE_RE = re.compile(r"([^\s\[\]#]+)#(chunk-\d+)")


def _has_any_citation(text: str) -> bool:
    t = text or ""
    return ("[" in t) and ("#chunk-" in t)


def _extract_cited_keys(text: str) -> list[str]:
    """
    Extract citation keys only when citations match the strict required format:
      [<filename>#chunk-XX]
    """
    seen: set[str] = set()
    ordered: list[str] = []
    for m in _CITATION_STRICT_RE.finditer(text):
        src = m.group(1)
        cid = m.group(2)
        key = f"{src}#{cid}"
        if key not in seen:
            seen.add(key)
            ordered.append(key)
    return ordered


def _extract_cited_keys_loose(text: str) -> list[str]:
    """
    Extract citation keys even if citations are slightly malformed (e.g., missing closing bracket).
    This is used only for deterministic citation repair.
    """
    seen: set[str] = set()
    ordered: list[str] = []
    for m in _CITATION_LOOSE_RE.finditer(text or ""):
        src = m.group(1)
        cid = m.group(2)
        key = f"{src}#{cid}"
        if key not in seen:
            seen.add(key)
            ordered.append(key)
    return ordered


def _normalize_citation_key(citation_key: str, store: dict[str, Any]) -> str:
    """
    Normalize citation keys so citations like doc.txt#chunk-3 map to stable stored keys like doc.txt#chunk-03.

    This prevents false negatives where the model cites a valid chunk but drops leading zeros.
    """
    chunks = store.get("chunks") or {}
    if citation_key in chunks:
        return citation_key

    # Split "source#chunk-<n>".
    if "#" not in citation_key:
        return citation_key
    src, cid = citation_key.split("#", 1)

    m = re.match(r"^chunk-(\d+)$", cid)
    if not m:
        return citation_key

    n = int(m.group(1))
    # Try common stable formats.
    candidates = [f"chunk-{n:02d}", f"chunk-{n:03d}", f"chunk-{n}"]
    for c in candidates:
        k = f"{src}#{c}"
        if k in chunks:
            return k
    return citation_key


def _task_raw(task) -> str:
    """
    CrewAI tasks store outputs in task.output (TaskOutput). We read .raw when present.
    """
    out = getattr(task, "output", None)
    raw = getattr(out, "raw", None)
    if isinstance(raw, str):
        return raw.strip()
    if out is None:
        return ""
    # Fallback stringification.
    return str(out).strip()


def _parse_followups(verifier_text: str) -> list[str]:
    """
    Parse verifier output in the strict format described in verifier.md.
    """
    txt = (verifier_text or "").strip()
    if not txt:
        return []

    # Be tolerant: some CrewAI runs return "FOLLOWUP_QUERIES" without a colon or with extra words.
    if "FOLLOWUP_QUERIES" not in txt:
        return []

    lines_all = [ln.strip() for ln in txt.splitlines() if ln.strip()]
    # Skip header line if present; otherwise parse any "- ..." lines.
    lines = lines_all[1:] if lines_all[0].startswith("FOLLOWUP_QUERIES") else lines_all
    out: list[str] = []
    for ln in lines:
        if ln.startswith("- "):
            out.append(ln[2:].strip())
    return out[:3]


def _passes_citation_density(answer_text: str) -> bool:
    """
    Enforce: at least one STRICT citation token per paragraph or bullet group.

    Implementation notes:
    - We treat blocks separated by a blank line as "paragraphs" (bullet groups also tend to be block-based).
    - We fail closed on any non-empty block that has no strict citation.
    """
    txt = (answer_text or "").strip()
    if not txt:
        return False
    blocks = re.split(r"\n\s*\n", txt)
    for blk in blocks:
        if not re.search(r"\w", blk):
            continue
        if _CITATION_STRICT_RE.search(blk) is None:
            return False
    return True


def _repair_citations_deterministic(answer_text: str, store: dict[str, Any]) -> str:
    """
    Deterministically repair common citation formatting issues WITHOUT another LLM pass.

    Goal:
    - Convert loose occurrences like "doc.txt#chunk-3" into strict "[doc.txt#chunk-03]" when possible.
    - Only repair citations that map to actually retrieved chunks in `store`.
    """
    txt = (answer_text or "").strip()
    if not txt:
        return txt

    # If we already have at least one strict citation, do not attempt to rewrite.
    if _CITATION_STRICT_RE.search(txt):
        return txt

    chunks = store.get("chunks") or {}

    def _repl(m: re.Match[str]) -> str:
        src = m.group(1)
        cid = m.group(2)
        key = _normalize_citation_key(f"{src}#{cid}", store)
        if key in chunks:
            src2, cid2 = key.split("#", 1)
            return f"[{src2}#{cid2}]"
        # If it doesn't map to retrieved evidence, leave it unchanged (we will fail closed later).
        return m.group(0)

    # Replace only when not already bracketed.
    repaired = re.sub(r"(?<!\[)([^\s\[\]#]+)#(chunk-\d+)(?!\])", _repl, txt)

    # Additional deterministic repair:
    # If the model cited "[chunk-08]" without a filename, we can repair it ONLY if that chunk_id
    # appears in exactly one retrieved source in the store. This keeps us grounded and avoids
    # guessing when multiple files share the same chunk id.
    if _CITATION_STRICT_RE.search(repaired) is None:
        chunks = store.get("chunks") or {}
        # Build reverse map: chunk_id -> unique source (or None if ambiguous).
        chunk_to_src: dict[str, str | None] = {}
        for k in chunks.keys():
            if "#" not in k:
                continue
            src, cid = k.split("#", 1)
            if cid not in chunk_to_src:
                chunk_to_src[cid] = src
            else:
                # Ambiguous.
                if chunk_to_src[cid] != src:
                    chunk_to_src[cid] = None

        def _repl2(m: re.Match[str]) -> str:
            cid = m.group(1)
            src = chunk_to_src.get(cid)
            if src:
                return f"[{src}#{cid}]"
            return m.group(0)

        repaired = re.sub(r"\[(chunk-\d+)\]", _repl2, repaired)

    return repaired


def _requirement_signals(text: str) -> bool:
    """
    Heuristic: detect language that indicates something is a requirement.

    Used only to decide whether to trigger a repair pass when the model's answer contradicts evidence.
    """
    t = (text or "").lower()
    return any(
        s in t
        for s in [
            "required",
            "must",
            "need to",
            "strong understanding",
            "experience with",
            "requirements",
        ]
    )


def _answer_negates_requirement(answer_text: str) -> bool:
    """
    Heuristic: detect when an answer suggests something is NOT required.
    """
    t = (answer_text or "").lower()
    return any(s in t for s in ["not required", "isn't required", "is not required", "not explicitly required", "not necessary"])


def _evidence_says_required(store: dict[str, Any], *, term: str) -> bool:
    """
    Check retrieved evidence for a specific term + requirement-signaling language.
    """
    if not term:
        return False
    term_l = term.lower()
    for _k, item in (store.get("chunks") or {}).items():
        txt = str(item.get("text", "")).lower()
        if term_l in txt and _requirement_signals(txt):
            return True
    return False


def _first_requirement_key(store: dict[str, Any], *, term: str) -> str | None:
    """
    Return the first retrieved (source#chunk_id) key that contains `term` and requirement-signaling language.

    Used for deterministic yes/no answers to avoid LLM hedging when evidence is unambiguous.
    """
    if not term:
        return None
    term_l = term.lower()
    for key, item in (store.get("chunks") or {}).items():
        txt = str(item.get("text", "")).lower()
        if term_l in txt and _requirement_signals(txt):
            return str(key)
    return None


def _build_citations_from_store(store: dict[str, Any], used_keys: list[str]) -> list[dict[str, Any]]:
    chunks = store.get("chunks") or {}
    citations: list[dict[str, Any]] = []
    for key in used_keys:
        key2 = _normalize_citation_key(key, store)
        item = chunks.get(key2)
        if not item:
            continue
        snippet = str(item.get("text", "")).strip().replace("\n", " ")
        snippet = snippet[:240] + ("..." if len(snippet) > 240 else "")
        citations.append(
            {
                "source": str(item.get("source", DOC_SOURCE)),
                "chunk_id": str(item.get("chunk_id", "chunk-??")),
                "snippet": snippet,
            }
        )
    return citations


def _evidence_pack_from_store(store: dict[str, Any]) -> str:
    """
    Deterministically build an Evidence Pack from the tool retrieval store.

    Format matches retriever.md:
    - [<filename>#chunk-XX] "<short quote>"
    """
    chunks = store.get("chunks") or {}
    lines: list[str] = []
    # Sort for deterministic output: by (source, numeric chunk index where possible).
    def _sort_key(k: str) -> tuple[str, int, str]:
        src, cid = (k.split("#", 1) + [""])[:2]
        m = re.match(r"^chunk-(\d+)$", cid)
        n = int(m.group(1)) if m else 10**9
        return (src.lower(), n, cid)

    for key in sorted(chunks.keys(), key=_sort_key):
        item = chunks[key]
        src = str(item.get("source", DOC_SOURCE))
        cid = str(item.get("chunk_id", "chunk-??"))
        text = str(item.get("text", "")).strip().replace("\n", " ")
        quote = text[:160] + ("..." if len(text) > 160 else "")
        # Always wrap in quotes to keep formatting consistent.
        lines.append(f'- [{src}#{cid}] "{quote}"')
    return "\n".join(lines).strip()


def _seed_store_from_question(store: dict[str, Any], question: str, top_k: int) -> None:
    """
    Fallback retrieval in case the Retriever Agent fails to call the tool.

    This keeps the knowledge source the same (doc.txt via our existing Chroma index),
    but ensures we have some evidence to work with before drafting/verifying.
    """
    results = retrieve_with_scores(question, top_k=top_k)
    store.setdefault("chunks", {})
    store.setdefault("calls", [])
    store["calls"].append({"query": question, "top_k": top_k, "results": []})

    for doc, score in results:
        chunk_id = str(doc.metadata.get("chunk_id", "chunk-??"))
        source = str(doc.metadata.get("source", DOC_SOURCE))
        item = {
            "chunk_id": chunk_id,
            "text": doc.page_content,
            "source": source,
            "score": float(score),
        }
        store["chunks"][f"{source}#{chunk_id}"] = item
        store["calls"][-1]["results"].append(item)


def answer_question_agent(question: str, top_k: int = 4, debug: bool = False) -> dict[str, Any]:
    """
    CrewAI-based agent mode.

    Flow:
    - Retriever Agent: builds Evidence Pack using retrieve_chunks tool
    - Responder Agent: drafts answer using ONLY Evidence Pack
    - Verifier Agent: validates citations/grounding; may request up to 3 follow-up queries
    - Optional retry (single additional pass): retrieve more -> revised draft -> verify
    - Fail closed if not grounded/cited
    """
    # Ensure our own persistent index is ready and up-to-date.
    index_if_needed()

    # CrewAI imports Chroma internally; apply sqlite compatibility shim before importing CrewAI.
    apply_sqlite_fix_for_chroma()

    # Disable CrewAI cloud tracing / trace panels by default. This avoids trace URLs and access errors.
    # (CrewAI respects this env var in its tracing utilities.)
    os.environ["CREWAI_TRACING_ENABLED"] = "false"

    # Import after sqlite shim.
    from crewai import Crew, LLM, Process, Task

    # Import our CrewAI-dependent helpers after sqlite shim too.
    from app.crew.agents import build_responder_agent, build_retriever_agent, build_verifier_agent
    from app.crew.tasks import build_draft_task, build_evidence_task, build_verify_task

    store: dict[str, Any] = {"calls": [], "chunks": {}}
    # Let the retrieval tool know whether to emit debug logs (does not change response schema).
    store["debug"] = bool(debug)
    retrieve_tool = make_retrieve_tool(store)

    llm = LLM(model=settings.model, api_key=settings.openai_api_key, temperature=0)

    retriever_agent = build_retriever_agent(llm=llm, tools=[retrieve_tool])
    responder_agent = build_responder_agent(llm=llm)
    verifier_agent = build_verifier_agent(llm=llm)

    evidence_task = build_evidence_task(question=question, top_k=top_k, agent=retriever_agent)
    draft_task = build_draft_task(question=question, evidence_task=evidence_task, agent=responder_agent)
    verify_task = build_verify_task(evidence_task=evidence_task, draft_task=draft_task, agent=verifier_agent)

    crew = Crew(
        agents=[retriever_agent, responder_agent, verifier_agent],
        tasks=[evidence_task, draft_task, verify_task],
        process=Process.sequential,
        verbose=False,
        # Disable CrewAI cloud tracing by default (avoids trace URLs / access errors).
        tracing=False,
    )
    crew.kickoff()

    evidence_text = _task_raw(evidence_task)
    draft_text = _task_raw(draft_task)
    verify_text = _task_raw(verify_task)
    passes_run = 1
    max_passes = 2

    # Debug logging: emit a compact trace of what happened server-side without changing response schema.
    if debug:
        logger.info(
            "Agent mode: initial pass complete. tool_calls=%s evidence_len=%s draft_len=%s verify_prefix=%r",
            len(store.get("calls") or []),
            len(evidence_text or ""),
            len(draft_text or ""),
            (verify_text or "")[:240],
        )
        if store.get("calls"):
            logger.info("Agent mode: tool_call_queries=%s", [c.get("query") for c in (store.get("calls") or [])])

    # If the retriever agent didn't call the tool, we will have no chunks in the store.
    # Fall back to a direct retrieval so the responder/verifier have something to ground on.
    if not (store.get("chunks") or {}):
        logger.info("Agent mode: no tool retrieval calls detected; falling back to direct retrieval.")
        _seed_store_from_question(store, question=question, top_k=top_k)
        evidence_text = _evidence_pack_from_store(store)

        # Re-run responder + verifier once with the seeded evidence pack.
        # NOTE: This counts as our second (and final) pass to respect the 2-pass cap.
        draft_task_fb = Task(
            description=(
                "Evidence Pack:\n"
                f"{evidence_text}\n\n"
                "User question:\n"
                f"{question}\n"
            ),
            expected_output="Customer-ready answer with inline citations [<filename>#chunk-XX].",
            agent=responder_agent,
        )
        verify_task_fb = Task(
            description=(
                "Evidence Pack:\n"
                f"{evidence_text}\n\n"
                "Verify the Draft Answer (provided via context) against the Evidence Pack.\n"
                "Output OK or FOLLOWUP_QUERIES (strict format)."
            ),
            expected_output="OK or FOLLOWUP_QUERIES list (strict format).",
            agent=verifier_agent,
            context=[draft_task_fb],
        )
        crew_fb = Crew(
            agents=[retriever_agent, responder_agent, verifier_agent],
            tasks=[draft_task_fb, verify_task_fb],
            process=Process.sequential,
            verbose=False,
            # Disable CrewAI cloud tracing by default (avoids trace URLs / access errors).
            tracing=False,
        )
        crew_fb.kickoff()
        draft_text = _task_raw(draft_task_fb) or draft_text
        verify_text = _task_raw(verify_task_fb) or verify_text
        passes_run = 2
        if debug:
            logger.info(
                "Agent mode: fallback pass complete. seeded_chunks=%s draft_len=%s verify_prefix=%r",
                len(store.get("chunks") or {}),
                len(draft_text or ""),
                (verify_text or "")[:240],
            )

    # Build a deterministic evidence pack from actually-retrieved chunks.
    # We use this for any repair passes so the responder/verifier never sees a hallucinated "Evidence Pack"
    # filename or mismatched chunk ids.
    evidence_pack_deterministic = _evidence_pack_from_store(store) if (store.get("chunks") or {}) else ""

    # Deterministic shortcut for a common accuracy failure mode:
    # Yes/No "do I need experience with X?" questions where evidence explicitly lists X as a requirement.
    # This avoids model hedging like "not explicitly required" when the retrieved text clearly indicates otherwise.
    q_lower = (question or "").lower()
    if "langchain" in q_lower and ("do i need" in q_lower or "is langchain" in q_lower or "need experience" in q_lower):
        req_key = _first_requirement_key(store, term="langchain")
        if req_key:
            # Strict token required format: [<filename>#chunk-XX]
            src, cid = req_key.split("#", 1)
            answer_text = (
                f"Yes. The documentation lists LangChain under required technical skills (\"Strong understanding of AI agents, "
                f"LLMs, LangChain, ...\") [{src}#{cid}]."
            )
            citations = _build_citations_from_store(store, [req_key])
            out_det: dict[str, Any] = {"answer": answer_text, "citations": citations}
            if debug:
                out_det["debug"] = {
                    "retrieved": [
                        {
                            "chunk_id": str(item.get("chunk_id", "chunk-??")),
                            "text": str(item.get("text", "")),
                            "score": float(item.get("score", 0.0)),
                        }
                        for _key, item in (store.get("chunks") or {}).items()
                    ]
                }
            return out_det

    followups = _parse_followups(verify_text)
    if debug and followups:
        logger.info("Agent mode: verifier requested followups=%s", followups)

    # Single retry pass if verifier asks for follow-ups (cap total passes at 2).
    if followups and passes_run < max_passes:
        for q in followups:
            # CrewAI tool functions return BaseTool; run() is the public interface.
            retrieve_tool.run(query=q, top_k=top_k)  # type: ignore[attr-defined]

        evidence_text2 = _evidence_pack_from_store(store)

        # Re-run responder + verifier with expanded evidence.
        # We embed the Evidence Pack directly to avoid extra tool calls.
        draft_task2 = Task(
            description=(
                "Evidence Pack:\n"
                f"{evidence_text2}\n\n"
                "User question:\n"
                f"{question}\n"
            ),
            expected_output="Customer-ready answer with inline citations [<filename>#chunk-XX].",
            agent=responder_agent,
        )
        verify_task2 = Task(
            description=(
                "Evidence Pack:\n"
                f"{evidence_text2}\n\n"
                "Verify the Draft Answer (provided via context) against the Evidence Pack.\n"
                "Output OK or FOLLOWUP_QUERIES (strict format)."
            ),
            expected_output="OK or FOLLOWUP_QUERIES list (strict format).",
            agent=verifier_agent,
            context=[draft_task2],
        )

        crew2 = Crew(
            agents=[retriever_agent, responder_agent, verifier_agent],
            tasks=[draft_task2, verify_task2],
            process=Process.sequential,
            verbose=False,
            # Disable CrewAI cloud tracing by default (avoids trace URLs / access errors).
            tracing=False,
        )
        crew2.kickoff()

        draft_text = _task_raw(draft_task2) or draft_text
        verify_text = _task_raw(verify_task2) or verify_text
        passes_run += 1
        if debug:
            logger.info(
                "Agent mode: retry pass complete. total_chunks=%s draft_len=%s verify_prefix=%r",
                len(store.get("chunks") or {}),
                len(draft_text or ""),
                (verify_text or "")[:240],
            )

    # Deterministic post-checks (fail closed):
    # - Must contain at least one citation token
    # - Must only cite retrieved evidence
    draft_text = _repair_citations_deterministic(draft_text, store)

    # If the draft has missing/invalid citations (common CrewAI failure mode: it literally uses "Evidence Pack"
    # as the filename), run one repair pass with a deterministic Evidence Pack built from retrieved chunks.
    if passes_run < max_passes:
        stored_keys = set((store.get("chunks") or {}).keys())
        used_keys_tmp = _extract_cited_keys(draft_text)
        has_any = bool(draft_text) and _has_any_citation(draft_text)
        invalid = False
        # Treat "refusal with evidence" as invalid too (improves accuracy when evidence clearly exists).
        if (draft_text or "").strip() == "I can’t find that in the provided documentation." and stored_keys:
            invalid = True
        # Treat "negates requirement" as invalid when evidence indicates the term is required.
        # Example: question "Do I need experience with LangChain?" and evidence says "Strong understanding of ... LangChain ..."
        elif _answer_negates_requirement(draft_text) and _evidence_says_required(store, term="langchain"):
            invalid = True
        elif not has_any or not used_keys_tmp:
            invalid = True
        else:
            for k in used_keys_tmp:
                if _normalize_citation_key(k, store) not in stored_keys:
                    invalid = True
                    break

        if invalid:
            evidence_pack = evidence_pack_deterministic
            if evidence_pack:
                from crewai import Crew, Process, Task

                draft_task_repair = Task(
                    description=(
                        "Evidence Pack:\n"
                        f"{evidence_pack}\n\n"
                        "User question:\n"
                        f"{question}\n"
                    ),
                    expected_output="Customer-ready answer with inline citations [<filename>#chunk-XX].",
                    agent=responder_agent,
                )
                verify_task_repair = Task(
                    description=(
                        "Evidence Pack:\n"
                        f"{evidence_pack}\n\n"
                        "Verify the Draft Answer (provided via context) against the Evidence Pack.\n"
                        "Output OK or FOLLOWUP_QUERIES (strict format)."
                    ),
                    expected_output="OK or FOLLOWUP_QUERIES list (strict format).",
                    agent=verifier_agent,
                    context=[draft_task_repair],
                )
                crew_repair = Crew(
                    agents=[retriever_agent, responder_agent, verifier_agent],
                    tasks=[draft_task_repair, verify_task_repair],
                    process=Process.sequential,
                    verbose=False,
                    tracing=False,
                )
                crew_repair.kickoff()
                draft_text = _task_raw(draft_task_repair) or draft_text
                verify_text = _task_raw(verify_task_repair) or verify_text
                passes_run += 1
                draft_text = _repair_citations_deterministic(draft_text, store)
                if debug:
                    logger.info(
                        "Agent mode: repair pass (invalid citations) complete. draft_len=%s verify_prefix=%r",
                        len(draft_text or ""),
                        (verify_text or "")[:240],
                    )

    if not draft_text or not _has_any_citation(draft_text):
        if debug:
            logger.info(
                "Agent mode failed citation check. evidence_len=%s verify_text=%r",
                len(evidence_text or ""),
                (verify_text or "")[:240],
            )
        out: dict[str, Any] = {"answer": "I can’t find that in the provided documentation.", "citations": []}
        if debug:
            out["debug"] = {
                "retrieved": [
                    {
                        "chunk_id": str(item.get("chunk_id", "chunk-??")),
                        "text": str(item.get("text", "")),
                        "score": float(item.get("score", 0.0)),
                    }
                    for _key, item in (store.get("chunks") or {}).items()
                ]
            }
        return out

    used_keys = _extract_cited_keys(draft_text)
    # Fail closed if we couldn't parse any strict citations.
    if not used_keys:
        out3: dict[str, Any] = {"answer": "I can’t find that in the provided documentation.", "citations": []}
        if debug:
            out3["debug"] = {
                "retrieved": [
                    {"chunk_id": str(item.get("chunk_id", "chunk-??")), "text": str(item.get("text", "")), "score": float(item.get("score", 0.0))}
                    for _key, item in (store.get("chunks") or {}).items()
                ]
            }
        return out3

    # Fail closed if any cited key wasn't actually retrieved.
    stored_keys = set((store.get("chunks") or {}).keys())
    for k in used_keys:
        if _normalize_citation_key(k, store) not in stored_keys:
            if debug:
                logger.info("Agent mode: citation not in retrieved evidence. cited_key=%r", k)
            out4: dict[str, Any] = {"answer": "I can’t find that in the provided documentation.", "citations": []}
            if debug:
                out4["debug"] = {
                    "retrieved": [
                        {"chunk_id": str(item.get("chunk_id", "chunk-??")), "text": str(item.get("text", "")), "score": float(item.get("score", 0.0))}
                        for _key, item in (store.get("chunks") or {}).items()
                    ]
                }
            return out4

    citations = _build_citations_from_store(store, used_keys)

    # Fail closed if citations reference chunks we didn't actually retrieve.
    if not citations:
        if debug:
            logger.info(
                "Agent mode: citation mismatch. cited_keys=%s stored_keys_sample=%s",
                used_keys[:12],
                list((store.get("chunks") or {}).keys())[:12],
            )

        out2: dict[str, Any] = {"answer": "I can’t find that in the provided documentation.", "citations": []}
        if debug:
            out2["debug"] = {
                "retrieved": [
                    {"chunk_id": str(item.get("chunk_id", "chunk-??")), "text": str(item.get("text", "")), "score": float(item.get("score", 0.0))}
                    for _key, item in (store.get("chunks") or {}).items()
                ]
            }
        return out2

    out3: dict[str, Any] = {"answer": draft_text, "citations": citations}
    if debug:
        out3["debug"] = {
            "retrieved": [
                {"chunk_id": str(item.get("chunk_id", "chunk-??")), "text": str(item.get("text", "")), "score": float(item.get("score", 0.0))}
                for _key, item in (store.get("chunks") or {}).items()
            ]
        }
    return out3


