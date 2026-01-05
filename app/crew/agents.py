from __future__ import annotations

from app.crew.prompts import load_crew_prompt


def build_retriever_agent(llm, tools):
    """
    Retriever agent: generates targeted retrieval queries and builds an Evidence Pack.
    """
    # Import lazily so we can apply the sqlite3 compatibility shim before CrewAI imports Chroma.
    from crewai import Agent

    return Agent(
        role="Retriever Agent",
        goal="Find the most relevant supporting evidence from doc.txt using the retrieval tool.",
        backstory="You are excellent at transforming questions into search queries and extracting short, high-signal quotes.",
        llm=llm,
        tools=tools,
        allow_delegation=False,
        verbose=False,
        system_template=load_crew_prompt("retriever"),
    )


def build_responder_agent(llm):
    """
    Responder agent: writes the final answer using ONLY the Evidence Pack, with citations.
    """
    # Import lazily so we can apply the sqlite3 compatibility shim before CrewAI imports Chroma.
    from crewai import Agent

    return Agent(
        role="Responder Agent",
        goal="Write a customer-ready answer grounded in evidence and properly cited.",
        backstory="You write concise, helpful support answers and always cite sources precisely.",
        llm=llm,
        allow_delegation=False,
        verbose=False,
        system_template=load_crew_prompt("responder"),
    )


def build_verifier_agent(llm):
    """
    Verifier agent: checks citations and grounding; may request follow-up retrieval queries.
    """
    # Import lazily so we can apply the sqlite3 compatibility shim before CrewAI imports Chroma.
    from crewai import Agent

    return Agent(
        role="Verifier Agent",
        goal="Ensure every factual claim is supported by evidence and citations are valid.",
        backstory="You are strict: if citations are missing or invalid, you request additional retrieval queries.",
        llm=llm,
        allow_delegation=False,
        verbose=False,
        system_template=load_crew_prompt("verifier"),
    )


