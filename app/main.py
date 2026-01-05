from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from app.core.logging import setup_logging
from app.core.types import AskRequest, AskResponse
from app.ingest import router as ingest_router
from app.rag import answer_question
from app.web import router as ui_router

setup_logging()
logger = logging.getLogger(__name__)

# NOTE: We reserve GET /docs for listing ingested documents (per project requirements).
# Move Swagger UI off /docs to avoid a route conflict.
app = FastAPI(
    # Public-facing name used in OpenAPI metadata.
    title="GroundedAI",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

# Serve local static assets for the browser UI (no external CDNs).
app.mount("/static", StaticFiles(directory="static"), name="static")

# Minimal UI at GET / (kept separate from API routes).
app.include_router(ui_router)

# Ingestion + docs/jobs endpoints (kept separate from /ask schemas).
app.include_router(ingest_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    try:
        result = answer_question(req.question, top_k=req.top_k, debug=req.debug)
        return AskResponse(**result)
    except Exception as e:
        logger.exception("Error answering question")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/ask_agent", response_model=AskResponse)
def ask_agent(req: AskRequest) -> AskResponse:
    """
    CrewAI-based agent mode. Request/response schema matches /ask.
    """
    try:
        # Lazy import so server can start even if CrewAI isn't installed yet.
        from app.crew.crew import answer_question_agent

        result = answer_question_agent(req.question, top_k=req.top_k, debug=req.debug)
        return AskResponse(**result)
    except Exception as e:
        logger.exception("Error answering question (agent mode)")
        raise HTTPException(status_code=500, detail=str(e)) from e


