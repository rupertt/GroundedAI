from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

# UI router kept separate from API routes so we don't risk breaking /ask and /health.
router = APIRouter()


def _project_root() -> Path:
    # app/web.py -> app/ -> project root
    return Path(__file__).resolve().parent.parent


templates = Jinja2Templates(directory=str(_project_root() / "templates"))


@router.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    """
    Serve the minimal browser UI. The UI calls POST /ask via fetch().
    """
    return templates.TemplateResponse("index.html", {"request": request})


