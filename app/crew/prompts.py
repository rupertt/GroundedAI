from __future__ import annotations

from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=16)
def load_crew_prompt(name: str) -> str:
    """
    Loads CrewAI prompt markdown from ./app/crew/prompts/{name}.md.
    """
    base_dir = Path(__file__).resolve().parent / "prompts"
    path = base_dir / f"{name}.md"
    return path.read_text(encoding="utf-8")


