from __future__ import annotations

import ipaddress
import logging
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from trafilatura import extract as trafilatura_extract

from app.rag import _data_dir, index_path_incremental, index_scan_incremental

logger = logging.getLogger(__name__)

router = APIRouter()

# ---- Simple in-memory job store (process-local)
_JOBS_LOCK = RLock()
_JOBS: dict[str, dict[str, Any]] = {}


def _raw_dir() -> Path:
    """
    Raw ingested files directory.
    """
    return _data_dir() / "raw"


def _now_ts() -> float:
    """
    Unix timestamp in seconds.
    """
    return time.time()


def _new_job() -> str:
    """
    Create a new job entry and return its job_id.
    """
    job_id = str(uuid.uuid4())
    with _JOBS_LOCK:
        _JOBS[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "progress": 0,
            "error": None,
            "created_at": _now_ts(),
            "finished_at": None,
        }
    return job_id


def _update_job(job_id: str, **fields: Any) -> None:
    """
    Update a job entry in place.
    """
    with _JOBS_LOCK:
        if job_id not in _JOBS:
            return
        _JOBS[job_id].update(fields)


def _finish_job(job_id: str, *, status: str, error: str | None = None) -> None:
    """
    Mark a job as finished.
    """
    _update_job(
        job_id,
        status=status,
        progress=100 if status == "succeeded" else _JOBS.get(job_id, {}).get("progress", 0),
        error=error,
        finished_at=_now_ts(),
    )


def _safe_filename(name: str) -> str:
    """
    Sanitize filenames to avoid path traversal + weird characters.
    """
    base = Path(name or "").name
    base = base.strip().replace("\x00", "")
    if not base:
        base = "upload.txt"
    # Normalize to a conservative character set.
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", base)
    base = re.sub(r"_+", "_", base)
    base = base.strip("._")
    return base or "upload.txt"


def _resolve_collision(path: Path) -> Path:
    """
    If filename exists, append _1, _2, ... before extension.
    """
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    i = 1
    while True:
        candidate = path.with_name(f"{stem}_{i}{suffix}")
        if not candidate.exists():
            return candidate
        i += 1


async def _save_upload_to_disk(file: UploadFile, *, max_bytes: int) -> tuple[Path, int]:
    """
    Save an uploaded file as-is to ./data/raw with a strict size limit.
    """
    _raw_dir().mkdir(parents=True, exist_ok=True)
    safe = _safe_filename(file.filename or "upload")
    suffix = Path(safe).suffix.lower()
    if suffix not in {".pdf", ".docx", ".txt"}:
        raise HTTPException(status_code=400, detail="Only .pdf, .docx, .txt uploads are supported.")

    dst = _resolve_collision(_raw_dir() / safe)
    total = 0
    try:
        with dst.open("wb") as f:
            while True:
                chunk = await file.read(1024 * 256)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise HTTPException(status_code=413, detail="File too large.")
                f.write(chunk)
    finally:
        try:
            await file.close()
        except Exception:
            pass

    return dst, total


def _is_private_ip(host: str) -> bool:
    """
    Basic SSRF protection: block private/local/link-local/multicast/etc.
    """
    try:
        ip = ipaddress.ip_address(host)
        return bool(
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        )
    except ValueError:
        # Not an IP literal; we can't safely resolve DNS here without extra complexity.
        return False


@dataclass(frozen=True)
class _FetchedPage:
    url: str
    html: str


def _fetch_url_html(url: str, *, timeout_s: float, max_bytes: int) -> _FetchedPage:
    """
    Fetch ONLY the provided URL (no crawling), with safety controls.
    """
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise HTTPException(status_code=400, detail="Only http/https URLs are allowed.")
    if not parsed.netloc:
        raise HTTPException(status_code=400, detail="Invalid URL.")
    # Block obvious IP-literal private ranges.
    host = (parsed.hostname or "").strip()
    if host and _is_private_ip(host):
        raise HTTPException(status_code=400, detail="Blocked host.")

    # Use a stable, explicit UA so upstreams can identify traffic from this app.
    headers = {"User-Agent": "GroundedAI/1.0 (+single-page ingestion)"}
    with httpx.Client(follow_redirects=True, timeout=timeout_s, headers=headers) as client:
        resp = client.get(url)
        resp.raise_for_status()

        # Streamed download size limiting isn't exposed on Response.content once read,
        # so we enforce size by checking the bytes length.
        content = resp.content
        if len(content) > max_bytes:
            raise HTTPException(status_code=413, detail="Downloaded content too large.")

        # Best-effort decode; BeautifulSoup + trafilatura can work with unicode.
        try:
            html = content.decode(resp.encoding or "utf-8", errors="ignore")
        except Exception:
            html = content.decode("utf-8", errors="ignore")
        return _FetchedPage(url=str(resp.url), html=html)


def _fallback_extract_text(html: str, *, max_chars: int) -> str:
    """
    Simple BeautifulSoup fallback extractor:
    - remove script/style/nav/footer/header
    - get visible text
    - normalize whitespace
    - cap output
    """
    soup = BeautifulSoup(html or "", "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        try:
            tag.decompose()
        except Exception:
            pass
    text = soup.get_text(separator=" ", strip=True)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_chars:
        text = text[:max_chars]
    return text


def _extract_main_text(html: str) -> str:
    """
    Extract main text from HTML with trafilatura, with a bs4 fallback.
    """
    txt = trafilatura_extract(
        html,
        output_format="txt",
        include_tables=True,
        include_links=False,
        favor_precision=True,
    )
    if txt is None or len((txt or "").strip()) < 200:
        txt = _fallback_extract_text(html, max_chars=200_000)
    return (txt or "").strip()


def _save_url_text(text: str, *, url: str) -> Path:
    """
    Save extracted text as a .txt file into ./data/raw.
    """
    _raw_dir().mkdir(parents=True, exist_ok=True)
    parsed = urlparse(url)
    host = (parsed.hostname or "site").replace(".", "_")
    ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    name = _safe_filename(f"url_{ts}_{host}.txt")
    dst = _resolve_collision(_raw_dir() / name)
    dst.write_text(text, encoding="utf-8")
    return dst


def _job_index_file(job_id: str, *, path: Path, source_type: str) -> None:
    """
    Background job: index a single file and update job status/progress.
    """
    try:
        _update_job(job_id, status="running", progress=10)
        res = index_path_incremental(path, source_type=source_type)
        _update_job(job_id, progress=90, result=res)
        _finish_job(job_id, status="succeeded")
    except Exception as e:
        logger.exception("Index job failed: job_id=%s path=%s", job_id, path)
        _finish_job(job_id, status="failed", error=str(e))


def _job_index_scan(job_id: str) -> None:
    """
    Background job: incremental scan.
    """
    try:
        _update_job(job_id, status="running", progress=10)
        res = index_scan_incremental()
        _update_job(job_id, progress=90, result=res)
        _finish_job(job_id, status="succeeded")
    except Exception as e:
        logger.exception("Index scan job failed: job_id=%s", job_id)
        _finish_job(job_id, status="failed", error=str(e))


@router.post("/ingest/upload")
async def ingest_upload(background_tasks: BackgroundTasks, file: UploadFile = File(...)) -> dict[str, Any]:
    """
    Upload a file to ./data/raw and queue background indexing.
    """
    job_id = _new_job()
    try:
        path, nbytes = await _save_upload_to_disk(file, max_bytes=10 * 1024 * 1024)
    except Exception as e:
        _finish_job(job_id, status="failed", error=str(e))
        raise

    background_tasks.add_task(_job_index_file, job_id, path=path, source_type="upload")
    return {"status": "ok", "saved_as": path.name, "job_id": job_id, "bytes": nbytes}


@router.post("/ingest/url")
def ingest_url(payload: dict[str, Any], background_tasks: BackgroundTasks) -> dict[str, Any]:
    """
    Fetch a single URL, extract main text, save into ./data/raw, and queue indexing.
    """
    url = str((payload or {}).get("url", "")).strip()
    if not url:
        raise HTTPException(status_code=400, detail="Missing url.")

    job_id = _new_job()
    try:
        page = _fetch_url_html(url, timeout_s=10.0, max_bytes=2 * 1024 * 1024)
        text = _extract_main_text(page.html)
        if len(text) < 200:
            raise HTTPException(status_code=400, detail="Failed to extract enough text from the URL.")
        saved = _save_url_text(text, url=page.url)
    except Exception as e:
        _finish_job(job_id, status="failed", error=str(e))
        raise

    background_tasks.add_task(_job_index_file, job_id, path=saved, source_type="url")
    return {"status": "ok", "saved_as": saved.name, "job_id": job_id, "chars": len(text)}


@router.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    """
    Get job status/progress for background indexing jobs.
    """
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found.")
        return job


@router.get("/docs")
def list_docs() -> list[dict[str, Any]]:
    """
    List files in ./data/raw (filename, size, modified time).
    """
    raw = _raw_dir()
    if not raw.exists():
        return []
    out: list[dict[str, Any]] = []
    for p in sorted(raw.iterdir(), key=lambda x: x.name.lower()):
        if not p.is_file():
            continue
        # Keep repository placeholders out of the UI.
        if p.name.startswith(".") or p.name == ".gitkeep":
            continue
        st = p.stat()
        out.append(
            {
                "filename": p.name,
                "size_bytes": int(st.st_size),
                "modified_at": float(st.st_mtime),
            }
        )
    return out


@router.post("/index")
def reindex_changed(background_tasks: BackgroundTasks) -> dict[str, Any]:
    """
    Optional endpoint: incremental scan/index of changed docs.
    """
    job_id = _new_job()
    background_tasks.add_task(_job_index_scan, job_id)
    return {"status": "ok", "job_id": job_id}


