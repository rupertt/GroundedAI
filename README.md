# rag-citations-app

Minimal local **Python 3.11** RAG web app that answers questions using **OpenAI + LangChain** with **grounded citations**.

## What it does

- **Indexes docs** into local persistent **Chroma** at `./data/index` (SQLite-backed).
  - Default: `./data/doc.txt` (single-doc mode)
  - Optional: `./data/raw/*` (multi-doc mode; supports `.txt`, `.md`, `.pdf`, `.docx`)
- **Chunks deterministically** with stable IDs: `chunk-00`, `chunk-01`, …
  - Hierarchical splitting by section headers where possible, then recursive splitting
  - Stores metadata: `source` (filename) and `section`
- **Retrieves** with improved diversity (MMR where available) and respects `top_k` throughout.
- **Answers** using only retrieved chunks and includes citations:
  - Single doc: `[doc.txt#chunk-03]`
  - Multi-doc: `[<filename>#chunk-03]`

If the documentation doesn’t contain enough info, the API returns:

`I can’t find that in the provided documentation.`

## Setup (Windows)

From PowerShell:

```powershell
cd "C:\Users\Rupert\Desktop\Coding Projects\Agentic AI\rag-citations-app"
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

Edit `.env` and set `OPENAI_API_KEY`.

Run:

```powershell
uvicorn app.main:app --reload
```

## Setup (Linux / WSL)

```bash
cd "rag-citations-app"
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and set `OPENAI_API_KEY`.

Run:

```bash
uvicorn app.main:app --reload
```

Open the UI:

- `http://127.0.0.1:8000/`

### If you see “unsupported version of sqlite3” (Chroma)

Some WSL distros ship an older SQLite library. This project includes `pysqlite3-binary` and automatically uses it when needed (see `app/rag.py`). If you still hit the error, reinstall deps:

```bash
pip install -U -r requirements.txt
```

## API

### Health

- `GET /health` → `{"status":"ok"}`

### Ask

- `POST /ask`

Request:

```json
{"question":"<string>","top_k":4,"debug":false}
```

Example curl:

```bash
curl -s -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What does this service do?","top_k":4,"debug":true}'
```

Response shape:

```json
{
  "answer": "<string>",
  "citations": [{"source":"doc.txt","chunk_id":"chunk-03","snippet":"..."}],
  "debug": {
    "retrieved": [{"chunk_id":"chunk-03","text":"...","score":0.12}]
  }
}
```

### Agent mode (CrewAI)

- `POST /ask_agent` (same request/response schema as `/ask`)

In the web UI (`/`), enable the **Agent mode** toggle to call `/ask_agent`.

## Adding new documents (multi-document mode)

You can add documentation in two ways:

### 1) Upload a file (background indexed)

- `POST /ingest/upload` (multipart/form-data field name: `file`)
- Allowed extensions: `.pdf`, `.docx`, `.txt`
- Saved as-is into `./data/raw/<safe_filename>`
- Returns quickly with a `job_id` and indexes in the background.

Example curl:

```bash
curl -s -X POST http://127.0.0.1:8000/ingest/upload \
  -F "file=@./some_doc.pdf"
```

### 2) Add a URL (single page only; background indexed)

- `POST /ingest/url` with JSON: `{"url":"https://..."}`
- Fetches ONLY the provided URL (no crawling), extracts main text, saves it as a `.txt` under `./data/raw/`, then indexes in the background.

Example curl:

```bash
curl -s -X POST http://127.0.0.1:8000/ingest/url \
  -H "Content-Type: application/json" \
  -d '{"url":"https://example.com/docs/page"}'
```

### Job status

- `GET /jobs/{job_id}` returns: `job_id`, `status`, `progress`, `error`, `created_at`, `finished_at`

### List docs

- `GET /docs` returns files present in `./data/raw` with `filename`, `size_bytes`, `modified_at`

### Reindex changed docs (optional)

- `POST /index` triggers an incremental scan in the background and returns a `job_id`

### Index persistence + incremental behavior

- The vector store is persisted under `./data/index`.
- Incremental state is tracked in `./data/index/manifest.json`.
- The app avoids full rebuilds: it deletes/reindexes only the files that changed (or were removed).

## Evals (measurable quality)

Lightweight eval cases live in `./evals/cases.json` and run via pytest.

Run the evals with pytest (requires `OPENAI_API_KEY`):

```bash
cd rag-citations-app
pytest -q
```

What the evals check:
- Response schema validates (`AskResponse`)
- For answerable questions:
  - At least one citation token exists (`[<filename>#chunk-XX]`)
  - All cited `chunk_id`s were retrieved (based on `debug.retrieved`)
- For out-of-scope questions:
  - Refusal text is exact: `I can’t find that in the provided documentation.`

Interpreting failures:
- If a test fails on missing citations, the app will now fail closed and refuse.
- If a test fails on “cited chunk_ids not retrieved”, it indicates hallucinated citations or a bug in retrieval/citation wiring.


