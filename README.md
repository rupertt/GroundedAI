# GroundedAI

GroundedAI is a minimal local **Python 3.11** RAG web app that answers questions using **OpenAI + LangChain** with **grounded citations** to the documentation you provide.

## What it does

- **Ingests + indexes** your docs into a local persistent **Chroma** store at `./data/index` (SQLite-backed).
  - Default example doc: `./data/doc.txt`
  - Multi-doc mode: files under `./data/raw/` (supports `.txt`, `.md`, `.pdf`, `.docx`)
- **Chunks deterministically** with stable IDs: `chunk-00`, `chunk-01`, …
- **Retrieves** relevant chunks and **answers only from retrieved text**.
- **Cites sources** inline:
  - Single doc: `[doc.txt#chunk-03]`
  - Multi-doc: `[<filename>#chunk-03]`

If the docs don’t contain enough info, the API returns:

`I can’t find that in the provided documentation.`

## Install (Linux / WSL)

```bash
cd rag-citations-app
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create `.env` and set `OPENAI_API_KEY`:

```bash
cp .env.example .env
```

Then edit `.env` and set `OPENAI_API_KEY`.

Run:

```bash
uvicorn app.main:app --reload
```

## Web UI

Open:

- `http://127.0.0.1:8000/`

In the UI you can:

- Ask questions (with citations)
- Toggle **Agent mode** (uses `/ask_agent`)
- Upload docs or add a URL
- Reindex changed docs and view job status

## API

### Ask

- `POST /ask`

Request:

```json
{"question":"<string>","top_k":4,"debug":false}
```

Example:

```bash
curl -s -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What does GroundedAI do?","top_k":4,"debug":true}'
```

### Agent mode

- `POST /ask_agent` (same request/response schema as `/ask`)

Example:

```bash
curl -s -X POST http://127.0.0.1:8000/ask_agent \
  -H "Content-Type: application/json" \
  -d '{"question":"Summarize the key concepts in the docs.","top_k":4,"debug":false}'
```

### Add documentation

- **Upload**: `POST /ingest/upload` (multipart field name: `file`)
- **Add URL**: `POST /ingest/url` with JSON `{"url":"https://..."}`

Examples:

```bash
curl -s -X POST http://127.0.0.1:8000/ingest/upload \
  -F "file=@./some_doc.pdf"
```

```bash
curl -s -X POST http://127.0.0.1:8000/ingest/url \
  -H "Content-Type: application/json" \
  -d '{"url":"https://example.com/docs/page"}'
```

### Jobs + docs

- **Job status**: `GET /jobs/{job_id}`
- **List docs**: `GET /docs` (files present in `./data/raw`)
- **Reindex changed docs**: `POST /index` (returns a `job_id`)

### If you see “unsupported version of sqlite3” (Chroma)

Some WSL distros ship an older SQLite library. This project includes `pysqlite3-binary` and automatically uses it when needed (see `app/rag.py`). If you still hit the error, reinstall deps:

```bash
pip install -U -r requirements.txt
```
