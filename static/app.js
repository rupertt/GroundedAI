// Minimal UI logic: POST /ask and render answer + citations (+ debug details).
// No external dependencies, no frameworks.

function el(id) {
  return document.getElementById(id);
}

function setLoading(isLoading) {
  const btn = el("askBtn");
  const status = el("status");
  btn.disabled = isLoading;
  status.textContent = isLoading ? "Asking..." : "";
}

function setIngestStatus(text) {
  el("ingestStatus").textContent = text || "";
}

function setJobStatus(text) {
  el("jobStatus").textContent = text || "";
}

function showResult() {
  el("result").classList.remove("hidden");
}

function clearResult() {
  el("answer").textContent = "";
  el("citations").innerHTML = "";
  el("debugRetrieved").innerHTML = "";
  el("debugDetails").classList.add("hidden");
}

function renderCitations(citations) {
  const ul = el("citations");
  ul.innerHTML = "";

  for (const c of citations || []) {
    const li = document.createElement("li");
    const title = document.createElement("div");
    // Include filename for multi-doc readability.
    title.textContent = `${c.source}#${c.chunk_id}`;

    const snippet = document.createElement("div");
    snippet.className = "snippet";
    snippet.textContent = c.snippet || "";

    li.appendChild(title);
    li.appendChild(snippet);
    ul.appendChild(li);
  }
}

function renderDebug(debug) {
  if (!debug || !debug.retrieved) {
    el("debugDetails").classList.add("hidden");
    return;
  }

  const container = el("debugRetrieved");
  container.innerHTML = "";

  for (const r of debug.retrieved) {
    const pre = document.createElement("pre");
    pre.textContent = `[${r.chunk_id}] (score=${r.score})\n${r.text}`;
    container.appendChild(pre);
  }

  el("debugDetails").classList.remove("hidden");
}

async function ask() {
  const question = el("question").value.trim();
  const topK = Number(el("top_k").value || "4");
  const debug = el("debug").checked;
  const agentMode = el("agent_mode").checked;

  if (!question) {
    el("status").textContent = "Please enter a question.";
    return;
  }

  setLoading(true);
  clearResult();

  try {
    const endpoint = agentMode ? "/ask_agent" : "/ask";
    const resp = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: question, top_k: topK, debug: debug }),
    });

    const data = await resp.json();
    if (!resp.ok) {
      throw new Error(data?.detail || `Request failed with status ${resp.status}`);
    }

    el("answer").textContent = data.answer || "";
    renderCitations(data.citations || []);
    renderDebug(data.debug);
    showResult();
  } catch (err) {
    el("status").textContent = `Error: ${err.message || String(err)}`;
  } finally {
    setLoading(false);
  }
}

async function fetchJson(url, options) {
  const resp = await fetch(url, options || {});
  let data = null;
  try {
    data = await resp.json();
  } catch (e) {
    data = null;
  }
  if (!resp.ok) {
    let detail = data?.detail;
    if (detail && typeof detail !== "string") {
      try {
        detail = JSON.stringify(detail);
      } catch (e) {
        detail = String(detail);
      }
    }
    const msg = detail || `Request failed with status ${resp.status}`;
    throw new Error(msg);
  }
  return data;
}

async function refreshDocs() {
  const docs = await fetchJson("/docs");
  const ul = el("docsList");
  ul.innerHTML = "";
  for (const d of docs || []) {
    const li = document.createElement("li");
    li.textContent = `${d.filename} (${d.size_bytes} bytes)`;
    ul.appendChild(li);
  }
  if (!docs || docs.length === 0) {
    const li = document.createElement("li");
    li.className = "subtle";
    li.textContent = "No docs found in ./data/raw yet.";
    ul.appendChild(li);
  }
}

async function pollJob(jobId) {
  setJobStatus(`Job ${jobId}: queued...`);
  while (true) {
    const job = await fetchJson(`/jobs/${jobId}`);
    const status = job.status || "unknown";
    const progress = job.progress ?? 0;
    const err = job.error;
    setJobStatus(
      `Job ${jobId}\nstatus: ${status}\nprogress: ${progress}%` +
        (err ? `\nerror: ${err}` : "")
    );
    if (status === "succeeded" || status === "failed") {
      break;
    }
    await new Promise((r) => setTimeout(r, 800));
  }
}

async function uploadDoc() {
  const fileInput = el("uploadFile");
  const file = fileInput.files && fileInput.files[0];
  if (!file) {
    setIngestStatus("Please choose a file to upload.");
    return;
  }

  setIngestStatus("Uploading...");
  setJobStatus("");
  const form = new FormData();
  form.append("file", file);
  const data = await fetchJson("/ingest/upload", { method: "POST", body: form });
  setIngestStatus(`Saved as ${data.saved_as}. Indexing in background...`);
  await pollJob(data.job_id);
  await refreshDocs();
  setIngestStatus("Done.");
}

async function addUrl() {
  const url = el("urlInput").value.trim();
  if (!url) {
    setIngestStatus("Please enter a URL.");
    return;
  }
  setIngestStatus("Fetching URL and extracting text...");
  setJobStatus("");
  const data = await fetchJson("/ingest/url", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url }),
  });
  setIngestStatus(`Saved as ${data.saved_as}. Indexing in background...`);
  await pollJob(data.job_id);
  await refreshDocs();
  setIngestStatus("Done.");
}

async function reindexChanged() {
  setIngestStatus("Starting incremental reindex...");
  setJobStatus("");
  const data = await fetchJson("/index", { method: "POST" });
  await pollJob(data.job_id);
  await refreshDocs();
  setIngestStatus("Done.");
}

function wireEvents() {
  el("askBtn").addEventListener("click", ask);
  el("question").addEventListener("keydown", (e) => {
    // Ctrl+Enter submits the question.
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
      ask();
    }
  });

  el("uploadBtn").addEventListener("click", () => {
    uploadDoc().catch((err) => {
      setIngestStatus(`Error: ${err.message || String(err)}`);
    });
  });

  el("urlBtn").addEventListener("click", () => {
    addUrl().catch((err) => {
      setIngestStatus(`Error: ${err.message || String(err)}`);
    });
  });

  el("refreshDocsBtn").addEventListener("click", () => {
    refreshDocs().catch((err) => {
      setIngestStatus(`Error: ${err.message || String(err)}`);
    });
  });

  el("reindexBtn").addEventListener("click", () => {
    reindexChanged().catch((err) => {
      setIngestStatus(`Error: ${err.message || String(err)}`);
    });
  });
}

wireEvents();
// Load docs list on initial page load.
refreshDocs().catch(() => {});


