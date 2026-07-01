# WikiCheck

Fact verification tool powered by Wikipedia. Enter a factual claim and WikiCheck retrieves relevant Wikipedia passages, ranks them by relevance, and runs NLI (Natural Language Inference) to produce a **Supports / Refutes / Not Enough Info** verdict.

## How it works

```
Claim → Query reformulation (LLM) → Wikipedia search → Passage reranking (cross-encoder) → NLI verification → Verdict
```

1. **Query reformulation** — an LLM rewrites the claim into search-optimised queries
2. **Wikipedia search** — queries hit the Wikipedia API across selected languages
3. **Reranking** — a cross-encoder model scores and filters passages by relevance
4. **NLI verification** — each passage is classified as Supports / Refutes / NEI; results are aggregated into a final verdict

## Project structure

```
backend/        FastAPI app — API endpoints + model inference
  app.py        Entry point, routes, static file serving
  schemas.py    Pydantic request/response models
  modules/      Wikipedia search, reranker, NLI verifier, LLM client
  .env          Environment config (see below)
web_app/        Frontend — single HTML file + support.js
requirements.txt
```

## Setup & deployment

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp backend/.env.example backend/.env
```

Edit `backend/.env`:

```
ACCESS_CODE=abc123        # 6-char gate code shown to users; leave empty to disable
LLM_PROVIDER=wikimedia    # or: openai
WIKIMEDIA_API_KEY=...     # JWT token for Wikimedia inference API
WIKIMEDIA_MODEL=qwen3-14b
# OPENAI_API_KEY=...      # only if LLM_PROVIDER=openai
```

### 3. Run

```bash
uvicorn backend.app:app --host 0.0.0.0 --port 80
```

> On Linux, port 80 requires root: `sudo uvicorn backend.app:app --host 0.0.0.0 --port 80`

The app is then available at `http://<your-server-ip>/`.

### Run in background (production)

```bash
nohup uvicorn backend.app:app --host 0.0.0.0 --port 80 &> wikicheck.log &
```

### Local development

```bash
uvicorn backend.app:app --reload --port 8000
# open http://localhost:8000
```
