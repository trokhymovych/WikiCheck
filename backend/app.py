from __future__ import annotations

import os
import pathlib

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

load_dotenv()

from .modules.aggregator import aggregate
from .modules.llm_client import OpenAIClient, WikimediaLLMClient, LLMClient
from .modules.nli_verifier import NLIVerifier
from .modules.query_reformulator import QueryReformulator
from .modules.reranker import CrossEncoderReranker
from .modules import wikipedia_search_v2
from .schemas import (
    AccessCodeRequest,
    CheckRequest, CheckResponse, Trace,
    SearchRequest, SearchResponse,
    RerankRequest, RerankResponse,
    VerifyRequest,
)

_provider = os.getenv("LLM_PROVIDER", "wikimedia")

def _build_llm() -> LLMClient:
    if _provider == "openai":
        return OpenAIClient(
            api_key=os.environ["OPENAI_API_KEY"],
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        )
    if _provider == "wikimedia":
        return WikimediaLLMClient(model=os.getenv("WIKIMEDIA_MODEL", "qwen3-14b"))
    raise ValueError(f"Unknown LLM_PROVIDER: {_provider!r}. Choose 'openai' or 'wikimedia'.")

_llm = _build_llm()
_reformulator = QueryReformulator(llm=_llm)
_verifier = NLIVerifier(llm=_llm)
_reranker = CrossEncoderReranker()

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="WikiCheck API")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


def _require_code(x_access_code: str = Header(default="")) -> None:
    expected = os.getenv("ACCESS_CODE", "")
    if expected and x_access_code != expected:
        raise HTTPException(status_code=403, detail="Invalid access code")


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://localhost:8000",
        "http://127.0.0.1",
        "http://127.0.0.1:8000",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "null",  # file:// origins
    ],
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)


# ── Stage 1: Query reformulation + Wikipedia search ───────────────────────────

@app.post("/api/search", response_model=SearchResponse, dependencies=[Depends(_require_code)])
@limiter.limit("3/minute")
def api_search(request: Request, req: SearchRequest) -> SearchResponse:
    queries = _reformulator.reformulate(req.claim)
    passages = wikipedia_search_v2.search(queries, req.languages)
    article_count = len({p.article for p in passages})
    return SearchResponse(
        queries=queries,
        passages=passages,
        article_count=article_count,
        passage_count=len(passages),
    )


# ── Stage 2: Passage reranking (cross-encoder relevance scoring + filtering) ──

@app.post("/api/rerank", response_model=RerankResponse, dependencies=[Depends(_require_code)])
@limiter.limit("3/minute")
def api_rerank(request: Request, req: RerankRequest) -> RerankResponse:
    ranked = _reranker.rerank(req.claim, req.passages)
    return RerankResponse(
        queries=req.queries,
        passages=ranked,
        passage_count=len(ranked),
    )


# ── Stage 3: NLI verification + aggregation ───────────────────────────────────

@app.post("/api/verify", response_model=CheckResponse, dependencies=[Depends(_require_code)])
@limiter.limit("3/minute")
def api_verify(request: Request, req: VerifyRequest) -> CheckResponse:
    nli_results = _verifier.verify(req.claim, req.passages)
    verdict, supporting, refuting = aggregate(nli_results)

    trace = Trace(
        claim=req.claim,
        rewritten_queries=req.queries,
        retrieved_articles=len({p.article for p in req.passages}),
        total_passages=len(req.passages),
        nli_breakdown={
            "supports": verdict.support_count,
            "refutes": verdict.refute_count,
            "nei": verdict.nei_count,
        },
    )

    return CheckResponse(
        verdict=verdict,
        supporting_evidence=supporting,
        refuting_evidence=refuting,
        trace=trace,
    )


# ── Convenience single-call endpoint ─────────────────────────────────────────

@app.post("/check", response_model=CheckResponse, dependencies=[Depends(_require_code)])
@limiter.limit("3/minute")
def check_claim(request: Request, req: CheckRequest) -> CheckResponse:
    queries = _reformulator.reformulate(req.claim)
    passages = wikipedia_search_v2.search(queries, req.languages)
    passages = _reranker.rerank(req.claim, passages)
    nli_results = _verifier.verify(req.claim, passages)
    verdict, supporting, refuting = aggregate(nli_results)

    trace = Trace(
        claim=req.claim,
        rewritten_queries=queries,
        retrieved_articles=len({p.article for p in passages}),
        total_passages=len(passages),
        nli_breakdown={
            "supports": verdict.support_count,
            "refutes": verdict.refute_count,
            "nei": verdict.nei_count,
        },
    )

    return CheckResponse(
        verdict=verdict,
        supporting_evidence=supporting,
        refuting_evidence=refuting,
        trace=trace,
    )


@app.post("/api/verify-access")
@limiter.limit("5/minute")
def verify_access(request: Request, req: AccessCodeRequest) -> dict:
    expected = os.getenv("ACCESS_CODE", "")
    if not expected or req.code == expected:
        return {"ok": True}
    return {"ok": False}


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


# ── Static frontend ───────────────────────────────────────────────────────────

_WEB = pathlib.Path(__file__).parent.parent / "web_app"

@app.get("/")
def index():
    return FileResponse(_WEB / "WikiCheck.dc.html")

@app.get("/support.js")
def support_js():
    return FileResponse(_WEB / "support.js")
