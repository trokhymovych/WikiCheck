from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel, Field


class Passage(BaseModel):
    article: str
    text: str
    section: str
    url: str
    language: str
    references: List[str] = Field(default_factory=list)


class AccessCodeRequest(BaseModel):
    code: str


class CheckRequest(BaseModel):
    claim: str
    languages: List[str] = Field(default=["en"])


# ── Stage 1: search ──────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    claim: str
    languages: List[str] = Field(default=["en"])


class SearchResponse(BaseModel):
    queries: List[str]
    passages: List[Passage]
    article_count: int
    passage_count: int


# ── Stage 2: rerank ──────────────────────────────────────────────────────────

class RerankRequest(BaseModel):
    claim: str
    passages: List[Passage]
    queries: List[str]


class RerankResponse(BaseModel):
    queries: List[str]
    passages: List[Passage]
    passage_count: int


# ── Stage 3: verify ───────────────────────────────────────────────────────────

class VerifyRequest(BaseModel):
    claim: str
    passages: List[Passage]
    queries: List[str]


# ── Shared output types ───────────────────────────────────────────────────────

class EvidenceCard(BaseModel):
    article: str
    text: str
    span: str
    confidence: str
    language: str
    section: str
    url: str
    references: List[str] = Field(default_factory=list)


class Verdict(BaseModel):
    label: str
    confidence: str
    support_count: int
    refute_count: int
    nei_count: int
    total_count: int


class Trace(BaseModel):
    claim: str
    rewritten_queries: List[str]
    retrieved_articles: int
    total_passages: int
    nli_breakdown: Dict[str, int]


class CheckResponse(BaseModel):
    verdict: Verdict
    supporting_evidence: List[EvidenceCard]
    refuting_evidence: List[EvidenceCard]
    trace: Trace
