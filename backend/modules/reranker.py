from __future__ import annotations

import os
from typing import List

from sentence_transformers import CrossEncoder

from ..schemas import Passage

_MODEL_NAME = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
_DEFAULT_THRESHOLD = float(os.getenv("RERANKER_THRESHOLD", "0.0"))


class CrossEncoderReranker:
    """Score (claim, passage) pairs and drop passages below a relevance threshold."""

    def __init__(
        self,
        model_name: str = _MODEL_NAME,
        threshold: float = _DEFAULT_THRESHOLD,
    ) -> None:
        self._model = CrossEncoder(model_name)
        self.threshold = threshold

    def rerank(self, claim: str, passages: List[Passage]) -> List[Passage]:
        if not passages:
            return []
        pairs = [(claim, p.text) for p in passages]
        scores = self._model.predict(pairs).tolist()
        scored = sorted(zip(scores, passages), key=lambda x: x[0], reverse=True)
        return [p for score, p in scored if score >= self.threshold]
