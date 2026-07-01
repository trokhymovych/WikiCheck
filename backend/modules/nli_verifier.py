from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Literal

from pydantic import BaseModel

from .llm_client import LLMClient
from ..schemas import Passage

_SYSTEM = (
    "You are a Natural Language Inference (NLI) model for fact-checking. "
    "Given a claim and a passage from Wikipedia, determine whether the passage "
    "SUPPORTS the claim, REFUTES it, or provides NOT ENOUGH INFO.\n\n"
    "Definitions:\n"
    "- SUPPORTS: The passage directly and fully confirms the specific claim is true. "
    "Partial matches do NOT count — if the passage only confirms part of the claim "
    "(e.g. a start date but not the end, a related but different quantity, or the same "
    "topic from a different angle), use NOT ENOUGH INFO instead. Pay close attention to "
    "temporal claims: 'construction started in X' does not confirm 'was built in X'.\n"
    "- REFUTES: The passage directly contradicts the claim — it states something "
    "explicitly incompatible with the claim. Do NOT use REFUTES merely because the "
    "passage mentions a related fact, a different aspect of the same topic, or an "
    "adjacent date/number that is not in conflict. A passage that is partially "
    "consistent with the claim, or that discusses the same subject without contradicting "
    "the specific claim, should be NOT ENOUGH INFO.\n"
    "- NOT ENOUGH INFO: The passage does not directly and fully confirm or contradict "
    "the claim. Use this when the passage is tangentially related, covers only part of "
    "the claim, discusses the same topic from a different angle, or lacks the specific "
    "information needed to fully verify the claim.\n\n"
    "Steps:\n"
    "1. Identify exactly what the claim asserts.\n"
    "2. Find the most relevant span in the passage.\n"
    "3. Ask: does the span directly confirm the claim, directly contradict it, or neither?\n"
    "4. Write your reasoning, then give your final label and confidence.\n\n"
    "Also extract the exact span (substring) from the passage most relevant to your decision. "
    "For NOT ENOUGH INFO, leave span empty. "
    "Return confidence 'low', 'medium', or 'high' based on how directly the passage addresses the claim."
)


class _NLIOutput(BaseModel):
    reasoning: str
    label: Literal["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
    confidence: Literal["low", "medium", "high"]
    span: str


@dataclass
class NLIResult:
    passage: Passage
    label: str
    confidence: str
    span: str


class NLIVerifier:
    def __init__(self, llm: LLMClient, max_workers: int = 8):
        self._llm = llm
        self._max_workers = max_workers

    def _verify_single(self, claim: str, passage: Passage) -> NLIResult:
        user = f"Claim: {claim}\n\nPassage: {passage.text}"
        output: _NLIOutput = self._llm.complete(
            system=_SYSTEM,
            user=user,
            response_format=_NLIOutput,
        )
        return NLIResult(
            passage=passage,
            label=output.label,
            confidence=output.confidence,
            span=output.span,
        )

    def verify(self, claim: str, passages: List[Passage]) -> List[NLIResult]:
        results: List[NLIResult] = []
        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            futures = {
                executor.submit(self._verify_single, claim, p): p for p in passages
            }
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception:
                    pass
        return results
