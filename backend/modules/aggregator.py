from __future__ import annotations

from typing import List, Tuple

from ..schemas import EvidenceCard, Verdict
from .nli_verifier import NLIResult

SUPPORTS = "SUPPORTS"
REFUTES = "REFUTES"
NEI = "NOT ENOUGH INFO"

_SCORE = {"low": 0.33, "medium": 0.66, "high": 1.0}


def _to_label(score: float) -> str:
    if score > 0.66:
        return "high"
    if score > 0.33:
        return "medium"
    return "low"


def _to_card(r: NLIResult) -> EvidenceCard:
    return EvidenceCard(
        article=r.passage.article,
        text=r.passage.text,
        span=r.span,
        confidence=r.confidence,
        language=r.passage.language,
        section=r.passage.section,
        url=r.passage.url,
        references=r.passage.references,
    )


def aggregate(
    results: List[NLIResult],
) -> Tuple[Verdict, List[EvidenceCard], List[EvidenceCard]]:
    supporting = [r for r in results if r.label == SUPPORTS]
    refuting = [r for r in results if r.label == REFUTES]
    nei = [r for r in results if r.label == NEI]

    support_score = sum(_SCORE[r.confidence] for r in supporting)
    refute_score = sum(_SCORE[r.confidence] for r in refuting)

    if support_score == 0 and refute_score == 0:
        final_label = NEI
        confidence = "low"
    elif support_score >= refute_score:
        final_label = SUPPORTS
        avg = support_score / len(supporting) if supporting else 0.0
        confidence = _to_label(avg)
    else:
        final_label = REFUTES
        avg = refute_score / len(refuting) if refuting else 0.0
        confidence = _to_label(avg)

    supporting_cards = [
        _to_card(r) for r in sorted(supporting, key=lambda r: _SCORE[r.confidence], reverse=True)
    ]
    refuting_cards = [
        _to_card(r) for r in sorted(refuting, key=lambda r: _SCORE[r.confidence], reverse=True)
    ]

    verdict = Verdict(
        label=final_label,
        confidence=confidence,
        support_count=len(supporting),
        refute_count=len(refuting),
        nei_count=len(nei),
        total_count=len(results),
    )

    return verdict, supporting_cards, refuting_cards
