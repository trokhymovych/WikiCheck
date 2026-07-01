from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field

from .llm_client import LLMClient

_SYSTEM = (
    "You are a named-entity extraction assistant for a Wikipedia fact-checking system. "
    "Given a factual claim, extract up to 5 named entities for Wikipedia search. Include:\n"
    "  (a) entities explicitly mentioned in the claim, and\n"
    "  (b) at most 1-2 well-known named concepts that directly bear on the truth of the "
    "claim's specific assertion (e.g. for 'The Earth is flat', 'Circumnavigation of the Earth' "
    "directly tests Earth's shape; for 'Shakespeare wrote Hamlet', no expansion is needed).\n"
    "For each entity provide:\n"
    "  - name: the entity name exactly as it would appear as a Wikipedia article title "
    "(use standard English capitalization; include disambiguating context only when the bare "
    "name is highly ambiguous, e.g. 'Mercury (planet)' vs 'Mercury (element)').\n"
    "  - entity_type: one of person, organization, location, event, work, concept.\n"
    "Do NOT include entities that are only loosely or thematically related to the claim. "
    "Do not repeat the same entity under different phrasings. "
    "Do not generate questions or verb phrases."
)


class _Entity(BaseModel):
    name: str = Field(
        description=(
            "The entity name exactly as it would appear as a Wikipedia article title. "
            "Use the most specific, unambiguous form (e.g. 'Eiffel Tower', not 'tower'). "
            "Use Wikipedia disambiguation format when needed, e.g. 'Mercury (planet)'."
        )
    )
    entity_type: Literal["person", "organization", "location", "event", "work", "concept"] = Field(
        description="The NER category of this entity."
    )


class _EntityList(BaseModel):
    entities: List[_Entity] = Field(
        description="Named entities explicitly mentioned in or directly referred to by the claim. Between 1 and 5.",
        min_length=1,
        max_length=5,
    )


class QueryReformulator:
    def __init__(self, llm: LLMClient):
        self._llm = llm

    def reformulate(self, claim: str) -> List[str]:
        result: _EntityList = self._llm.complete(
            system=_SYSTEM,
            user=f"Claim: {claim}",
            response_format=_EntityList,
        )
        return [entity.name for entity in result.entities]
