from typing import Optional, Union
from pydantic import BaseModel  # type: ignore


# API data types handling
class TextRequest(BaseModel):
    text: str


class Token(BaseModel):
    access_token: str
    token_type: str


class NLIRequest(BaseModel):
    claim: str
    hypothesis: str


class NLIResponse(BaseModel):
    label: str
    contradiction_prob: float
    entailment_prob: float
    neutral_prob: float


class ClaimFactCheckResponse(BaseModel):
    claim: str
    text: str
    article: str
    label: str
    contradiction_prob: float
    entailment_prob: float
    neutral_prob: float


class FactCheckResponse(BaseModel):
    predicted_label: str
    predicted_evidence: list[list[str]]


class TokenData(BaseModel):
    username: Union[str, None] = None


class User(BaseModel):
    username: str
    email: Union[str, None] = None
    full_name: Union[str, None] = None
    disabled: Union[bool, None] = None
    credits: int = 0


class UserInDB(User):
    hashed_password: str
