from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Any, Type

from openai import OpenAI
from pydantic import BaseModel


class LLMClient(ABC):
    @abstractmethod
    def complete(self, system: str, user: str, response_format: Type[BaseModel]) -> Any:
        ...


class OpenAIClient(LLMClient):
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self._client = OpenAI(api_key=api_key)
        self.model = model

    def complete(self, system: str, user: str, response_format: Type[BaseModel]) -> Any:
        response = self._client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format=response_format,
        )
        return response.choices[0].message.parsed


class WikimediaLLMClient(LLMClient):
    def __init__(self, model: str = "qwen3-14b"):
        base_url = os.environ["WIKIMEDIA_INFERENCE_BASE"].format(model=model)
        api_key = os.environ["WIKIMEDIA_API_KEY"]
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self.model = model

    def complete(self, system: str, user: str, response_format: Type[BaseModel]) -> Any:
        schema_str = json.dumps(response_format.model_json_schema(), indent=2)
        augmented_system = (
            f"{system}\n\nRespond with valid JSON matching this schema:\n{schema_str}"
        )
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": augmented_system},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
        )
        return response_format.model_validate_json(response.choices[0].message.content)
