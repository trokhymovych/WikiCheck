from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Any, Type

import httpx
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
    BASE_URL = "https://api.wikimedia.org/service/lw/inference/v1/models/{model}/openai/v1/chat/completions"

    def __init__(self, model: str = "qwen3-14b"):
        self._url = self.BASE_URL.format(model=model)
        self.model = model

    def complete(self, system: str, user: str, response_format: Type[BaseModel]) -> Any:
        schema_str = json.dumps(response_format.model_json_schema(), indent=2)
        augmented_system = (
            f"{system}\n\nRespond with valid JSON matching this schema:\n{schema_str}"
        )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": augmented_system},
                {"role": "user", "content": user},
            ],
            "response_format": {"type": "json_object"},
            "stream": False,
        }
        headers = {
            "Content-Type": "application/json",
            "User-Agent": os.environ["WIKIMEDIA_USER_AGENT"],
        }
        response = httpx.post(self._url, json=payload, headers=headers)
        if not response.is_success:
            raise RuntimeError(f"Wikimedia API {response.status_code}: {response.text}")
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        return response_format.model_validate_json(content)
