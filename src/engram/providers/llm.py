"""LLM provider interface and implementations for consolidation."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """Abstract LLM provider for consolidation stages that need intelligence."""

    @abstractmethod
    def complete(self, prompt: str, system: str = "", max_tokens: int = 1024) -> str:
        """Get a completion from the LLM."""
        ...

    @abstractmethod
    def extract_json(self, prompt: str, system: str = "") -> dict:
        """Get a JSON-structured completion."""
        ...


class NoLLMProvider(LLMProvider):
    """Stub provider when no LLM is configured. Consolidation degrades gracefully."""

    def complete(self, prompt: str, system: str = "", max_tokens: int = 1024) -> str:
        return ""

    def extract_json(self, prompt: str, system: str = "") -> dict:
        return {}


class LiteLLMProvider(LLMProvider):
    """LiteLLM-based provider (optional dependency)."""

    def __init__(self, model: str = "gpt-4o-mini", **kwargs: Any):
        try:
            import litellm
            self._litellm = litellm
        except ImportError:
            raise ImportError("Install litellm: pip install engram[llm]")
        self.model = model
        self.kwargs = kwargs

    def complete(self, prompt: str, system: str = "", max_tokens: int = 1024) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = self._litellm.completion(
            model=self.model, messages=messages, max_tokens=max_tokens, **self.kwargs
        )
        return response.choices[0].message.content or ""

    def extract_json(self, prompt: str, system: str = "") -> dict:
        import json
        text = self.complete(prompt, system, max_tokens=2048)
        # Try to extract JSON from the response
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1])
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON in the response
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start:end])
                except json.JSONDecodeError:
                    pass
            logger.warning("failed to extract JSON from LLM response")
            return {}
