"""Text generation adapters used by the Naive RAG pipeline."""

from __future__ import annotations

import os
from typing import Protocol
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv is not None:
    project_root = Path(__file__).resolve().parents[2]
    load_dotenv(project_root / ".env")


DEFAULT_GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.6-flash",
)


class TextGenerator(Protocol):
    """Minimal interface required by the RAG pipeline."""

    def generate(self, prompt: str) -> str:
        ...


class GeminiTextGenerator:
    """Generate grounded answers with the Google Gen AI SDK."""

    def __init__(
        self,
        model_name: str = DEFAULT_GEMINI_MODEL,
        client: object | None = None,
        temperature: float = 0.1,
    ):
        if not model_name.strip():
            raise ValueError("model_name cannot be empty.")

        if not 0.0 <= temperature <= 2.0:
            raise ValueError(
                "temperature must be between 0.0 and 2.0."
            )

        self.model_name = model_name
        self.temperature = temperature
        self._client = client

    @property
    def client(self):
        """Create the Gemini client only when generation is requested."""

        if self._client is None:
            try:
                from google import genai
            except ImportError as exc:
                raise RuntimeError(
                    "Google Gen AI SDK is not installed. Run: "
                    'pip install "google-genai>=1.0,<2.0"'
                ) from exc

            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "GEMINI_API_KEY is not set."
                )

            self._client = genai.Client(
                api_key=api_key
            )

        return self._client

    def generate(self, prompt: str) -> str:
        normalized_prompt = prompt.strip()

        if not normalized_prompt:
            raise ValueError("The generation prompt cannot be empty.")

        try:
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError(
                "Google Gen AI SDK is not installed."
            ) from exc

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=normalized_prompt,
            config=types.GenerateContentConfig(
                temperature=self.temperature,
            ),
        )

        answer = (response.text or "").strip()

        if not answer:
            raise RuntimeError(
                "Gemini returned an empty answer."
            )

        return answer
