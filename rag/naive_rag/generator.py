"""Text generation adapters used by the RAG pipelines."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Protocol

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


@dataclass(frozen=True, slots=True)
class GenerationUsage:
    """Token usage reported by one model generation call."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class TextGenerator(Protocol):
    """Minimal interface required by the RAG pipelines."""

    def generate(self, prompt: str) -> str:
        ...


class GeminiTextGenerator:
    """Generate grounded answers and retain API-reported token usage."""

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
        self._usage_history: list[GenerationUsage] = []

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

    @property
    def last_usage(self) -> GenerationUsage:
        """Return usage for the most recent generation."""

        if not self._usage_history:
            return GenerationUsage()
        return self._usage_history[-1]

    @property
    def usage_totals(self) -> GenerationUsage:
        """Return cumulative usage since the last reset."""

        return GenerationUsage(
            input_tokens=sum(
                item.input_tokens
                for item in self._usage_history
            ),
            output_tokens=sum(
                item.output_tokens
                for item in self._usage_history
            ),
            total_tokens=sum(
                item.total_tokens
                for item in self._usage_history
            ),
        )

    def reset_usage(self) -> None:
        """Clear accumulated usage before an evaluation run."""

        self._usage_history.clear()

    def generate(self, prompt: str) -> str:
        normalized_prompt = prompt.strip()

        if not normalized_prompt:
            raise ValueError(
                "The generation prompt cannot be empty."
            )

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

        self._usage_history.append(
            self._extract_usage(response)
        )

        answer = (response.text or "").strip()

        if not answer:
            raise RuntimeError(
                "Gemini returned an empty answer."
            )

        return answer

    @staticmethod
    def _extract_usage(response: object) -> GenerationUsage:
        usage = getattr(
            response,
            "usage_metadata",
            None,
        )

        if usage is None:
            return GenerationUsage()

        input_tokens = int(
            getattr(
                usage,
                "prompt_token_count",
                0,
            )
            or 0
        )
        output_tokens = int(
            getattr(
                usage,
                "candidates_token_count",
                None,
            )
            or getattr(
                usage,
                "response_token_count",
                0,
            )
            or 0
        )
        total_tokens = int(
            getattr(
                usage,
                "total_token_count",
                0,
            )
            or (input_tokens + output_tokens)
        )

        return GenerationUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )
