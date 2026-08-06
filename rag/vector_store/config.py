"""Environment-based Qdrant configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class VectorStoreSettings:
    """Connection and collection settings for Qdrant."""

    url: str = os.getenv(
        "QDRANT_URL",
        "http://127.0.0.1:6333",
    )
    api_key: str | None = os.getenv("QDRANT_API_KEY") or None
    collection_name: str = os.getenv(
        "QDRANT_COLLECTION",
        "swiftrail_knowledge",
    )
    vector_size: int = int(
        os.getenv("QDRANT_VECTOR_SIZE", "384")
    )
    default_top_k: int = int(
        os.getenv("QDRANT_TOP_K", "5")
    )

    def validate(self) -> None:
        if not self.url.strip():
            raise ValueError("QDRANT_URL cannot be empty.")

        if not self.collection_name.strip():
            raise ValueError(
                "QDRANT_COLLECTION cannot be empty."
            )

        if self.vector_size < 1:
            raise ValueError(
                "QDRANT_VECTOR_SIZE must be positive."
            )

        if self.default_top_k < 1:
            raise ValueError(
                "QDRANT_TOP_K must be positive."
            )
