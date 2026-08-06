"""Load, validate, chunk, embed, and store the Swiftrail corpus."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any

from rag.chunking.chunker import MarkdownChunker
from rag.embeddings.embedder import ChunkEmbedder
from rag.loading.loader import CorpusLoader
from rag.metadata.schema import (
    ChunkMetadataSchema,
    DocumentMetadataSchema,
)
from rag.vector_store.qdrant_store import QdrantVectorStore


@dataclass(frozen=True, slots=True)
class IngestionReport:
    """Summary produced after one ingestion run."""

    collection_name: str
    documents_loaded: int
    chunks_created: int
    chunks_embedded: int
    points_uploaded: int
    points_stored: int
    vector_size: int
    recreated_collection: bool
    duration_seconds: float
    collection_info: dict[str, Any]


class IngestionPipeline:
    """Orchestrate every stage required to populate Qdrant."""

    def __init__(
        self,
        loader: Any | None = None,
        chunker: Any | None = None,
        embedder: Any | None = None,
        vector_store: Any | None = None,
    ):
        self.loader = loader or CorpusLoader()
        self.chunker = chunker or MarkdownChunker()
        self.embedder = embedder or ChunkEmbedder()
        self.vector_store = (
            vector_store or QdrantVectorStore()
        )

    def run(
        self,
        *,
        recreate_collection: bool = False,
    ) -> IngestionReport:
        """Run the complete ingestion pipeline once."""

        started_at = perf_counter()

        documents = self.loader.load()
        if not documents:
            raise RuntimeError(
                "The corpus loader returned no documents."
            )

        self._validate_documents(documents)

        chunks = self.chunker.chunk_documents(documents)
        if not chunks:
            raise RuntimeError(
                "The chunker returned no chunks."
            )

        self._validate_chunks(chunks)

        embedded_chunks = self.embedder.embed_chunks(chunks)

        if len(embedded_chunks) != len(chunks):
            raise RuntimeError(
                "The number of embedded chunks does not match "
                "the number of source chunks."
            )

        vector_size = self._validate_vector_sizes(
            embedded_chunks
        )

        expected_vector_size = int(
            self.vector_store.settings.vector_size
        )
        if vector_size != expected_vector_size:
            raise ValueError(
                "Embedding and Qdrant vector sizes do not match. "
                f"Embeddings use {vector_size}, while Qdrant "
                f"expects {expected_vector_size}."
            )

        self.vector_store.health_check()
        self.vector_store.ensure_collection(
            recreate=recreate_collection
        )

        uploaded = self.vector_store.upsert(
            embedded_chunks
        )
        if uploaded != len(embedded_chunks):
            raise RuntimeError(
                "Qdrant did not accept every embedded chunk."
            )

        stored = self.vector_store.count()

        if recreate_collection and stored != uploaded:
            raise RuntimeError(
                "The recreated collection contains an unexpected "
                "number of points."
            )

        if stored < uploaded:
            raise RuntimeError(
                "The collection contains fewer points than were "
                "uploaded during this run."
            )

        duration = perf_counter() - started_at

        return IngestionReport(
            collection_name=(
                self.vector_store.collection_name
            ),
            documents_loaded=len(documents),
            chunks_created=len(chunks),
            chunks_embedded=len(embedded_chunks),
            points_uploaded=uploaded,
            points_stored=stored,
            vector_size=vector_size,
            recreated_collection=recreate_collection,
            duration_seconds=round(duration, 3),
            collection_info=(
                self.vector_store.collection_info()
            ),
        )

    @staticmethod
    def _validate_documents(
        documents: list[Any],
    ) -> None:
        document_ids: set[str] = set()

        for document in documents:
            validated = (
                DocumentMetadataSchema.model_validate(
                    document.metadata
                )
            )

            if validated.doc_id in document_ids:
                raise ValueError(
                    "Duplicate document ID found during ingestion: "
                    f"{validated.doc_id}"
                )

            document_ids.add(validated.doc_id)

            if not document.text.strip():
                raise ValueError(
                    f"Document {validated.doc_id} has empty text."
                )

            if len(document.checksum) != 64:
                raise ValueError(
                    f"Document {validated.doc_id} has an invalid "
                    "SHA-256 checksum."
                )

    @staticmethod
    def _validate_chunks(
        chunks: list[Any],
    ) -> None:
        chunk_ids: set[str] = set()

        for chunk in chunks:
            if chunk.chunk_id in chunk_ids:
                raise ValueError(
                    "Duplicate chunk ID found during ingestion: "
                    f"{chunk.chunk_id}"
                )

            chunk_ids.add(chunk.chunk_id)

            if not chunk.text.strip():
                raise ValueError(
                    f"Chunk {chunk.chunk_id} has empty text."
                )

            metadata = (
                asdict(chunk.metadata)
                if hasattr(
                    chunk.metadata,
                    "__dataclass_fields__",
                )
                else chunk.metadata
            )

            ChunkMetadataSchema.model_validate(metadata)

    @staticmethod
    def _validate_vector_sizes(
        embedded_chunks: list[Any],
    ) -> int:
        if not embedded_chunks:
            raise RuntimeError(
                "The embedder returned no vectors."
            )

        vector_sizes = {
            len(embedded.vector)
            for embedded in embedded_chunks
        }

        if len(vector_sizes) != 1:
            raise ValueError(
                "The embedder returned inconsistent vector sizes."
            )

        vector_size = vector_sizes.pop()

        if vector_size < 1:
            raise ValueError(
                "The embedder returned empty vectors."
            )

        return vector_size


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Ingest the Swiftrail corpus into Qdrant."
        )
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help=(
            "Delete and rebuild the configured Qdrant "
            "collection before uploading the corpus."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    report = IngestionPipeline().run(
        recreate_collection=args.recreate
    )

    print("Swiftrail ingestion completed.")
    print(
        json.dumps(
            asdict(report),
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
