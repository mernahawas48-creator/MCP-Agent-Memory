"""Run an end-to-end vector-store smoke test against local Qdrant."""

from __future__ import annotations

import json

from rag.chunking.chunker import MarkdownChunker
from rag.embeddings.embedder import ChunkEmbedder
from rag.loading.loader import CorpusLoader
from rag.metadata.schema import SearchFilters
from rag.vector_store.qdrant_store import QdrantVectorStore


def main() -> None:
    documents = CorpusLoader().load()
    chunks = MarkdownChunker().chunk_documents(documents)

    embedder = ChunkEmbedder()
    embedded_chunks = embedder.embed_chunks(chunks)

    store = QdrantVectorStore()
    store.health_check()
    store.ensure_collection(recreate=True)
    uploaded = store.upsert(embedded_chunks)

    query = "Who can release a severe credit hold?"
    query_vector = embedder.embed_query(query)

    results = store.search(
        query_vector,
        SearchFilters(
            role="finance_manager",
            statuses=("active",),
        ),
        top_k=3,
    )

    print(f"Loaded documents: {len(documents)}")
    print(f"Created chunks: {len(chunks)}")
    print(f"Uploaded points: {uploaded}")
    print(f"Stored points: {store.count()}")
    print("Collection:")
    print(json.dumps(store.collection_info(), indent=2))

    print("\nSearch results:")
    for result in results:
        print(
            f"- score={result.score:.4f} | "
            f"{result.metadata['doc_id']} | "
            f"{result.metadata['section_id']}"
        )

    assert uploaded == 22
    assert store.count() == 22
    assert results
    assert all(
        "finance_manager"
        in result.metadata["access_roles"]
        for result in results
    )

    print("\nVector database smoke test passed.")


if __name__ == "__main__":
    main()
