from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
from qdrant_client import models

from rag.metadata.schema import SearchFilters
from rag.vector_store.config import VectorStoreSettings
from rag.vector_store.qdrant_store import QdrantVectorStore


@dataclass(frozen=True, slots=True)
class FakeChunkMetadata:
    doc_id: str = "credit_hold_policy"
    title: str = "Credit Hold Policy"
    version: str = "1.0"
    effective_date: str = "2026-08-01"
    status: str = "active"
    department: str = "finance"
    document_type: str = "policy"
    access_roles: tuple[str, ...] = (
        "sales_rep",
        "finance_manager",
    )
    source_path: str = (
        "documents/credit_hold_policy.md"
    )
    section_id: str = "CH-3"
    section_title: str = "Severe Release"
    chunk_index: int = 2
    source_checksum: str = "a" * 64
    keywords: tuple[str, ...] = (
        "credit hold",
        "severe",
    )


@dataclass(frozen=True, slots=True)
class FakeEmbeddedChunk:
    chunk_id: str
    vector: tuple[float, ...]
    text: str
    metadata: FakeChunkMetadata


class FakeQdrantClient:
    def __init__(self):
        self.exists = False
        self.calls: list[str] = []
        self.index_fields: list[str] = []
        self.upserted_points = []
        self.last_filter = None

    def collection_exists(self, collection_name):
        self.calls.append("collection_exists")
        return self.exists

    def create_collection(self, **kwargs):
        self.calls.append("create_collection")
        self.exists = True
        self.create_kwargs = kwargs

    def delete_collection(self, **kwargs):
        self.calls.append("delete_collection")
        self.exists = False

    def get_collection(self, **kwargs):
        self.calls.append("get_collection")
        return SimpleNamespace(
            payload_schema={
                field: object()
                for field in self.index_fields
            },
            config=SimpleNamespace(
                params=SimpleNamespace(
                    vectors=SimpleNamespace(
                        size=384,
                        distance=models.Distance.COSINE,
                    )
                ),
                hnsw_config=SimpleNamespace(
                    m=16,
                    ef_construct=100,
                ),
            ),
            points_count=len(self.upserted_points),
            indexed_vectors_count=len(
                self.upserted_points
            ),
        )

    def create_payload_index(
        self,
        collection_name,
        field_name,
        field_schema,
        wait,
    ):
        self.calls.append(
            f"create_payload_index:{field_name}"
        )
        self.index_fields.append(field_name)

    def upsert(
        self,
        collection_name,
        points,
        wait,
    ):
        self.calls.append("upsert")
        self.upserted_points.extend(points)

    def query_points(self, **kwargs):
        self.calls.append("query_points")
        self.last_filter = kwargs["query_filter"]

        point = self.upserted_points[0]
        return SimpleNamespace(
            points=[
                SimpleNamespace(
                    id=point.id,
                    score=0.91,
                    payload=point.payload,
                )
            ]
        )

    def count(self, **kwargs):
        return SimpleNamespace(
            count=len(self.upserted_points)
        )

    def get_collections(self):
        return SimpleNamespace(collections=[])


def _settings() -> VectorStoreSettings:
    return VectorStoreSettings(
        url="http://test-qdrant:6333",
        collection_name="test_collection",
        vector_size=384,
        default_top_k=5,
    )


def _embedded_chunk() -> FakeEmbeddedChunk:
    return FakeEmbeddedChunk(
        chunk_id=(
            "8fe0dfbf-6e58-5817-a3e9-8fe48cd1b383"
        ),
        vector=(1.0,) + (0.0,) * 383,
        text=(
            "Only a finance manager may release "
            "a severe credit hold."
        ),
        metadata=FakeChunkMetadata(),
    )


def test_collection_and_indexes_are_created_before_upsert():
    client = FakeQdrantClient()
    store = QdrantVectorStore(
        settings=_settings(),
        client=client,
    )

    store.ensure_collection()
    uploaded = store.upsert([_embedded_chunk()])

    assert uploaded == 1
    assert client.calls.index(
        "create_collection"
    ) < client.calls.index("upsert")

    for field_name in store.PAYLOAD_INDEXES:
        assert (
            f"create_payload_index:{field_name}"
            in client.calls
        )

    assert client.calls.index(
        "create_payload_index:access_roles"
    ) < client.calls.index("upsert")


def test_upsert_preserves_text_and_metadata_payload():
    client = FakeQdrantClient()
    store = QdrantVectorStore(
        settings=_settings(),
        client=client,
    )

    store.ensure_collection()
    store.upsert([_embedded_chunk()])

    payload = client.upserted_points[0].payload

    assert payload["doc_id"] == "credit_hold_policy"
    assert payload["section_id"] == "CH-3"
    assert payload["status"] == "active"
    assert payload["access_roles"] == [
        "sales_rep",
        "finance_manager",
    ]
    assert "severe credit hold" in payload["text"]


def test_search_uses_role_and_status_filters():
    client = FakeQdrantClient()
    store = QdrantVectorStore(
        settings=_settings(),
        client=client,
    )

    store.ensure_collection()
    store.upsert([_embedded_chunk()])

    results = store.search(
        (1.0,) + (0.0,) * 383,
        SearchFilters(
            role="finance_manager",
            statuses=("active",),
            section_ids=("CH-3",),
        ),
        top_k=3,
    )

    assert len(results) == 1
    assert results[0].metadata["section_id"] == "CH-3"

    filter_data = client.last_filter.model_dump()
    filter_text = str(filter_data)

    assert "access_roles" in filter_text
    assert "finance_manager" in filter_text
    assert "status" in filter_text
    assert "active" in filter_text
    assert "section_id" in filter_text
    assert "CH-3" in filter_text


def test_wrong_vector_size_is_rejected():
    store = QdrantVectorStore(
        settings=_settings(),
        client=FakeQdrantClient(),
    )

    with pytest.raises(
        ValueError,
        match="Vector size",
    ):
        store.search(
            [1.0, 0.0],
            SearchFilters(role="sales_rep"),
        )
