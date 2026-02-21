"""
Integration tests for the OpenSearch storage backends.

Requires a running OpenSearch instance at https://localhost:9200
with credentials admin:Admin@1234 (or set OPENSEARCH_* env vars).

Run with:
    python3.11 -m pytest tests/test_opensearch.py -v
"""

import os
import json
import asyncio
from datetime import datetime

import aiohttp
import numpy as np
import pytest
import pytest_asyncio

# ── Configure OpenSearch connection for tests ──────────────────────────────
os.environ.setdefault("OPENSEARCH_URI", "https://localhost:9200")
os.environ.setdefault("OPENSEARCH_USERNAME", "admin")
os.environ.setdefault("OPENSEARCH_PASSWORD", "Admin@1234")
os.environ.setdefault("OPENSEARCH_VERIFY_CERTS", "false")
# Use a dedicated test database prefix to avoid collisions
os.environ["OPENSEARCH_DATABASE"] = "minirag_pytest"

from minirag.base import DocStatus, DocProcessingStatus
from minirag.utils import EmbeddingFunc
from minirag.kg.opensearch_impl import (
    OpenSearchKVStorage,
    OpenSearchVectorStorage,
    OpenSearchGraphStorage,
    OpenSearchDocStatusStorage,
)


# ── Helpers ────────────────────────────────────────────────────────────────

OS_BASE = os.environ["OPENSEARCH_URI"].rstrip("/")


async def _os_request(session, method, path, body=None):
    url = f"{OS_BASE}/{path}"
    kw = {}
    if body is not None:
        kw["json"] = body
        kw["headers"] = {"Content-Type": "application/json"}
    async with session.request(method, url, **kw) as resp:
        try:
            return resp.status, await resp.json()
        except Exception:
            return resp.status, await resp.text()


async def _delete_test_indices(session):
    """Delete all minirag_pytest_* indices."""
    status, data = await _os_request(
        session, "GET", "_cat/indices?h=index&format=json"
    )
    if isinstance(data, list):
        for idx in data:
            name = idx.get("index", "")
            if name.startswith("minirag_pytest_"):
                await _os_request(session, "DELETE", name)


async def _refresh(session, pattern="minirag_pytest_*"):
    await _os_request(session, "POST", f"{pattern}/_refresh")


def _make_embedding_func(dim=8):
    """Create a deterministic embedding function for tests."""

    async def _embed(texts):
        # Deterministic: hash-based embedding so same text => same vector
        result = []
        for t in texts:
            np.random.seed(hash(t) % (2**31))
            result.append(np.random.rand(dim).astype(np.float32))
        return np.array(result)

    return EmbeddingFunc(embedding_dim=dim, max_token_size=512, func=_embed)


# ── Fixtures ───────────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def os_session():
    """aiohttp session for direct OpenSearch verification."""
    conn = aiohttp.TCPConnector(ssl=False)
    auth = aiohttp.BasicAuth(
        os.environ["OPENSEARCH_USERNAME"], os.environ["OPENSEARCH_PASSWORD"]
    )
    session = aiohttp.ClientSession(auth=auth, connector=conn)
    # Clean up indices before each test
    await _delete_test_indices(session)
    yield session
    # Clean up after test
    await _delete_test_indices(session)
    await session.close()


@pytest_asyncio.fixture
async def kv_storage(os_session):
    global_config = {}
    embed_func = _make_embedding_func()
    storage = OpenSearchKVStorage(
        namespace="test_ns",
        global_config=global_config,
        embedding_func=embed_func,
    )
    yield storage


@pytest_asyncio.fixture
async def vector_storage(os_session):
    embed_func = _make_embedding_func(dim=8)
    global_config = {
        "embedding_batch_num": 32,
        "vector_db_storage_cls_kwargs": {
            "cosine_better_than_threshold": 0.2,
        },
    }
    storage = OpenSearchVectorStorage(
        namespace="test_vec",
        global_config=global_config,
        embedding_func=embed_func,
        meta_fields={"source"},
    )
    yield storage


@pytest_asyncio.fixture
async def graph_storage(os_session):
    global_config = {}
    storage = OpenSearchGraphStorage(
        namespace="test_graph",
        global_config=global_config,
    )
    yield storage


@pytest_asyncio.fixture
async def doc_status_storage(os_session):
    global_config = {}
    embed_func = _make_embedding_func()
    storage = OpenSearchDocStatusStorage(
        namespace="test_docstatus",
        global_config=global_config,
        embedding_func=embed_func,
    )
    yield storage


# ═══════════════════════════════════════════════════════════════════════════
# KV Storage Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestOpenSearchKVStorage:

    @pytest.mark.asyncio
    async def test_upsert_and_get_by_id(self, kv_storage):
        await kv_storage.upsert({"key1": {"name": "Alice", "age": 30}})
        await kv_storage.index_done_callback()

        result = await kv_storage.get_by_id("key1")
        assert result is not None
        assert result["name"] == "Alice"
        assert result["age"] == 30

    @pytest.mark.asyncio
    async def test_get_by_id_missing(self, kv_storage):
        result = await kv_storage.get_by_id("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_by_ids(self, kv_storage):
        await kv_storage.upsert({
            "a": {"val": 1},
            "b": {"val": 2},
            "c": {"val": 3},
        })
        await kv_storage.index_done_callback()

        results = await kv_storage.get_by_ids(["a", "c", "missing"])
        assert len(results) == 3
        assert results[0] is not None and results[0]["val"] == 1
        assert results[1] is not None and results[1]["val"] == 3
        assert results[2] is None

    @pytest.mark.asyncio
    async def test_get_by_ids_with_fields(self, kv_storage):
        await kv_storage.upsert({"d1": {"name": "Doc1", "content": "Hello", "extra": "x"}})
        await kv_storage.index_done_callback()

        results = await kv_storage.get_by_ids(["d1"], fields={"name", "content"})
        assert len(results) == 1
        assert "name" in results[0]
        assert "content" in results[0]
        assert "extra" not in results[0]

    @pytest.mark.asyncio
    async def test_get_by_ids_empty(self, kv_storage):
        results = await kv_storage.get_by_ids([])
        assert results == []

    @pytest.mark.asyncio
    async def test_all_keys(self, kv_storage):
        await kv_storage.upsert({
            "k1": {"x": 1},
            "k2": {"x": 2},
            "k3": {"x": 3},
        })
        await kv_storage.index_done_callback()

        keys = await kv_storage.all_keys()
        assert set(keys) == {"k1", "k2", "k3"}

    @pytest.mark.asyncio
    async def test_all_keys_empty(self, kv_storage):
        keys = await kv_storage.all_keys()
        assert keys == []

    @pytest.mark.asyncio
    async def test_filter_keys(self, kv_storage):
        await kv_storage.upsert({"exists1": {"v": 1}, "exists2": {"v": 2}})
        await kv_storage.index_done_callback()

        missing = await kv_storage.filter_keys(["exists1", "exists2", "new1", "new2"])
        assert missing == {"new1", "new2"}

    @pytest.mark.asyncio
    async def test_filter_keys_empty(self, kv_storage):
        result = await kv_storage.filter_keys([])
        assert result == set()

    @pytest.mark.asyncio
    async def test_upsert_overwrites(self, kv_storage):
        await kv_storage.upsert({"k": {"val": "original"}})
        await kv_storage.index_done_callback()
        await kv_storage.upsert({"k": {"val": "updated"}})
        await kv_storage.index_done_callback()

        result = await kv_storage.get_by_id("k")
        assert result["val"] == "updated"

    @pytest.mark.asyncio
    async def test_drop(self, kv_storage):
        await kv_storage.upsert({"x": {"v": 1}, "y": {"v": 2}})
        await kv_storage.index_done_callback()
        assert len(await kv_storage.all_keys()) == 2

        await kv_storage.drop()
        assert len(await kv_storage.all_keys()) == 0

    @pytest.mark.asyncio
    async def test_upsert_empty(self, kv_storage):
        # Should not raise
        await kv_storage.upsert({})


# ═══════════════════════════════════════════════════════════════════════════
# Vector Storage Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestOpenSearchVectorStorage:

    @pytest.mark.asyncio
    async def test_upsert_and_query(self, vector_storage):
        data = {
            "doc1": {"content": "The quick brown fox jumps over the lazy dog"},
            "doc2": {"content": "A fast red fox leaps across a sleepy hound"},
            "doc3": {"content": "Quantum computing uses qubits for computation"},
        }
        await vector_storage.upsert(data)
        await vector_storage.index_done_callback()

        results = await vector_storage.query("fox jumping over dog", top_k=3)
        assert isinstance(results, list)
        # Should return results (exact count depends on min_score threshold)
        for r in results:
            assert "id" in r
            assert "distance" in r
            assert "content" in r

    @pytest.mark.asyncio
    async def test_upsert_with_meta_fields(self, vector_storage):
        data = {
            "doc_meta": {
                "content": "Test document with metadata",
                "source": "unit_test",
            },
        }
        await vector_storage.upsert(data)
        await vector_storage.index_done_callback()

        results = await vector_storage.query("test document", top_k=5)
        if results:
            assert results[0].get("source") == "unit_test"

    @pytest.mark.asyncio
    async def test_upsert_empty(self, vector_storage):
        # Should not raise
        await vector_storage.upsert({})

    @pytest.mark.asyncio
    async def test_query_returns_distance(self, vector_storage):
        await vector_storage.upsert({
            "v1": {"content": "hello world"},
        })
        await vector_storage.index_done_callback()

        results = await vector_storage.query("hello world", top_k=1)
        if results:
            # Same text should have high cosine similarity
            assert results[0]["distance"] > 0.0

    @pytest.mark.asyncio
    async def test_query_no_results(self, vector_storage):
        # Empty index — query should return empty list
        results = await vector_storage.query("anything", top_k=5)
        assert results == []


# ═══════════════════════════════════════════════════════════════════════════
# Graph Storage Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestOpenSearchGraphStorage:

    @pytest.mark.asyncio
    async def test_upsert_and_get_node(self, graph_storage):
        await graph_storage.upsert_node(
            "ALICE", {"entity_type": "PERSON", "description": "A researcher"}
        )
        await graph_storage.index_done_callback()

        node = await graph_storage.get_node("ALICE")
        assert node is not None
        assert node["entity_type"] == "PERSON"
        assert node["description"] == "A researcher"

    @pytest.mark.asyncio
    async def test_get_node_missing(self, graph_storage):
        result = await graph_storage.get_node("NONEXISTENT")
        assert result is None

    @pytest.mark.asyncio
    async def test_has_node(self, graph_storage):
        assert await graph_storage.has_node("BOB") is False

        await graph_storage.upsert_node("BOB", {"entity_type": "PERSON"})
        await graph_storage.index_done_callback()

        assert await graph_storage.has_node("BOB") is True

    @pytest.mark.asyncio
    async def test_upsert_and_get_edge(self, graph_storage):
        await graph_storage.upsert_node("SRC", {"entity_type": "PERSON"})
        await graph_storage.upsert_node("TGT", {"entity_type": "ORG"})
        await graph_storage.upsert_edge(
            "SRC", "TGT", {"relationship": "works_at", "weight": "0.9"}
        )
        await graph_storage.index_done_callback()

        edge = await graph_storage.get_edge("SRC", "TGT")
        assert edge is not None
        assert edge["relationship"] == "works_at"

    @pytest.mark.asyncio
    async def test_get_edge_symmetric(self, graph_storage):
        """Edge lookup should work regardless of argument order (sorted key)."""
        await graph_storage.upsert_edge("AAA", "ZZZ", {"rel": "linked"})
        await graph_storage.index_done_callback()

        edge_fwd = await graph_storage.get_edge("AAA", "ZZZ")
        edge_rev = await graph_storage.get_edge("ZZZ", "AAA")
        assert edge_fwd is not None
        assert edge_rev is not None
        assert edge_fwd["rel"] == edge_rev["rel"]

    @pytest.mark.asyncio
    async def test_has_edge(self, graph_storage):
        assert await graph_storage.has_edge("X", "Y") is False

        await graph_storage.upsert_edge("X", "Y", {"rel": "test"})
        await graph_storage.index_done_callback()

        assert await graph_storage.has_edge("X", "Y") is True
        assert await graph_storage.has_edge("Y", "X") is True  # symmetric

    @pytest.mark.asyncio
    async def test_node_degree(self, graph_storage):
        await graph_storage.upsert_node("CENTER", {"entity_type": "THING"})
        await graph_storage.upsert_edge("CENTER", "N1", {"rel": "a"})
        await graph_storage.upsert_edge("CENTER", "N2", {"rel": "b"})
        await graph_storage.upsert_edge("CENTER", "N3", {"rel": "c"})
        await graph_storage.index_done_callback()

        degree = await graph_storage.node_degree("CENTER")
        assert degree == 3

    @pytest.mark.asyncio
    async def test_edge_degree(self, graph_storage):
        await graph_storage.upsert_edge("P", "Q", {"rel": "x"})
        await graph_storage.upsert_edge("P", "R", {"rel": "y"})
        await graph_storage.upsert_edge("Q", "S", {"rel": "z"})
        await graph_storage.index_done_callback()

        # P has degree 2, Q has degree 2 => edge_degree(P, Q) = 4
        degree = await graph_storage.edge_degree("P", "Q")
        assert degree == 4

    @pytest.mark.asyncio
    async def test_get_node_edges(self, graph_storage):
        await graph_storage.upsert_edge("HUB", "SPOKE1", {"rel": "a"})
        await graph_storage.upsert_edge("HUB", "SPOKE2", {"rel": "b"})
        await graph_storage.index_done_callback()

        edges = await graph_storage.get_node_edges("HUB")
        assert edges is not None
        assert len(edges) == 2
        # Each edge should be a (source, target) tuple
        for src, tgt in edges:
            assert "HUB" in (src, tgt)

    @pytest.mark.asyncio
    async def test_get_node_edges_none_for_missing(self, graph_storage):
        edges = await graph_storage.get_node_edges("NOWHERE")
        assert edges is None

    @pytest.mark.asyncio
    async def test_get_types(self, graph_storage):
        await graph_storage.upsert_node("E1", {"entity_type": "PERSON"})
        await graph_storage.upsert_node("E2", {"entity_type": "LOCATION"})
        await graph_storage.upsert_node("E3", {"entity_type": "PERSON"})
        await graph_storage.index_done_callback()

        lowered, original = await graph_storage.get_types()
        assert set(original) == {"PERSON", "LOCATION"}
        assert set(lowered) == {"person", "location"}

    @pytest.mark.asyncio
    async def test_get_types_empty(self, graph_storage):
        lowered, original = await graph_storage.get_types()
        assert lowered == []
        assert original == []

    @pytest.mark.asyncio
    async def test_delete_node(self, graph_storage):
        await graph_storage.upsert_node("DEL_NODE", {"entity_type": "TEMP"})
        await graph_storage.upsert_edge("DEL_NODE", "OTHER", {"rel": "link"})
        await graph_storage.index_done_callback()

        assert await graph_storage.has_node("DEL_NODE") is True

        await graph_storage.delete_node("DEL_NODE")
        await graph_storage.index_done_callback()

        assert await graph_storage.has_node("DEL_NODE") is False
        # Related edges should also be deleted
        assert await graph_storage.has_edge("DEL_NODE", "OTHER") is False

    @pytest.mark.asyncio
    async def test_upsert_node_overwrites(self, graph_storage):
        await graph_storage.upsert_node("N", {"entity_type": "V1", "desc": "old"})
        await graph_storage.index_done_callback()
        await graph_storage.upsert_node("N", {"entity_type": "V2", "desc": "new"})
        await graph_storage.index_done_callback()

        node = await graph_storage.get_node("N")
        assert node["desc"] == "new"
        assert node["entity_type"] == "V2"

    @pytest.mark.asyncio
    async def test_get_node_from_types(self, graph_storage):
        await graph_storage.upsert_node("ALICE", {"entity_type": "PERSON", "role": "dev"})
        await graph_storage.upsert_node("BOB", {"entity_type": "PERSON", "role": "pm"})
        await graph_storage.upsert_node("ACME", {"entity_type": "ORG", "role": "corp"})
        await graph_storage.index_done_callback()

        persons = await graph_storage.get_node_from_types(["PERSON"])
        assert len(persons) == 2
        names = {p["entity_name"] for p in persons}
        assert names == {"ALICE", "BOB"}
        # Check properties are included
        for p in persons:
            assert "role" in p

    @pytest.mark.asyncio
    async def test_get_node_from_types_empty(self, graph_storage):
        result = await graph_storage.get_node_from_types([])
        assert result == []

    @pytest.mark.asyncio
    async def test_get_neighbors_within_k_hops(self, graph_storage):
        # Build a small graph: A -> B -> C
        await graph_storage.upsert_node("A", {"entity_type": "T"})
        await graph_storage.upsert_node("B", {"entity_type": "T"})
        await graph_storage.upsert_node("C", {"entity_type": "T"})
        await graph_storage.upsert_edge("A", "B", {"rel": "1"})
        await graph_storage.upsert_edge("B", "C", {"rel": "2"})
        await graph_storage.index_done_callback()

        # 1 hop from A should reach B
        paths_1 = await graph_storage.get_neighbors_within_k_hops("A", 1)
        neighbors_1 = {p[-1] for p in paths_1}
        assert "B" in neighbors_1

        # 2 hops from A should reach C
        paths_2 = await graph_storage.get_neighbors_within_k_hops("A", 2)
        endpoints = {p[-1] for p in paths_2}
        assert "C" in endpoints

    @pytest.mark.asyncio
    async def test_get_neighbors_missing_node(self, graph_storage):
        result = await graph_storage.get_neighbors_within_k_hops("GHOST", 2)
        assert result == []

    @pytest.mark.asyncio
    async def test_edge_id_is_deterministic(self, graph_storage):
        """The sorted edge ID should be the same regardless of argument order."""
        eid1 = graph_storage._edge_id("ZZZ", "AAA")
        eid2 = graph_storage._edge_id("AAA", "ZZZ")
        assert eid1 == eid2
        assert eid1 == "AAA::ZZZ"


# ═══════════════════════════════════════════════════════════════════════════
# DocStatus Storage Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestOpenSearchDocStatusStorage:

    def _make_doc(self, status, content="test content"):
        now = datetime.now().isoformat()
        return {
            "content": content,
            "content_summary": content[:100],
            "content_length": len(content),
            "status": status,
            "created_at": now,
            "updated_at": now,
        }

    @pytest.mark.asyncio
    async def test_upsert_and_get_by_id(self, doc_status_storage):
        doc = self._make_doc(DocStatus.PENDING)
        await doc_status_storage.upsert({"doc1": doc})

        result = await doc_status_storage.get_by_id("doc1")
        assert result is not None
        assert result["status"] == "pending"
        assert result["content"] == "test content"

    @pytest.mark.asyncio
    async def test_get_by_id_missing(self, doc_status_storage):
        result = await doc_status_storage.get_by_id("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_all_keys(self, doc_status_storage):
        await doc_status_storage.upsert({
            "d1": self._make_doc(DocStatus.PENDING),
            "d2": self._make_doc(DocStatus.PROCESSED),
        })

        keys = await doc_status_storage.all_keys()
        assert set(keys) == {"d1", "d2"}

    @pytest.mark.asyncio
    async def test_filter_keys_returns_unprocessed(self, doc_status_storage):
        """filter_keys should return keys not in storage OR not processed."""
        await doc_status_storage.upsert({
            "processed_doc": self._make_doc(DocStatus.PROCESSED),
            "pending_doc": self._make_doc(DocStatus.PENDING),
            "failed_doc": self._make_doc(DocStatus.FAILED),
        })

        result = await doc_status_storage.filter_keys(
            {"processed_doc", "pending_doc", "failed_doc", "new_doc"}
        )
        # processed_doc should NOT be in result (already processed)
        # pending_doc, failed_doc, new_doc should be in result
        assert "processed_doc" not in result
        assert "pending_doc" in result
        assert "failed_doc" in result
        assert "new_doc" in result

    @pytest.mark.asyncio
    async def test_filter_keys_empty(self, doc_status_storage):
        result = await doc_status_storage.filter_keys([])
        assert result == set()

    @pytest.mark.asyncio
    async def test_get_docs_by_status(self, doc_status_storage):
        await doc_status_storage.upsert({
            "p1": self._make_doc(DocStatus.PENDING, "Pending doc 1"),
            "p2": self._make_doc(DocStatus.PENDING, "Pending doc 2"),
            "f1": self._make_doc(DocStatus.FAILED, "Failed doc"),
            "done1": self._make_doc(DocStatus.PROCESSED, "Done doc"),
        })

        pending = await doc_status_storage.get_docs_by_status(DocStatus.PENDING)
        assert len(pending) == 2
        assert set(pending.keys()) == {"p1", "p2"}
        for doc_id, doc in pending.items():
            assert isinstance(doc, DocProcessingStatus)
            assert doc.status == DocStatus.PENDING

        failed = await doc_status_storage.get_docs_by_status(DocStatus.FAILED)
        assert len(failed) == 1
        assert "f1" in failed
        assert failed["f1"].status == DocStatus.FAILED

        processed = await doc_status_storage.get_docs_by_status(DocStatus.PROCESSED)
        assert len(processed) == 1
        assert "done1" in processed

    @pytest.mark.asyncio
    async def test_get_failed_docs(self, doc_status_storage):
        await doc_status_storage.upsert({
            "ok": self._make_doc(DocStatus.PROCESSED),
            "bad": self._make_doc(DocStatus.FAILED, "Broken doc"),
        })

        failed = await doc_status_storage.get_failed_docs()
        assert len(failed) == 1
        assert "bad" in failed
        assert failed["bad"].content == "Broken doc"

    @pytest.mark.asyncio
    async def test_get_pending_docs(self, doc_status_storage):
        await doc_status_storage.upsert({
            "waiting": self._make_doc(DocStatus.PENDING, "Waiting doc"),
            "done": self._make_doc(DocStatus.PROCESSED),
        })

        pending = await doc_status_storage.get_pending_docs()
        assert len(pending) == 1
        assert "waiting" in pending
        assert pending["waiting"].content == "Waiting doc"

    @pytest.mark.asyncio
    async def test_get_status_counts(self, doc_status_storage):
        await doc_status_storage.upsert({
            "a": self._make_doc(DocStatus.PENDING),
            "b": self._make_doc(DocStatus.PENDING),
            "c": self._make_doc(DocStatus.PROCESSED),
            "d": self._make_doc(DocStatus.FAILED),
        })

        counts = await doc_status_storage.get_status_counts()
        assert counts.get("pending", 0) == 2
        assert counts.get("processed", 0) == 1
        assert counts.get("failed", 0) == 1

    @pytest.mark.asyncio
    async def test_get_status_counts_empty(self, doc_status_storage):
        counts = await doc_status_storage.get_status_counts()
        assert counts == {}

    @pytest.mark.asyncio
    async def test_status_transition(self, doc_status_storage):
        """Simulate a document going through the processing pipeline."""
        now = datetime.now().isoformat()

        # Step 1: enqueue as pending
        await doc_status_storage.upsert({
            "doc_x": self._make_doc(DocStatus.PENDING, "Some content"),
        })
        pending = await doc_status_storage.get_docs_by_status(DocStatus.PENDING)
        assert "doc_x" in pending

        # Step 2: mark as processed
        await doc_status_storage.upsert({
            "doc_x": {
                "content": "Some content",
                "content_summary": "Some content",
                "content_length": 12,
                "status": DocStatus.PROCESSED,
                "created_at": now,
                "updated_at": datetime.now().isoformat(),
                "chunks_count": 3,
            },
        })
        processed = await doc_status_storage.get_docs_by_status(DocStatus.PROCESSED)
        assert "doc_x" in processed
        assert processed["doc_x"].chunks_count == 3

        # Pending should now be empty
        pending = await doc_status_storage.get_docs_by_status(DocStatus.PENDING)
        assert "doc_x" not in pending

    @pytest.mark.asyncio
    async def test_drop(self, doc_status_storage):
        await doc_status_storage.upsert({
            "x": self._make_doc(DocStatus.PENDING),
            "y": self._make_doc(DocStatus.PROCESSED),
        })
        assert len(await doc_status_storage.all_keys()) == 2

        await doc_status_storage.drop()
        assert len(await doc_status_storage.all_keys()) == 0

    @pytest.mark.asyncio
    async def test_get_by_ids(self, doc_status_storage):
        await doc_status_storage.upsert({
            "d1": self._make_doc(DocStatus.PENDING, "first"),
            "d2": self._make_doc(DocStatus.FAILED, "second"),
        })

        results = await doc_status_storage.get_by_ids(["d1", "d2", "d3"])
        assert len(results) == 3
        assert results[0] is not None
        assert results[0]["content"] == "first"
        assert results[1] is not None
        assert results[1]["content"] == "second"
        assert results[2] is None


# ═══════════════════════════════════════════════════════════════════════════
# Cross-store Integration Test
# ═══════════════════════════════════════════════════════════════════════════


class TestOpenSearchCrossStore:

    @pytest.mark.asyncio
    async def test_all_stores_create_indices(self, os_session):
        """Verify that initializing all 4 stores creates the expected indices."""
        embed_func = _make_embedding_func(dim=8)
        gc = {
            "embedding_batch_num": 32,
            "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.2},
        }

        kv = OpenSearchKVStorage(
            namespace="full_docs", global_config=gc, embedding_func=embed_func
        )
        vec = OpenSearchVectorStorage(
            namespace="entities", global_config=gc, embedding_func=embed_func
        )
        graph = OpenSearchGraphStorage(namespace="graph", global_config=gc)
        doc = OpenSearchDocStatusStorage(
            namespace="docstatus", global_config=gc, embedding_func=embed_func
        )

        # Trigger lazy initialization on all stores
        await kv.upsert({"test": {"v": 1}})
        await vec.upsert({"test": {"content": "test"}})
        await graph.upsert_node("TEST", {"entity_type": "TEST"})
        await doc.upsert({"test": {
            "content": "test", "content_summary": "test",
            "content_length": 4, "status": DocStatus.PENDING,
            "created_at": "now", "updated_at": "now",
        }})

        await _refresh(os_session)

        # Verify indices exist
        status, data = await _os_request(
            os_session, "GET",
            "_cat/indices/minirag_pytest_*?h=index&format=json",
        )
        index_names = {idx["index"] for idx in data}
        assert "minirag_pytest_kv_full_docs" in index_names
        assert "minirag_pytest_vec_entities" in index_names
        assert "minirag_pytest_graph_nodes" in index_names
        assert "minirag_pytest_graph_edges" in index_names
        assert "minirag_pytest_docstatus" in index_names

    @pytest.mark.asyncio
    async def test_index_naming_uses_database_prefix(self, os_session):
        """Verify that OPENSEARCH_DATABASE env var controls index prefix."""
        embed_func = _make_embedding_func()
        gc = {}
        kv = OpenSearchKVStorage(
            namespace="myns", global_config=gc, embedding_func=embed_func,
        )
        # Index name should be {database}_kv_{namespace}
        assert kv._index_name == "minirag_pytest_kv_myns"
