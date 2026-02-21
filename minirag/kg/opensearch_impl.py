from __future__ import annotations

import os
import json
import ssl
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple, Union

import aiohttp
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from dotenv import load_dotenv

from ..base import (
    BaseKVStorage,
    BaseVectorStorage,
    BaseGraphStorage,
    DocStatusStorage,
    DocStatus,
    DocProcessingStatus,
)
from ..utils import logger

# Load .env from the current folder; OS env vars take precedence
load_dotenv(dotenv_path=".env", override=False)

RETRY_EXCEPTIONS = (
    aiohttp.ClientError,
    aiohttp.ServerDisconnectedError,
    ConnectionResetError,
    OSError,
    TimeoutError,
)

READ_RETRY = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(RETRY_EXCEPTIONS),
    reraise=True,
)

WRITE_RETRY = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(RETRY_EXCEPTIONS),
    reraise=True,
)


# ── Shared connection helper ────────────────────────────────────────────────


def _create_os_session() -> tuple[str, aiohttp.ClientSession]:
    """Read OPENSEARCH_* env vars and return (base_url, session)."""
    uri = os.environ.get("OPENSEARCH_URI")
    if not uri:
        raise ValueError("OPENSEARCH_URI environment variable is required")
    username = os.environ.get("OPENSEARCH_USERNAME", "")
    password = os.environ.get("OPENSEARCH_PASSWORD", "")
    verify_certs = os.environ.get("OPENSEARCH_VERIFY_CERTS", "true").lower() in (
        "true",
        "1",
        "yes",
        "on",
    )

    base_url = uri.rstrip("/")

    auth = None
    if username:
        auth = aiohttp.BasicAuth(username, password)

    ssl_context: ssl.SSLContext | bool = True
    if not verify_certs:
        ssl_context = False

    connector = aiohttp.TCPConnector(ssl=ssl_context)
    session = aiohttp.ClientSession(
        auth=auth,
        connector=connector,
        headers={"Content-Type": "application/json"},
    )
    return base_url, session


def _get_database() -> str:
    """Return the database prefix for index names."""
    return os.environ.get("OPENSEARCH_DATABASE", "minirag")


# ── Shared HTTP helpers (used by all classes) ───────────────────────────────


async def _request(
    session: aiohttp.ClientSession,
    base_url: str,
    method: str,
    path: str,
    body: dict | None = None,
    params: dict | None = None,
) -> dict:
    """Issue an HTTP request to OpenSearch and return the JSON response."""
    url = f"{base_url}/{path}"
    kwargs: dict[str, Any] = {}
    if body is not None:
        kwargs["json"] = body
    if params is not None:
        kwargs["params"] = params
    async with session.request(method, url, **kwargs) as resp:
        text = await resp.text()
        if resp.status >= 400:
            logger.error(
                f"OpenSearch {method} {path} returned {resp.status}: {text}"
            )
        try:
            result = json.loads(text)
            result["_status"] = resp.status
            return result
        except json.JSONDecodeError:
            return {"_raw": text, "_status": resp.status}


async def _bulk_ndjson(
    session: aiohttp.ClientSession,
    base_url: str,
    index: str,
    actions: list[str],
) -> dict:
    """Send a _bulk request with NDJSON body to the specified index."""
    body_str = "\n".join(actions) + "\n"
    url = f"{base_url}/{index}/_bulk"
    async with session.post(
        url,
        data=body_str,
        headers={"Content-Type": "application/x-ndjson"},
    ) as resp:
        text = await resp.text()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"_raw": text, "_status": resp.status}


async def _ensure_index(
    session: aiohttp.ClientSession,
    base_url: str,
    index_name: str,
    mapping: dict,
) -> None:
    """Create an index if it does not already exist."""
    resp = await _request(session, base_url, "HEAD", index_name)
    if resp.get("_status") == 200:
        logger.debug(f"Index '{index_name}' already exists")
        return
    create_resp = await _request(session, base_url, "PUT", index_name, body=mapping)
    if create_resp.get("acknowledged"):
        logger.info(f"Created index '{index_name}'")
    elif "resource_already_exists_exception" in str(create_resp):
        logger.debug(f"Index '{index_name}' already exists (race)")
    else:
        logger.warning(f"Index '{index_name}' creation response: {create_resp}")


# ═══════════════════════════════════════════════════════════════════════════
# KV storage backend using OpenSearch
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class OpenSearchKVStorage(BaseKVStorage):

    _session: aiohttp.ClientSession = field(default=None, repr=False, init=False)
    _base_url: str = field(default="", repr=False, init=False)
    _index_name: str = field(default="", repr=False, init=False)
    _initialized: bool = field(default=False, repr=False, init=False)

    def __post_init__(self):
        database = _get_database()
        self._index_name = f"{database}_kv_{self.namespace}"

    async def _ensure_session(self):
        if self._initialized:
            return
        self._base_url, self._session = _create_os_session()
        mapping = {
            "settings": {"number_of_shards": 1, "number_of_replicas": 0},
            "mappings": {"dynamic": True},
        }
        await _ensure_index(self._session, self._base_url, self._index_name, mapping)
        logger.info(
            f"OpenSearch KV storage '{self._index_name}' ready at {self._base_url}"
        )
        self._initialized = True

    @READ_RETRY
    async def all_keys(self) -> list[str]:
        await self._ensure_session()
        keys = []
        body: dict[str, Any] = {
            "query": {"match_all": {}},
            "size": 10000,
            "_source": False,
            "sort": [{"_id": "asc"}],
        }
        search_after = None
        while True:
            if search_after is not None:
                body["search_after"] = search_after
            resp = await _request(
                self._session, self._base_url, "POST",
                f"{self._index_name}/_search", body=body,
            )
            hits = resp.get("hits", {}).get("hits", [])
            if not hits:
                break
            for hit in hits:
                keys.append(hit["_id"])
            search_after = hits[-1].get("sort")
            if search_after is None:
                break
        return keys

    @READ_RETRY
    async def get_by_id(self, id: str) -> Union[dict, None]:
        await self._ensure_session()
        resp = await _request(
            self._session, self._base_url, "GET",
            f"{self._index_name}/_doc/{id}",
        )
        if not resp.get("found"):
            return None
        return resp["_source"]

    @READ_RETRY
    async def get_by_ids(
        self, ids: list[str], fields: Union[set[str], None] = None
    ) -> list[Union[dict, None]]:
        await self._ensure_session()
        if not ids:
            return []
        resp = await _request(
            self._session, self._base_url, "POST",
            f"{self._index_name}/_mget", body={"ids": ids},
        )
        found: dict[str, dict] = {}
        for doc in resp.get("docs", []):
            if doc.get("found"):
                source = doc["_source"]
                if fields:
                    source = {k: v for k, v in source.items() if k in fields}
                found[doc["_id"]] = source
        return [found.get(id) for id in ids]

    @READ_RETRY
    async def filter_keys(self, data: list[str]) -> set[str]:
        await self._ensure_session()
        if not data:
            return set()
        resp = await _request(
            self._session, self._base_url, "POST",
            f"{self._index_name}/_mget", body={"ids": list(data)},
        )
        existing = set()
        for doc in resp.get("docs", []):
            if doc.get("found"):
                existing.add(doc["_id"])
        return set(data) - existing

    @WRITE_RETRY
    async def upsert(self, data: dict[str, dict]) -> None:
        await self._ensure_session()
        if not data:
            return
        actions = []
        for k, v in data.items():
            doc = dict(v)
            doc.pop("_id", None)
            actions.append(json.dumps({"index": {"_id": k}}))
            actions.append(json.dumps(doc))
        resp = await _bulk_ndjson(
            self._session, self._base_url, self._index_name, actions
        )
        if resp.get("errors"):
            failed = [
                item
                for item in resp.get("items", [])
                if "error" in item.get("index", {})
            ]
            if failed:
                logger.error(
                    f"Bulk KV upsert had {len(failed)} errors: {failed[:3]}"
                )

    @WRITE_RETRY
    async def drop(self) -> None:
        await self._ensure_session()
        await _request(
            self._session, self._base_url, "POST",
            f"{self._index_name}/_delete_by_query",
            body={"query": {"match_all": {}}},
            params={"refresh": "true"},
        )

    async def index_done_callback(self) -> None:
        await self._ensure_session()
        await _request(
            self._session, self._base_url, "POST",
            f"{self._index_name}/_refresh",
        )


# ═══════════════════════════════════════════════════════════════════════════
# Vector storage backend using OpenSearch k-NN plugin
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class OpenSearchVectorStorage(BaseVectorStorage):

    _session: aiohttp.ClientSession = field(default=None, repr=False, init=False)
    _base_url: str = field(default="", repr=False, init=False)
    _index_name: str = field(default="", repr=False, init=False)
    _initialized: bool = field(default=False, repr=False, init=False)
    cosine_better_than_threshold: float = field(default=0.2, repr=False, init=False)

    def __post_init__(self):
        database = _get_database()
        self._index_name = f"{database}_vec_{self.namespace}"
        kwargs = self.global_config.get("vector_db_storage_cls_kwargs", {})
        self.cosine_better_than_threshold = kwargs.get(
            "cosine_better_than_threshold", 0.2
        )

    async def _ensure_session(self):
        if self._initialized:
            return
        self._base_url, self._session = _create_os_session()

        embedding_dim = self.embedding_func.embedding_dim
        mapping = {
            "settings": {
                "index": {
                    "knn": True,
                    "number_of_shards": 1,
                    "number_of_replicas": 0,
                }
            },
            "mappings": {
                "properties": {
                    "vector": {
                        "type": "knn_vector",
                        "dimension": embedding_dim,
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "lucene",
                        },
                    },
                    "id": {"type": "keyword"},
                    "content": {"type": "text"},
                    "created_at": {"type": "long"},
                }
            },
        }
        await _ensure_index(self._session, self._base_url, self._index_name, mapping)
        logger.info(
            f"OpenSearch vector storage '{self._index_name}' ready at {self._base_url}"
        )
        self._initialized = True

    @WRITE_RETRY
    async def upsert(self, data: dict[str, dict]) -> None:
        await self._ensure_session()
        if not data:
            return

        import asyncio
        import time
        import numpy as np

        current_time = int(time.time())

        # Batch compute embeddings
        contents = [v["content"] for v in data.values()]
        max_batch = self.global_config.get("embedding_batch_num", 32)
        batches = [
            contents[i : i + max_batch]
            for i in range(0, len(contents), max_batch)
        ]
        embedding_tasks = [self.embedding_func(batch) for batch in batches]
        embeddings_list = await asyncio.gather(*embedding_tasks)
        embeddings = np.concatenate(embeddings_list)

        # Build bulk actions
        actions = []
        for idx, (doc_id, doc_data) in enumerate(data.items()):
            doc = {
                "vector": embeddings[idx].tolist(),
                "id": doc_id,
                "content": doc_data.get("content", ""),
                "created_at": current_time,
            }
            # Include any meta_fields
            for mf in self.meta_fields:
                if mf in doc_data:
                    doc[mf] = doc_data[mf]
            actions.append(json.dumps({"index": {"_id": doc_id}}))
            actions.append(json.dumps(doc))

        resp = await _bulk_ndjson(
            self._session, self._base_url, self._index_name, actions
        )
        if resp.get("errors"):
            failed = [
                item
                for item in resp.get("items", [])
                if "error" in item.get("index", {})
            ]
            if failed:
                logger.error(
                    f"Bulk vector upsert had {len(failed)} errors: {failed[:3]}"
                )

    @READ_RETRY
    async def query(self, query: str, top_k: int) -> list[dict]:
        await self._ensure_session()

        embedding_result = await self.embedding_func([query])
        embedding = embedding_result[0]

        # Convert cosine threshold to OpenSearch min_score
        # OpenSearch cosinesimil: score = 1 / (2 - cosine_similarity)
        min_score = 1.0 / (2.0 - self.cosine_better_than_threshold)

        body = {
            "size": top_k,
            "query": {
                "knn": {
                    "vector": {
                        "vector": (
                            embedding.tolist()
                            if hasattr(embedding, "tolist")
                            else list(embedding)
                        ),
                        "min_score": min_score,
                    }
                }
            },
            "_source": {"excludes": ["vector"]},
        }

        resp = await _request(
            self._session, self._base_url, "POST",
            f"{self._index_name}/_search", body=body,
        )

        results = []
        for hit in resp.get("hits", {}).get("hits", []):
            source = hit["_source"]
            # Convert OS score back to cosine similarity
            os_score = hit["_score"]
            cosine_sim = 2.0 - 1.0 / os_score
            results.append(
                {
                    **source,
                    "id": source.get("id", hit["_id"]),
                    "distance": cosine_sim,
                }
            )
        return results

    async def index_done_callback(self) -> None:
        await self._ensure_session()
        await _request(
            self._session, self._base_url, "POST",
            f"{self._index_name}/_refresh",
        )


# ═══════════════════════════════════════════════════════════════════════════
# Graph storage backend using OpenSearch (pure REST, no Cypher)
# ═══════════════════════════════════════════════════════════════════════════


GRAPH_NODES_MAPPING = {
    "settings": {"number_of_shards": 1, "number_of_replicas": 0},
    "mappings": {
        "properties": {
            "id": {"type": "keyword"},
            "entity_type": {"type": "keyword"},
            "properties": {"type": "flat_object"},
        }
    },
}

GRAPH_EDGES_MAPPING = {
    "settings": {"number_of_shards": 1, "number_of_replicas": 0},
    "mappings": {
        "properties": {
            "id": {"type": "keyword"},
            "source": {"type": "keyword"},
            "target": {"type": "keyword"},
            "properties": {"type": "flat_object"},
        }
    },
}


@dataclass
class OpenSearchGraphStorage(BaseGraphStorage):

    _session: aiohttp.ClientSession = field(default=None, repr=False, init=False)
    _base_url: str = field(default="", repr=False, init=False)
    _nodes_index: str = field(default="", repr=False, init=False)
    _edges_index: str = field(default="", repr=False, init=False)
    _initialized: bool = field(default=False, repr=False, init=False)

    def __post_init__(self):
        database = _get_database()
        self._nodes_index = f"{database}_graph_nodes"
        self._edges_index = f"{database}_graph_edges"

    async def _ensure_session(self):
        if self._initialized:
            return
        self._base_url, self._session = _create_os_session()
        await _ensure_index(
            self._session, self._base_url, self._nodes_index, GRAPH_NODES_MAPPING
        )
        await _ensure_index(
            self._session, self._base_url, self._edges_index, GRAPH_EDGES_MAPPING
        )
        logger.info(
            f"OpenSearch graph storage ready at {self._base_url} "
            f"(nodes={self._nodes_index}, edges={self._edges_index})"
        )
        self._initialized = True

    @staticmethod
    def _edge_id(src: str, tgt: str) -> str:
        """Sorted pair key for idempotent undirected edge upserts."""
        a, b = sorted([src, tgt])
        return f"{a}::{b}"

    # ── get_types ────────────────────────────────────────────────────────

    @READ_RETRY
    async def get_types(self) -> tuple[list[str], list[str]]:
        await self._ensure_session()
        resp = await _request(
            self._session, self._base_url, "POST",
            f"{self._nodes_index}/_search",
            body={
                "size": 0,
                "aggs": {
                    "entity_types": {
                        "terms": {"field": "entity_type", "size": 10000}
                    }
                },
            },
        )
        buckets = (
            resp.get("aggregations", {})
            .get("entity_types", {})
            .get("buckets", [])
        )
        original = [b["key"] for b in buckets]
        lowered = [t.lower() for t in original]
        return lowered, original

    # ── node CRUD ────────────────────────────────────────────────────────

    @READ_RETRY
    async def has_node(self, node_id: str) -> bool:
        await self._ensure_session()
        resp = await _request(
            self._session, self._base_url, "HEAD",
            f"{self._nodes_index}/_doc/{node_id}",
        )
        return resp.get("_status") == 200

    @READ_RETRY
    async def get_node(self, node_id: str) -> Union[dict, None]:
        await self._ensure_session()
        resp = await _request(
            self._session, self._base_url, "GET",
            f"{self._nodes_index}/_doc/{node_id}",
        )
        if not resp.get("found"):
            return None
        return resp["_source"].get("properties", {})

    @WRITE_RETRY
    async def upsert_node(self, node_id: str, node_data: dict[str, str]) -> None:
        await self._ensure_session()
        entity_type = node_data.get("entity_type", "UNKNOWN")
        doc = {
            "id": node_id,
            "entity_type": entity_type,
            "properties": node_data,
        }
        await _request(
            self._session, self._base_url, "PUT",
            f"{self._nodes_index}/_doc/{node_id}", body=doc,
        )

    @WRITE_RETRY
    async def delete_node(self, node_id: str) -> None:
        await self._ensure_session()
        # Delete connected edges first
        await _request(
            self._session, self._base_url, "POST",
            f"{self._edges_index}/_delete_by_query",
            body={
                "query": {
                    "bool": {
                        "should": [
                            {"term": {"source": node_id}},
                            {"term": {"target": node_id}},
                        ],
                        "minimum_should_match": 1,
                    }
                }
            },
        )
        # Delete the node itself
        await _request(
            self._session, self._base_url, "DELETE",
            f"{self._nodes_index}/_doc/{node_id}",
        )

    # ── edge CRUD ────────────────────────────────────────────────────────

    @READ_RETRY
    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        await self._ensure_session()
        edge_id = self._edge_id(source_node_id, target_node_id)
        resp = await _request(
            self._session, self._base_url, "HEAD",
            f"{self._edges_index}/_doc/{edge_id}",
        )
        return resp.get("_status") == 200

    @READ_RETRY
    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> Union[dict, None]:
        await self._ensure_session()
        edge_id = self._edge_id(source_node_id, target_node_id)
        resp = await _request(
            self._session, self._base_url, "GET",
            f"{self._edges_index}/_doc/{edge_id}",
        )
        if not resp.get("found"):
            return None
        return resp["_source"].get("properties", {})

    @WRITE_RETRY
    async def upsert_edge(
        self,
        source_node_id: str,
        target_node_id: str,
        edge_data: dict[str, str],
    ) -> None:
        await self._ensure_session()
        edge_id = self._edge_id(source_node_id, target_node_id)
        doc = {
            "id": edge_id,
            "source": source_node_id,
            "target": target_node_id,
            "properties": edge_data,
        }
        await _request(
            self._session, self._base_url, "PUT",
            f"{self._edges_index}/_doc/{edge_id}", body=doc,
        )

    # ── degree / traversal ───────────────────────────────────────────────

    @READ_RETRY
    async def node_degree(self, node_id: str) -> int:
        await self._ensure_session()
        resp = await _request(
            self._session, self._base_url, "POST",
            f"{self._edges_index}/_count",
            body={
                "query": {
                    "bool": {
                        "should": [
                            {"term": {"source": node_id}},
                            {"term": {"target": node_id}},
                        ],
                        "minimum_should_match": 1,
                    }
                }
            },
        )
        return resp.get("count", 0)

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        src_degree = await self.node_degree(src_id)
        tgt_degree = await self.node_degree(tgt_id)
        return src_degree + tgt_degree

    @READ_RETRY
    async def get_node_edges(
        self, source_node_id: str
    ) -> Union[list[tuple[str, str]], None]:
        await self._ensure_session()
        resp = await _request(
            self._session, self._base_url, "POST",
            f"{self._edges_index}/_search",
            body={
                "query": {
                    "bool": {
                        "should": [
                            {"term": {"source": source_node_id}},
                            {"term": {"target": source_node_id}},
                        ],
                        "minimum_should_match": 1,
                    }
                },
                "size": 10000,
                "_source": ["source", "target"],
            },
        )
        edges = []
        for hit in resp.get("hits", {}).get("hits", []):
            src = hit["_source"]["source"]
            tgt = hit["_source"]["target"]
            edges.append((src, tgt))
        return edges if edges else None

    # ── lifecycle ────────────────────────────────────────────────────────

    async def index_done_callback(self) -> None:
        await self._ensure_session()
        await _request(
            self._session, self._base_url, "POST",
            f"{self._nodes_index}/_refresh",
        )
        await _request(
            self._session, self._base_url, "POST",
            f"{self._edges_index}/_refresh",
        )


# ═══════════════════════════════════════════════════════════════════════════
# Document status storage backend using OpenSearch
# ═══════════════════════════════════════════════════════════════════════════

DOC_STATUS_MAPPING = {
    "settings": {"number_of_shards": 1, "number_of_replicas": 0},
    "mappings": {
        "properties": {
            "status": {"type": "keyword"},
            "content_summary": {"type": "text"},
            "content_length": {"type": "integer"},
            "created_at": {"type": "keyword"},
            "updated_at": {"type": "keyword"},
            "chunks_count": {"type": "integer"},
            "error": {"type": "text"},
        }
    },
}


@dataclass
class OpenSearchDocStatusStorage(DocStatusStorage):

    _session: aiohttp.ClientSession = field(default=None, repr=False, init=False)
    _base_url: str = field(default="", repr=False, init=False)
    _index_name: str = field(default="", repr=False, init=False)
    _initialized: bool = field(default=False, repr=False, init=False)

    def __post_init__(self):
        database = _get_database()
        self._index_name = f"{database}_docstatus"

    async def _ensure_session(self):
        if self._initialized:
            return
        self._base_url, self._session = _create_os_session()
        await _ensure_index(
            self._session, self._base_url, self._index_name, DOC_STATUS_MAPPING
        )
        logger.info(
            f"OpenSearch DocStatus storage '{self._index_name}' ready at {self._base_url}"
        )
        self._initialized = True

    # ── KV base operations ───────────────────────────────────────────────

    @READ_RETRY
    async def all_keys(self) -> list[str]:
        await self._ensure_session()
        keys = []
        body: dict[str, Any] = {
            "query": {"match_all": {}},
            "size": 10000,
            "_source": False,
            "sort": [{"_id": "asc"}],
        }
        search_after = None
        while True:
            if search_after is not None:
                body["search_after"] = search_after
            resp = await _request(
                self._session, self._base_url, "POST",
                f"{self._index_name}/_search", body=body,
            )
            hits = resp.get("hits", {}).get("hits", [])
            if not hits:
                break
            for hit in hits:
                keys.append(hit["_id"])
            search_after = hits[-1].get("sort")
            if search_after is None:
                break
        return keys

    @READ_RETRY
    async def get_by_id(self, id: str) -> Union[dict, None]:
        await self._ensure_session()
        resp = await _request(
            self._session, self._base_url, "GET",
            f"{self._index_name}/_doc/{id}",
        )
        if not resp.get("found"):
            return None
        return resp["_source"]

    @READ_RETRY
    async def get_by_ids(
        self, ids: list[str], fields: Union[set[str], None] = None
    ) -> list[Union[dict, None]]:
        await self._ensure_session()
        if not ids:
            return []
        resp = await _request(
            self._session, self._base_url, "POST",
            f"{self._index_name}/_mget", body={"ids": ids},
        )
        found: dict[str, dict] = {}
        for doc in resp.get("docs", []):
            if doc.get("found"):
                source = doc["_source"]
                if fields:
                    source = {k: v for k, v in source.items() if k in fields}
                found[doc["_id"]] = source
        return [found.get(id) for id in ids]

    @READ_RETRY
    async def filter_keys(self, data: list[str]) -> set[str]:
        await self._ensure_session()
        if not data:
            return set()
        resp = await _request(
            self._session, self._base_url, "POST",
            f"{self._index_name}/_mget", body={"ids": list(data)},
        )
        existing = set()
        for doc in resp.get("docs", []):
            if doc.get("found"):
                existing.add(doc["_id"])
        return set(data) - existing

    @WRITE_RETRY
    async def upsert(self, data: dict[str, dict]) -> None:
        await self._ensure_session()
        if not data:
            return
        actions = []
        for k, v in data.items():
            doc = dict(v)
            doc.pop("_id", None)
            actions.append(json.dumps({"index": {"_id": k}}))
            actions.append(json.dumps(doc))
        await _bulk_ndjson(
            self._session, self._base_url, self._index_name, actions
        )

    @WRITE_RETRY
    async def drop(self) -> None:
        await self._ensure_session()
        await _request(
            self._session, self._base_url, "POST",
            f"{self._index_name}/_delete_by_query",
            body={"query": {"match_all": {}}},
            params={"refresh": "true"},
        )

    async def index_done_callback(self) -> None:
        await self._ensure_session()
        await _request(
            self._session, self._base_url, "POST",
            f"{self._index_name}/_refresh",
        )

    # ── DocStatus-specific queries ───────────────────────────────────────

    @READ_RETRY
    async def get_status_counts(self) -> dict[str, int]:
        await self._ensure_session()
        resp = await _request(
            self._session, self._base_url, "POST",
            f"{self._index_name}/_search",
            body={
                "size": 0,
                "aggs": {
                    "status_counts": {
                        "terms": {"field": "status", "size": 100}
                    }
                },
            },
        )
        counts: dict[str, int] = {}
        for bucket in (
            resp.get("aggregations", {})
            .get("status_counts", {})
            .get("buckets", [])
        ):
            counts[bucket["key"]] = bucket["doc_count"]
        return counts

    @READ_RETRY
    async def get_failed_docs(self) -> dict[str, DocProcessingStatus]:
        await self._ensure_session()
        resp = await _request(
            self._session, self._base_url, "POST",
            f"{self._index_name}/_search",
            body={
                "query": {"term": {"status": "failed"}},
                "size": 10000,
            },
        )
        result: dict[str, DocProcessingStatus] = {}
        for hit in resp.get("hits", {}).get("hits", []):
            try:
                result[hit["_id"]] = DocProcessingStatus(**hit["_source"])
            except (KeyError, TypeError) as e:
                logger.error(
                    f"Missing required field for doc {hit['_id']}: {e}"
                )
        return result

    @READ_RETRY
    async def get_pending_docs(self) -> dict[str, DocProcessingStatus]:
        await self._ensure_session()
        resp = await _request(
            self._session, self._base_url, "POST",
            f"{self._index_name}/_search",
            body={
                "query": {"term": {"status": "pending"}},
                "size": 10000,
            },
        )
        result: dict[str, DocProcessingStatus] = {}
        for hit in resp.get("hits", {}).get("hits", []):
            try:
                result[hit["_id"]] = DocProcessingStatus(**hit["_source"])
            except (KeyError, TypeError) as e:
                logger.error(
                    f"Missing required field for doc {hit['_id']}: {e}"
                )
        return result
