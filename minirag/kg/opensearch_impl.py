from __future__ import annotations

import os
import json
import ssl
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple, Union

import aiohttp
import asyncio
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
    timeout = aiohttp.ClientTimeout(total=120, connect=10, sock_read=60)
    session = aiohttp.ClientSession(
        auth=auth,
        connector=connector,
        headers={"Content-Type": "application/json"},
        timeout=timeout,
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
    max_retries: int = 2,
) -> dict:
    """Issue an HTTP request to OpenSearch and return the JSON response."""
    url = f"{base_url}/{path}"
    kwargs: dict[str, Any] = {}
    if body is not None:
        kwargs["json"] = body
    if params is not None:
        kwargs["params"] = params
    last_err = None
    for attempt in range(max_retries + 1):
        try:
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
        except (asyncio.TimeoutError, aiohttp.ClientError) as e:
            last_err = e
            if attempt < max_retries:
                logger.warning(f"OpenSearch {method} {path} attempt {attempt+1} failed: {e}, retrying...")
                await asyncio.sleep(1)
    raise last_err


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
# Graph storage backend using OpenSearch Graph Plugin (Cypher interface)
# ═══════════════════════════════════════════════════════════════════════════


def _create_os_client():
    """Create an AsyncOpenSearch client from OPENSEARCH_* env vars."""
    import ssl as _ssl
    uri = os.environ.get("OPENSEARCH_URI", "https://localhost:9200")
    username = os.environ.get("OPENSEARCH_USERNAME", "admin")
    password = os.environ.get("OPENSEARCH_PASSWORD", "")
    verify_certs = os.environ.get("OPENSEARCH_VERIFY_CERTS", "true").lower() in (
        "true", "1", "yes",
    )
    use_ssl = uri.startswith("https")

    ssl_context = None
    if use_ssl and not verify_certs:
        ssl_context = _ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = _ssl.CERT_NONE

    from opensearchpy import AsyncOpenSearch
    return AsyncOpenSearch(
        hosts=[uri.rstrip("/")],
        http_auth=(username, password) if username else None,
        use_ssl=use_ssl,
        verify_certs=verify_certs,
        ssl_context=ssl_context,
        ssl_show_warn=False,
        timeout=30,
        max_retries=3,
        retry_on_timeout=True,
    )


def _graph_database_name() -> str:
    """Build graph plugin database name from OPENSEARCH_DATABASE env var."""
    import hashlib as _hl, re as _re
    raw = os.environ.get("OPENSEARCH_DATABASE", "minirag")
    name = _re.sub(r"[^a-z0-9_-]", "_", raw.lower()).lstrip("_-") or "g"
    if name[0].isdigit():
        name = "g" + name
    suffix = _hl.sha256(raw.encode()).hexdigest()[:8]
    return f"{name}-{suffix}"


@dataclass
class OpenSearchGraphStorage(BaseGraphStorage):
    """Graph storage using the OpenSearch Graph Plugin with Cypher queries.

    All node/edge operations go through ``POST _plugins/_cypher``.
    The plugin manages two underlying indices automatically:
    ``{database}-lpg-nodes`` and ``{database}-lpg-edges``.
    """

    _client: Any = field(default=None, repr=False, init=False)
    _database_name: str = field(default="", repr=False, init=False)
    _database_ready: bool = field(default=False, repr=False, init=False)

    def __post_init__(self):
        self._database_name = _graph_database_name()

    # ── Cypher execution ─────────────────────────────────────────────────

    @staticmethod
    def _clean_id(node_id: str) -> str:
        """Strip surrounding quotes and leading hyphens from node IDs."""
        return node_id.strip().strip('"').lstrip("-").strip()

    async def _execute_cypher(self, query: str, params: dict | None = None) -> dict:
        """Execute a Cypher query against the graph plugin endpoint."""
        import asyncio as _aio
        body: dict[str, Any] = {"query": query, "database": self._database_name}
        if params:
            body["parameters"] = params
        for attempt in range(3):
            try:
                return await self._client.transport.perform_request(
                    "POST", "/_plugins/_cypher", body=body
                )
            except Exception as e:
                if attempt < 2:
                    await _aio.sleep(2 ** attempt)
                else:
                    logger.error(f"Cypher failed after 3 attempts: {e}\nQuery: {query}")
                    raise

    @staticmethod
    def _rows(resp: dict) -> list[list]:
        """Normalise Cypher response to positional row lists."""
        columns = resp.get("columns", [])
        data = resp.get("data", [])
        if not columns or not data:
            return []
        return [[item.get(col) for col in columns] for item in data]

    # ── Database lifecycle ───────────────────────────────────────────────

    async def _ensure_session(self):
        if self._database_ready:
            return
        if self._client is None:
            self._client = _create_os_client()
        # Create graph database if it doesn't exist
        dim = self.embedding_func.embedding_dim if self.embedding_func else 1024
        db_body = {
            "embedding": {
                "dimension": dim, "field": "embedding",
                "engine": "faiss", "space_type": "cosinesimil",
            },
            "schema": {
                "nodes": {
                    "entity_id": {"type": "keyword"},
                    "entity_type": {"type": "keyword"},
                    "description": {"type": "text"},
                },
                "edges": {
                    "weight": {"type": "float"},
                    "description": {"type": "text"},
                    "keywords": {"type": "text"},
                },
                "strict": False,
            },
        }
        try:
            await self._client.transport.perform_request(
                "PUT", f"/_plugins/_graph/database/{self._database_name}", body=db_body,
            )
            logger.info(f"Created graph database: {self._database_name}")
        except Exception as e:
            err = str(e).lower()
            if "already exists" in err or "already_exists" in err or "resource_already_exists" in err:
                logger.debug(f"Graph database already exists: {self._database_name}")
            elif "creation failed" in err:
                # Graph plugin returns 500 "Creation failed" when DB already exists
                logger.debug(f"Graph database likely already exists: {self._database_name}")
            else:
                raise
            # Migrate: ensure promoted fields exist on pre-schema databases
            await self._ensure_promoted_fields(db_body.get("schema", {}))
        self._database_ready = True
        logger.info(f"OpenSearch graph plugin storage ready (db={self._database_name})")

    async def _ensure_promoted_fields(self, schema: dict) -> None:
        """Add promoted fields to existing indices that lack _meta.schema,
        then backfill existing documents so promoted top-level fields are populated."""
        for suffix, key in [("lpg-nodes", "nodes"), ("lpg-edges", "edges")]:
            fields = schema.get(key, {})
            if not fields:
                continue
            index = f"{self._database_name}-{suffix}"
            try:
                mapping = await self._client.indices.get_mapping(index=index)
                existing = mapping.get(index, {}).get("mappings", {})
                if existing.get("_meta", {}).get("schema"):
                    continue  # already has promoted schema
                # Build PUT mapping body with promoted fields + _meta
                props = {}
                for fname, fdef in fields.items():
                    if isinstance(fdef, dict):
                        props[fname] = fdef
                    else:
                        props[fname] = {"type": fdef}
                body: dict = {"properties": props}
                body["_meta"] = {"schema": {key: {f: (d if isinstance(d, str) else d.get("type", "keyword")) for f, d in fields.items()}, "strict": schema.get("strict", False)}}
                await self._client.indices.put_mapping(index=index, body=body)
                logger.info(f"Migrated {index}: added promoted fields {list(fields.keys())}")
                # Backfill: copy properties.X → top-level X for existing docs
                await self._backfill_promoted_fields(index, list(fields.keys()))
            except Exception as e:
                logger.warning(f"Failed to migrate promoted fields for {index}: {e}")

    async def _backfill_promoted_fields(self, index: str, field_names: list[str]) -> None:
        """Use update_by_query to copy nested properties.* values to top-level promoted fields."""
        # Build a Painless script that promotes each field
        lines = []
        for f in field_names:
            lines.append(
                f"if (ctx._source.containsKey('properties') && ctx._source.properties.containsKey('{f}')) "
                f"{{ ctx._source['{f}'] = ctx._source.properties['{f}']; }}"
            )
        script = " ".join(lines)
        try:
            resp = await self._client.update_by_query(
                index=index,
                body={
                    "script": {"source": script, "lang": "painless"},
                    "query": {"match_all": {}},
                },
                refresh=True,
            )
            updated = resp.get("updated", 0)
            logger.info(f"Backfilled {updated} docs in {index} with promoted fields {field_names}")
        except Exception as e:
            logger.warning(f"Failed to backfill promoted fields for {index}: {e}")

    # ── get_types ────────────────────────────────────────────────────────

    async def get_types(self) -> tuple[list[str], list[str]]:
        await self._ensure_session()
        try:
            resp = await self._execute_cypher(
                "MATCH (n:Entity) RETURN DISTINCT n.entity_type AS t"
            )
            original = [r[0] for r in self._rows(resp) if r[0]]
            lowered = [t.lower() for t in original]
            return lowered, original
        except Exception:
            return [], []

    # ── node CRUD ────────────────────────────────────────────────────────

    async def has_node(self, node_id: str) -> bool:
        await self._ensure_session()
        try:
            resp = await self._execute_cypher(
                "MATCH (n:Entity {entity_id: $id}) RETURN count(n) > 0 AS exists",
                {"id": self._clean_id(node_id)},
            )
            rows = self._rows(resp)
            return bool(rows[0][0]) if rows else False
        except Exception:
            return False

    async def get_node(self, node_id: str) -> Union[dict, None]:
        await self._ensure_session()
        try:
            resp = await self._execute_cypher(
                "MATCH (n:Entity {entity_id: $id}) RETURN properties(n) AS props",
                {"id": self._clean_id(node_id)},
            )
            rows = self._rows(resp)
            if rows and rows[0][0]:
                props = rows[0][0]
                props.pop("embedding", None)
                return props
            return None
        except Exception:
            return None

    async def upsert_node(self, node_id: str, node_data: dict[str, str]) -> None:
        await self._ensure_session()
        node_id = self._clean_id(node_id)
        if not node_id:
            return
        props = {}
        for k, v in node_data.items():
            if k in ("_id", "embedding"):
                continue
            props[k] = v.strip('"') if isinstance(v, str) else v
        props["entity_id"] = node_id
        import re as _re
        entity_type = props.get("entity_type", "")
        label_clause = ""
        if entity_type:
            safe_type = _re.sub(r"[^a-zA-Z0-9_]", "_", entity_type)
            label_clause = f", n:`{safe_type}`"
        await self._execute_cypher(
            f"MERGE (n:Entity {{entity_id: $id}}) "
            f"ON CREATE SET n += $props{label_clause} "
            f"ON MATCH SET n += $props{label_clause}",
            {"id": node_id, "props": props},
        )
        # Refresh so subsequent MERGEs can find this node
        try:
            await self._client.indices.refresh(index=f"{self._database_name}-lpg-nodes")
        except Exception:
            pass

    async def delete_node(self, node_id: str) -> None:
        await self._ensure_session()
        try:
            await self._execute_cypher(
                "MATCH (n:Entity {entity_id: $id}) DETACH DELETE n",
                {"id": self._clean_id(node_id)},
            )
        except Exception as e:
            logger.error(f"Error deleting node {node_id}: {e}")

    # ── edge CRUD ────────────────────────────────────────────────────────

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        await self._ensure_session()
        try:
            resp = await self._execute_cypher(
                "MATCH (a:Entity {entity_id: $src})-[r:DIRECTED]->(b:Entity {entity_id: $tgt}) "
                "RETURN count(r) > 0 AS exists",
                {"src": self._clean_id(source_node_id), "tgt": self._clean_id(target_node_id)},
            )
            rows = self._rows(resp)
            return bool(rows[0][0]) if rows else False
        except Exception:
            return False

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> Union[dict, None]:
        await self._ensure_session()
        try:
            resp = await self._execute_cypher(
                "MATCH (a:Entity {entity_id: $src})-[r:DIRECTED]->(b:Entity {entity_id: $tgt}) "
                "RETURN properties(r) AS props LIMIT 1",
                {"src": self._clean_id(source_node_id), "tgt": self._clean_id(target_node_id)},
            )
            rows = self._rows(resp)
            return rows[0][0] if rows else None
        except Exception:
            return None

    async def upsert_edge(
        self,
        source_node_id: str,
        target_node_id: str,
        edge_data: dict[str, str],
    ) -> None:
        await self._ensure_session()
        source_node_id = self._clean_id(source_node_id)
        target_node_id = self._clean_id(target_node_id)
        if not source_node_id or not target_node_id:
            return
        props = {}
        for k, v in edge_data.items():
            if k == "_id":
                continue
            props[k] = v.strip('"') if isinstance(v, str) else v
        if "weight" in props:
            try:
                props["weight"] = float(props["weight"])
            except (ValueError, TypeError):
                props["weight"] = 1.0
        params = {"src": source_node_id, "tgt": target_node_id, "props": props}
        # Check existence first (MERGE on edges is O(E) in the graph plugin)
        resp = await self._execute_cypher(
            "MATCH (s:Entity {entity_id: $src})-[r:DIRECTED]->(t:Entity {entity_id: $tgt}) "
            "RETURN count(r) AS cnt", params,
        )
        exists = resp.get("data", [{}])[0].get("cnt", 0) > 0
        if exists:
            await self._execute_cypher(
                "MATCH (s:Entity {entity_id: $src})-[r:DIRECTED]->(t:Entity {entity_id: $tgt}) "
                "SET r += $props", params,
            )
        else:
            # CREATE + SET split: graph plugin drops SET in same statement as CREATE
            await self._execute_cypher(
                "MATCH (s:Entity {entity_id: $src}), (t:Entity {entity_id: $tgt}) "
                "CREATE (s)-[r:DIRECTED]->(t)", params,
            )
            if props:
                await self._execute_cypher(
                    "MATCH (s:Entity {entity_id: $src})-[r:DIRECTED]->(t:Entity {entity_id: $tgt}) "
                    "SET r += $props", params,
                )

    # ── degree / traversal ───────────────────────────────────────────────

    async def node_degree(self, node_id: str) -> int:
        await self._ensure_session()
        try:
            resp = await self._execute_cypher(
                "MATCH (n:Entity {entity_id: $id})-[r:DIRECTED]-() "
                "RETURN count(r) AS degree",
                {"id": self._clean_id(node_id)},
            )
            rows = self._rows(resp)
            return int(rows[0][0]) if rows else 0
        except Exception:
            return 0

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        return await self.node_degree(src_id) + await self.node_degree(tgt_id)

    async def get_node_edges(
        self, source_node_id: str
    ) -> Union[list[tuple[str, str]], None]:
        await self._ensure_session()
        try:
            resp = await self._execute_cypher(
                "MATCH (n:Entity {entity_id: $id})-[r:DIRECTED]-(c:Entity) "
                "RETURN n.entity_id AS src, c.entity_id AS tgt",
                {"id": self._clean_id(source_node_id)},
            )
            rows = self._rows(resp)
            return [(r[0], r[1]) for r in rows] if rows else None
        except Exception:
            return None

    # ── type-based node query ───────────────────────────────────────────

    async def get_node_from_types(self, type_list) -> list[dict]:
        """Return all nodes whose entity_type is in *type_list*."""
        await self._ensure_session()
        if not type_list:
            return []
        try:
            resp = await self._execute_cypher(
                "MATCH (n:Entity) WHERE n.entity_type IN $types "
                "RETURN n.entity_id AS eid, properties(n) AS props",
                {"types": list(type_list)},
            )
            result = []
            for row in self._rows(resp):
                props = row[1] or {}
                props.pop("embedding", None)
                result.append({**props, "entity_name": row[0]})
            return result
        except Exception:
            return []

    # ── k-hop traversal ────────────────────────────────────────────────

    async def get_neighbors_within_k_hops(
        self, source_node_id: str, k: int
    ) -> list[tuple]:
        """BFS traversal returning paths (as tuples of node IDs) up to k hops."""
        await self._ensure_session()
        if not await self.has_node(source_node_id):
            return []

        from ..utils import merge_tuples
        import copy

        # Hop 1: direct neighbours
        edges = await self.get_node_edges(source_node_id) or []
        source_edge = [
            (source_node_id, tgt if src == source_node_id else src)
            for src, tgt in edges
        ]

        count = 1
        while count < k:
            count += 1
            sc_edge = copy.deepcopy(source_edge)
            source_edge = []
            for pair in sc_edge:
                last_node = pair[-1]
                append_edges = await self.get_node_edges(last_node) or []
                append_tuples = [
                    (last_node, tgt if src == last_node else src)
                    for src, tgt in append_edges
                ]
                for t in merge_tuples([pair], append_tuples):
                    source_edge.append(t)
        return source_edge

    # ── lifecycle ────────────────────────────────────────────────────────

    async def index_done_callback(self) -> None:
        await self._ensure_session()
        try:
            await self._client.indices.refresh(index=f"{self._database_name}-lpg-nodes")
        except Exception:
            pass
        try:
            await self._client.indices.refresh(index=f"{self._database_name}-lpg-edges")
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════
# Docgraph storage backend — document-derived graph mode
# ═══════════════════════════════════════════════════════════════════════════


def _default_authz_claims() -> dict:
    """Default authz claims for docgraph ingest (no real auth in MiniRAG)."""
    return {"backend_roles": [os.environ.get("OPENSEARCH_AUTHZ_ROLE", "admin")]}


@dataclass
class OpenSearchDocgraphStorage(OpenSearchGraphStorage):
    """Graph storage using the OpenSearch docgraph (document-derived graph) API.

    Uses ``mode: docgraph`` database with evidence model:
        Document → HAS_CHUNK → Chunk → MENTIONS → Entity
                                     → ASSERTS  → RelFact

    Write path uses ``_plugins/_graph/docgraph/{db}/_ingest`` and ``_extract``.
    Read path uses evidence-mediated Cypher (all Entity/RelFact access goes
    through Chunk).
    """

    # Buffers for batched docgraph _extract calls
    _pending_entities: dict = field(default_factory=dict, repr=False, init=False)
    _pending_relations: dict = field(default_factory=dict, repr=False, init=False)
    _known_chunks: set = field(default_factory=set, repr=False, init=False)
    _known_docs: set = field(default_factory=set, repr=False, init=False)

    # ── Database lifecycle (override: docgraph mode) ─────────────────────

    async def _ensure_session(self):
        if self._database_ready:
            return
        if self._client is None:
            self._client = _create_os_client()
        dim = self.embedding_func.embedding_dim if self.embedding_func else 1024
        db_body = {
            "mode": "docgraph",
            "embedding": {
                "dimension": dim, "field": "embedding",
                "engine": "faiss", "space_type": "cosinesimil",
            },
            "schema": {
                "nodes": {
                    "entity_id": {"type": "keyword"},
                    "entity_type": {"type": "keyword"},
                    "description": {"type": "text"},
                },
                "edges": {
                    "weight": {"type": "float"},
                    "description": {"type": "text"},
                    "keywords": {"type": "text"},
                },
                "strict": False,
            },
        }
        try:
            await self._client.transport.perform_request(
                "PUT", f"/_plugins/_graph/database/{self._database_name}", body=db_body,
            )
            logger.info(f"Created docgraph database: {self._database_name}")
        except Exception as e:
            err = str(e).lower()
            if any(k in err for k in ("already exists", "already_exists", "resource_already_exists", "creation failed")):
                logger.debug(f"Docgraph database already exists: {self._database_name}")
            else:
                raise
        self._database_ready = True
        logger.info(f"OpenSearch docgraph storage ready (db={self._database_name})")

    # ── Docgraph REST helpers ────────────────────────────────────────────

    async def _docgraph_request(self, action: str, body: dict) -> dict:
        """POST _plugins/_graph/docgraph/{db}/{action}"""
        import asyncio as _aio
        path = f"/_plugins/_graph/docgraph/{self._database_name}/{action}"
        for attempt in range(3):
            try:
                return await self._client.transport.perform_request(
                    "POST", path, body=body,
                )
            except Exception as e:
                info = getattr(e, 'info', None) or str(e)
                logger.warning(f"Docgraph {action} attempt {attempt+1}: {info}")
                if attempt < 2:
                    await _aio.sleep(2 ** attempt)
                else:
                    logger.error(f"Docgraph {action} failed: {info}")
                    raise

    async def _ensure_doc_and_chunk(self, chunk_id: str, doc_id: str | None = None):
        """Ensure Document and Chunk nodes exist via docgraph upsert APIs."""
        # Use a default document if none provided
        if not doc_id:
            doc_id = "default"

        if doc_id not in self._known_docs:
            await self._docgraph_request("_upsert_document", {
                "document_id": doc_id,
                "authz_claims": _default_authz_claims(),
            })
            self._known_docs.add(doc_id)

        if chunk_id not in self._known_chunks:
            await self._docgraph_request("_upsert_chunk", {
                "chunk_id": chunk_id,
                "document_id": doc_id,
                "authz_claims": _default_authz_claims(),
            })
            self._known_chunks.add(chunk_id)

    # ── Write path (buffered → _extract) ─────────────────────────────────

    async def upsert_node(self, node_id: str, node_data: dict[str, str]) -> None:
        await self._ensure_session()
        node_id = self._clean_id(node_id)
        if not node_id:
            return

        # Parse source_id to get chunk keys
        source_id = node_data.get("source_id", "")
        chunk_ids = [c.strip() for c in source_id.split("<SEP>") if c.strip()] if source_id else ["unknown"]

        entity_obj = {
            "entity_id": node_id,
            "labels": ["Entity"],
            "properties": {
                "entity_id": node_id,
                "entity_type": node_data.get("entity_type", ""),
                "description": node_data.get("description", ""),
                "source_id": source_id,
            },
        }

        for cid in chunk_ids:
            self._pending_entities.setdefault(cid, {})[node_id] = entity_obj

    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str],
    ) -> None:
        await self._ensure_session()
        source_node_id = self._clean_id(source_node_id)
        target_node_id = self._clean_id(target_node_id)
        if not source_node_id or not target_node_id:
            return

        source_id = edge_data.get("source_id", "")
        chunk_ids = [c.strip() for c in source_id.split("<SEP>") if c.strip()] if source_id else ["unknown"]

        weight = edge_data.get("weight", 1.0)
        try:
            weight = float(weight)
        except (ValueError, TypeError):
            weight = 1.0

        rel_id = f"{source_node_id}--{target_node_id}"
        rel_obj = {
            "relation_id": rel_id,
            "type": "DIRECTED",
            "source_entity_id": source_node_id,
            "target_entity_id": target_node_id,
            "properties": {
                "weight": weight,
                "description": edge_data.get("description", ""),
                "keywords": edge_data.get("keywords", ""),
                "source_id": source_id,
            },
        }

        for cid in chunk_ids:
            self._pending_relations.setdefault(cid, {})[rel_id] = rel_obj

    async def _flush_pending(self):
        """Flush all buffered entities/relations via docgraph _extract."""
        all_chunk_ids = set(self._pending_entities.keys()) | set(self._pending_relations.keys())
        # Create all docs and chunks first
        for cid in all_chunk_ids:
            await self._ensure_doc_and_chunk(cid)
        # Refresh so chunks are visible to _extract
        try:
            await self._client.indices.refresh(index=f"{self._database_name}-lpg-nodes")
        except Exception:
            pass
        # Now extract entities/relations
        for cid in all_chunk_ids:
            entities = list(self._pending_entities.get(cid, {}).values())
            relations = list(self._pending_relations.get(cid, {}).values())
            if entities or relations:
                await self._docgraph_request("_extract", {
                    "chunk_id": cid,
                    "entities": entities,
                    "relations": relations,
                })
        self._pending_entities.clear()
        self._pending_relations.clear()

    async def index_done_callback(self) -> None:
        await self._ensure_session()
        await self._flush_pending()
        try:
            await self._client.indices.refresh(index=f"{self._database_name}-lpg-nodes")
        except Exception:
            pass
        try:
            await self._client.indices.refresh(index=f"{self._database_name}-lpg-edges")
        except Exception:
            pass

    # ── Read path (evidence-mediated queries) ────────────────────────────

    async def has_node(self, node_id: str) -> bool:
        await self._ensure_session()
        try:
            resp = await self._execute_cypher(
                "MATCH (c:Chunk)-[:MENTIONS]->(n:Entity) "
                "WHERE n.entity_id = $id "
                "RETURN count(n) > 0 AS exists",
                {"id": self._clean_id(node_id)},
            )
            rows = self._rows(resp)
            return bool(rows[0][0]) if rows else False
        except Exception:
            return False

    async def get_node(self, node_id: str) -> Union[dict, None]:
        await self._ensure_session()
        try:
            resp = await self._execute_cypher(
                "MATCH (c:Chunk)-[:MENTIONS]->(n:Entity) "
                "WHERE n.entity_id = $id "
                "RETURN properties(n) AS props LIMIT 1",
                {"id": self._clean_id(node_id)},
            )
            rows = self._rows(resp)
            if rows and rows[0][0]:
                props = rows[0][0]
                props.pop("embedding", None)
                return props
            return None
        except Exception:
            return None

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        await self._ensure_session()
        try:
            resp = await self._execute_cypher(
                "MATCH (c:Chunk)-[:ASSERTS]->(rf:RelFact)"
                "-[:SOURCE_ENTITY]->(a:Entity), "
                "(rf)-[:TARGET_ENTITY]->(b:Entity) "
                "WHERE a.entity_id = $src AND b.entity_id = $tgt "
                "RETURN count(rf) > 0 AS exists",
                {"src": self._clean_id(source_node_id), "tgt": self._clean_id(target_node_id)},
            )
            rows = self._rows(resp)
            return bool(rows[0][0]) if rows else False
        except Exception:
            return False

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> Union[dict, None]:
        await self._ensure_session()
        try:
            resp = await self._execute_cypher(
                "MATCH (c:Chunk)-[:ASSERTS]->(rf:RelFact)"
                "-[:SOURCE_ENTITY]->(a:Entity), "
                "(rf)-[:TARGET_ENTITY]->(b:Entity) "
                "WHERE a.entity_id = $src AND b.entity_id = $tgt "
                "RETURN properties(rf) AS props LIMIT 1",
                {"src": self._clean_id(source_node_id), "tgt": self._clean_id(target_node_id)},
            )
            rows = self._rows(resp)
            return rows[0][0] if rows else None
        except Exception:
            return None

    async def node_degree(self, node_id: str) -> int:
        await self._ensure_session()
        try:
            resp = await self._execute_cypher(
                "MATCH (c:Chunk)-[:ASSERTS]->(rf:RelFact)-[:SOURCE_ENTITY]->(n:Entity) "
                "WHERE n.entity_id = $id "
                "RETURN count(rf) AS deg",
                {"id": self._clean_id(node_id)},
            )
            rows_src = self._rows(resp)
            deg_src = int(rows_src[0][0]) if rows_src else 0

            resp2 = await self._execute_cypher(
                "MATCH (c:Chunk)-[:ASSERTS]->(rf:RelFact)-[:TARGET_ENTITY]->(n:Entity) "
                "WHERE n.entity_id = $id "
                "RETURN count(rf) AS deg",
                {"id": self._clean_id(node_id)},
            )
            rows_tgt = self._rows(resp2)
            deg_tgt = int(rows_tgt[0][0]) if rows_tgt else 0

            return deg_src + deg_tgt
        except Exception:
            return 0

    async def get_node_edges(
        self, source_node_id: str
    ) -> Union[list[tuple[str, str]], None]:
        await self._ensure_session()
        try:
            clean_id = self._clean_id(source_node_id)
            resp1 = await self._execute_cypher(
                "MATCH (c:Chunk)-[:ASSERTS]->(rf:RelFact)"
                "-[:SOURCE_ENTITY]->(a:Entity), "
                "(rf)-[:TARGET_ENTITY]->(b:Entity) "
                "WHERE a.entity_id = $id "
                "RETURN a.entity_id AS src, b.entity_id AS tgt",
                {"id": clean_id},
            )
            resp2 = await self._execute_cypher(
                "MATCH (c:Chunk)-[:ASSERTS]->(rf:RelFact)"
                "-[:TARGET_ENTITY]->(a:Entity), "
                "(rf)-[:SOURCE_ENTITY]->(b:Entity) "
                "WHERE a.entity_id = $id "
                "RETURN b.entity_id AS src, a.entity_id AS tgt",
                {"id": clean_id},
            )
            edges = []
            for r in self._rows(resp1):
                edges.append((r[0], r[1]))
            for r in self._rows(resp2):
                edges.append((r[0], r[1]))
            return edges if edges else None
        except Exception:
            return None

    async def get_types(self) -> tuple[list[str], list[str]]:
        await self._ensure_session()
        try:
            resp = await self._execute_cypher(
                "MATCH (c:Chunk)-[:MENTIONS]->(n:Entity) "
                "RETURN DISTINCT n.entity_type AS t"
            )
            original = [r[0] for r in self._rows(resp) if r[0]]
            lowered = [t.lower() for t in original]
            return lowered, original
        except Exception:
            return [], []

    async def get_node_from_types(self, type_list) -> list[dict]:
        await self._ensure_session()
        if not type_list:
            return []
        try:
            resp = await self._execute_cypher(
                "MATCH (c:Chunk)-[:MENTIONS]->(n:Entity) "
                "WHERE n.entity_type IN $types "
                "RETURN n.entity_id AS eid, properties(n) AS props",
                {"types": list(type_list)},
            )
            result = []
            for row in self._rows(resp):
                props = row[1] or {}
                props.pop("embedding", None)
                result.append({**props, "entity_name": row[0]})
            return result
        except Exception:
            return []

    # ── Delete path ──────────────────────────────────────────────────────

    async def delete_node(self, node_id: str) -> None:
        """Withdraw documents that sourced this entity, or fall back to Cypher."""
        await self._ensure_session()
        try:
            # Find source documents for this entity via evidence chain
            resp = await self._execute_cypher(
                "MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk)-[:MENTIONS]->(n:Entity) "
                "WHERE n.entity_id = $id "
                "RETURN DISTINCT d.id AS doc_id",
                {"id": self._clean_id(node_id)},
            )
            doc_ids = [r[0] for r in self._rows(resp) if r[0]]
            for doc_id in doc_ids:
                try:
                    await self._docgraph_request("_withdraw_document", {"document_id": doc_id})
                except Exception as e:
                    logger.warning(f"Failed to withdraw document {doc_id}: {e}")
        except Exception as e:
            logger.error(f"Error deleting node {node_id}: {e}")


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

    @WRITE_RETRY
    async def upsert(self, data: dict[str, dict]) -> None:
        await self._ensure_session()
        if not data:
            return
        # Serialize enum values to strings for JSON storage
        actions = []
        for k, v in data.items():
            doc = dict(v)
            doc.pop("_id", None)
            for field_name, field_val in doc.items():
                if isinstance(field_val, DocStatus):
                    doc[field_name] = field_val.value
            actions.append(json.dumps({"index": {"_id": k}}))
            actions.append(json.dumps(doc))
        # Use refresh=true so data is immediately searchable
        body_str = "\n".join(actions) + "\n"
        url = f"{self._base_url}/{self._index_name}/_bulk?refresh=true"
        async with self._session.post(
            url,
            data=body_str,
            headers={"Content-Type": "application/x-ndjson"},
        ) as resp:
            text = await resp.text()
            try:
                result = json.loads(text)
            except json.JSONDecodeError:
                result = {"_raw": text}
        if result.get("errors"):
            failed = [
                item
                for item in result.get("items", [])
                if "error" in item.get("index", {})
            ]
            if failed:
                logger.error(
                    f"Bulk DocStatus upsert had {len(failed)} errors: {failed[:3]}"
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

    @READ_RETRY
    async def filter_keys(self, data) -> set[str]:
        """Return keys that should be processed (not in storage or not successfully processed)."""
        await self._ensure_session()
        if not data:
            return set()
        data_list = list(data)
        resp = await _request(
            self._session, self._base_url, "POST",
            f"{self._index_name}/_mget",
            body={
                "docs": [
                    {"_id": k, "_source": ["status"]} for k in data_list
                ]
            },
        )
        result = set()
        for i, doc in enumerate(resp.get("docs", [])):
            key = data_list[i]
            if not doc.get("found"):
                result.add(key)
            else:
                status = doc.get("_source", {}).get("status")
                if status != DocStatus.PROCESSED.value:
                    result.add(key)
        return result

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
    async def get_docs_by_status(
        self, status: DocStatus
    ) -> dict[str, DocProcessingStatus]:
        """Get all documents with a specific status."""
        await self._ensure_session()
        resp = await _request(
            self._session, self._base_url, "POST",
            f"{self._index_name}/_search",
            body={
                "query": {"term": {"status": status.value}},
                "size": 10000,
            },
        )
        result: dict[str, DocProcessingStatus] = {}
        for hit in resp.get("hits", {}).get("hits", []):
            try:
                data = hit["_source"].copy()
                # If content is missing, use content_summary as content
                if "content" not in data and "content_summary" in data:
                    data["content"] = data["content_summary"]
                # Convert status string to DocStatus enum
                if "status" in data and isinstance(data["status"], str):
                    data["status"] = DocStatus(data["status"])
                # Remove fields not in DocProcessingStatus
                data.pop("_status", None)
                result[hit["_id"]] = DocProcessingStatus(**data)
            except (KeyError, TypeError, ValueError) as e:
                logger.error(
                    f"Missing required field for doc {hit['_id']}: {e}"
                )
        return result

    async def get_failed_docs(self) -> dict[str, DocProcessingStatus]:
        return await self.get_docs_by_status(DocStatus.FAILED)

    async def get_pending_docs(self) -> dict[str, DocProcessingStatus]:
        return await self.get_docs_by_status(DocStatus.PENDING)
