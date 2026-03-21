"""
Test script: MiniRAG with Bedrock LLM + OpenSearch storage backends.

Prerequisites:
  - OpenSearch running at https://localhost:9200 (with admin:Admin@1234)
  - AWS credentials configured (IAM role or env vars) with Bedrock access
  - pip install aioboto3

Usage:
  python3.11 test_bedrock_opensearch.py
"""

import os
import asyncio
import json
import sys

# ── OpenSearch config ────────────────────────────────────────────────────────
os.environ["OPENSEARCH_URI"] = "https://localhost:9200"
os.environ["OPENSEARCH_USERNAME"] = "admin"
os.environ["OPENSEARCH_PASSWORD"] = "Admin@1234"
os.environ["OPENSEARCH_VERIFY_CERTS"] = "false"
os.environ["OPENSEARCH_DATABASE"] = "minirag_test"

# ── AWS config ───────────────────────────────────────────────────────────────
os.environ.setdefault("AWS_DEFAULT_REGION", "us-west-2")

# ── Models ───────────────────────────────────────────────────────────────────
BEDROCK_LLM_MODEL = "anthropic.claude-3-5-sonnet-20241022-v2:0"
BEDROCK_EMBED_MODEL = "amazon.titan-embed-text-v2:0"
TITAN_V2_DIM = 1024

WORKING_DIR = "./minirag_test_workdir"
os.makedirs(WORKING_DIR, exist_ok=True)

SAMPLE_DOC = """
Albert Einstein (14 March 1879 - 18 April 1955) was a German-born theoretical
physicist who is widely held as one of the most influential scientists in history.
He is best known for developing the theory of relativity, and he also made
important contributions to quantum mechanics. His mass-energy equivalence
formula E = mc squared, which arises from relativity theory, has been called
the world's most famous equation. He received the 1921 Nobel Prize in Physics
for his discovery of the law of the photoelectric effect, a pivotal step in
the development of quantum theory.

Einstein was born in the German Empire but moved to Switzerland in 1895,
renouncing his German citizenship the following year. He graduated from
the Swiss Federal Polytechnic in Zurich in 1900. He was employed at the
Swiss Patent Office in Bern from 1902 to 1909. In 1905, he published four
groundbreaking papers during his annus mirabilis (miracle year), which
brought him to the notice of the academic world.

Einstein moved to Berlin in 1914. In November 1915, he completed the general
theory of relativity. In 1933, while visiting the United States, Adolf Hitler
came to power in Germany. Opposed to the persecution, Einstein decided not
to return and settled in the United States. He became an American citizen in
1940. On the eve of World War II, he endorsed a letter to President Franklin
D. Roosevelt alerting him to the potential German nuclear weapons program and
recommending that the US begin similar research.
"""

TEST_QUERIES = [
    "What is Einstein famous for?",
    "Where did Einstein work before academia?",
    "When did Einstein become an American citizen?",
]

OS_BASE = "https://localhost:9200"
OS_AUTH = ("admin", "Admin@1234")


async def os_request(session, method, path, json_body=None):
    """Helper to issue requests to OpenSearch."""
    url = f"{OS_BASE}/{path}"
    kwargs = {}
    if json_body is not None:
        kwargs["json"] = json_body
        kwargs["headers"] = {"Content-Type": "application/json"}
    async with session.request(method, url, **kwargs) as resp:
        try:
            return resp.status, await resp.json()
        except Exception:
            return resp.status, await resp.text()


async def cleanup_indices(session):
    """Delete all minirag_test_* indices and graph plugin databases."""
    print("=== Cleaning up previous test indices ===")
    _, data = await os_request(session, "GET", "_cat/indices?h=index&format=json")
    if isinstance(data, list):
        for idx in data:
            name = idx.get("index", "")
            if name.startswith("minirag_test"):
                status, _ = await os_request(session, "DELETE", name)
                print(f"  Deleted index: {name} -> {status}")
    # Also delete graph plugin database
    from minirag.kg.opensearch_impl import _graph_database_name
    db_name = _graph_database_name()
    status, _ = await os_request(session, "DELETE", f"_plugins/_graph/database/{db_name}")
    print(f"  Deleted graph database: {db_name} -> {status}")


async def refresh_all(session):
    """Refresh all minirag_test* indices so doc counts are accurate."""
    await os_request(session, "POST", "minirag_test*/_refresh")


async def check_indices(session):
    """Print doc counts for all minirag_test* indices (including graph plugin)."""
    print("\n=== Phase 2: Verifying OpenSearch indices ===")
    await refresh_all(session)
    _, data = await os_request(
        session, "GET", "_cat/indices/minirag_test*?h=index,docs.count&format=json"
    )
    if not isinstance(data, list) or not data:
        print("  ERROR: No minirag_test indices found!")
        return False
    for idx in sorted(data, key=lambda x: x.get("index", "")):
        print(f"  {idx['index']}: {idx.get('docs.count', '?')} docs")
    return True


async def check_graph(session):
    """Print sample graph nodes and edges via Cypher."""
    print("\n=== Phase 4: Checking graph data ===")
    from minirag.kg.opensearch_impl import _graph_database_name
    db_name = _graph_database_name()
    # Count nodes
    status, data = await os_request(
        session, "POST", "_plugins/_cypher",
        json_body={"query": "MATCH (n:Entity) RETURN count(n) AS cnt", "database": db_name},
    )
    cnt = data.get("data", [{}])[0].get("cnt", 0) if status == 200 else "?"
    print(f"  Graph nodes: {cnt}")
    # Count edges
    status, data = await os_request(
        session, "POST", "_plugins/_cypher",
        json_body={"query": "MATCH ()-[r:DIRECTED]->() RETURN count(r) AS cnt", "database": db_name},
    )
    cnt = data.get("data", [{}])[0].get("cnt", 0) if status == 200 else "?"
    print(f"  Graph edges: {cnt}")
    # Sample nodes
    status, data = await os_request(
        session, "POST", "_plugins/_cypher",
        json_body={"query": "MATCH (n:Entity) RETURN n.entity_id AS eid, n.entity_type AS t LIMIT 5", "database": db_name},
    )
    if status == 200:
        for row in data.get("data", []):
            print(f"    Node: {row.get('eid')} (type={row.get('t', '?')})")


async def main():
    import aiohttp
    from minirag import MiniRAG, QueryParam
    from minirag.utils import EmbeddingFunc

    # ── Build LLM function ───────────────────────────────────────────────
    # Use aioboto3 Converse API directly to avoid bedrock module's env-var bug
    # when running with IAM roles (no explicit AWS_ACCESS_KEY_ID set).
    import copy as _copy

    async def llm_func(
        prompt,
        system_prompt=None,
        history_messages=[],
        keyword_extraction=False,
        **kwargs,
    ):
        from minirag.utils import locate_json_string_body_from_string

        kwargs.pop("hashing_kv", None)
        kwargs.pop("keyword_extraction", None)

        messages = []
        for msg in history_messages:
            m = _copy.copy(msg)
            m["content"] = [{"text": m["content"]}]
            messages.append(m)
        messages.append({"role": "user", "content": [{"text": prompt}]})

        args = {"modelId": BEDROCK_LLM_MODEL, "messages": messages}
        if system_prompt:
            args["system"] = [{"text": system_prompt}]

        # Map inference params
        inference_params_map = {
            "max_tokens": "maxTokens",
            "top_p": "topP",
            "stop_sequences": "stopSequences",
        }
        inf_keys = set(kwargs) & {"max_tokens", "temperature", "top_p", "stop_sequences"}
        if inf_keys:
            args["inferenceConfig"] = {}
            for param in inf_keys:
                args["inferenceConfig"][inference_params_map.get(param, param)] = kwargs.pop(param)

        session = aioboto3.Session()
        async with session.client("bedrock-runtime") as client:
            response = await client.converse(**args)

        result = response["output"]["message"]["content"][0]["text"]
        if keyword_extraction:
            return locate_json_string_body_from_string(result)
        return result

    # ── Build embedding function ─────────────────────────────────────────
    # Work around bedrock_embed's env-var bug when using IAM roles:
    # it tries os.environ["KEY"] = os.environ.get("KEY", None) which fails.
    import aioboto3
    import numpy as np

    async def embed_func(texts):
        session = aioboto3.Session()
        async with session.client("bedrock-runtime") as client:
            embeddings = []
            for text in texts:
                body = json.dumps(
                    {"inputText": text, "embeddingTypes": ["float"]}
                )
                response = await client.invoke_model(
                    modelId=BEDROCK_EMBED_MODEL,
                    body=body,
                    accept="application/json",
                    contentType="application/json",
                )
                response_body = await response["body"].json()
                embeddings.append(response_body["embedding"])
            return np.array(embeddings)

    # ── OpenSearch helper session for verification ───────────────────────
    conn = aiohttp.TCPConnector(ssl=False)
    auth = aiohttp.BasicAuth(*OS_AUTH)
    os_session = aiohttp.ClientSession(auth=auth, connector=conn)

    try:
        await cleanup_indices(os_session)

        # ── Create MiniRAG instance ──────────────────────────────────────
        print("\n=== Creating MiniRAG with Bedrock + OpenSearch ===")
        rag = MiniRAG(
            working_dir=WORKING_DIR,
            llm_model_func=llm_func,
            llm_model_name=BEDROCK_LLM_MODEL,
            llm_model_max_token_size=4096,
            embedding_func=EmbeddingFunc(
                embedding_dim=TITAN_V2_DIM,
                max_token_size=8192,
                func=embed_func,
            ),
            embedding_batch_num=20,
            # OpenSearch storage backends
            kv_storage="OpenSearchKVStorage",
            vector_storage="OpenSearchVectorStorage",
            graph_storage="OpenSearchGraphStorage",
            doc_status_storage="OpenSearchDocStatusStorage",
            # Cosine threshold for vector search
            vector_db_storage_cls_kwargs={
                "cosine_better_than_threshold": 0.2,
            },
            enable_llm_cache=False,
        )

        # ── Phase 1: Insert document ─────────────────────────────────────
        print("\n=== Phase 1: Inserting document ===")
        await rag.ainsert(SAMPLE_DOC)
        print("  Document inserted successfully.")

        # ── Phase 2: Check indices ───────────────────────────────────────
        ok = await check_indices(os_session)
        if not ok:
            sys.exit(1)

        # ── Phase 3: Query ───────────────────────────────────────────────
        print("\n=== Phase 3: Running queries ===")
        for q in TEST_QUERIES:
            print(f"\n  Q: {q}")
            try:
                answer = await rag.aquery(q, param=QueryParam(mode="mini"))
                answer_clean = answer.replace("\n", " ").strip()
                print(f"  A: {answer_clean[:300]}")
            except Exception as e:
                print(f"  ERROR: {type(e).__name__}: {e}")

        # ── Phase 4: Verify graph ────────────────────────────────────────
        await check_graph(os_session)

    finally:
        await os_session.close()

    print("\n=== Test complete ===")


if __name__ == "__main__":
    asyncio.run(main())
