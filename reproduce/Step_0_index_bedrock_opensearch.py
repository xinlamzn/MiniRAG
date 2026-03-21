"""
Step 0: Index the LiHua-World dataset into MiniRAG using Bedrock + OpenSearch Graph Plugin.

Usage:
    python reproduce/Step_0_index_bedrock_opensearch.py
    python reproduce/Step_0_index_bedrock_opensearch.py --datapath ./dataset/LiHua-World/data/LiHua-World/

Environment variables:
    AWS_DEFAULT_REGION          (default: us-west-2)
    OPENSEARCH_URI              (default: https://localhost:9200)
    OPENSEARCH_USERNAME         (default: admin)
    OPENSEARCH_PASSWORD         (default: Admin@1234)
    OPENSEARCH_VERIFY_CERTS     (default: false)
    OPENSEARCH_DATABASE         (default: minirag_reproduce)
    BEDROCK_LLM_MODEL           (default: us.anthropic.claude-sonnet-4-20250514-v1:0)
    BEDROCK_EMBED_MODEL         (default: amazon.titan-embed-text-v2:0)
"""

import os
import sys
import json
import time
import asyncio
import argparse
import copy

import aioboto3
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Config ───────────────────────────────────────────────────────────────
os.environ.setdefault("AWS_DEFAULT_REGION", "us-west-2")
os.environ.setdefault("OPENSEARCH_URI", "https://localhost:9200")
os.environ.setdefault("OPENSEARCH_USERNAME", "admin")
os.environ.setdefault("OPENSEARCH_PASSWORD", "Admin@1234")
os.environ.setdefault("OPENSEARCH_VERIFY_CERTS", "false")
os.environ.setdefault("OPENSEARCH_DATABASE", "minirag_reproduce")

BEDROCK_LLM_MODEL = os.environ.get("BEDROCK_LLM_MODEL", "us.anthropic.claude-sonnet-4-20250514-v1:0")
BEDROCK_EMBED_MODEL = os.environ.get("BEDROCK_EMBED_MODEL", "amazon.titan-embed-text-v2:0")
TITAN_V2_DIM = 1024

from minirag import MiniRAG
from minirag.utils import EmbeddingFunc, locate_json_string_body_from_string


# ── Bedrock LLM ──────────────────────────────────────────────────────────
async def llm_func(prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs):
    kwargs.pop("hashing_kv", None)
    kwargs.pop("keyword_extraction", None)

    messages = []
    for msg in history_messages:
        m = copy.copy(msg)
        m["content"] = [{"text": m["content"]}]
        messages.append(m)
    messages.append({"role": "user", "content": [{"text": prompt}]})

    args = {"modelId": BEDROCK_LLM_MODEL, "messages": messages}
    if system_prompt:
        args["system"] = [{"text": system_prompt}]

    inference_params_map = {"max_tokens": "maxTokens", "top_p": "topP", "stop_sequences": "stopSequences"}
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


# ── Bedrock Embedding ────────────────────────────────────────────────────
async def embed_func(texts):
    session = aioboto3.Session()
    async with session.client("bedrock-runtime") as client:
        embeddings = []
        for text in texts:
            body = json.dumps({"inputText": text, "embeddingTypes": ["float"]})
            response = await client.invoke_model(
                modelId=BEDROCK_EMBED_MODEL, body=body,
                accept="application/json", contentType="application/json",
            )
            response_body = await response["body"].json()
            embeddings.append(response_body["embedding"])
        return np.array(embeddings)


# ── Main ─────────────────────────────────────────────────────────────────
def find_txt_files(root_path):
    txt_files = []
    for root, dirs, files in os.walk(root_path):
        for file in sorted(files):
            if file.endswith(".txt"):
                txt_files.append(os.path.join(root, file))
    return txt_files


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", default="./dataset/LiHua-World/data/LiHua-World/")
    parser.add_argument("--workingdir", default="./minirag_reproduce_workdir")
    args = parser.parse_args()

    os.makedirs(args.workingdir, exist_ok=True)

    rag = MiniRAG(
        working_dir=args.workingdir,
        llm_model_func=llm_func,
        llm_model_name=BEDROCK_LLM_MODEL,
        llm_model_max_token_size=4096,
        embedding_func=EmbeddingFunc(embedding_dim=TITAN_V2_DIM, max_token_size=8192, func=embed_func),
        embedding_batch_num=20,
        kv_storage="OpenSearchKVStorage",
        vector_storage="OpenSearchVectorStorage",
        graph_storage="OpenSearchGraphStorage",
        doc_status_storage="OpenSearchDocStatusStorage",
        vector_db_storage_cls_kwargs={"cosine_better_than_threshold": 0.2},
        enable_llm_cache=False,
    )

    txt_files = find_txt_files(args.datapath)
    print(f"Found {len(txt_files)} text files in {args.datapath}")

    t0 = time.time()
    for i, filepath in enumerate(txt_files):
        with open(filepath) as f:
            content = f.read()
        print(f"\n[{i+1}/{len(txt_files)}] Inserting {os.path.basename(filepath)} ({len(content)} chars)")
        try:
            await rag.ainsert(content)
        except Exception as e:
            print(f"  ERROR: {e}")

    elapsed = time.time() - t0
    print(f"\n=== Indexing complete: {len(txt_files)} files in {elapsed:.0f}s ({elapsed/60:.1f}m) ===")


if __name__ == "__main__":
    asyncio.run(main())
