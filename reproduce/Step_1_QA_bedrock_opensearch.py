"""
Step 1: Query MiniRAG using Bedrock + OpenSearch Graph Plugin on the LiHua-World dataset.

Usage:
    python reproduce/Step_1_QA_bedrock_opensearch.py
    python reproduce/Step_1_QA_bedrock_opensearch.py --mode mini
    python reproduce/Step_1_QA_bedrock_opensearch.py --mode naive --max-queries 10

Environment variables: same as Step_0_index_bedrock_opensearch.py
"""

import os
import sys
import csv
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

from minirag import MiniRAG, QueryParam
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
async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--querypath", default="./dataset/LiHua-World/qa/query_set.csv")
    parser.add_argument("--workingdir", default="./minirag_reproduce_workdir")
    parser.add_argument("--outputpath", default="./reproduce/lihua_bedrock_opensearch_results.json")
    parser.add_argument("--mode", default="mini", choices=["mini", "naive", "light"])
    parser.add_argument("--max-queries", type=int, default=0, help="Limit queries (0=all)")
    args = parser.parse_args()

    os.makedirs(args.workingdir, exist_ok=True)
    os.makedirs(os.path.dirname(args.outputpath), exist_ok=True)

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

    # Load questions
    questions, gold_answers = [], []
    with open(args.querypath, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            questions.append(row["Question"])
            gold_answers.append(row["Gold Answer"])

    total = len(questions)
    if args.max_queries > 0:
        questions = questions[:args.max_queries]
        gold_answers = gold_answers[:args.max_queries]
    print(f"Querying {len(questions)} of {total} questions (mode={args.mode})")

    results = []
    errors = []
    t0 = time.time()

    for i, (q, ga) in enumerate(zip(questions, gold_answers)):
        print(f"\n[{i+1}/{len(questions)}] Q: {q[:100]}...")
        try:
            answer = await rag.aquery(q, param=QueryParam(mode=args.mode))
            answer_clean = answer.replace("\n", " ").strip()
            print(f"  A: {answer_clean[:200]}")
            results.append({"question": q, "gold_answer": ga, "answer": answer, "error": None})
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"question": q, "gold_answer": ga, "answer": None, "error": str(e)})
            errors.append({"question": q, "error": str(e)})

    elapsed = time.time() - t0

    # Save results
    with open(args.outputpath, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    error_path = args.outputpath.replace(".json", "_errors.json")
    with open(error_path, "w", encoding="utf-8") as f:
        json.dump(errors, f, ensure_ascii=False, indent=2)

    answered = sum(1 for r in results if r["answer"])
    print(f"\n=== Query complete: {answered}/{len(questions)} answered, {len(errors)} errors in {elapsed:.0f}s ({elapsed/60:.1f}m) ===")
    print(f"Results: {args.outputpath}")
    print(f"Errors:  {error_path}")


if __name__ == "__main__":
    asyncio.run(main())
