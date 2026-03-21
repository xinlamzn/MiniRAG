"""
MiniRAG Reproduce: Index + Query using Bedrock + OpenSearch Graph Plugin.

Usage:
    python reproduce/run_bedrock_opensearch.py                    # Both steps
    python reproduce/run_bedrock_opensearch.py --skip-index       # Query only
    python reproduce/run_bedrock_opensearch.py --skip-query       # Index only
    python reproduce/run_bedrock_opensearch.py --max-queries 10   # Limit queries

Environment variables:
    OPENSEARCH_URI, OPENSEARCH_USERNAME, OPENSEARCH_PASSWORD, OPENSEARCH_DATABASE
    BEDROCK_LLM_MODEL, BEDROCK_EMBED_MODEL, AWS_DEFAULT_REGION
"""

import os, sys, csv, json, time, asyncio, argparse, copy
import aioboto3, numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("AWS_DEFAULT_REGION", "us-west-2")
os.environ.setdefault("OPENSEARCH_URI", "https://localhost:9200")
os.environ.setdefault("OPENSEARCH_USERNAME", "admin")
os.environ.setdefault("OPENSEARCH_PASSWORD", "Admin@1234")
os.environ.setdefault("OPENSEARCH_VERIFY_CERTS", "false")
os.environ.setdefault("OPENSEARCH_DATABASE", "minirag_reproduce")

BEDROCK_LLM_MODEL = os.environ.get("BEDROCK_LLM_MODEL", "us.anthropic.claude-sonnet-4-20250514-v1:0")
BEDROCK_EMBED_MODEL = os.environ.get("BEDROCK_EMBED_MODEL", "amazon.titan-embed-text-v2:0")

from minirag import MiniRAG, QueryParam
from minirag.utils import EmbeddingFunc, locate_json_string_body_from_string


async def llm_func(prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs):
    kwargs.pop("hashing_kv", None); kwargs.pop("keyword_extraction", None)
    messages = [{"role": m["role"], "content": [{"text": m["content"]}]} for m in history_messages]
    messages.append({"role": "user", "content": [{"text": prompt}]})
    args = {"modelId": BEDROCK_LLM_MODEL, "messages": messages}
    if system_prompt:
        args["system"] = [{"text": system_prompt}]
    param_map = {"max_tokens": "maxTokens", "top_p": "topP", "stop_sequences": "stopSequences"}
    inf_keys = set(kwargs) & {"max_tokens", "temperature", "top_p", "stop_sequences"}
    if inf_keys:
        args["inferenceConfig"] = {param_map.get(p, p): kwargs.pop(p) for p in inf_keys}
    async with aioboto3.Session().client("bedrock-runtime") as client:
        resp = await client.converse(**args)
    result = resp["output"]["message"]["content"][0]["text"]
    return locate_json_string_body_from_string(result) if keyword_extraction else result


async def embed_func(texts):
    async with aioboto3.Session().client("bedrock-runtime") as client:
        embeddings = []
        for text in texts:
            resp = await client.invoke_model(
                modelId=BEDROCK_EMBED_MODEL,
                body=json.dumps({"inputText": text, "embeddingTypes": ["float"]}),
                accept="application/json", contentType="application/json",
            )
            embeddings.append((await resp["body"].json())["embedding"])
        return np.array(embeddings)


def make_rag(workingdir):
    os.makedirs(workingdir, exist_ok=True)
    return MiniRAG(
        working_dir=workingdir,
        llm_model_func=llm_func,
        llm_model_name=BEDROCK_LLM_MODEL,
        llm_model_max_token_size=4096,
        embedding_func=EmbeddingFunc(embedding_dim=1024, max_token_size=8192, func=embed_func),
        embedding_batch_num=20,
        kv_storage="OpenSearchKVStorage",
        vector_storage="OpenSearchVectorStorage",
        graph_storage="OpenSearchGraphStorage",
        doc_status_storage="OpenSearchDocStatusStorage",
        vector_db_storage_cls_kwargs={"cosine_better_than_threshold": 0.2},
        enable_llm_cache=False,
    )


async def run_index(rag, datapath):
    txt_files = sorted(
        os.path.join(r, f)
        for r, _, files in os.walk(datapath)
        for f in files if f.endswith(".txt")
    )
    print(f"\n{'='*60}\nStep 0: Indexing {len(txt_files)} files from {datapath}\n{'='*60}")
    t0 = time.time()
    for i, fp in enumerate(txt_files):
        with open(fp) as f:
            content = f.read()
        print(f"[{i+1}/{len(txt_files)}] {os.path.basename(fp)} ({len(content)} chars)")
        try:
            await rag.ainsert(content)
        except Exception as e:
            print(f"  ERROR: {e}")
    elapsed = time.time() - t0
    print(f"\nIndexing done: {len(txt_files)} files in {elapsed:.0f}s ({elapsed/60:.1f}m)")
    return elapsed


async def run_query(rag, querypath, outputpath, mode, max_queries):
    questions, gold_answers = [], []
    with open(querypath, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            questions.append(row["Question"])
            gold_answers.append(row["Gold Answer"])
    total = len(questions)
    if max_queries > 0:
        questions, gold_answers = questions[:max_queries], gold_answers[:max_queries]

    print(f"\n{'='*60}\nStep 1: Querying {len(questions)}/{total} questions (mode={mode})\n{'='*60}")
    os.makedirs(os.path.dirname(outputpath), exist_ok=True)
    results, errors = [], []
    t0 = time.time()
    for i, (q, ga) in enumerate(zip(questions, gold_answers)):
        print(f"[{i+1}/{len(questions)}] {q[:100]}...")
        try:
            answer = await rag.aquery(q, param=QueryParam(mode=mode))
            print(f"  A: {answer.replace(chr(10), ' ')[:200]}")
            results.append({"question": q, "gold_answer": ga, "answer": answer})
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"question": q, "gold_answer": ga, "answer": None, "error": str(e)})
            errors.append({"question": q, "error": str(e)})

    elapsed = time.time() - t0
    with open(outputpath, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    error_path = outputpath.replace(".json", "_errors.json")
    with open(error_path, "w", encoding="utf-8") as f:
        json.dump(errors, f, ensure_ascii=False, indent=2)

    answered = sum(1 for r in results if r.get("answer"))
    print(f"\nQuery done: {answered}/{len(questions)} answered, {len(errors)} errors in {elapsed:.0f}s ({elapsed/60:.1f}m)")
    print(f"Results: {outputpath}")
    return elapsed


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", default="./dataset/LiHua-World/data/LiHua-World/")
    parser.add_argument("--querypath", default="./dataset/LiHua-World/qa/query_set.csv")
    parser.add_argument("--workingdir", default="./minirag_reproduce_workdir")
    parser.add_argument("--outputpath", default="./reproduce/lihua_bedrock_opensearch_results.json")
    parser.add_argument("--mode", default="mini", choices=["mini", "naive", "light"])
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--skip-index", action="store_true")
    parser.add_argument("--skip-query", action="store_true")
    args = parser.parse_args()

    rag = make_rag(args.workingdir)
    index_time = query_time = 0

    if not args.skip_index:
        index_time = await run_index(rag, args.datapath)
    if not args.skip_query:
        query_time = await run_query(rag, args.querypath, args.outputpath, args.mode, args.max_queries)

    print(f"\n{'='*60}\nTotal: index={index_time:.0f}s query={query_time:.0f}s\n{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
