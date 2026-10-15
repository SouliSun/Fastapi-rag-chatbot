"""
Evaluate ingest.py's RAG with RAGAS.
Put this next to ingest.py / clean_data.py and run:  python evaluate1.py
pip install ragas datasets python-dotenv langchain-openai langchain-huggingface
"""
import json, os, time
from collections import defaultdict

from dotenv import load_dotenv
from datasets import load_dataset
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from ragas import EvaluationDataset, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    NonLLMContextPrecisionWithReference,
    NonLLMContextRecall,
    FactualCorrectness,
)
from ragas.run_config import RunConfig

from ingest import generate_answer, GROQ_API_KEY, GROQ_BASE_URL, MODEL_NAME  # importing runs indexing

K = 3
JUDGE_MODEL = "qwen/qwen3.8-27b"      # Fresh daily token quota on Groq
CACHE = "rag_outputs.json"           # generated answers are cached so reruns skip the LLM calls
load_dotenv()

norm = lambda s: " ".join(s.lower().split())
fmt = lambda items: [f"Q: {r['question']}\nA: {r['answer']}" for r in items]  # same format the LLM sees

# ---- 1) Run the chatbot on every eval question (cached) ----
items = json.load(open("ragas_eval_set.json", encoding="utf-8"))
cache = json.load(open(CACHE, encoding="utf-8")) if os.path.exists(CACHE) else {}

rows = []
for it in items:
    key = str(it["id"])
    if key not in cache:
        out = generate_answer(it["user_input"], k=K)
        cache[key] = {
            "response": out["answer"],
            "fallback": out["fallback"],
            # what the LLM saw; if the bot fell back, keep the raw top-k so metrics have something to score
            "retrieved_contexts": fmt(out["contexts"] if out["contexts"] else out["results"]),
            "retrieved_questions": [r["question"] for r in out["results"]],  # pre-threshold top-k
        }
        json.dump(cache, open(CACHE, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
        time.sleep(1)  # be gentle with Groq rate limits
    rows.append({**it, **cache[key]})

# ---- 2) Deterministic checks (no judge LLM needed) ----
ans = [r for r in rows if r["answerable"]]
una = [r for r in rows if not r["answerable"]]

hits, rr = [], []
for r in ans:
    srcs = {norm(q) for q in r["source_questions"]}
    ranks = [i for i, q in enumerate(r["retrieved_questions"], 1) if norm(q) in srcs]
    hits.append(bool(ranks)); rr.append(1 / ranks[0] if ranks else 0)
print(f"\nHit@{K} (a source FAQ is in top-{K}): {sum(hits)/len(hits):.2%}   MRR: {sum(rr)/len(rr):.3f}")
print(f"False refusals on answerable questions: {sum(r['fallback'] for r in ans)}/{len(ans)}")
print(f"Fallback triggered on unanswerable/out-of-scope: {sum(r['fallback'] for r in una)}/{len(una)}")

# ---- 3) RAGAS on answerable questions ----
llm = LangchainLLMWrapper(ChatOpenAI(model=JUDGE_MODEL, api_key=GROQ_API_KEY,
                                     base_url=GROQ_BASE_URL, temperature=0))
emb = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=MODEL_NAME))  # same embedder as your RAG
ds = EvaluationDataset.from_list([{
    "user_input": r["user_input"],
    "retrieved_contexts": r["retrieved_contexts"],
    "response": r["response"],
    "reference": r["reference"],
    "reference_contexts": [r["reference"]],
} for r in ans])

res = evaluate(
    ds,
    metrics=[
        Faithfulness(),
        ResponseRelevancy(strictness=1),
        NonLLMContextPrecisionWithReference(),
        NonLLMContextRecall(),
        FactualCorrectness(),
    ],
    llm=llm,
    embeddings=emb,
    run_config=RunConfig(max_workers=2, timeout=180, max_retries=10, max_wait=90),
)
df = res.to_pandas()
df["query_type"] = [r["query_type"] for r in ans]
df["id"] = [r["id"] for r in ans]
df.to_csv("ragas_results.csv", index=False)
print("\nMean scores by query type:")
print(df.groupby("query_type").mean(numeric_only=True).round(3))
print("\nOverall:\n", df.mean(numeric_only=True).round(3))

# ---- 4) Unanswerable: read these by hand ----
print("\n=== Unanswerable / out-of-scope: check for invented details ===")
for r in una:
    print(f"\n[{r['id']}] {r['user_input']}\n  fallback={r['fallback']}\n  bot: {r['response']}")