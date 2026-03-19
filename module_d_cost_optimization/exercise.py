"""
Module D Exercise: Production-Grade Cost Optimization
-------------------------------------------------------
Build a production-grade cost measurement and optimization pipeline
for the FinTech multi-agent support system.

Segments covered:
  14. Token counting & cost logging (enhanced)
  15. Before/after cost comparison (production-grade)

Production features you will build:
  - Structured JSON logging with trace IDs
  - Per-intent cost and latency breakdown
  - Semantic caching with hit-rate measurement
  - Cost threshold alerting and budget tracking
  - Audit logging (JSONL) for fintech compliance
  - Quality regression testing after optimization
  - Reranking for improved RAG retrieval
"""

import os
import sys
import json
import time
import uuid
import logging
import math
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()

os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")

sys.path.insert(0, str(Path(__file__).parent.parent / "project"))
from fintech_support_agent import build_support_agent, ask

# ---------------------------------------------------------------------------
# TODO 1: Import tiktoken, get_openai_callback, and OpenAIEmbeddings
# ---------------------------------------------------------------------------
# import tiktoken
# from langchain_community.callbacks.manager import get_openai_callback
# from langchain_openai import OpenAIEmbeddings


# Test queries covering all agent paths
TEST_QUERIES = [
    "What is the overdraft fee?",
    "What credit score do I need for a personal loan?",
    "How much does a domestic wire transfer cost?",
    "How long do I have to report unauthorized transactions?",
    "What is the monthly fee for a Premium Checking account?",
    "What is the balance on ACC-12345?",
    "Show me recent transactions for ACC-67890.",
    "This is terrible service! I want to speak to a manager!",
]

# Quality regression: expected terms in responses for key queries
QUALITY_CHECKS = {
    "What is the overdraft fee?": ["overdraft", "fee"],
    "What credit score do I need for a personal loan?": ["credit", "loan"],
    "What is the balance on ACC-12345?": ["balance", "12450", "12,450"],
    "This is terrible service! I want to speak to a manager!": [
        "support", "specialist", "follow",
    ],
}


# ===================================================================
# PROVIDED: Structured JSON Logging
# ===================================================================
class JSONFormatter(logging.Formatter):
    """JSON log formatter for structured observability pipelines."""
    def format(self, record):
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "extra"):
            entry.update(record.extra)
        return json.dumps(entry)


app_logger = logging.getLogger("fintech.cost")
app_logger.setLevel(logging.INFO)
app_logger.propagate = False
_console = logging.StreamHandler()
_console.setFormatter(JSONFormatter())
app_logger.addHandler(_console)

AUDIT_LOG_PATH = Path(__file__).parent / "audit_log.jsonl"
audit_logger = logging.getLogger("fintech.audit")
audit_logger.setLevel(logging.INFO)
audit_logger.propagate = False
_file = logging.FileHandler(AUDIT_LOG_PATH, mode="w", encoding="utf-8")
_file.setFormatter(JSONFormatter())
audit_logger.addHandler(_file)


# ===================================================================
# PROVIDED: Semantic Cache
# ===================================================================
class SemanticCache:
    """
    In-memory semantic cache using embedding cosine similarity.
    Production replacement: Redis + vector index, or GPTCache.
    """

    def __init__(self, embedding_model, threshold=0.95):
        self.embeddings = embedding_model
        self.threshold = threshold
        self._store: list[tuple[list[float], str, dict]] = []
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _cosine_sim(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        return dot / (na * nb) if na and nb else 0.0

    def get(self, query: str) -> dict | None:
        if not self._store:
            self.misses += 1
            return None
        qvec = self.embeddings.embed_query(query)
        for cvec, _cq, cresult in self._store:
            if self._cosine_sim(qvec, cvec) >= self.threshold:
                self.hits += 1
                return cresult
        self.misses += 1
        return None

    def put(self, query: str, result: dict):
        vec = self.embeddings.embed_query(query)
        self._store.append((vec, query, result))

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


# ===================================================================
# PROVIDED: Cost Tracker with Alerting
# ===================================================================
class CostTracker:
    """Tracks token usage, cost, and latency per query and per intent."""

    def __init__(self, daily_budget: float = 1.0, per_query_alert: float = 0.01):
        self.daily_budget = daily_budget
        self.per_query_alert = per_query_alert
        self.records: list[dict] = []
        self.alerts: list[str] = []

    def record(self, *, trace_id: str, query: str, intent: str,
               prompt_tokens: int, completion_tokens: int,
               cost: float, latency_ms: float, cache_hit: bool = False):
        entry = {
            "trace_id": trace_id, "query": query, "intent": intent,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost": cost, "latency_ms": latency_ms, "cache_hit": cache_hit,
        }
        self.records.append(entry)
        if cost > self.per_query_alert:
            self.alerts.append(
                f"HIGH COST: ${cost:.6f} > ${self.per_query_alert} "
                f"(trace={trace_id}, intent={intent})"
            )
        cumulative = sum(r["cost"] for r in self.records)
        if cumulative > self.daily_budget * 0.8:
            self.alerts.append(
                f"BUDGET WARNING: ${cumulative:.4f} > 80% of "
                f"${self.daily_budget} daily budget"
            )

    @property
    def total_cost(self):
        return sum(r["cost"] for r in self.records)

    @property
    def avg_cost(self):
        return self.total_cost / len(self.records) if self.records else 0

    @property
    def avg_prompt(self):
        return (sum(r["prompt_tokens"] for r in self.records)
                / len(self.records)) if self.records else 0

    @property
    def avg_completion(self):
        return (sum(r["completion_tokens"] for r in self.records)
                / len(self.records)) if self.records else 0

    @property
    def avg_latency(self):
        return (sum(r["latency_ms"] for r in self.records)
                / len(self.records)) if self.records else 0

    def intent_summary(self) -> dict:
        by_intent: dict[str, list[dict]] = defaultdict(list)
        for r in self.records:
            by_intent[r["intent"]].append(r)
        summary = {}
        for intent, recs in by_intent.items():
            n = len(recs)
            summary[intent] = {
                "count": n,
                "avg_cost": sum(r["cost"] for r in recs) / n,
                "avg_prompt": sum(r["prompt_tokens"] for r in recs) / n,
                "avg_completion": sum(r["completion_tokens"] for r in recs) / n,
                "avg_latency_ms": sum(r["latency_ms"] for r in recs) / n,
            }
        return summary


# ===================================================================
# SEGMENT 14: TOKEN COUNTING
# ===================================================================
print("=" * 70)
print("SEGMENT 14: TOKEN COUNTING")
print("=" * 70)

supervisor_prompt = (
    "Classify the customer query into exactly one category:\n"
    "- \"policy\" — general questions about account fees, loans, transfers, "
    "fraud policies, or banking products\n"
    "- \"account_status\" — requests to check balance, view transactions, "
    "or look up a SPECIFIC account (usually contains an account number like ACC-XXXXX)\n"
    "- \"escalation\" — complaints, frustration, requests for a manager, "
    "fraud reports, or complex issues needing human attention\n\n"
    "Respond with ONLY the category name."
)

# ---------------------------------------------------------------------------
# TODO 2: Count tokens in supervisor_prompt with tiktoken
#
# Use tiktoken.encoding_for_model("gpt-4o-mini") to get the encoder.
# Count tokens in supervisor_prompt and print the result.
# Print the hidden cost: tokens * 1000 queries/day.
#
# NOTE: LangSmith also captures per-run token counts in every trace.
# Open your LangSmith dashboard to compare — you'll see tokens broken
# down by individual LLM calls (supervisor vs agent). This module adds
# alerting, caching, and audit logs on top of that foundation.
# ---------------------------------------------------------------------------
# YOUR CODE HERE
print("Complete TODO 2 to count tokens.\n")


# ===================================================================
# SEGMENT 15: BEFORE / AFTER COMPARISON
# ===================================================================

# ---------------------------------------------------------------------------
# TODO 3: Build the BASELINE pipeline
#
# Use build_support_agent() with:
#   - collection_name="exercise_baseline"
#   - chunk_size=1000, chunk_overlap=100
#   - top_k=5
# ---------------------------------------------------------------------------
baseline_agent = None  # Replace with build_support_agent(...)

# YOUR CODE HERE


# ---------------------------------------------------------------------------
# TODO 4: Implement the measurement loop for BASELINE
#
# For each query in TEST_QUERIES:
#   1. Generate a trace_id: uuid.uuid4().hex[:12]
#   2. Measure latency with time.perf_counter()
#   3. Use get_openai_callback() to capture tokens and cost
#   4. Call ask(baseline_app, query)
#   5. Use tracker.record(...) to store all metrics
#   6. Log to audit_logger with trace_id, query, intent, tokens, cost
#   7. Print per-query: intent, prompt tokens, completion, cost, latency
#
# Use a CostTracker instance with daily_budget=1.0, per_query_alert=0.005
#
# Store results as list of (query, result_dict) for quality checks later.
# ---------------------------------------------------------------------------
baseline_tracker = CostTracker(daily_budget=1.0, per_query_alert=0.005)
baseline_results = []

if baseline_agent is not None:
    baseline_app = baseline_agent["app"]
    print(f"\n{'=' * 70}")
    print(f"MEASURING: BEFORE: Baseline (chunk=1000, k=5)")
    print(f"{'=' * 70}")

    # YOUR MEASUREMENT LOOP HERE

else:
    print("Complete TODO 3 first.")


# ---------------------------------------------------------------------------
# TODO 5: Build the OPTIMIZED pipeline with reranking
#
# Use build_support_agent() with:
#   - collection_name="exercise_optimized"
#   - chunk_size=400, chunk_overlap=50
#   - top_k=3
#   - enable_reranking=True, rerank_fetch_k=6
# ---------------------------------------------------------------------------
optimized_agent = None  # Replace with build_support_agent(...)

# YOUR CODE HERE


# ---------------------------------------------------------------------------
# TODO 6: Measure the OPTIMIZED pipeline
#
# Same measurement loop as TODO 4, but with the optimized agent.
# Use a fresh CostTracker instance.
# Store results as list of (query, result_dict).
# ---------------------------------------------------------------------------
optimized_tracker = CostTracker(daily_budget=1.0, per_query_alert=0.005)
optimized_results = []

if optimized_agent is not None:
    optimized_app = optimized_agent["app"]
    print(f"\n{'=' * 70}")
    print(f"MEASURING: AFTER: Optimized (chunk=400, k=3, reranked)")
    print(f"{'=' * 70}")

    # YOUR MEASUREMENT LOOP HERE

else:
    print("Complete TODO 5 first.")


# ---------------------------------------------------------------------------
# TODO 7: Demonstrate semantic caching
#
# 1. Create an OpenAIEmbeddings model (text-embedding-3-small)
# 2. Create a SemanticCache with the embeddings model (threshold=0.95)
# 3. Pre-populate the cache from optimized_results:
#      for query, result in optimized_results:
#          cache.put(query, result)
# 4. Run the same TEST_QUERIES again with the cache enabled
#    (check cache first, skip LLM on hit, record 0 cost)
# 5. Print cache hit rate
# ---------------------------------------------------------------------------
cached_tracker = CostTracker(daily_budget=1.0, per_query_alert=0.005)
cached_results = []

if optimized_results:
    print("\n\nDemonstrating semantic caching...")
    # YOUR CODE HERE
else:
    print("Complete TODOs 5-6 first for caching demo.")


# ---------------------------------------------------------------------------
# TODO 8: Print the 3-way comparison table
#
# Show: BASELINE | OPTIMIZED | + CACHE | Savings %
# Metrics: avg prompt, avg completion, avg cost, avg latency, total cost
# Also print cache hit rate.
#
# Use safe_pct helper to avoid division by zero:
#   def safe_pct(before_val, after_val):
#       return (before_val - after_val) / before_val * 100 if before_val else 0
# ---------------------------------------------------------------------------
print(f"\n{'=' * 85}")
print("FULL COMPARISON: BASELINE -> OPTIMIZED -> OPTIMIZED + CACHE")
print(f"{'=' * 85}")

# YOUR CODE HERE — build and print comparison table
print("Complete TODOs above, then build comparison table here.")


# ---------------------------------------------------------------------------
# TODO 9: Print per-intent cost breakdown
#
# Use baseline_tracker.intent_summary() to get per-intent stats.
# Print a table with: Intent, Count, Avg Prompt, Avg Completion,
#                      Avg Cost, Avg Latency
#
# This reveals which intent paths are most expensive to optimize.
# ---------------------------------------------------------------------------
print(f"\n{'=' * 85}")
print("PER-INTENT COST BREAKDOWN (Baseline)")
print(f"{'=' * 85}")

# YOUR CODE HERE
print("Complete TODOs above, then print intent breakdown here.")


# ---------------------------------------------------------------------------
# TODO 10: Quality regression check + projected savings
#
# Part A — Quality regression:
#   For each query in QUALITY_CHECKS:
#     - Get baseline and optimized responses
#     - Check if any expected_terms appear in each response (case-insensitive)
#     - Print PASS/FAIL for optimized config
#     - Track overall pass/fail
#
# Part B — Projected savings:
#   At 1,000 queries/day, calculate daily/monthly/annual savings for
#   both optimized and optimized+cache configurations.
#
# Part C — Audit summary:
#   Print the audit log file path and explain its contents/purpose.
# ---------------------------------------------------------------------------
print(f"\n{'=' * 85}")
print("QUALITY REGRESSION CHECK")
print(f"{'=' * 85}")

# YOUR CODE HERE — Part A: quality regression

print(f"\n{'=' * 85}")
print("PROJECTED SAVINGS")
print(f"{'=' * 85}")

# YOUR CODE HERE — Part B: projected savings

print(f"\n{'=' * 85}")
print("AUDIT & COMPLIANCE")
print(f"{'=' * 85}")

# YOUR CODE HERE — Part C: audit summary
print("Complete all TODOs to see full production-grade output.")
