"""
Module D Demo: Production-Grade Cost Optimization
---------------------------------------------------
Industry-standard cost measurement, optimization, and monitoring
for the FinTech multi-agent support system.

Segments covered:
  14. Token counting & cost logging (enhanced with structured observability)
  15. Before/after cost comparison (production-grade)

Production features demonstrated:
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

import tiktoken
from langchain_community.callbacks.manager import get_openai_callback
from langchain_openai import OpenAIEmbeddings

load_dotenv()

os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")

sys.path.insert(0, str(Path(__file__).parent.parent / "project"))
from fintech_support_agent import build_support_agent, ask

# ---------------------------------------------------------------------------
# Test queries covering all agent paths
# ---------------------------------------------------------------------------
TEST_QUERIES = [
    # --- Policy queries (most expensive — supervisor + RAG) ---
    "What is the overdraft fee?",
    "What credit score do I need for a personal loan?",
    "How much does a domestic wire transfer cost?",
    "How long do I have to report unauthorized transactions?",
    "What is the monthly fee for a Premium Checking account?",
    # --- Account queries (moderate — supervisor + account lookup) ---
    "What is the balance on ACC-12345?",
    "Show me recent transactions for ACC-67890.",
    # --- Escalation queries (cheapest — supervisor + handoff) ---
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
# PRODUCTION INFRASTRUCTURE
# ===================================================================

# ---------------------------------------------------------------------------
# Structured JSON Logging
# ---------------------------------------------------------------------------
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


# Application logger — structured console output
app_logger = logging.getLogger("fintech.cost")
app_logger.setLevel(logging.INFO)
app_logger.propagate = False
_console = logging.StreamHandler()
_console.setFormatter(JSONFormatter())
app_logger.addHandler(_console)

# Audit logger — compliance JSONL file
AUDIT_LOG_PATH = Path(__file__).parent / "audit_log.jsonl"
audit_logger = logging.getLogger("fintech.audit")
audit_logger.setLevel(logging.INFO)
audit_logger.propagate = False
_file = logging.FileHandler(AUDIT_LOG_PATH, mode="w", encoding="utf-8")
_file.setFormatter(JSONFormatter())
audit_logger.addHandler(_file)


# ---------------------------------------------------------------------------
# Semantic Cache
# ---------------------------------------------------------------------------
class SemanticCache:
    """
    In-memory semantic cache using embedding cosine similarity.

    Production replacement: Redis + vector index, or GPTCache.
    This demonstrates the pattern with the same embedding model
    used for retrieval.
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
        """Return cached result if a semantically similar query exists."""
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


# ---------------------------------------------------------------------------
# Cost Tracker with Per-Intent Breakdown & Alerting
# ---------------------------------------------------------------------------
class CostTracker:
    """
    Tracks token usage, cost, and latency per query and per intent.
    Raises alerts when cost thresholds are exceeded.
    """

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

        # --- Alerting ---
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
        """Per-intent aggregation for cost breakdown dashboards."""
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

encoder = tiktoken.encoding_for_model("gpt-4o-mini")

samples = {
    "Simple query": "What is the overdraft fee?",
    "Complex query": (
        "I've been waiting 3 weeks for my fraud dispute to be resolved "
        "and nobody is helping me!"
    ),
    "Supervisor system prompt": (
        "Classify the customer query into exactly one category:\n"
        "- \"policy\" — general questions about account fees, loans, transfers, "
        "fraud policies, or banking products\n"
        "- \"account_status\" — requests to check balance, view transactions, "
        "or look up a SPECIFIC account (usually contains an account number like ACC-XXXXX)\n"
        "- \"escalation\" — complaints, frustration, requests for a manager, "
        "fraud reports, or complex issues needing human attention\n\n"
        "Respond with ONLY the category name."
    ),
}

for label, text in samples.items():
    count = len(encoder.encode(text))
    print(f"  {label:30s} -> {count:4d} tokens")

sup_tokens = len(encoder.encode(samples["Supervisor system prompt"]))
print(f"\n  Hidden cost: {sup_tokens} tokens x every call "
      f"= {sup_tokens * 1000:,} tokens/day at 1K queries")

print(f"\n  NOTE: LangSmith also captures token counts per LLM call in every trace.")
print(f"  Open your LangSmith dashboard to see prompt/completion tokens")
print(f"  broken down by individual runs (supervisor vs agent).")
print(f"  Local measurement (this module) adds alerting, caching, and audit logs")
print(f"  on top of what LangSmith provides out of the box.")


# ===================================================================
# MEASUREMENT ENGINE
# ===================================================================
def run_measurement(agent_components, config_name, tracker, cache=None):
    """
    Run all test queries with full production instrumentation:
    - Trace ID per query
    - Latency measurement
    - Cost tracking with per-intent breakdown
    - Audit logging (JSONL)
    - Optional semantic cache

    Returns list of (query, result_dict) for quality checks.
    """
    app = agent_components["app"]
    results = []

    print(f"\n{'=' * 70}")
    print(f"MEASURING: {config_name}")
    print(f"{'=' * 70}")

    for i, query in enumerate(TEST_QUERIES, 1):
        trace_id = uuid.uuid4().hex[:12]

        # --- Check cache first ---
        if cache is not None:
            cached_result = cache.get(query)
            if cached_result is not None:
                tracker.record(
                    trace_id=trace_id, query=query,
                    intent=cached_result.get("intent", "?"),
                    prompt_tokens=0, completion_tokens=0,
                    cost=0.0, latency_ms=0.0, cache_hit=True,
                )
                results.append((query, cached_result))
                print(f"  Q{i:02d} [CACHE HIT       ] | "
                      f"Prompt:     0 | Completion:    0 | "
                      f"$0.000000 | 0ms")
                audit_logger.info("query_processed", extra={
                    "extra": {
                        "trace_id": trace_id, "query": query,
                        "cache_hit": True, "config": config_name,
                    }
                })
                continue

        # --- LLM call with cost + latency tracking ---
        start = time.perf_counter()
        with get_openai_callback() as cb:
            result = ask(app, query)
        elapsed_ms = (time.perf_counter() - start) * 1000

        intent = result.get("intent", "?")
        results.append((query, result))

        tracker.record(
            trace_id=trace_id, query=query, intent=intent,
            prompt_tokens=cb.prompt_tokens,
            completion_tokens=cb.completion_tokens,
            cost=cb.total_cost, latency_ms=elapsed_ms,
        )

        # Populate cache for future hits
        if cache is not None:
            cache.put(query, result)

        print(f"  Q{i:02d} [{intent:15s}] | "
              f"Prompt: {cb.prompt_tokens:5d} | "
              f"Completion: {cb.completion_tokens:4d} | "
              f"${cb.total_cost:.6f} | {elapsed_ms:.0f}ms")

        # Structured log
        app_logger.info("query_completed", extra={
            "extra": {
                "trace_id": trace_id, "intent": intent,
                "prompt_tokens": cb.prompt_tokens,
                "completion_tokens": cb.completion_tokens,
                "cost": cb.total_cost,
                "latency_ms": round(elapsed_ms, 1),
                "config": config_name,
            },
        })

        # Audit log (compliance)
        audit_logger.info("query_processed", extra={
            "extra": {
                "trace_id": trace_id, "query": query, "intent": intent,
                "prompt_tokens": cb.prompt_tokens,
                "completion_tokens": cb.completion_tokens,
                "cost": cb.total_cost,
                "latency_ms": round(elapsed_ms, 1),
                "response_length": len(result.get("response", "")),
                "config": config_name, "cache_hit": False,
            },
        })

    # --- Print summary ---
    total_p = sum(r["prompt_tokens"] for r in tracker.records)
    total_c = sum(r["completion_tokens"] for r in tracker.records)
    print(f"\n  TOTALS    | Prompt: {total_p:5d} | "
          f"Completion: {total_c:4d} | ${tracker.total_cost:.6f}")
    print(f"  AVG/QUERY | Prompt: {tracker.avg_prompt:5.0f} | "
          f"Completion: {tracker.avg_completion:4.0f} | "
          f"${tracker.avg_cost:.6f} | {tracker.avg_latency:.0f}ms")

    return results


# ===================================================================
# SEGMENT 15: BEFORE / AFTER COMPARISON
# ===================================================================

# --- BEFORE: Baseline ---
print("\n\nBuilding BASELINE pipeline (chunk=1000, k=5, no reranking)...")
baseline = build_support_agent(
    collection_name="cost_baseline",
    chunk_size=1000, chunk_overlap=100, top_k=5,
)
baseline_tracker = CostTracker(daily_budget=1.0, per_query_alert=0.005)
baseline_results = run_measurement(
    baseline, "BEFORE: Baseline (chunk=1000, k=5)", baseline_tracker,
)

# --- AFTER: Optimized + Reranking ---
print("\n\nBuilding OPTIMIZED pipeline (chunk=400, k=3, reranking enabled)...")
optimized = build_support_agent(
    collection_name="cost_optimized",
    chunk_size=400, chunk_overlap=50, top_k=3,
    enable_reranking=True, rerank_fetch_k=6,
)
optimized_tracker = CostTracker(daily_budget=1.0, per_query_alert=0.005)
optimized_results = run_measurement(
    optimized, "AFTER: Optimized (chunk=400, k=3, reranked)", optimized_tracker,
)

# --- AFTER + CACHE: Pre-populate cache, then rerun ---
print("\n\nDemonstrating semantic caching (pre-populated from optimized run)...")
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
cache = SemanticCache(embeddings_model, threshold=0.95)

# Pre-populate cache from previous optimized results (no extra LLM cost)
for query, result in optimized_results:
    cache.put(query, result)

cached_tracker = CostTracker(daily_budget=1.0, per_query_alert=0.005)
cached_results = run_measurement(
    optimized, "AFTER + CACHE: Optimized with semantic cache",
    cached_tracker, cache=cache,
)


# ===================================================================
# 3-WAY COMPARISON TABLE
# ===================================================================
print(f"\n{'=' * 85}")
print("FULL COMPARISON: BASELINE -> OPTIMIZED -> OPTIMIZED + CACHE")
print(f"{'=' * 85}")

b, o, c = baseline_tracker, optimized_tracker, cached_tracker


def safe_pct(before_val, after_val):
    return (before_val - after_val) / before_val * 100 if before_val else 0


print(f"\n{'Metric':<28} {'BASELINE':>12} {'OPTIMIZED':>12} "
      f"{'+ CACHE':>12} {'Savings':>10}")
print("-" * 85)
print(f"{'Avg prompt tokens':<28} {b.avg_prompt:>12.0f} "
      f"{o.avg_prompt:>12.0f} {c.avg_prompt:>12.0f} "
      f"{safe_pct(b.avg_prompt, o.avg_prompt):>9.1f}%")
print(f"{'Avg completion tokens':<28} {b.avg_completion:>12.0f} "
      f"{o.avg_completion:>12.0f} {c.avg_completion:>12.0f}")
print(f"{'Avg cost per query':<28} ${b.avg_cost:>11.6f} "
      f"${o.avg_cost:>11.6f} ${c.avg_cost:>11.6f} "
      f"{safe_pct(b.avg_cost, o.avg_cost):>9.1f}%")
print(f"{'Avg latency (ms)':<28} {b.avg_latency:>12.0f} "
      f"{o.avg_latency:>12.0f} {c.avg_latency:>12.0f} "
      f"{safe_pct(b.avg_latency, o.avg_latency):>9.1f}%")
print(f"{'Total cost (8 queries)':<28} ${b.total_cost:>11.6f} "
      f"${o.total_cost:>11.6f} ${c.total_cost:>11.6f}")

print(f"\n  Cache stats: {cache.hit_rate:.0%} hit rate "
      f"({cache.hits} hits / {cache.hits + cache.misses} lookups)")
print(f"  -> With warm cache, cost drops to $0 for repeated/similar queries")


# ===================================================================
# PER-INTENT COST BREAKDOWN
# ===================================================================
print(f"\n{'=' * 85}")
print("PER-INTENT COST BREAKDOWN (Baseline)")
print(f"{'=' * 85}")

intent_stats = baseline_tracker.intent_summary()
print(f"\n{'Intent':<18} {'Count':>6} {'Avg Prompt':>12} {'Avg Compl':>10} "
      f"{'Avg Cost':>12} {'Avg Latency':>12}")
print("-" * 75)
for intent, stats in sorted(intent_stats.items()):
    print(f"{intent:<18} {stats['count']:>6} {stats['avg_prompt']:>12.0f} "
          f"{stats['avg_completion']:>10.0f} ${stats['avg_cost']:>11.6f} "
          f"{stats['avg_latency_ms']:>10.0f}ms")

print(f"\n  -> Policy queries cost the most (RAG context tokens).")
print(f"  -> Escalation queries are cheapest (no retrieval, short response).")
print(f"  -> Focus optimization effort on the most expensive intent paths.")


# ===================================================================
# COST ALERTS
# ===================================================================
all_alerts = baseline_tracker.alerts + optimized_tracker.alerts
if all_alerts:
    print(f"\n{'=' * 85}")
    print(f"COST ALERTS ({len(all_alerts)} triggered)")
    print(f"{'=' * 85}")
    for alert in all_alerts:
        print(f"  WARNING: {alert}")
else:
    print(f"\n  No cost alerts triggered (all queries within thresholds).")


# ===================================================================
# QUALITY REGRESSION CHECK
# ===================================================================
print(f"\n{'=' * 85}")
print("QUALITY REGRESSION CHECK")
print(f"{'=' * 85}")

baseline_resp_map = {q: r.get("response", "") for q, r in baseline_results}
optimized_resp_map = {q: r.get("response", "") for q, r in optimized_results}

regression_passed = True
for query, expected_terms in QUALITY_CHECKS.items():
    b_resp = baseline_resp_map.get(query, "")
    o_resp = optimized_resp_map.get(query, "")
    b_ok = any(t.lower() in b_resp.lower() for t in expected_terms)
    o_ok = any(t.lower() in o_resp.lower() for t in expected_terms)

    status = "PASS" if o_ok else "FAIL"
    if not o_ok:
        regression_passed = False

    short_q = query[:50]
    print(f"  [{status}] \"{short_q}\"")
    print(f"         Expected one of {expected_terms} | "
          f"Baseline={'found' if b_ok else 'MISSING'} | "
          f"Optimized={'found' if o_ok else 'MISSING'}")

overall = "ALL CHECKS PASSED" if regression_passed else "REGRESSION DETECTED"
print(f"\n  Overall: {overall}")
print(f"  -> Always verify quality before deploying cost optimizations.")
print(f"  -> For full evaluation, run Module B on the optimized config.")


# ===================================================================
# PROJECTED SAVINGS
# ===================================================================
print(f"\n{'=' * 85}")
print("PROJECTED SAVINGS")
print(f"{'=' * 85}")

qpd = 1000
save_opt = (b.avg_cost - o.avg_cost) * qpd
save_cache = (b.avg_cost - c.avg_cost) * qpd

print(f"\n  At {qpd:,} queries/day:")
print(f"  {'':30s} {'Optimized':>14} {'+ Cache':>14}")
print(f"  {'-' * 60}")
print(f"  {'Daily savings':<30} ${save_opt:>13.4f} ${save_cache:>13.4f}")
print(f"  {'Monthly savings':<30} ${save_opt * 30:>13.2f} "
      f"${save_cache * 30:>13.2f}")
print(f"  {'Annual savings':<30} ${save_opt * 365:>13.2f} "
      f"${save_cache * 365:>13.2f}")


# ===================================================================
# AUDIT LOG SUMMARY
# ===================================================================
print(f"\n{'=' * 85}")
print("AUDIT & COMPLIANCE")
print(f"{'=' * 85}")
print(f"  Audit log written to: {AUDIT_LOG_PATH}")
print(f"  Format: JSON Lines (one structured record per query)")
print(f"  Fields: trace_id, query, intent, tokens, cost, latency, config")
print(f"  Use for: compliance reporting, cost attribution, anomaly detection")
print(f"  Retention: configure per regulatory requirements (fintech: 5-7 years)")


# ===================================================================
# BATCH API DEMO (Pattern 4: 50% discount for non-real-time workloads)
# ===================================================================
print(f"\n{'=' * 85}")
print("BATCH API DEMO (Pattern 4: 50% Discount for Non-Real-Time Workloads)")
print(f"{'=' * 85}")
print("""
OpenAI's Batch API provides 50% cost reduction for workloads that don't
need real-time responses. Results are returned within 24 hours.

  Normal:  $0.15/M input,  $0.60/M output  (GPT-4o-mini)
  Batch:   $0.075/M input, $0.30/M output   -> 50% savings

Use cases for fintech:
  - Running evaluation datasets (Module B) overnight
  - Batch-processing customer feedback or survey responses
  - Generating training/fine-tuning data from policy documents
  - Periodic compliance report summarization
""")

# Demonstrate how to create and submit a batch request.
# This uses the OpenAI client directly (not LangChain) since the Batch API
# is a platform feature for async job submission.

from openai import OpenAI

batch_client = OpenAI()

# Prepare batch requests: each is a chat completion in JSONL format
batch_queries = [
    "What is the overdraft fee?",
    "What credit score do I need for a personal loan?",
    "What is the international wire transfer fee?",
]

batch_requests = []
for i, query in enumerate(batch_queries):
    batch_requests.append({
        "custom_id": f"batch-query-{i}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You are a helpful banking assistant for SecureBank."},
                {"role": "user", "content": query},
            ],
            "max_tokens": 200,
        },
    })

# Write batch input file
batch_input_path = Path(__file__).parent / "batch_input.jsonl"
with open(batch_input_path, "w", encoding="utf-8") as f:
    for req in batch_requests:
        f.write(json.dumps(req) + "\n")

print(f"  Batch input file written: {batch_input_path}")
print(f"  Contains {len(batch_requests)} requests")

# Upload and submit the batch
try:
    batch_file = batch_client.files.create(
        file=open(batch_input_path, "rb"),
        purpose="batch",
    )
    print(f"  Uploaded file ID: {batch_file.id}")

    batch_job = batch_client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": "fintech-cost-demo-batch", "module": "D"},
    )
    print(f"  Batch job ID: {batch_job.id}")
    print(f"  Status: {batch_job.status}")
    print(f"\n  The batch will complete within 24 hours.")
    print(f"  Check status with: client.batches.retrieve('{batch_job.id}')")
    print(f"  Once complete, download results with:")
    print(f"    client.files.content(batch_job.output_file_id)")

    # Cost projection
    normal_cost_per_m_input = 0.15
    batch_cost_per_m_input = 0.075
    est_tokens = len(batch_requests) * 300  # rough estimate per request
    normal_est = est_tokens / 1_000_000 * normal_cost_per_m_input
    batch_est = est_tokens / 1_000_000 * batch_cost_per_m_input
    print(f"\n  Estimated cost comparison for this batch:")
    print(f"    Normal API: ~${normal_est:.6f}")
    print(f"    Batch API:  ~${batch_est:.6f}  (50% savings)")

    # At scale projection
    daily_batch_queries = 10_000
    daily_tokens = daily_batch_queries * 300
    daily_normal = daily_tokens / 1_000_000 * (normal_cost_per_m_input + 0.60)
    daily_batch = daily_tokens / 1_000_000 * (batch_cost_per_m_input + 0.30)
    annual_savings = (daily_normal - daily_batch) * 365
    print(f"\n  At-scale projection ({daily_batch_queries:,} non-real-time queries/day):")
    print(f"    Annual normal API cost:  ${daily_normal * 365:>10.2f}")
    print(f"    Annual batch API cost:   ${daily_batch * 365:>10.2f}")
    print(f"    Annual savings:          ${annual_savings:>10.2f}")

except Exception as e:
    print(f"\n  Batch API submission failed: {e}")
    print(f"  This is expected if your API key doesn't support the Batch API.")
    print(f"  The batch_input.jsonl file was still created for reference.")
    print(f"\n  To submit manually:")
    print(f"    1. Upload: client.files.create(file=open('batch_input.jsonl', 'rb'), purpose='batch')")
    print(f"    2. Submit: client.batches.create(input_file_id=file.id, endpoint='/v1/chat/completions', completion_window='24h')")
    print(f"    3. Check:  client.batches.retrieve(batch_id)")


# ===================================================================
# KEY TAKEAWAYS
# ===================================================================
print(f"\n{'=' * 85}")
print("KEY TAKEAWAYS")
print(f"{'=' * 85}")
print("""
Production cost optimization requires FIVE layers:

1. MEASUREMENT  - Structured per-intent cost + latency tracking with trace IDs
                  (not just total cost across all queries)

2. OPTIMIZATION - Smaller chunks (1000->400), fewer retrievals (k=5->3),
                  reranking (fetch 6, keep best 3), model routing

3. CACHING      - Semantic cache eliminates repeated LLM calls entirely
                  (100% cost reduction on cache hits)

4. MONITORING   - Cost caps, per-query alerts, budget tracking
                  (prevent runaway costs before they happen)

5. COMPLIANCE   - Audit logging with trace IDs for every query
                  (required in fintech for regulatory review)

CRITICAL: Run quality regression checks after EVERY optimization change.
Cost savings are worthless if answer quality degrades.

What we optimized:
  - chunk_size: 1000 -> 400 (less context per chunk)
  - chunk_overlap: 100 -> 50 (less duplicated text)
  - top_k: 5 -> 3 (fewer retrieved documents)
  - Reranking: fetch 6, score by relevance, keep best 3

What else to consider in production:
  - Prompt caching (Anthropic/OpenAI) - reuse cached system prompts
  - Model routing - cheap model for simple queries, powerful for complex
  - Batch API - 50% discount for non-real-time workloads
  - Persistent vector stores (Pinecone, Weaviate) instead of in-memory Chroma

LangSmith connection (Module A):
  Every query in this demo is also traced in LangSmith with per-run token counts.
  Open your LangSmith dashboard to see the same data broken down by individual
  LLM calls (supervisor vs agent). This module adds structured alerting, caching,
  audit logging, and comparison analysis on top of that foundation.
""")
