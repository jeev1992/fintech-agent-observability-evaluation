"""
Module D Demo: Cost Optimization
----------------------------------
Measure token usage and cost for two configurations of the FinTech
multi-agent system, then compare side by side.

Segments:
  1. Token counting with tiktoken  (local awareness)
  2. Before / After cost comparison (get_openai_callback)

Everything else - per-run token breakdowns, trace trees, latency,
per-intent analysis - lives in your LangSmith dashboard.
"""

import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

import tiktoken
from langchain_community.callbacks.manager import get_openai_callback

load_dotenv()

os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")

sys.path.insert(0, str(Path(__file__).parent.parent / "project"))
from fintech_support_agent import build_support_agent, ask


# Test queries - one per intent path so LangSmith traces show all routes.
TEST_QUERIES = [
    # Policy (RAG - most expensive: supervisor + retriever + generation)
    "What is the overdraft fee?",
    "What credit score do I need for a personal loan?",
    "How much does a domestic wire transfer cost?",
    "How long do I have to report unauthorized transactions?",
    "What is the monthly fee for a Premium Checking account?",
    # Account status (moderate: supervisor + account lookup)
    "What is the balance on ACC-12345?",
    "Show me recent transactions for ACC-67890.",
    # Escalation (cheapest: supervisor + short message)
    "This is terrible service! I want to speak to a manager!",
]

# Quick quality smoke-test: at least one term should appear in the response.
QUALITY_CHECKS = {
    "What is the overdraft fee?": ["overdraft", "fee"],
    "What credit score do I need for a personal loan?": ["credit", "loan"],
    "What is the balance on ACC-12345?": ["balance", "12450", "12,450"],
}


# ===================================================================
# SEGMENT 1: TOKEN COUNTING
# ===================================================================
print("=" * 60)
print("SEGMENT 1: TOKEN COUNTING WITH TIKTOKEN")
print("=" * 60)

encoder = tiktoken.encoding_for_model("gpt-4o-mini")

samples = {
    "Simple query": "What is the overdraft fee?",
    "Complex query": (
        "I've been waiting 3 weeks for my fraud dispute to be "
        "resolved and nobody is helping me!"
    ),
    "Supervisor system prompt": (
        "Classify the customer query into exactly one category:\n"
        "- \"policy\" - general questions about account fees, loans, transfers, "
        "fraud policies, or banking products\n"
        "- \"account_status\" - requests to check balance, view transactions, "
        "or look up a SPECIFIC account (usually contains an account number "
        "like ACC-XXXXX)\n"
        "- \"escalation\" - complaints, frustration, requests for a manager, "
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
print(f"\n  TIP: Open LangSmith to see exact per-call token counts in every trace.")


# ===================================================================
# Helper: run queries and collect cost via get_openai_callback
# ===================================================================
def measure(agent_components, label):
    """Run TEST_QUERIES, print per-query cost, return totals."""
    app = agent_components["app"]
    results = []

    print(f"\n{'=' * 60}")
    print(f"MEASURING: {label}")
    print(f"{'=' * 60}")

    total_prompt = 0
    total_completion = 0
    total_cost = 0.0

    for i, query in enumerate(TEST_QUERIES, 1):
        start = time.perf_counter()
        with get_openai_callback() as cb:
            result = ask(app, query)
        elapsed_ms = (time.perf_counter() - start) * 1000

        intent = result.get("intent", "?")
        results.append((query, result))
        total_prompt += cb.prompt_tokens
        total_completion += cb.completion_tokens
        total_cost += cb.total_cost

        print(f"  Q{i:02d} [{intent:15s}] | "
              f"Prompt: {cb.prompt_tokens:5d} | "
              f"Completion: {cb.completion_tokens:4d} | "
              f"${cb.total_cost:.6f} | {elapsed_ms:.0f}ms")

    n = len(TEST_QUERIES)
    print(f"\n  TOTALS    | Prompt: {total_prompt:5d} | "
          f"Completion: {total_completion:4d} | ${total_cost:.6f}")
    print(f"  AVG/QUERY | Prompt: {total_prompt / n:5.0f} | "
          f"Completion: {total_completion / n:4.0f} | "
          f"${total_cost / n:.6f}")

    return total_cost, total_prompt, total_completion, results


# ===================================================================
# SEGMENT 2: BEFORE / AFTER COMPARISON
# ===================================================================
print(f"\n\n{'=' * 60}")
print("SEGMENT 2: BEFORE / AFTER COST COMPARISON")
print("=" * 60)

# --- BASELINE: large chunks, more retrieved docs ---
print("\nBuilding BASELINE pipeline (chunk=1000, k=5)...")
baseline = build_support_agent(
    collection_name="cost_baseline",
    chunk_size=1000, chunk_overlap=100, top_k=5,
)
b_cost, b_prompt, b_compl, b_results = measure(
    baseline, "BEFORE - Baseline (chunk=1000, k=5)",
)

# --- OPTIMIZED: smaller chunks, fewer retrieved docs ---
print("\nBuilding OPTIMIZED pipeline (chunk=400, k=3)...")
optimized = build_support_agent(
    collection_name="cost_optimized",
    chunk_size=400, chunk_overlap=50, top_k=3,
)
o_cost, o_prompt, o_compl, o_results = measure(
    optimized, "AFTER - Optimized (chunk=400, k=3)",
)


# ===================================================================
# COMPARISON TABLE
# ===================================================================
def safe_pct(before, after):
    return (before - after) / before * 100 if before else 0


n = len(TEST_QUERIES)

print(f"\n{'=' * 60}")
print("COMPARISON")
print(f"{'=' * 60}")
print(f"\n{'Metric':<26} {'BASELINE':>12} {'OPTIMIZED':>12} {'Savings':>10}")
print("-" * 62)
print(f"{'Avg prompt tokens':<26} {b_prompt / n:>12.0f} "
      f"{o_prompt / n:>12.0f} "
      f"{safe_pct(b_prompt, o_prompt):>9.1f}%")
print(f"{'Avg completion tokens':<26} {b_compl / n:>12.0f} "
      f"{o_compl / n:>12.0f}")
print(f"{'Avg cost / query':<26} ${b_cost / n:>11.6f} "
      f"${o_cost / n:>11.6f} "
      f"{safe_pct(b_cost, o_cost):>9.1f}%")
print(f"{'Total cost (8 queries)':<26} ${b_cost:>11.6f} "
      f"${o_cost:>11.6f}")


# ===================================================================
# QUALITY SMOKE TEST
# ===================================================================
print(f"\n{'=' * 60}")
print("QUALITY SMOKE TEST")
print(f"{'=' * 60}")

o_resp_map = {q: r.get("response", "") for q, r in o_results}
all_pass = True
for query, terms in QUALITY_CHECKS.items():
    resp = o_resp_map.get(query, "")
    found = any(t.lower() in resp.lower() for t in terms)
    status = "PASS" if found else "FAIL"
    if not found:
        all_pass = False
    print(f"  [{status}] {query[:50]}")

print(f"\n  Overall: {'ALL PASSED' if all_pass else 'REGRESSION DETECTED'}")
print(f"  For full evaluation, run Module B on the optimized config.")


# ===================================================================
# PROJECTED SAVINGS
# ===================================================================
qpd = 1000
daily_saving = (b_cost / n - o_cost / n) * qpd

print(f"\n{'=' * 60}")
print("PROJECTED SAVINGS (at 1,000 queries/day)")
print(f"{'=' * 60}")
print(f"  Daily:   ${daily_saving:>8.4f}")
print(f"  Monthly: ${daily_saving * 30:>8.2f}")
print(f"  Annual:  ${daily_saving * 365:>8.2f}")


# ===================================================================
# KEY TAKEAWAYS
# ===================================================================
print(f"\n{'=' * 60}")
print("KEY TAKEAWAYS")
print(f"{'=' * 60}")
print("""
1. MEASURE FIRST - Use get_openai_callback() + LangSmith traces
   to see exactly where tokens are spent.

2. TUNE RAG PARAMS - chunk_size and top_k are the biggest levers.
   Smaller chunks + fewer docs = fewer prompt tokens = lower cost.

3. VERIFY QUALITY - Always run quality checks after optimization.
   Cost savings are worthless if answers degrade.

4. USE LANGSMITH - Open your dashboard to see per-run token
   breakdowns, per-intent costs, and trace comparisons.

5. ADDITIONAL PATTERNS (production):
   - Semantic caching (skip LLM for repeated queries)
   - Model routing (cheap model for simple intents)
   - Batch API (50% discount for non-real-time workloads)
   - Prompt caching (reuse cached system prompts)
""")
