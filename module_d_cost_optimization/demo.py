"""
Module D Demo: Cost Optimization & Wrap-Up
---------------------------------------------
Demonstrates token counting and before/after cost comparison
for the FinTech multi-agent support system.

Segments covered:
  14. Token counting & cost logging
  15. Before/after cost comparison
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

import tiktoken
from langchain.callbacks import get_openai_callback

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


# ---------------------------------------------------------------------------
# SEGMENT 14: Token counting with tiktoken
# ---------------------------------------------------------------------------
print("=" * 60)
print("SEGMENT 14: TOKEN COUNTING")
print("=" * 60)

encoder = tiktoken.encoding_for_model("gpt-4o-mini")

# Show how many tokens different components use
samples = {
    "Simple query": "What is the overdraft fee?",
    "Complex query": "I've been waiting 3 weeks for my fraud dispute to be resolved and nobody is helping me!",
    "System prompt (supervisor)": (
        "Classify the customer query into exactly one category:\n"
        "- \"policy\" — general questions about account fees, loans, transfers, "
        "fraud policies, or banking products\n"
        "- \"account_status\" — requests to check balance, view transactions, "
        "or look up a SPECIFIC account\n"
        "- \"escalation\" — complaints, frustration, or requests for a manager\n\n"
        "Respond with ONLY the category name."
    ),
}

for label, text in samples.items():
    token_count = len(encoder.encode(text))
    print(f"  {label:30s} → {token_count:4d} tokens")

print(f"\n  Remember: system prompts are sent on EVERY call → hidden cost multiplier")


# ---------------------------------------------------------------------------
# Cost measurement helper
# ---------------------------------------------------------------------------
def measure_cost(agent_components, config_name: str) -> dict:
    """Run test queries and capture total token usage and cost."""
    app = agent_components["app"]

    print(f"\n{'=' * 60}")
    print(f"MEASURING: {config_name}")
    print(f"{'=' * 60}")

    total_prompt = 0
    total_completion = 0
    total_cost = 0.0

    for i, query in enumerate(TEST_QUERIES, 1):
        with get_openai_callback() as cb:
            result = ask(app, query)

        total_prompt += cb.prompt_tokens
        total_completion += cb.completion_tokens
        total_cost += cb.total_cost

        intent = result.get("intent", "?")
        print(f"  Q{i:02d} [{intent:15s}] | "
              f"Prompt: {cb.prompt_tokens:5d} | "
              f"Completion: {cb.completion_tokens:4d} | "
              f"Cost: ${cb.total_cost:.6f}")

    n = len(TEST_QUERIES)
    avg_prompt = total_prompt / n
    avg_completion = total_completion / n
    avg_cost = total_cost / n

    print(f"\n  TOTALS   | Prompt: {total_prompt:5d} | "
          f"Completion: {total_completion:4d} | Cost: ${total_cost:.6f}")
    print(f"  AVERAGES | Prompt: {avg_prompt:5.0f} | "
          f"Completion: {avg_completion:4.0f} | Cost/Query: ${avg_cost:.6f}")

    return {
        "config": config_name,
        "avg_prompt": avg_prompt,
        "avg_completion": avg_completion,
        "avg_cost": avg_cost,
        "total_cost": total_cost,
    }


# ---------------------------------------------------------------------------
# SEGMENT 15: BEFORE — Baseline configuration
# ---------------------------------------------------------------------------
print("\n\nBuilding BASELINE pipeline (chunk=1000, k=5)...")
baseline = build_support_agent(
    collection_name="cost_baseline",
    chunk_size=1000,
    chunk_overlap=100,
    top_k=5,
)
baseline_results = measure_cost(baseline, "BEFORE: Baseline (chunk=1000, k=5)")

# ---------------------------------------------------------------------------
# SEGMENT 15: AFTER — Optimized configuration
# ---------------------------------------------------------------------------
print("\n\nBuilding OPTIMIZED pipeline (chunk=400, k=3)...")
optimized = build_support_agent(
    collection_name="cost_optimized",
    chunk_size=400,
    chunk_overlap=50,
    top_k=3,
)
optimized_results = measure_cost(optimized, "AFTER: Optimized (chunk=400, k=3)")

# ---------------------------------------------------------------------------
# SEGMENT 15: Side-by-side comparison
# ---------------------------------------------------------------------------
print(f"\n{'=' * 65}")
print("BEFORE / AFTER COMPARISON")
print(f"{'=' * 65}")

b = baseline_results
o = optimized_results

prompt_pct = (b["avg_prompt"] - o["avg_prompt"]) / b["avg_prompt"] * 100
cost_pct = (b["avg_cost"] - o["avg_cost"]) / b["avg_cost"] * 100

print(f"\n{'Metric':<28} {'BEFORE':>12} {'AFTER':>12} {'Savings':>10}")
print("-" * 65)
print(f"{'Avg prompt tokens':<28} {b['avg_prompt']:>12.0f} "
      f"{o['avg_prompt']:>12.0f} {prompt_pct:>9.1f}%")
print(f"{'Avg completion tokens':<28} {b['avg_completion']:>12.0f} "
      f"{o['avg_completion']:>12.0f}")
print(f"{'Avg cost per query':<28} ${b['avg_cost']:>11.6f} "
      f"${o['avg_cost']:>11.6f} {cost_pct:>9.1f}%")
print(f"{'Total cost (8 queries)':<28} ${b['total_cost']:>11.6f} "
      f"${o['total_cost']:>11.6f}")

# Projected savings
qpd = 1000
daily_savings = (b["avg_cost"] - o["avg_cost"]) * qpd
print(f"\nAt {qpd:,} queries/day:")
print(f"  Daily savings:  ${daily_savings:.4f}")
print(f"  Monthly savings: ${daily_savings * 30:.2f}")
print(f"  Annual savings:  ${daily_savings * 365:.2f}")

print(f"\n{'=' * 65}")
print("KEY TAKEAWAYS")
print(f"{'=' * 65}")
print("""
1. Reducing k (5→3) and chunk_size (1000→400) cuts prompt tokens significantly
2. Cost savings compound: 1,000 queries/day adds up fast
3. Multi-agent overhead (supervisor call) is small vs RAG context savings
4. CRITICAL: Run Module B evaluation on the optimized config to verify quality!
   Cost savings mean nothing if answer quality degrades.

What we optimized:
  - chunk_size: 1000 → 400 (less context per chunk, fewer redundant tokens)
  - chunk_overlap: 100 → 50 (less duplicated text between chunks)
  - top_k: 5 → 3 (fewer retrieved documents, less context for the LLM)

What else could be optimized (architecture patterns):
  - Use gpt-4o-mini for supervisor (it's already cheapest model; could use even cheaper)
  - Prompt caching (Anthropic/OpenAI) — reuse cached system prompts
  - Semantic caching — cache responses for similar queries
  - Model routing — send simple queries to cheap model, complex to expensive
  - Batch API — 50% discount for non-real-time workloads
""")
