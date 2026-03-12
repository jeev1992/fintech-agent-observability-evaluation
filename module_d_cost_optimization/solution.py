"""
Module D Solution: Cost Optimization & Wrap-Up
-------------------------------------------------
Full working solution: token counting, before/after cost comparison.
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

TEST_QUERIES = [
    "What is the overdraft fee?",
    "What credit score do I need for a personal loan?",
    "How much does a domestic wire transfer cost?",
    "How long do I have to report unauthorized transactions?",
    "What is the balance on ACC-12345?",
    "Show me recent transactions for ACC-67890.",
    "What is the monthly fee for a Premium Checking account?",
    "I want to speak to a manager!",
]

# ===================================================================
# SEGMENT 14: TOKEN COUNTING
# ===================================================================

print("=" * 60)
print("SEGMENT 14: TOKEN COUNTING")
print("=" * 60)

# --- SOLUTION 2: Count tokens with tiktoken ---
encoder = tiktoken.encoding_for_model("gpt-4o-mini")

supervisor_prompt = (
    "Classify the customer query into exactly one category:\n"
    "- \"policy\" — general questions about account fees, loans, transfers, "
    "fraud policies, or banking products\n"
    "- \"account_status\" — requests to check balance, view transactions, "
    "or look up a SPECIFIC account\n"
    "- \"escalation\" — complaints, frustration, or requests for a manager\n\n"
    "Respond with ONLY the category name."
)

token_count = len(encoder.encode(supervisor_prompt))
print(f"  Supervisor system prompt: {token_count} tokens")
print(f"  This is sent on EVERY query → {token_count * 1000:,} tokens/day at 1K queries")


# ===================================================================
# SEGMENT 15: BEFORE / AFTER COMPARISON
# ===================================================================

def measure(app, label):
    """Run test queries and return average token stats."""
    print(f"\n{'=' * 60}")
    print(f"MEASURING: {label}")
    print(f"{'=' * 60}")

    totals = {"prompt": 0, "completion": 0, "cost": 0.0}

    for i, query in enumerate(TEST_QUERIES, 1):
        with get_openai_callback() as cb:
            result = ask(app, query)
        totals["prompt"] += cb.prompt_tokens
        totals["completion"] += cb.completion_tokens
        totals["cost"] += cb.total_cost
        intent = result.get("intent", "?")
        print(f"  Q{i} [{intent:15s}] | prompt={cb.prompt_tokens:5d} | "
              f"completion={cb.completion_tokens:4d} | ${cb.total_cost:.6f}")

    n = len(TEST_QUERIES)
    stats = {
        "label": label,
        "avg_prompt": totals["prompt"] / n,
        "avg_completion": totals["completion"] / n,
        "avg_cost": totals["cost"] / n,
        "total_cost": totals["cost"],
    }

    print(f"\n  TOTALS   | prompt={totals['prompt']:5d} | "
          f"completion={totals['completion']:4d} | ${totals['cost']:.6f}")
    print(f"  AVERAGES | prompt={stats['avg_prompt']:5.0f} | "
          f"completion={stats['avg_completion']:4.0f} | ${stats['avg_cost']:.6f}")

    return stats


# --- SOLUTION 3: Build BASELINE ---
print("\nBuilding BASELINE pipeline...")
baseline_agent = build_support_agent(
    collection_name="sol_d_baseline",
    chunk_size=1000,
    chunk_overlap=100,
    top_k=5,
)

# --- SOLUTION 4: Measure BASELINE ---
before = measure(baseline_agent["app"], "BEFORE: Baseline (chunk=1000, k=5)")

# --- SOLUTION 5: Build OPTIMIZED ---
print("\nBuilding OPTIMIZED pipeline...")
optimized_agent = build_support_agent(
    collection_name="sol_d_optimized",
    chunk_size=400,
    chunk_overlap=50,
    top_k=3,
)

# --- SOLUTION 6: Measure OPTIMIZED ---
after = measure(optimized_agent["app"], "AFTER: Optimized (chunk=400, k=3)")

# --- SOLUTION 7: Comparison ---
print(f"\n{'=' * 65}")
print("BEFORE / AFTER COMPARISON")
print(f"{'=' * 65}")

prompt_pct = ((before["avg_prompt"] - after["avg_prompt"])
              / before["avg_prompt"] * 100)
cost_pct = ((before["avg_cost"] - after["avg_cost"])
            / before["avg_cost"] * 100)

print(f"\n{'Metric':<28} {'BEFORE':>12} {'AFTER':>12} {'Savings':>10}")
print("-" * 65)
print(f"{'Avg prompt tokens':<28} {before['avg_prompt']:>12.0f} "
      f"{after['avg_prompt']:>12.0f} {prompt_pct:>9.1f}%")
print(f"{'Avg completion tokens':<28} {before['avg_completion']:>12.0f} "
      f"{after['avg_completion']:>12.0f}")
print(f"{'Avg cost per query':<28} ${before['avg_cost']:>11.6f} "
      f"${after['avg_cost']:>11.6f} {cost_pct:>9.1f}%")
print(f"{'Total cost (8 queries)':<28} ${before['total_cost']:>11.6f} "
      f"${after['total_cost']:>11.6f}")

qpd = 1000
daily_savings = (before["avg_cost"] - after["avg_cost"]) * qpd
print(f"\nAt {qpd:,} queries/day:")
print(f"  Daily savings:  ${daily_savings:.4f}")
print(f"  Monthly savings: ${daily_savings * 30:.2f}")
print(f"  Annual savings:  ${daily_savings * 365:.2f}")

print(f"\nIMPORTANT: Run Module B evaluation on optimized config to verify quality!")
print(f"Cost savings mean nothing if answer quality degrades.")
