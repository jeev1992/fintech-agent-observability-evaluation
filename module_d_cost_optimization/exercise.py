"""
Module D Exercise: Cost Optimization & Wrap-Up
-------------------------------------------------
Measure token usage for the FinTech multi-agent system,
then apply optimizations and compare before/after results.

Segments covered:
  14. Token counting & cost logging
  15. Before/after cost comparison
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")

sys.path.insert(0, str(Path(__file__).parent.parent / "project"))
from fintech_support_agent import build_support_agent, ask

# ---------------------------------------------------------------------------
# TODO 1: Import tiktoken and get_openai_callback
# ---------------------------------------------------------------------------
# YOUR IMPORTS HERE


# ---------------------------------------------------------------------------

# Test queries covering all agent paths
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


# ---------------------------------------------------------------------------
# TODO 2: Count tokens locally with tiktoken
#
# Use tiktoken to count tokens for the supervisor's system prompt.
# How many tokens is it? This gets sent on EVERY call.
#
# encoder = tiktoken.encoding_for_model("gpt-4o-mini")
# tokens = encoder.encode("your text here")
# print(len(tokens))
# ---------------------------------------------------------------------------
print("=" * 60)
print("SEGMENT 14: TOKEN COUNTING")
print("=" * 60)

supervisor_prompt = (
    "Classify the customer query into exactly one category:\n"
    "- \"policy\" — general questions about account fees, loans, transfers, "
    "fraud policies, or banking products\n"
    "- \"account_status\" — requests to check balance, view transactions, "
    "or look up a SPECIFIC account\n"
    "- \"escalation\" — complaints, frustration, or requests for a manager\n\n"
    "Respond with ONLY the category name."
)

# YOUR CODE HERE — count tokens in supervisor_prompt
print("Complete TODO 2 to count tokens.\n")


# ---------------------------------------------------------------------------
# TODO 3: Build the BASELINE pipeline (BEFORE)
#
# Use these parameters:
#   - collection_name="exercise_baseline"
#   - chunk_size=1000, chunk_overlap=100
#   - top_k=5
# ---------------------------------------------------------------------------
baseline_agent = None  # Replace with build_support_agent(...)

# YOUR CODE HERE


# ---------------------------------------------------------------------------
# TODO 4: Measure token usage for the BASELINE
#
# For each query in TEST_QUERIES:
#   - Use get_openai_callback() context manager
#   - Call ask(baseline_app, query) inside the callback
#   - Print: query type (intent), prompt tokens, completion tokens, cost
#   - Track totals
#
# Print averages at the end.
# Store results for comparison (avg_prompt, avg_completion, avg_cost)
# ---------------------------------------------------------------------------
baseline_stats = {"avg_prompt": 0, "avg_completion": 0, "avg_cost": 0}

if baseline_agent is not None:
    baseline_app = baseline_agent["app"]
    print("\n=== BEFORE: BASELINE MEASUREMENT ===")
    # YOUR CODE HERE
else:
    print("Complete TODO 3 first.")


# ---------------------------------------------------------------------------
# TODO 5: Build the OPTIMIZED pipeline (AFTER)
#
# Use these parameters:
#   - collection_name="exercise_optimized"
#   - chunk_size=400, chunk_overlap=50
#   - top_k=3
# ---------------------------------------------------------------------------
optimized_agent = None  # Replace with build_support_agent(...)

# YOUR CODE HERE


# ---------------------------------------------------------------------------
# TODO 6: Measure token usage for the OPTIMIZED pipeline
# Same as TODO 4 but with the optimized agent.
# Store results for comparison.
# ---------------------------------------------------------------------------
optimized_stats = {"avg_prompt": 0, "avg_completion": 0, "avg_cost": 0}

if optimized_agent is not None:
    optimized_app = optimized_agent["app"]
    print("\n=== AFTER: OPTIMIZED MEASUREMENT ===")
    # YOUR CODE HERE
else:
    print("Complete TODO 5 first.")


# ---------------------------------------------------------------------------
# TODO 7: Calculate and print the BEFORE / AFTER comparison
#
# Show a side-by-side table:
#   - Metric | BEFORE | AFTER | Savings %
#   - Average prompt tokens
#   - Average completion tokens
#   - Average cost per query
#
# Also calculate projected annual savings at 1,000 queries/day.
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("BEFORE / AFTER COMPARISON")
print("=" * 65)

# YOUR CODE HERE — calculate percentage reductions and print table
print("Complete TODOs above, then calculate comparison here.")
