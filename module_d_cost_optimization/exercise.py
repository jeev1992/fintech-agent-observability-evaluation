"""
Module D Exercise: Cost Optimization
---------------------------------------
Measure and optimize the cost of the FinTech multi-agent system.

Segments:
  1. Token counting with tiktoken
  2. Before / After cost comparison using get_openai_callback

Use LangSmith to explore per-run token breakdowns, trace trees,
and per-intent cost differences - no custom infrastructure needed.
"""

import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")

sys.path.insert(0, str(Path(__file__).parent.parent / "project"))
from fintech_support_agent import build_support_agent, ask

# ---------------------------------------------------------------------------
# TODO 1: Import tiktoken and get_openai_callback
# ---------------------------------------------------------------------------
# import tiktoken
# from langchain_community.callbacks.manager import get_openai_callback


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

# Quality smoke-test: expected terms in responses for key queries
QUALITY_CHECKS = {
    "What is the overdraft fee?": ["overdraft", "fee"],
    "What credit score do I need for a personal loan?": ["credit", "loan"],
    "What is the balance on ACC-12345?": ["balance", "12450", "12,450"],
}


# ===================================================================
# SEGMENT 1: TOKEN COUNTING
# ===================================================================
print("=" * 60)
print("SEGMENT 1: TOKEN COUNTING")
print("=" * 60)

supervisor_prompt = (
    "Classify the customer query into exactly one category:\n"
    "- \"policy\" - general questions about account fees, loans, transfers, "
    "fraud policies, or banking products\n"
    "- \"account_status\" - requests to check balance, view transactions, "
    "or look up a SPECIFIC account (usually contains an account number like ACC-XXXXX)\n"
    "- \"escalation\" - complaints, frustration, requests for a manager, "
    "fraud reports, or complex issues needing human attention\n\n"
    "Respond with ONLY the category name."
)

# ---------------------------------------------------------------------------
# TODO 2: Count tokens in supervisor_prompt with tiktoken
#
# Steps:
#   1. Get the encoder: tiktoken.encoding_for_model("gpt-4o-mini")
#   2. Encode the supervisor_prompt and print the token count
#   3. Print the hidden cost: tokens * 1000 queries/day
# ---------------------------------------------------------------------------
# YOUR CODE HERE
print("Complete TODO 2 to count tokens.\n")


# ===================================================================
# SEGMENT 2: BEFORE / AFTER COMPARISON
# ===================================================================

# ---------------------------------------------------------------------------
# TODO 3: Build BASELINE and OPTIMIZED agents, measure cost
#
# Steps:
#   a) Build the BASELINE agent:
#        build_support_agent(collection_name="exercise_baseline",
#                            chunk_size=1000, chunk_overlap=100, top_k=5)
#
#   b) Build the OPTIMIZED agent:
#        build_support_agent(collection_name="exercise_optimized",
#                            chunk_size=400, chunk_overlap=50, top_k=3)
#
#   c) For each agent, loop through TEST_QUERIES:
#        - Use get_openai_callback() to capture tokens and cost:
#            with get_openai_callback() as cb:
#                result = ask(app, query)
#        - Track totals for prompt_tokens, completion_tokens, total_cost
#        - Print per-query: intent, prompt tokens, completion, cost
#
#   d) Print totals and averages for each configuration.
# ---------------------------------------------------------------------------
# YOUR CODE HERE
print("Complete TODO 3 to run the before/after comparison.\n")


# ---------------------------------------------------------------------------
# TODO 4: Print a comparison table and quality check
#
# Steps:
#   a) Print a side-by-side comparison of BASELINE vs OPTIMIZED:
#        Metric           | BASELINE | OPTIMIZED | Savings %
#        Avg prompt tokens
#        Avg cost / query
#        Total cost
#
#   b) Quality smoke test: for each query in QUALITY_CHECKS, verify
#      that at least one expected term appears in the optimized response.
#      Print PASS/FAIL for each.
#
#   c) Print projected savings at 1,000 queries/day (daily/monthly/annual).
# ---------------------------------------------------------------------------
# YOUR CODE HERE
print("Complete TODO 4 to see comparison and quality check.\n")

print("\nTIP: Open your LangSmith dashboard to see per-run token breakdowns,")
print("trace trees, and per-intent cost analysis for every query above.")
