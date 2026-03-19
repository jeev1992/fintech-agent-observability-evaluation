"""
Module A Solution: Agent Observability with LangSmith
-------------------------------------------------------
Full working solution: LangSmith setup, running traced queries,
inspecting trace trees, error tracing, and tagging runs.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- SOLUTION 1: Enable LangSmith tracing ---
os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")

tracing_enabled = os.environ.get("LANGCHAIN_TRACING_V2", "false") == "true"
print(f"LangSmith tracing enabled: {tracing_enabled}")

sys.path.insert(0, str(Path(__file__).parent.parent / "project"))
from fintech_support_agent import build_support_agent, ask

# --- SOLUTION 2: Build the multi-agent pipeline ---
print("\nBuilding FinTech support agent...")
agent = build_support_agent(collection_name="observability_solution")
app = agent["app"]
print("Pipeline ready.\n")

# --- SOLUTION 3: Run 3 queries covering all agent types ---
print("=" * 60)
print("SEGMENT 2: FIRST TRACES")
print("=" * 60)

queries = [
    "What is the overdraft fee?",
    "What is the balance on ACC-12345?",
    "I need to speak to a manager about fraud on my account!",
]

for query in queries:
    print(f"\nQuery: {query}")
    result = ask(app, query)
    print(f"  Intent: {result['intent']}")
    print(f"  Answer: {result['response'][:200]}...")
    print(f"  Sources: {result['retrieved_sources']}")

# --- SOLUTION 4: Trace inspection answers ---
# (Actual values depend on your run — these are typical patterns)

# Query 1 (Policy - "What is the overdraft fee?"):
#   Agent selected: policy_agent
#   Total LLM calls: 2 (classify_intent + policy RAG generation)
#   Most expensive call tokens: ~1,000-1,500 prompt (the RAG call with context)
#   Total latency: ~1,500-2,500ms
#   Retrieved documents: account_fees.md (primary), possibly loan_policy.md

# Query 2 (Account - "What is the balance on ACC-12345?"):
#   Agent selected: account_agent
#   Total LLM calls: 2 (classify_intent + account response generation)
#   Most expensive call tokens: ~200-400 prompt (account data is small)
#   Total latency: ~800-1,200ms

# Query 3 (Escalation - "I need to speak to a manager..."):
#   Agent selected: escalation_agent
#   Total LLM calls: 2 (classify_intent + escalation response)
#   Most expensive call tokens: ~100-200 prompt (no context needed)
#   Total latency: ~600-1,000ms

# --- SOLUTION 5: Error tracing ---
print("\n" + "=" * 60)
print("SEGMENT 3: ERROR TRACING")
print("=" * 60)

error_query = "What is the balance on ACC-99999?"
print(f"\nQuery: {error_query}")
result = ask(app, error_query)
print(f"  Intent: {result['intent']}")
print(f"  Answer: {result['response'][:200]}...")

# Findings:
#   Supervisor routed to: account_agent (correct — it detected ACC-XXXXX pattern)
#   Error origin: account_agent node — MOCK_ACCOUNTS.get("ACC-99999") returns None
#   Error flow: The agent returns a "couldn't find account" message directly
#                (not an exception — graceful handling). The trace shows the
#                account_agent run completing with the fallback response.

# --- SOLUTION 6: Tagged runs ---
print("\n" + "=" * 60)
print("BONUS: TAGGED RUNS")
print("=" * 60)

tagged_queries = [
    ("policy", "What is the overdraft fee?"),
    ("account", "What is the balance on ACC-12345?"),
    ("escalation", "I need to speak to a manager about fraud!"),
]

for tag, query in tagged_queries:
    print(f"\n[Tag: {tag}] Query: {query}")
    result = app.invoke(
        {
            "query": query,
            "intent": "",
            "response": "",
            "context": "",
            "retrieved_sources": [],
        },
        config={"tags": [f"agent-type:{tag}", "exercise-solution"]},
    )
    print(f"  Intent: {result['intent']}")
    print(f"  Answer: {result['response'][:150]}...")

print("\n>>> Open LangSmith and filter by tag 'exercise-solution'")
print(">>> Compare token usage and latency across agent types.")
print(">>> Policy queries should use the most tokens.")
