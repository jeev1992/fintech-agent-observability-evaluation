"""
Module A Demo: Agent Observability with LangSmith
----------------------------------------------------
Demonstrates why observability matters for multi-agent systems.
Shows how silent failures occur and how LangSmith traces reveal them.

Segments covered:
  1. Silent failure demo — agent gives a plausible but wrong answer
  4. Monitoring overview — tagging runs, viewing aggregate data
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Ensure LangSmith tracing is enabled
os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")

sys.path.insert(0, str(Path(__file__).parent.parent / "project"))
from fintech_support_agent import build_support_agent, ask

# ---------------------------------------------------------------------------
# 1. Build the multi-agent pipeline with tracing enabled
# ---------------------------------------------------------------------------
# Use chunk_size=200 to make retrieval fragile — tiny fragments mean key
# details like "$35 per transaction, maximum 3 per day ($105)" often get
# split across chunks, causing partial or wrong answers.
print("Building FinTech support agent with LangSmith tracing...")
agent = build_support_agent(collection_name="observability_demo", chunk_size=200, chunk_overlap=20)
app = agent["app"]
print("Pipeline ready. All runs will be traced to LangSmith.\n")

# ---------------------------------------------------------------------------
# 2. SEGMENT 1: Silent failure demo
#    These queries are designed to RELIABLY produce wrong or incomplete
#    answers. Each has a specific failure mode that traces will reveal.
# ---------------------------------------------------------------------------
print("=" * 60)
print("SEGMENT 1: SILENT FAILURE DEMO")
print("=" * 60)

# Each query has an expected answer and explanation of what can go wrong.
tricky_queries = [
    {
        "query": "How much does overdraft protection cost?",
        "expected": "$12 per transfer from linked savings account",
        "trap": (
            "RETRIEVAL FAILURE: 'Overdraft protection' ($12/transfer) is a different "
            "product than the 'overdraft fee' ($35/transaction). With small chunks, "
            "the retriever might return the $35 chunk instead of the $12 chunk. "
            "The agent confidently answers with the WRONG number."
        ),
    },
    {
        "query": "I'm really upset about being charged $105 in one day! What is your overdraft policy?",
        "expected": "Overdraft fee is $35/transaction, max 3/day ($105). Should route to policy_agent.",
        "trap": (
            "ROUTING FAILURE: The emotional language ('really upset') may trick the "
            "supervisor into routing to escalation_agent instead of policy_agent. "
            "You'll get an empathetic apology instead of the actual fee breakdown."
        ),
    },
    {
        "query": "Does my account ACC-12345 qualify for the monthly fee waiver?",
        "expected": "Yes — ACC-12345 has $12,450.75 (above $1,500 threshold for Premium Checking waiver)",
        "trap": (
            "MULTI-HOP FAILURE: Answering correctly requires BOTH an account lookup "
            "(to get the balance) AND a policy lookup (to check the $1,500 waiver "
            "threshold). The supervisor routes to only ONE agent, so the answer "
            "will be incomplete — either the balance or the waiver rule, not both."
        ),
    },
    {
        "query": "How much does a replacement debit card cost?",
        "expected": "$5 standard / $25 expedited (account_fees.md) — BUT fraud_policy.md says FREE for fraud cases",
        "trap": (
            "CONFLICTING SOURCES: account_fees.md says '$5.00' while fraud_policy.md "
            "says 'Free'. The agent will cite whichever document the retriever "
            "returns first, without mentioning the other. The answer is technically "
            "correct but misleadingly incomplete."
        ),
    },
]

for item in tricky_queries:
    print(f"\nQuery: {item['query']}")
    result = ask(app, item["query"])
    print(f"Intent:   {result['intent']}")
    print(f"Answer:   {result['response'][:200]}")
    print(f"Sources:  {result['retrieved_sources']}")
    print(f"Expected: {item['expected']}")
    print(f"Trap:     {item['trap']}")
    print("-" * 60)

print("\n>>> WHAT TO DO NOW:")
print(">>>   1. Open LangSmith (https://smith.langchain.com)")
print(">>>   2. Find these 4 traces in the 'Runs' tab")
print(">>>   3. For each trace, click into the run tree and answer:")
print(">>>      - Which agent did the supervisor route to?")
print(">>>      - What documents did the retriever return (if any)?")
print(">>>      - Does the retrieved context actually contain the answer?")
print(">>>      - Did the LLM hallucinate, or was the context itself wrong?")
print(">>>")
print(">>>   This is the debugging superpower of observability:")
print(">>>   wrong answer → open trace → find the exact failing step.\n")

# ---------------------------------------------------------------------------
# 3. SEGMENT 4: Tagging runs for comparison
#    Show how to tag runs so you can filter them in the dashboard.
# ---------------------------------------------------------------------------
print("=" * 60)
print("SEGMENT 4: TAGGING RUNS FOR MONITORING")
print("=" * 60)

# Run the same queries with tags for easy filtering
tagged_queries = [
    ("policy", "What is the overdraft fee?"),
    ("account", "What is the balance on ACC-12345?"),
    ("escalation", "I'm furious! Someone stole money from my account!"),
]

for tag, query in tagged_queries:
    print(f"\n[Tag: {tag}] Query: {query}")
    # Tags appear in LangSmith and can be used for filtering
    result = app.invoke(
        {
            "query": query,
            "intent": "",
            "response": "",
            "context": "",
            "retrieved_sources": [],
        },
        config={"tags": [f"agent-type:{tag}", "demo-monitoring"]},
    )
    print(f"Intent: {result['intent']}")
    print(f"Answer: {result['response'][:150]}...")

print("\n>>> In LangSmith, filter by tag 'demo-monitoring'")
print(">>> Compare latency and token usage across agent types.")
print(">>> Policy queries use the most tokens (RAG context).")
print(">>> Escalation queries are cheapest (no retrieval).")

# ---------------------------------------------------------------------------
# 4. Token and latency breakdown
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("KEY TAKEAWAYS")
print("=" * 60)
print("""
1. Every query generates 2+ LLM calls (supervisor + agent)
2. Policy queries are most expensive (supervisor + retriever + LLM with context)
3. Traces show EXACTLY where failures occur — which run, which input/output
4. Tags let you slice monitoring data by agent type, model version, etc.
5. LangSmith captures all of this automatically for LangChain/LangGraph

Next: In the exercise, you'll set up tracing yourself and inspect traces.
""")
