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
print("Building FinTech support agent with LangSmith tracing...")
agent = build_support_agent(collection_name="observability_demo")
app = agent["app"]
print("Pipeline ready. All runs will be traced to LangSmith.\n")

# ---------------------------------------------------------------------------
# 2. SEGMENT 1: Silent failure demo
#    Run a query where the agent might give a plausible but wrong answer.
#    Without traces, you can't tell WHERE it went wrong.
# ---------------------------------------------------------------------------
print("=" * 60)
print("SEGMENT 1: SILENT FAILURE DEMO")
print("=" * 60)

# This query is designed to be tricky — the agent might hallucinate
# a wrong number or retrieve the wrong policy document.
tricky_queries = [
    # Could confuse overdraft fee ($35) with overdraft protection transfer fee ($12)
    "How much does overdraft protection cost?",
    # Could hallucinate a rate not in our docs
    "What's the interest rate on a personal loan?",
    # Might retrieve fraud policy instead of transfer policy
    "Can I reverse a wire transfer?",
]

for query in tricky_queries:
    print(f"\nQuery: {query}")
    result = ask(app, query)
    print(f"Intent: {result['intent']}")
    print(f"Answer: {result['response'][:200]}...")
    print(f"Sources: {result['retrieved_sources']}")
    print("-" * 40)

print("\n>>> NOW: Open LangSmith (https://smith.langchain.com)")
print(">>> Find these traces and inspect the run tree.")
print(">>> Can you identify where each answer's information came from?")
print(">>> Is any answer wrong? Which step caused it?\n")

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
