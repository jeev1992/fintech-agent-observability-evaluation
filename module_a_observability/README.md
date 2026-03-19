# Module A — Agent Observability (`module_a_observability`)

This is where the workshop starts. You need a free LangSmith account and your `.env` file configured.

The goal of this module is to show *why* you can't evaluate or secure what you can't see — and to give you the debugging superpower of hierarchical traces for multi-agent systems.

---

## What this module teaches

> "You can't evaluate or secure what you can't see."

Agents chain LLM calls, tool invocations, and branching decisions. Without tracing, debugging is guesswork. A customer asks "What's the overdraft fee?" and the agent says "$25" — but the policy says **$35**. Where did it go wrong?

- Did the supervisor route to the wrong agent?
- Did the retriever return the wrong document?
- Did the LLM hallucinate despite having the correct context?

Without traces, you'd guess. With LangSmith, you see *exactly* which step failed.

---

## File breakdown

### `demo.py` — Silent failures and monitoring

This is the instructor-led demo. It does two things:

1. **Silent failure demo** — Runs 4 deliberately tricky queries that each produce a different type of failure. The agent gives plausible-sounding answers, but the demo prints the expected correct answer alongside so you can spot the gap. Then you open LangSmith to find WHERE it went wrong.

2. **Tagging runs** — Re-runs queries with tags (`agent-type:policy`, `agent-type:account`, `agent-type:escalation`) so you can filter them in the LangSmith monitoring dashboard.

**The 4 failure modes demonstrated:**

| Query | Failure Mode | What Goes Wrong |
|---|---|---|
| "How much does overdraft protection cost?" | Retrieval | Retriever returns $35 fee chunk instead of $12 protection chunk |
| "I'm really upset about $105! What is your overdraft policy?" | Routing | Emotional tone tricks supervisor into escalation instead of policy |
| "Does ACC-12345 qualify for the fee waiver?" | Multi-hop | Needs both account lookup AND policy lookup, but supervisor picks only one |
| "How much does a replacement debit card cost?" | Conflicting sources | account_fees.md says $5, fraud_policy.md says Free |

**What to do after running it:**
- Open LangSmith and find the 4 traces
- For each trace, click into the run tree and identify: which agent was selected, what documents were retrieved, and whether the context actually contained the correct answer
- This is the core skill: wrong answer → open trace → find the exact failing step

### `exercise.py` — Your turn: setup, trace, debug

Six TODOs that walk you through the full observability workflow:

| TODO | What you do |
|------|-------------|
| 1 | Enable LangSmith tracing (set `LANGCHAIN_TRACING_V2=true`) |
| 2 | Build the multi-agent pipeline |
| 3 | Run 3 queries (policy, account, escalation) and print results |
| 4 | Open LangSmith UI — answer 5 questions about each trace (which agent, how many LLM calls, token counts, latency, retrieved docs) |
| 5 | Inject a deliberate failure (non-existent account ACC-99999) and trace the error path |
| 6 | (Bonus) Tag your runs and filter them in the dashboard |

### `solution.py` — Reference implementation

Complete working code with all 6 TODOs solved, plus written answers to the trace inspection questions.

### `notes.md` — Concepts

Covers: observability vs logging vs monitoring, trace anatomy (trace → runs → spans), what to look for in a trace, monitoring dashboards, sampling strategies, and Langfuse as an open-source alternative.

---

## How to run

You need `OPENAI_API_KEY` and `LANGCHAIN_API_KEY` set in your `.env` file. Get a free LangSmith account at [smith.langchain.com](https://smith.langchain.com).

```bash
# Run from the project root directory

# Part 1: Watch the demo (instructor runs tricky queries, reveals traces)
python module_a_observability/demo.py

# Part 2: Complete the exercise (setup tracing, run queries, inspect traces)
python module_a_observability/exercise.py

# Part 3: Check against the solution
python module_a_observability/solution.py
```

**What to expect from `demo.py`:**
- It runs 4 tricky queries and 3 tagged queries
- All runs are sent to LangSmith automatically
- Open https://smith.langchain.com to see the trace tree for each query
- Policy queries show 2+ LLM calls (supervisor + RAG); escalation queries are cheapest

**What to expect from `exercise.py`:**
- It prints prompts for each TODO
- You fill in the code, run it, then inspect traces in the LangSmith UI
- The trace inspection questions (TODO 4) are answered by reading the UI, not by code

---

## Quick mental model

- A **trace** is one customer query, end-to-end. It contains multiple **runs**.
- A **run** is one step: an LLM call, a retriever call, or a tool call. Runs form a parent-child tree.
- Every query through our agent generates **at least 2 runs**: supervisor (classify intent) + specialist agent.
- Policy queries are the most expensive because they include retriever + LLM-with-context.
- Observability is the foundation for Modules B, C, and D — everything that follows references traces.
