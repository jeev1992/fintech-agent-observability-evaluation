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

1. **Silent failure demo** — Runs tricky queries (e.g., "How much does overdraft protection cost?") where the agent gives plausible but potentially wrong answers. Then you open LangSmith and find the exact failure point in the trace tree.

2. **Tagging runs** — Re-runs queries with tags (`agent-type:policy`, `agent-type:account`, `agent-type:escalation`) so you can filter them in the LangSmith monitoring dashboard.

**What to watch for when you run it:**
- Does the agent confuse the overdraft fee ($35) with the overdraft protection transfer fee ($12)?
- Can you find the trace in LangSmith and identify which run produced the wrong number?
- Which agent type uses the most tokens?

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
# Run from the Week 8/ directory

# Part 1: Watch the demo (instructor runs tricky queries, reveals traces)
python module_a_observability/demo.py

# Part 2: Complete the exercise (setup tracing, run queries, inspect traces)
python module_a_observability/exercise.py

# Part 3: Check against the solution
python module_a_observability/solution.py
```

**What to expect from `demo.py`:**
- It runs 3 tricky queries and 3 tagged queries
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
