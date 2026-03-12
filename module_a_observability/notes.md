# Agent Observability with LangSmith
## A Complete Guide to Tracing and Debugging Multi-Agent Systems

---

## Table of Contents

1. [Why Observability Is Non-Negotiable](#1-why-observability-is-non-negotiable)
2. [Observability vs Logging vs Monitoring](#2-observability-vs-logging-vs-monitoring)
3. [LangSmith: What It Is and What It Isn't](#3-langsmith-what-it-is-and-what-it-isnt)
4. [Setting Up LangSmith Tracing](#4-setting-up-langsmith-tracing)
5. [Anatomy of a Trace](#5-anatomy-of-a-trace)
6. [Reading a Multi-Agent Trace Tree](#6-reading-a-multi-agent-trace-tree)
7. [Debugging with Traces: A Walkthrough](#7-debugging-with-traces-a-walkthrough)
8. [The Monitoring Dashboard](#8-the-monitoring-dashboard)
9. [Tagging, Filtering, and Sampling](#9-tagging-filtering-and-sampling)
10. [Common Misconceptions](#10-common-misconceptions)
11. [Alternatives to LangSmith](#11-alternatives-to-langsmith)
12. [How Our FinTech Agent Uses Observability](#12-how-our-fintech-agent-uses-observability)

---

## 1. Why Observability Is Non-Negotiable

Agents are not simple input→output functions. They chain LLM calls, tool invocations, and branching decisions. When something goes wrong, the failure is often **silent** — the agent returns a plausible-sounding but wrong answer.

### The Silent Failure Problem

Consider our FinTech support agent. A customer asks:

> "What's the overdraft fee?"

The agent returns: *"The overdraft fee is $25 per transaction."*

Sounds reasonable. But the actual policy says **$35 per transaction**.

Without observability, you'd have to guess where it went wrong:

```
Possible failure points (and you can't tell which one without a trace):

  ┌─ Did the supervisor route to the WRONG agent?
  │    e.g., sent to escalation_agent instead of policy_agent
  │
  ├─ Did the retriever return the WRONG documents?
  │    e.g., retrieved loan_policy.md instead of account_fees.md
  │
  ├─ Did the retriever return the RIGHT documents but rank them poorly?
  │    e.g., account_fees.md was result #4 out of 5, buried in noise
  │
  ├─ Did the LLM HALLUCINATE despite having the correct context?
  │    e.g., context says "$35" but model generated "$25" from training data
  │
  └─ Did the LLM get CONFUSED by too much context?
       e.g., retrieved 5 chunks, one mentions "$25 wire transfer fee",
       model confused it with the overdraft fee
```

With a LangSmith trace, you open the run tree and see *exactly* which step produced the wrong output. No guessing.

### The Multi-Agent Amplification

Single-chain RAG apps have 1–2 failure points. Our multi-agent system has 5+:

```
Customer Query
  │
  ├── [1] Supervisor classifies intent ──── Could misroute
  │
  ├── [2] Retriever embeds query ────────── Could embed poorly
  │
  ├── [3] Vector search returns docs ────── Could return wrong docs
  │
  ├── [4] Context is formatted ──────────── Could truncate key info
  │
  ├── [5] LLM generates answer ─────────── Could hallucinate
  │
  └── [6] Response returned ────────────── Could contain PII
```

Each step is a potential failure point. Debugging without traces = debugging blind.

---

## 2. Observability vs Logging vs Monitoring

These three terms get confused constantly. They serve different purposes:

```
┌─────────────────┬────────────────────────────────┬───────────────────────────┐
│                 │ What It Captures               │ When You Use It           │
├─────────────────┼────────────────────────────────┼───────────────────────────┤
│ LOGGING         │ Individual events              │ "Did error X happen?"     │
│                 │ print(), logger.info()         │ Grep through log files    │
│                 │ Flat list of messages           │ After-the-fact forensics  │
├─────────────────┼────────────────────────────────┼───────────────────────────┤
│ MONITORING      │ Aggregate metrics over time    │ "Is the system healthy?"  │
│                 │ Avg latency, error rate, p99   │ Dashboards, alerts        │
│                 │ Trends across many requests     │ Detect patterns           │
├─────────────────┼────────────────────────────────┼───────────────────────────┤
│ OBSERVABILITY   │ Structured, hierarchical       │ "WHY did this specific    │
│                 │ per-request traces             │  request fail?"           │
│                 │ Parent-child run trees          │ Debug individual requests │
│                 │ Token counts, latency, I/O     │ Root-cause analysis       │
└─────────────────┴────────────────────────────────┴───────────────────────────┘
```

### Real-World Analogy: The Hospital

- **Logging** = Nurse's notes ("Patient complained of headache at 3pm")
- **Monitoring** = Hospital dashboard ("Average ER wait time: 45 min, bed occupancy: 87%")
- **Observability** = Full patient chart ("This patient's blood work at 2pm showed X, the MRI at 3pm showed Y, which explains why the headache started at 3pm")

Logging tells you *something* happened.
Monitoring tells you *how often* it happens.
Observability tells you *exactly what happened and why*.

**For agents, you need all three. But observability is the foundation** — you can't build good monitoring or evaluation without structured traces.

---

## 3. LangSmith: What It Is and What It Isn't

### What LangSmith IS

LangSmith is a **platform for LLM application observability, evaluation, and monitoring**. It captures structured traces of every LLM call, tool invocation, and decision in your application.

```
Your Agent Code
    │
    ├── LangChain / LangGraph (auto-instrumented)
    │       │
    │       └── LANGCHAIN_TRACING_V2=true
    │           Automatically sends traces to LangSmith.
    │           No code changes needed.
    │
    └── Raw OpenAI / Anthropic calls (manual instrumentation)
            │
            ├── wrap_openai()     → wraps the client
            └── @traceable        → decorator for functions
    │
    ▼
LangSmith Platform (cloud-hosted)
    ├── Trace Storage        → stores run trees with full I/O
    ├── Trace Viewer         → hierarchical run tree UI
    ├── Monitoring Dashboard → latency, cost, error aggregates
    ├── Evaluation Framework → datasets, evaluators, experiments
    └── Annotation Queues    → human-in-the-loop review
```

### What LangSmith IS NOT

| Misconception | Reality |
|---------------|---------|
| "LangSmith IS LangChain" | LangSmith is a **standalone platform**. It works with any framework. You can use LangSmith without LangChain. |
| "LangSmith requires code changes" | For LangChain/LangGraph, set one env var. That's it. |
| "LangSmith is only for development" | It works in production too — with sampling to control cost. |
| "LangSmith replaces logging" | No. Keep your logs. LangSmith adds structured traces on top. |

### Free Tier

- **Developer plan:** 1 seat, 5,000 traces/month, free
- Enough for development and small-scale evaluation
- For production: paid plans with higher limits

---

## 4. Setting Up LangSmith Tracing

### Step 1: Create an Account

Go to [smith.langchain.com](https://smith.langchain.com) and sign up for the free Developer plan.

### Step 2: Get Your API Key

Navigate to Settings → API Keys → Create API Key

### Step 3: Set Environment Variables

```bash
# In your .env file:
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_pt_xxxxxxxxxxxxx
LANGCHAIN_PROJECT=fintech-support-agent    # optional: organizes traces
```

Or in Python:

```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "lsv2_pt_xxxxxxxxxxxxx"
```

### Step 3: Run Your Agent

That's it. If you're using LangChain or LangGraph, every chain invocation, LLM call, retriever call, and tool use is automatically captured.

```python
from fintech_support_agent import build_support_agent, ask

agent = build_support_agent()
app = agent["app"]

result = ask(app, "What is the overdraft fee?")
# This trace is now in LangSmith. Go look at it.
```

### For Non-LangChain Code

If you're using raw OpenAI calls:

```python
from langsmith.wrappers import wrap_openai
from openai import OpenAI

client = wrap_openai(OpenAI())
# Now all client.chat.completions.create() calls are traced
```

Or use the `@traceable` decorator for any function:

```python
from langsmith import traceable

@traceable
def my_custom_function(query: str) -> str:
    # This function and its children will appear in traces
    ...
```

---

## 5. Anatomy of a Trace

### Key Terminology

```
TRACE
  │
  ├── The full lifecycle of one request (one customer query)
  ├── Contains multiple RUNS
  └── Has a unique trace ID
  
RUN
  │
  ├── One step in the trace (one LLM call, one tool call, one retriever call)
  ├── Has input, output, start/end time, token counts
  ├── Has a parent-child relationship to other runs
  └── Types: "llm", "chain", "tool", "retriever"

SPAN
  │
  └── Sometimes used interchangeably with "run" in OpenTelemetry contexts
```

### What Each Run Captures

Every run in LangSmith stores:

```
┌──────────────────────────────────────────────────────┐
│ RUN: classify_intent (type: chain)                   │
├──────────────────────────────────────────────────────┤
│                                                      │
│  Input:     {"query": "What is the overdraft fee?"}  │
│  Output:    {"intent": "policy"}                     │
│                                                      │
│  Start:     2026-03-15T14:23:01.234Z                 │
│  End:       2026-03-15T14:23:01.567Z                 │
│  Latency:   333ms                                    │
│                                                      │
│  Child Runs:                                         │
│    └── ChatOpenAI (type: llm)                        │
│        ├── Prompt tokens:  87                        │
│        ├── Completion tokens: 3                      │
│        ├── Model: gpt-4o-mini                        │
│        └── Output: "policy"                          │
│                                                      │
│  Tags:      ["agent-type:supervisor"]                │
│  Metadata:  {"model": "gpt-4o-mini"}                 │
│                                                      │
└──────────────────────────────────────────────────────┘
```

### Token Counts on Tool Calls

**Important nuance:** Tool calls themselves don't consume tokens. But the LLM call that *decides* to use a tool does. And the LLM call that *processes* the tool result does.

```
LLM Call #1: "Should I use the retriever?" → 87 prompt tokens, 3 completion tokens
  │
  └── Retriever Call: vector search → 0 tokens (it's a database query)
  │
LLM Call #2: "Here's the context, now answer" → 1,240 prompt tokens, 85 completion tokens
```

---

## 6. Reading a Multi-Agent Trace Tree

Here's what a full trace looks like for a policy query through our FinTech support agent:

```
TRACE: "What is the overdraft fee?"  (total: 1,410ms, $0.000234)
│
├── [Chain] classify_intent                    333ms
│   └── [LLM] ChatOpenAI                      320ms
│       ├── Input: system prompt + "What is the overdraft fee?"
│       ├── Output: "policy"
│       ├── Prompt: 87 tokens
│       └── Completion: 3 tokens
│
└── [Chain] policy_agent                       1,077ms
    ├── [Retriever] Chroma                     180ms
    │   ├── Input: "What is the overdraft fee?"
    │   └── Output: [account_fees.md chunk 3, account_fees.md chunk 1, ...]
    │
    ├── [Function] format_docs                 2ms
    │
    └── [LLM] ChatOpenAI                       890ms
        ├── Input: system prompt + formatted context + question
        ├── Output: "The overdraft fee is $35 per transaction..."
        ├── Prompt: 1,240 tokens
        └── Completion: 85 tokens
```

For an account query:

```
TRACE: "What is the balance on ACC-12345?"  (total: 890ms, $0.000089)
│
├── [Chain] classify_intent                    340ms
│   └── [LLM] ChatOpenAI
│       ├── Output: "account_status"
│       ├── Prompt: 87 tokens
│       └── Completion: 3 tokens
│
└── [Chain] account_agent                      550ms
    └── [LLM] ChatOpenAI
        ├── Input: system prompt + account JSON + question
        ├── Output: "Account ACC-12345 has a balance of $12,450.75..."
        ├── Prompt: 320 tokens
        └── Completion: 65 tokens
```

Notice: **No retriever run** in the account trace — the account agent doesn't use RAG; it uses the mock database directly.

For an escalation query:

```
TRACE: "I'm furious! Nobody is helping!"  (total: 720ms, $0.000045)
│
├── [Chain] classify_intent                    310ms
│   └── [LLM] ChatOpenAI
│       └── Output: "escalation"
│
└── [Chain] escalation_agent                   410ms
    └── [LLM] ChatOpenAI
        ├── Input: system prompt + query (NO context)
        ├── Output: "I sincerely apologize..."
        ├── Prompt: 105 tokens
        └── Completion: 95 tokens
```

**Key insight:** Escalation is the cheapest path (no retrieval, minimal prompt). Policy is the most expensive (retrieval + full context). The supervisor call is cheap but runs on EVERY query.

---

## 7. Debugging with Traces: A Walkthrough

### Scenario: The Agent Says "$25" but the Policy Says "$35"

**Step 1:** Find the trace in LangSmith for the query "What is the overdraft fee?"

**Step 2:** Check the supervisor output → Did it route to `policy_agent`? If not, that's the bug.

**Step 3:** Check the retriever output → Did it return `account_fees.md`? Does the chunk contain "$35"?

```
If retriever returned account_fees.md with "$35" in the chunk:
  → The retriever is fine. The LLM hallucinated.
  → Fix: Strengthen the system prompt, add "quote exact numbers from context"

If retriever returned loan_policy.md instead:
  → Retrieval failure. Wrong document.
  → Fix: Improve embeddings, adjust chunk size, increase k

If retriever returned account_fees.md but a chunk WITHOUT the fee:
  → Chunking problem. The fee was in a different chunk that wasn't retrieved.
  → Fix: Adjust chunk_size/overlap, or increase k to retrieve more chunks
```

**Step 4:** Check the LLM input → Is the full system prompt + context + question what you expected?

**Step 5:** Check the LLM output → Does "$25" appear? Where did it come from?

This is "the new stack trace" for agent developers. Instead of stepping through code, you step through the run tree.

---

## 8. The Monitoring Dashboard

LangSmith's monitoring view aggregates trace data across many requests:

### What You Can See

```
┌─────────────────────────────────────────────────────────────┐
│ MONITORING DASHBOARD                                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Latency (last 24h):                                        │
│    p50: 890ms | p95: 2,100ms | p99: 3,400ms                │
│                                                             │
│  Token Usage:                                               │
│    Total prompt: 234,500 | Total completion: 45,200         │
│    Avg prompt/request: 1,172 | Avg completion/request: 226  │
│                                                             │
│  Error Rate: 0.3% (2 failures in 667 traces)                │
│                                                             │
│  Cost: $0.47 total | $0.000703 per trace                    │
│                                                             │
│  Filters:                                                   │
│    [Model ▾] [Tag ▾] [Time range ▾] [Latency > X ▾]        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Monitoring vs Observability

**Critical distinction:** Monitoring dashboards show averages. Averages hide individual failures.

```
Example:
  Average latency: 900ms ← looks fine!
  
  But: 95% of requests are 400ms (account/escalation queries)
       5% of requests are 8,000ms (policy queries with 5 chunks)
  
  The dashboard says "900ms average." The user who waited 8 seconds is furious.
```

Use monitoring for **trend detection** (is latency increasing over time?). Use per-trace observability for **debugging** (why was THIS request slow?).

---

## 9. Tagging, Filtering, and Sampling

### Tagging Runs

Add tags to runs for easy filtering:

```python
result = app.invoke(
    {"query": query, "intent": "", "response": "", "context": "", "retrieved_sources": []},
    config={"tags": ["agent-type:policy", "version:v2", "experiment:prompt-tuning"]}
)
```

In LangSmith, filter by tag to compare:
- All `agent-type:policy` traces vs `agent-type:escalation`
- `version:v1` vs `version:v2` of your prompts
- `experiment:prompt-tuning` to isolate a specific test run

### Adding Metadata

```python
config={"metadata": {"model": "gpt-4o-mini", "chunk_size": 1000, "k": 5}}
```

Metadata is searchable in the LangSmith UI. Use it to record configuration parameters for each run.

### Sampling in Production

Don't trace 100% of traffic in production:

```
Development:  100% tracing  → see everything
Staging:      100% tracing  → see everything
Production:   10-20% tracing → saves cost, stays within free tier
```

The free tier gives you 5,000 traces/month. At 1,000 queries/day, that's ~5 days of full tracing. Sample 15% and you cover the full month.

---

## 10. Common Misconceptions

### ❌ Misconception 1: "Observability = logging"

**Reality:** Logging captures flat events. Observability captures **structured, hierarchical traces** with parent-child relationships, token counts, latency waterfall, and full input/output at every step. They solve different problems.

### ❌ Misconception 2: "LangSmith only works with LangChain"

**Reality:** LangSmith works with **any framework**. For LangChain/LangGraph, it's automatic (one env var). For raw OpenAI, use `wrap_openai()`. For arbitrary code, use `@traceable`. You can even send custom spans via the SDK.

### ❌ Misconception 3: "Traces are just for debugging"

**Reality:** Traces are the foundation for:
- **Evaluation** (Module B) — curate failing traces into test datasets
- **Guardrails** (Module C) — guardrail invocations appear in traces
- **Cost monitoring** (Module D) — token usage comes from trace data

### ❌ Misconception 4: "Trace everything in production"

**Reality:** Full tracing in production is expensive (storage, API calls, latency overhead). Sample 10–20% of traffic. Use 100% only during debugging sessions or experiments.

### ❌ Misconception 5: "A run and a trace are the same thing"

**Reality:** One **trace** contains many **runs**. The trace is the full request lifecycle. Each run is one step (LLM call, tool call, retriever call). Runs form a parent-child tree within the trace.

---

## 11. Alternatives to LangSmith

| Tool | Type | Key Difference |
|------|------|----------------|
| **Langfuse** | Open-source (MIT) | Self-hostable, 50K observations/month free. Good for teams wanting full control. |
| **OpenTelemetry (OTel)** | Standard protocol | Industry-standard tracing spec. LangSmith has OTel interop. Not LLM-specific. |
| **Arize Phoenix** | Open-source | Focus on LLM evaluation and guardrails. |
| **Weights & Biases** | MLOps platform | Broader ML focus, not LLM-trace-specific. |

LangSmith's advantage: tight integration with LangChain/LangGraph (zero-config tracing), plus built-in evaluation framework. If you're already using LangChain, LangSmith is the path of least resistance.

---

## 12. How Our FinTech Agent Uses Observability

Every concept maps directly to our SecureBank support agent:

```
CONCEPT                     HOW OUR FINTECH AGENT USES IT
─────────────────────────────────────────────────────────────────────────
Trace per request           Every customer query generates one trace
                            with 2+ runs (supervisor + specialist agent)

Run tree hierarchy          Supervisor → Policy Agent → Retriever → LLM
                            Parent-child relationships show data flow

Token counting              Each LLM run captures prompt + completion tokens
                            Policy queries use 10x more tokens than escalation

Latency waterfall           Retriever calls add 150-200ms
                            LLM generation adds 500-1,000ms
                            Total: 700-2,500ms depending on agent type

Tagging                     Tag by agent type (policy/account/escalation)
                            to compare performance across query types

Error tracing               Non-existent account (ACC-99999) shows
                            account_agent returning fallback response
                            — visible as a normal run, not an exception

Monitoring                  Dashboard shows policy queries dominate cost
                            Escalation queries are cheapest
                            Supervisor call is cheap but ubiquitous
```

---

## Summary

| Concept | Key Takeaway |
|---------|-------------|
| **Observability** | Structured, hierarchical per-request traces — not just logs or metrics |
| **Trace** | Full lifecycle of one customer query (contains multiple runs) |
| **Run** | One step: LLM call, retriever call, or tool call |
| **LangSmith setup** | One env var (`LANGCHAIN_TRACING_V2=true`) for LangChain/LangGraph |
| **Debugging** | Open the run tree, find the step that produced wrong output |
| **Monitoring** | Aggregates (latency, cost, errors) — hides individual failures |
| **Tagging** | Attach labels to runs for filtering and comparison |
| **Sampling** | 10-20% in production to control cost |
| **Foundation** | Observability enables evaluation (B), guardrails (C), and cost monitoring (D) |

---

*Next: [Module B — Evaluation with LangSmith + DeepEval →](../module_b_evaluation/notes.md)*
