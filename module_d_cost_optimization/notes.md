# Cost Optimization & Wrap-Up
## A Complete Guide to Measuring and Reducing Multi-Agent System Costs

---

## Table of Contents

1. [Why Cost Matters for Multi-Agent Systems](#1-why-cost-matters-for-multi-agent-systems)
2. [Token Economics: The Basics](#2-token-economics-the-basics)
3. [The Multi-Agent Cost Structure](#3-the-multi-agent-cost-structure)
4. [Counting Tokens with tiktoken](#4-counting-tokens-with-tiktoken)
5. [Measuring Cost with get_openai_callback](#5-measuring-cost-with-get_openai_callback)
6. [The Before/After Methodology](#6-the-beforeafter-methodology)
7. [Optimization Pattern 1: Reduce Retrieval Context](#7-optimization-pattern-1-reduce-retrieval-context)
8. [Optimization Pattern 2: Model Routing](#8-optimization-pattern-2-model-routing)
9. [Optimization Pattern 3: Caching](#9-optimization-pattern-3-caching)
10. [Optimization Pattern 4: Batch API](#10-optimization-pattern-4-batch-api)
11. [The Cost-Quality Trade-off](#11-the-cost-quality-tradeoff)
12. [Common Misconceptions](#12-common-misconceptions)
13. [The Full Production Architecture](#13-the-full-production-architecture)

---

## 1. Why Cost Matters for Multi-Agent Systems

Multi-agent systems are inherently more expensive than single-chain apps because **every customer query generates at least 2 LLM calls**: the supervisor (intent classification) plus the specialist agent.

### Quick Math

```
Single-chain RAG:
  1 LLM call per query
  ~1,000 prompt tokens + ~100 completion tokens
  Cost: ~$0.0002 per query (GPT-4o-mini)

Multi-agent system (our FinTech agent):
  2+ LLM calls per query (supervisor + agent)
  Policy path: ~1,400 prompt tokens + ~90 completion tokens (across both calls)
  Cost: ~$0.0003 per query (GPT-4o-mini)

At 1,000 queries/day:
  Single-chain: $0.20/day  → $73/year
  Multi-agent:  $0.30/day  → $110/year

At 100,000 queries/day:
  Single-chain: $20/day    → $7,300/year
  Multi-agent:  $30/day    → $10,950/year
  Difference:               $3,650/year
```

The difference compounds. And that's with the cheapest model (GPT-4o-mini). With GPT-4o:

```
Multi-agent with GPT-4o:
  ~1,400 prompt tokens × $2.50/M + ~90 completion tokens × $10.00/M
  = $0.0035 + $0.0009 = $0.0044 per query
  
At 100,000 queries/day:
  $440/day → $160,600/year
```

Cost optimization at scale is the difference between a viable product and burning cash.

---

## 2. Token Economics: The Basics

### What Is a Token?

A token is the smallest unit of text that an LLM processes. For English text:

```
~1 token ≈ 0.75 words (English)
~1 token ≈ 4 characters (English)

Examples:
  "Hello"         → 1 token
  "overdraft"     → 1 token
  "fee"           → 1 token
  "The overdraft fee is $35 per transaction." → ~10 tokens
```

Different models use different tokenizers. Always use the model-specific tokenizer (e.g., `tiktoken` for OpenAI models).

### Input vs Output Token Pricing

This is the most important cost fact most engineers miss:

```
┌─────────────────┬──────────────────┬──────────────────┬──────────────────┐
│ Model           │ Input Price      │ Output Price     │ Output Multiple  │
│                 │ (per 1M tokens)  │ (per 1M tokens)  │                  │
├─────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ GPT-4o-mini     │ $0.15            │ $0.60            │ 4x more          │
│ GPT-4o          │ $2.50            │ $10.00           │ 4x more          │
│ Claude Sonnet   │ $3.00            │ $15.00           │ 5x more          │
│ Claude Opus     │ $15.00           │ $75.00           │ 5x more          │
└─────────────────┴──────────────────┴──────────────────┴──────────────────┘
```

**Output tokens cost 4–5x more than input tokens.** Always optimize output length first.

### The System Prompt Tax

```
HIDDEN COST MULTIPLIER:

Your supervisor system prompt: ~90 tokens
Your policy agent system prompt: ~120 tokens

Every query sends the supervisor prompt: 90 tokens × 1 call  =  90 tokens
Policy queries also send the policy prompt: 120 tokens × 1 call = 120 tokens

Total system prompt overhead per policy query: 210 tokens
At 1,000 queries/day: 210,000 tokens/day JUST for system prompts

That's ~$0.03/day on GPT-4o-mini, or ~$0.53/day on GPT-4o.
Not huge — but it's tokens you pay for on EVERY call without any information gain.
```

---

## 3. The Multi-Agent Cost Structure

Here's what a single query costs through our FinTech support agent:

```
POLICY QUERY: "What is the overdraft fee?"
─────────────────────────────────────────────

Call 1: SUPERVISOR (classify intent)
  ┌─ System prompt:  ~90 tokens (input)
  ├─ User query:     ~10 tokens (input)
  └─ Response:       ~3 tokens (output) → "policy"
  Total: ~100 input + ~3 output

Call 2: POLICY AGENT (RAG generation)
  ┌─ System prompt:  ~120 tokens (input)
  ├─ Retrieved docs: ~600-1,200 tokens (input) ← THIS IS THE BIG ONE
  ├─ User query:     ~10 tokens (input)
  └─ Response:       ~50-150 tokens (output)
  Total: ~730-1,330 input + ~50-150 output

GRAND TOTAL: ~830-1,430 input + ~53-153 output
COST: ~$0.0002 - $0.0003 (GPT-4o-mini)
```

```
ACCOUNT QUERY: "What is the balance on ACC-12345?"
──────────────────────────────────────────────────

Call 1: SUPERVISOR → ~100 input + ~3 output
Call 2: ACCOUNT AGENT → ~300 input (account JSON) + ~70 output

GRAND TOTAL: ~400 input + ~73 output
COST: ~$0.0001 (GPT-4o-mini)
```

```
ESCALATION QUERY: "I'm furious! Nobody is helping!"
────────────────────────────────────────────────────

Call 1: SUPERVISOR → ~100 input + ~3 output
Call 2: ESCALATION AGENT → ~120 input + ~100 output

GRAND TOTAL: ~220 input + ~103 output
COST: ~$0.0001 (GPT-4o-mini)
```

**Key insight:** The policy agent is the most expensive path because of the **retrieved context**. That's where optimization has the highest impact.

---

## 4. Counting Tokens with tiktoken

### What Is tiktoken?

`tiktoken` is OpenAI's token counting library. It lets you count tokens **locally** without making an API call.

```python
import tiktoken

# Get the encoder for your model
encoder = tiktoken.encoding_for_model("gpt-4o-mini")

# Count tokens in a string
text = "What is the overdraft fee?"
tokens = encoder.encode(text)
print(len(tokens))  # 7

# See the actual tokens
print(tokens)        # [3923, 374, 279, 927, 55806, 11307, 30]
print([encoder.decode([t]) for t in tokens])
# ['What', ' is', ' the', ' over', 'draft', ' fee', '?']
```

### Practical Uses

```python
# Count tokens in your system prompt (this goes on EVERY call)
supervisor_prompt = "Classify the customer query into exactly one category..."
print(f"Supervisor prompt: {len(encoder.encode(supervisor_prompt))} tokens")
# This many tokens are billed on every single query

# Count tokens in a retrieved document chunk
chunk = "The overdraft fee is $35 per transaction..."
print(f"Chunk: {len(encoder.encode(chunk))} tokens")
# Multiply by k to see total retrieval context cost

# Estimate cost before running
prompt_tokens = 1200
completion_tokens = 100
cost = (prompt_tokens * 0.15 / 1_000_000) + (completion_tokens * 0.60 / 1_000_000)
print(f"Estimated cost: ${cost:.6f}")
```

---

## 5. Measuring Cost with get_openai_callback

### What It Does

LangChain's `get_openai_callback()` is a context manager that captures token usage and cost for all OpenAI calls within its scope:

```python
from langchain_community.callbacks.manager import get_openai_callback

with get_openai_callback() as cb:
    result = ask(app, "What is the overdraft fee?")

print(f"Prompt tokens:     {cb.prompt_tokens}")
print(f"Completion tokens: {cb.completion_tokens}")
print(f"Total tokens:      {cb.total_tokens}")
print(f"Total cost:        ${cb.total_cost:.6f}")
```

### What It Captures

```
cb.prompt_tokens      → Total input tokens across ALL LLM calls in scope
cb.completion_tokens  → Total output tokens across ALL LLM calls in scope
cb.total_tokens       → Sum of prompt + completion
cb.total_cost         → Estimated cost based on model pricing
cb.successful_requests → Number of successful API calls
```

**Important:** This captures tokens across ALL LLM calls within the `with` block. For our multi-agent system, that's the supervisor call + the specialist agent call combined.

### LangSmith Also Captures Token Usage

Because `LANGCHAIN_TRACING_V2=true` is set, every LLM call is also traced to LangSmith with per-run token counts. Open the LangSmith dashboard and click any trace to see:

- **Per-run breakdown**: supervisor call tokens vs agent call tokens (not just the combined total)
- **Latency waterfall**: which call took the longest
- **Full I/O**: exact prompts and completions sent/received

So why do we also measure locally with `get_openai_callback()`? Because LangSmith gives you visibility; this module adds **action**: cost threshold alerts, budget tracking, per-intent breakdowns, semantic caching, audit logging, and automated before/after comparison tables. They complement each other.

### Per-Query Cost Logger

```python
def measure(app, queries, label):
    totals = {"prompt": 0, "completion": 0, "cost": 0.0}

    for query in queries:
        with get_openai_callback() as cb:
            result = ask(app, query)
        totals["prompt"] += cb.prompt_tokens
        totals["completion"] += cb.completion_tokens
        totals["cost"] += cb.total_cost

    n = len(queries)
    return {
        "avg_prompt": totals["prompt"] / n,
        "avg_completion": totals["completion"] / n,
        "avg_cost": totals["cost"] / n,
    }
```

---

## 6. The Before/After Methodology

This is the core technique in this module. It's simple and powerful:

```
STEP 1: BASELINE
  Run a fixed set of test queries through the current (unoptimized) agent.
  Record: tokens per query, cost per query, quality scores.

STEP 2: OPTIMIZE
  Change ONE variable (chunk_size, k, model, prompt length, etc.).

STEP 3: MEASURE
  Run the SAME test queries through the optimized agent.
  Record: tokens per query, cost per query, quality scores.

STEP 4: COMPARE
  Side-by-side table showing before vs after.
  Calculate: % savings in tokens, % savings in cost.
  Project: Annual savings at expected query volume.

STEP 5: VERIFY QUALITY
  Run Module B evaluators on the optimized config.
  If quality dropped, the savings aren't worth it.
```

### The Comparison Table

```
Metric                   BEFORE           AFTER            Savings
────────────────────────────────────────────────────────────────────
Avg prompt tokens        1,240            780              37.1%
Avg completion tokens    88               85               3.4%
Avg cost per query       $0.000312        $0.000198        36.5%
Total cost (8 queries)   $0.002496        $0.001584        36.5%

At 1,000 queries/day:
  Daily savings:   $0.1140
  Monthly savings: $3.42
  Annual savings:  $41.61
```

---

## 7. Optimization Pattern 1: Reduce Retrieval Context

The highest-impact, lowest-effort optimization. Most of the token cost in policy queries comes from the retrieved document context.

### Lever A: Reduce k (number of retrieved documents)

```
k=5:  5 chunks × ~250 tokens each = ~1,250 prompt tokens
k=3:  3 chunks × ~250 tokens each = ~750 prompt tokens
                                      ─────────────
                                      Savings: 500 tokens (40%)
```

**Trade-off:** Lower k means fewer retrieved documents. If the relevant document was at position #4 or #5, you'll miss it. Check MRR (Module B) after reducing k.

### Lever B: Reduce chunk_size

```
chunk_size=1000:  3 chunks × ~250 tokens = ~750 tokens
chunk_size=400:   3 chunks × ~100 tokens = ~300 tokens
                                            ─────────
                                            Savings: 450 tokens (60%)
```

**Trade-off:** Smaller chunks may split relevant information across two chunks, requiring the LLM to piece things together. Or they may lose context needed for a complete answer.

### Lever C: Context compression

Instead of sending raw document chunks, use an LLM to compress them first:

```
Before compression: "The overdraft fee is $35 per transaction. We will not charge
                     an overdraft fee if your account is overdrawn by $5 or less.
                     Maximum overdraft fees per day: 3 transactions ($105 maximum).
                     Overdraft protection transfer fee: $12 per transfer..."
                     (~60 tokens)

After compression:  "Overdraft: $35/txn, max 3/day ($105), protection transfer $12,
                     waived if overdrawn ≤$5"
                     (~25 tokens)
```

This adds an LLM call (cost) but reduces the main generation prompt significantly. Worth it only at very high volumes.

---

## 8. Optimization Pattern 2: Model Routing

Use the cheapest model that works for each task:

```
┌─────────────────────┬──────────────────┬──────────────────────────────┐
│ Task                │ Model Choice     │ Why                          │
├─────────────────────┼──────────────────┼──────────────────────────────┤
│ Supervisor          │ GPT-4o-mini      │ Intent classification is     │
│ (classify intent)   │                  │ simple; cheapest model works │
├─────────────────────┼──────────────────┼──────────────────────────────┤
│ Policy Agent        │ GPT-4o-mini      │ RAG answers are constrained  │
│ (RAG generation)    │ or GPT-4o        │ by context; mini often works │
├─────────────────────┼──────────────────┼──────────────────────────────┤
│ Account Agent       │ GPT-4o-mini      │ Formatting account data is   │
│ (data formatting)   │                  │ straightforward              │
├─────────────────────┼──────────────────┼──────────────────────────────┤
│ Escalation Agent    │ GPT-4o           │ Empathy and nuance may       │
│ (empathetic handoff)│ or GPT-4o-mini   │ benefit from a better model  │
└─────────────────────┴──────────────────┴──────────────────────────────┘
```

### Advanced: Query Complexity Routing

```
Simple query: "What is the overdraft fee?"
  → Route to GPT-4o-mini (cheap, fast, sufficient)

Complex query: "Compare the total annual cost of Premium vs Basic Checking
                for someone who maintains a $2,000 balance and uses ATMs 5x/month"
  → Route to GPT-4o (reasoning required)
```

**Caveat:** The classifier that determines query complexity itself costs something. Start with a single cheap model for everything; only add routing when you've proven the quality difference justifies the complexity.

---

## 9. Optimization Pattern 3: Caching

### Prompt Caching (Provider-Level)

Some providers cache the prefix of your prompt:

```
Anthropic prompt caching:
  First call:  ~120 token system prompt → billed at full price
  Second call: Same system prompt → billed at 10% (cached)

  Savings: 90% on repeated system prompt tokens
  Automatically applied when same prefix is reused
```

OpenAI also offers prompt caching for certain models. This is particularly effective for our use case because the **system prompt is identical on every call**.

### Semantic Caching (Application-Level)

Cache responses for similar queries:

```
Query 1: "What is the overdraft fee?"
  → Generate, cache response with embedding

Query 2: "How much is the overdraft charge?"
  → Embedding is similar to Query 1
  → Return cached response (no LLM call)

Savings: 100% on cache hits
Hit rates: 15-60% depending on query diversity
```

**Requires:** A vector database (like Chroma or Redis) to store embeddings and responses. Infrastructure investment.

---

## 10. Optimization Pattern 4: Batch API

OpenAI's Batch API offers **50% discount** for non-real-time workloads:

```
Normal API:  $0.15/M input, $0.60/M output (GPT-4o-mini)
Batch API:   $0.075/M input, $0.30/M output
             50% savings on everything

Trade-off: Results returned within 24 hours, not in real-time
```

**Use for:**
- Running evaluation datasets (Module B) — you don't need real-time results
- Batch-processing customer feedback
- Generating training data

**Don't use for:**
- Real-time customer support (our primary use case)
- Anything where immediate response matters

---

## 11. The Cost-Quality Trade-off

This is the most important principle in cost optimization:

```
                  HIGH QUALITY
                       │
                       │     ★ Target: Here
                       │     (Pareto frontier)
                       │    ╱
                       │   ╱
                       │  ╱       × Wasteful
                       │ ╱        (high cost, good quality
                       │╱          but not better than ★)
  ─────────────────────┼───────────────────── HIGH COST
                      ╱│
                     ╱ │
                    ╱  │
           × Bad  ╱   │
  (cheap but wrong)    │
                       │
                  LOW QUALITY
```

**Seek the Pareto frontier:** Maximum quality for minimum cost. Every optimization should be plotted on this curve.

### The Verify Step

```
NEVER skip this:

After EVERY optimization:
  1. Run Module B evaluation dataset
  2. Check: routing accuracy, faithfulness, correctness, MRR
  3. If any metric dropped significantly → revert the optimization
  4. If quality is maintained → keep the savings

Cost savings with quality degradation = false savings
```

---

## 12. Common Misconceptions

### ❌ Misconception 1: "Token count = word count"

**Reality:** ~1 token ≈ 0.75 words in English. A 1,000-word document is ~1,333 tokens. Different models use different tokenizers — always count with the model-specific tool (`tiktoken` for OpenAI).

### ❌ Misconception 2: "The LLM call is the expensive part"

**Reality:** For RAG-based agents, the **retrieved context** is the expensive part. The system prompt + query is typically 100-200 tokens. The retrieved documents can be 500-1,500 tokens. Reducing retrieval context has the highest impact.

### ❌ Misconception 3: "Prompt caching and response caching are the same"

**Reality:** Completely different mechanisms:
- **Prompt caching** (Anthropic/OpenAI): Provider caches the prefix of your prompt. Automatic, reduces input cost.
- **Response caching** (semantic caching): You cache complete responses for similar queries. Requires your own vector DB. Eliminates the LLM call entirely on cache hits.

### ❌ Misconception 4: "GPT-4o-mini is always sufficient"

**Reality:** For our support agent, GPT-4o-mini works well because answers are constrained by retrieved context or database data. But for complex reasoning, nuanced empathy, or tasks requiring long-horizon planning, a more powerful model may be necessary. Always measure quality when switching models.

### ❌ Misconception 5: "Model routing is a quick fix"

**Reality:** Model routing requires a classifier to determine query complexity. That classifier itself costs something (tokens, latency). And it's a new component that can fail. Start with prompt caching (free/cheap) and retrieval reduction (simple) before building routing infrastructure.

### ❌ Misconception 6: "Cost optimization is a one-time event"

**Reality:** Model prices change, query patterns shift, and new models launch regularly. Review costs monthly. What was optimal last month may not be optimal today.

---

## 13. The Full Production Architecture

This is everything from all four modules combined:

```
┌──────────────────────────────────────────────────────────────────────┐
│                     PRODUCTION ARCHITECTURE                         │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Customer Query                                                      │
│      │                                                               │
│      ▼                                                               │
│  ┌────────────────────────────────┐                                  │
│  │  MODULE C: INPUT GUARDRAILS   │                                  │
│  │  • OpenAI Moderation API      │                                  │
│  │  • Regex keyword guard        │                                  │
│  │  • LLM injection classifier   │                                  │
│  │  • Presidio PII redaction     │                                  │
│  └──────────────┬─────────────────┘                                  │
│                 │                                                     │
│                 ▼                                                     │
│  ┌────────────────────────────────┐     ┌──────────────────────┐     │
│  │  MULTI-AGENT GRAPH            │     │  MODULE A:           │     │
│  │                               │────►│  LANGSMITH TRACING   │     │
│  │  Supervisor → Agent           │     │  (every run traced)  │     │
│  │  (optimized: Module D)        │     └──────────────────────┘     │
│  └──────────────┬─────────────────┘                                  │
│                 │                                                     │
│                 ▼                                                     │
│  ┌────────────────────────────────┐                                  │
│  │  MODULE C: OUTPUT GUARDRAILS  │                                  │
│  │  • Guardrails AI validators   │                                  │
│  │  • Presidio PII redaction     │                                  │
│  └──────────────┬─────────────────┘                                  │
│                 │                                                     │
│                 ▼                                                     │
│  Response to User                                                    │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │  MODULE B: CONTINUOUS EVALUATION                            │    │
│  │  • Curate failing traces → evaluation dataset               │    │
│  │  • Run evaluators in CI/CD (pytest + DeepEval)             │    │
│  │  • A/B test prompt changes                                  │    │
│  └──────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │  MODULE D: COST MONITORING                                  │    │
│  │  • Track tokens and cost per query                          │    │
│  │  • Monthly before/after reviews                             │    │
│  │  • Alert on cost spikes                                     │    │
│  └──────────────────────────────────────────────────────────────┘    │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### Metrics Dashboard

```
METRIC              MODULE    TARGET        ACTION IF MISSED
───────────────────────────────────────────────────────────────
Routing accuracy    B         > 95%         Fix supervisor prompt
Retrieval MRR       B         > 0.8         Adjust chunking/embeddings
Faithfulness        B         > 0.8         Add "quote from context" to prompt
Correctness         B         > 0.8         Improve retrieval + prompt
Empathy (G-Eval)    B         > 0.7         Improve escalation prompt
PII leak rate       C         0%            Add more Presidio entities
Avg cost/query      D         Within budget Reduce k, chunk_size, or switch model
p95 latency         A         < 3s          Optimize retrieval, consider caching
```

### This Is Ongoing

Production hardening is never "done." It's a continuous loop:

```
Deploy → Observe (traces) → Evaluate (datasets) → Guard (validators)
  → Optimize (cost) → Deploy → ...
```

---

## Summary

| Concept | Key Takeaway |
|---------|-------------|
| **Multi-agent cost** | Every query = 2+ LLM calls. Policy path is most expensive (RAG context). |
| **Output > input cost** | Output tokens cost 4-5x more. Optimize output length first. |
| **System prompt tax** | System prompts are billed on every call. Hidden cost multiplier. |
| **tiktoken** | Count tokens locally without API calls. Use model-specific encoder. |
| **get_openai_callback** | LangChain context manager captures tokens and cost per call. |
| **Before/after** | Change one variable, measure same queries, compare. Always verify quality. |
| **Reduce k** | Highest impact, lowest effort. 5→3 can save 20-40% prompt tokens. |
| **Model routing** | Cheap model for simple tasks. Infrastructure investment — do last. |
| **Caching** | Prompt caching (free, provider-level). Semantic caching (needs vector DB). |
| **Batch API** | 50% discount for non-real-time workloads. Use for eval datasets. |
| **Cost-quality** | Seek the Pareto frontier. Cheap but wrong = false savings. |

---

*Previous: [Module C — Output Guardrails ←](../module_c_guardrails/notes.md)*
