# Module D — Cost Optimization & Wrap-Up (`module_d_cost_optimization`)

This is the final module. It answers the question every production team asks: "How much does this cost per query, and how do we reduce it without breaking quality?"

The approach is simple: measure the baseline, apply optimizations, measure again, compare side by side.

---

## What this module teaches

> "A system that's cheap but wrong is worthless."

Multi-agent systems are expensive because every customer query generates **at least 2 LLM calls** (supervisor + specialist agent). Policy queries generate 3+ calls (supervisor + retriever embedding + LLM with full context).

The cost structure you'll discover:

| Component | Prompt Tokens | Why |
|-----------|---------------|-----|
| Supervisor | ~85–100 | System prompt + query → single word output |
| Policy Agent (RAG) | ~800–1,500 | System prompt + retrieved context + query |
| Account Agent | ~200–400 | System prompt + account JSON |
| Escalation Agent | ~80–120 | System prompt + query (no context needed) |

Two surprising facts:
1. **Output tokens cost 2–8x more** than input tokens
2. **System prompts count as input on EVERY call** — a 200-token system prompt across 2 calls = 400 extra tokens per query

---

## File breakdown

### `demo.py` — Token counting and before/after comparison

Does three things:

1. **Token counting** — Uses `tiktoken` to count tokens in the supervisor's system prompt. Shows this hidden cost multiplier.

2. **BEFORE measurement** — Runs 8 queries through the baseline config (`chunk_size=1000, k=5`), captures tokens and cost per query.

3. **AFTER measurement** — Runs the same 8 queries through the optimized config (`chunk_size=400, k=3`), then prints a side-by-side comparison table with savings percentages and projected annual savings at 1,000 queries/day.

**What to watch for when you run it:**
- How many tokens does the supervisor system prompt use?
- Which query type (policy/account/escalation) costs the most?
- What's the percentage reduction in prompt tokens?

### `exercise.py` — Build your own cost comparison

Seven TODOs:

| TODO | What you do |
|------|-------------|
| 1 | Import `tiktoken` and `get_openai_callback` |
| 2 | Count tokens in the supervisor system prompt |
| 3 | Build the BASELINE pipeline (`chunk=1000, k=5`) |
| 4 | Measure token usage for all 8 queries (BEFORE) |
| 5 | Build the OPTIMIZED pipeline (`chunk=400, k=3`) |
| 6 | Measure token usage for all 8 queries (AFTER) |
| 7 | Calculate and print the before/after comparison table |

### `solution.py` — Reference implementation

Complete working code for all 7 TODOs. Prints a clean comparison table with percentage savings and projected annual savings.

### `notes.md` — Concepts

Covers: token economics (input vs output pricing), multi-agent cost structure, `tiktoken` and `get_openai_callback()`, optimization patterns (model routing, caching, smaller chunks, lower k), and the cost-quality trade-off.

---

## How to run

No extra dependencies — `tiktoken` is already in `requirements.txt`.

```bash
# Run from the Week 8/ directory

# Part 1: Watch the before/after demo
python module_d_cost_optimization/demo.py

# Part 2: Build your own cost comparison
python module_d_cost_optimization/exercise.py

# Part 3: Check against the solution
python module_d_cost_optimization/solution.py
```

**What to expect from `demo.py`:**
- Counts tokens in the supervisor prompt (~80–100 tokens)
- Runs 8 queries through BASELINE config — prints per-query token/cost breakdown
- Runs 8 queries through OPTIMIZED config — prints per-query token/cost breakdown
- Prints a side-by-side comparison table with savings percentages
- Shows projected annual savings at 1,000 queries/day

**What to expect from `solution.py`:**
- Same output structure as demo, built from scratch via the TODOs

---

## What was optimized (and what else could be)

```
What we changed:
  chunk_size: 1000 → 400   (less context per chunk, fewer redundant tokens)
  chunk_overlap: 100 → 50  (less duplicated text between chunks)
  top_k: 5 → 3             (fewer retrieved documents in the prompt)

What else could be optimized (architecture patterns):
  +-- Model routing         -> cheap model for simple queries, expensive for complex
  +-- Prompt caching        -> reuse cached system prompts (provider-dependent)
  +-- Semantic caching      -> cache responses for similar queries (needs vector DB)
  +-- Batch API             -> 50% discount for non-real-time workloads
```

---

## Quick mental model

- Every query = **at least 2 LLM calls**. Policy queries = 3+. This is the hidden cost of multi-agent architectures.
- **Reduce k first** (5→3 is easy, high impact). Then shrink chunks. Then consider model routing.
- **Always measure quality alongside cost.** Run Module B evaluators on the optimized config. If routing accuracy or faithfulness drops, the savings aren't worth it.
- The cost comparison in this module is the same methodology you'd use in production: baseline → change one variable → measure → compare.
