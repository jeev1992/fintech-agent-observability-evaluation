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

### `demo.py` — Production-grade token counting and before/after comparison

Does five things:

1. **Token counting** — Uses `tiktoken` to count tokens in the supervisor's system prompt. Shows the hidden cost multiplier.

2. **BEFORE measurement** — Runs 8 queries through the baseline config (`chunk_size=1000, k=5`), captures tokens, cost, latency per query with trace IDs and audit logging.

3. **AFTER measurement** — Runs the same 8 queries through the optimized config (`chunk_size=400, k=3, reranking enabled`), then measures with the same instrumentation.

4. **AFTER + CACHE measurement** — Pre-populates a semantic cache from optimized results, reruns queries to demonstrate 100% cache hit rate and $0 LLM cost.

5. **Production analysis** — Prints a 3-way comparison table (baseline / optimized / cached), per-intent cost breakdown, cost alerts, quality regression check, projected annual savings, and audit log summary.

**Production features demonstrated:**
- Structured JSON logging with trace IDs
- Per-intent cost and latency breakdown
- Semantic caching with hit-rate measurement
- Cost threshold alerting and budget tracking
- Audit logging (JSONL) for fintech compliance
- Quality regression testing after optimization
- Reranking for improved RAG retrieval

### `exercise.py` — Build your own production cost pipeline

Ten TODOs:

| TODO | What you do |
|------|-------------|
| 1 | Import `tiktoken`, `get_openai_callback`, and `OpenAIEmbeddings` |
| 2 | Count tokens in the supervisor system prompt with tiktoken |
| 3 | Build the BASELINE pipeline (`chunk=1000, k=5`) |
| 4 | Implement measurement loop with trace IDs, latency, cost tracking, audit logging |
| 5 | Build the OPTIMIZED pipeline (`chunk=400, k=3, reranking=True`) |
| 6 | Measure the OPTIMIZED pipeline |
| 7 | Demonstrate semantic caching (pre-populate + rerun) |
| 8 | Print 3-way comparison table (baseline / optimized / cached) |
| 9 | Print per-intent cost breakdown |
| 10 | Quality regression check + projected savings + audit summary |

Infrastructure classes are provided: `JSONFormatter`, `SemanticCache`, `CostTracker`.

### `solution.py` — Reference implementation

Complete working code for all 10 TODOs. Prints the full production-grade output including comparison tables, per-intent breakdown, cost alerts, quality regression, projected savings, and audit log.

### `notes.md` — Concepts

Covers: token economics (input vs output pricing), multi-agent cost structure, `tiktoken` and `get_openai_callback()`, optimization patterns (model routing, caching, smaller chunks, lower k, reranking), the cost-quality trade-off, structured logging with trace IDs, audit logging for fintech compliance, semantic caching, cost caps and budget alerting, quality regression testing, and per-intent cost breakdown.

---

## How to run

No extra dependencies — `tiktoken` is already in `requirements.txt`.

```bash
# Run from the project root directory

# Part 1: Watch the before/after demo
python module_d_cost_optimization/demo.py

# Part 2: Build your own cost comparison
python module_d_cost_optimization/exercise.py

# Part 3: Check against the solution
python module_d_cost_optimization/solution.py
```

**What to expect from `demo.py`:**
- Counts tokens in the supervisor prompt (~80–100 tokens)
- Runs 8 queries through BASELINE config — prints per-query token/cost/latency with trace IDs
- Runs 8 queries through OPTIMIZED config (reranking enabled) — same instrumentation
- Runs CACHED pass — shows 100% cache hit rate, $0 LLM cost
- Prints 3-way comparison table (baseline / optimized / cached) with savings
- Prints per-intent cost breakdown (policy vs account vs escalation)
- Shows any cost alerts triggered
- Runs quality regression check on optimized responses
- Shows projected annual savings at 1,000 queries/day
- Writes audit log to `audit_log.jsonl`

**What to expect from `solution.py`:**
- Same output structure as demo, built from scratch via the 10 TODOs
- Writes audit log to `audit_log_solution.jsonl`

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
