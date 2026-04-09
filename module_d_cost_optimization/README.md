# Module D - Cost Optimization (`module_d_cost_optimization`)

The final module. Answers the question every production team asks: *"How much does this cost per query, and how do we reduce it?"*

---

## What this module teaches

Multi-agent systems are expensive because every query generates **at least 2 LLM calls** (supervisor + specialist agent). Policy queries with RAG are the most expensive path.

| Component | Prompt Tokens | Why |
|-----------|---------------|-----|
| Supervisor | ~85-100 | System prompt + query -> single word output |
| Policy Agent (RAG) | ~800-1,500 | System prompt + retrieved context + query |
| Account Agent | ~200-400 | System prompt + account JSON |
| Escalation Agent | ~80-120 | System prompt + query (no context needed) |

The approach: **measure baseline -> apply optimizations -> measure again -> compare**.

---

## File breakdown

### `demo.py` - Instructor walkthrough

1. **Token counting** - Uses `tiktoken` to count tokens in the supervisor system prompt. Shows the hidden cost multiplier.
2. **Before/After measurement** - Runs 8 queries through baseline config (`chunk=1000, k=5`) and optimized config (`chunk=400, k=3`), using `get_openai_callback()` to capture tokens and cost.
3. **Comparison table** - Side-by-side baseline vs optimized with savings %.
4. **Quality smoke test** - Verifies optimized responses still contain expected terms.
5. **Projected savings** - Extrapolates to daily/monthly/annual at 1K queries/day.

Points students to **LangSmith** for per-run token breakdowns, trace trees, and per-intent cost analysis.

### `exercise.py` - Student exercise (4 TODOs)

| TODO | What you do |
|------|-------------|
| 1 | Import `tiktoken` and `get_openai_callback` |
| 2 | Count tokens in the supervisor system prompt |
| 3 | Build baseline + optimized agents, measure cost with `get_openai_callback` |
| 4 | Print comparison table, quality check, and projected savings |

### `solution.py` - Reference implementation

Complete working code for all 4 TODOs.

### `notes.md` - Concepts

Token economics, multi-agent cost structure, `tiktoken`, `get_openai_callback()`, optimization patterns, and the cost-quality trade-off.

---

## How to run

```bash
# From the project root

# Watch the demo
python module_d_cost_optimization/demo.py

# Do the exercise
python module_d_cost_optimization/exercise.py

# Check against the solution
python module_d_cost_optimization/solution.py
```

## What to explore in LangSmith

After running the demo or solution, open your LangSmith dashboard to see:

- **Per-run token counts** - prompt vs completion tokens for each LLM call
- **Trace trees** - how many LLM calls each query path generates
- **Per-intent cost differences** - policy (expensive) vs escalation (cheap)
- **Latency breakdown** - where time is spent across the agent graph
