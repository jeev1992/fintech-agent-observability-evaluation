# Module B — Evaluation with LangSmith + DeepEval (`module_b_evaluation`)

This is the largest module. It covers the full evaluation stack for a multi-agent system — from creating test datasets to computing retrieval quality to using LLM-as-judge and open-source metrics.

The key insight: you can't evaluate a multi-agent system like a single-chain app. Each agent has different success criteria. Routing errors cascade — if the supervisor sends a policy question to the account agent, it doesn't matter how good the account agent is.

---

## What this module teaches

> "You can't improve what you can't measure."

Six evaluation techniques, each targeting a different layer of the system:

| Technique | What it measures | Layer |
|-----------|-----------------|-------|
| Evaluation datasets | Ground truth for comparison | Foundation |
| Custom evaluators | Routing accuracy, keyword correctness | Supervisor |
| LLM-as-judge | Faithfulness, correctness | Policy + Account agents |
| MRR (Mean Reciprocal Rank) | How quickly the retriever finds the right doc | Retriever |
| DeepEval metrics | Faithfulness, hallucination, answer relevancy | End-to-end |
| G-Eval | Custom criteria in plain English (e.g., empathy) | Escalation agent |

---

## File breakdown

### `demo.py` — Dataset creation and A/B experiments

Does two things:

1. **Creates a LangSmith dataset** called `fintech-demo-eval` with 15 labeled examples covering all agent paths (policy, account, escalation, out-of-scope).

2. **Runs two experiments** against the same dataset with routing and keyword correctness evaluators:
   - **v1-baseline**: `chunk_size=100` (tiny fragments — numbers split from context)
   - **v2-improved**: `chunk_size=1500` (full sections intact — all details preserved)
   
   One change, one variable. This demonstrates the **observe → tweak → re-evaluate** improvement loop.

**What to watch for when you run it:**
- Does every example get routed to the correct agent?
- Which examples score lowest on keyword correctness?
- Can you see the comparison in the LangSmith UI?

### `exercise.py` — Build the full evaluation stack

Eight TODOs covering all six techniques:

| TODO | What you do |
|------|-------------|
| 1 | Implement `run_agent()` — wrap the multi-agent graph for evaluation |
| 2 | Write `routing_evaluator()` — compare predicted vs expected intent |
| 3 | Write `faithfulness_evaluator()` — LLM-as-judge, score 0–1 |
| 4 | Write `correctness_evaluator()` — LLM-as-judge, compare to reference |
| 5 | Run `evaluate()` in LangSmith with all evaluators |
| 6 | Compute MRR across 10 retrieval queries |
| 7 | Run DeepEval metrics (faithfulness, hallucination, answer relevancy) |
| 8 | Build a G-Eval empathy metric for escalation responses |

### `solution.py` — Reference implementation

Complete working code for all 8 TODOs: evaluators, MRR computation, DeepEval assert_test, and G-Eval empathy scoring.

### `notes.md` — Concepts

Covers: why labeled data matters for multi-agent systems, dataset format, MRR formula and interpretation, DeepEval vs LangSmith comparison, G-Eval criteria design, and the observe→evaluate→compare→improve loop.

---

## How to run

Each file is self-contained — they can be run in any order. Each file creates its own LangSmith dataset (`fintech-demo-eval`, `fintech-exercise-eval`, `fintech-solution-eval`) so experiments never collide.

```bash
# Run from the project root directory

# Part 1: Create dataset + run A/B experiments
python module_b_evaluation/demo.py

# Part 2: Complete all evaluation exercises
python module_b_evaluation/exercise.py

# Part 3: Check against the solution
python module_b_evaluation/solution.py
```

**What to expect from `demo.py`:**
- Creates a dataset with 15 examples in LangSmith
- Runs the agent on all examples **3 times each** (`num_repetitions=3`) to smooth out LLM non-determinism — single runs are noisy, averaged scores give reliable comparisons
- Prints routing accuracy and keyword correctness per example
- To compare side-by-side:
  1. Open LangSmith → **Datasets & Experiments** (left sidebar) → click **`fintech-demo-eval`**
  2. On the **Experiments** tab, check the boxes next to both experiments
  3. Click the **Compare** button that appears at the bottom of the page
  4. This opens a row-by-row comparison of every example's scores across both experiments

**Expected results:**

| Experiment | keyword_correctness | routing_accuracy | What changed |
|---|---|---|---|
| `demo-v1-baseline` | ~0.66 | 1.00 | `chunk_size=100` — policy docs shredded into tiny pieces |
| `demo-v2-improved` | ~0.73 | 1.00 | `chunk_size=1500` — full policy sections stay intact |

Key observations:
- **Routing accuracy = 1.00** in both — the supervisor correctly classifies every query.
- **Keyword correctness improves in v2** — same model, same prompt, same `top_k`, only chunk size changes. With `chunk_size=100`, sentences like "$35 per transaction, maximum 3 per day ($105)" get split across chunks, so the LLM only sees partial info. With `chunk_size=1500`, full policy sections stay together.
- **This is the evaluation hill-climbing loop**: observe a low score → hypothesize a fix (better chunking) → re-evaluate → confirm improvement.

> **Why this works**: The `keyword_correctness` evaluator uses regex to extract numbers and dollar amounts from the expected answer, then checks if they appear in the actual response. With `chunk_size=100`, retrieved chunks are so small they often miss the exact figures. With `chunk_size=1500`, the full context with all numbers is preserved in each chunk.

**What to expect from `solution.py`:**
- Runs all LangSmith evaluators (routing, faithfulness, correctness)
- Computes MRR across 10 queries and prints the score
- Runs DeepEval metrics if installed (`pip install deepeval`)
- Runs G-Eval empathy metric on escalation queries
- DeepEval and G-Eval sections gracefully skip if not installed

---

## What problem does each evaluator solve?

```
Multi-Agent System
  |
  +-- Supervisor         -> routing_evaluator         (did it classify correctly?)
  |
  +-- Policy Agent       -> faithfulness_evaluator    (is it grounded in context?)
  |                      -> MRR                       (did retriever find the right doc?)
  |
  +-- Account Agent      -> correctness_evaluator    (do account details match?)
  |
  +-- Escalation Agent   -> G-Eval (empathy)          (is it genuinely empathetic?)
  |
  +-- End-to-end         -> DeepEval (hallucination)  (did it make things up?)
```

---

## Quick mental model

- **Routing accuracy** is the most critical metric — wrong agent = wrong answer, regardless of agent quality.
- **MRR** measures retrieval, not generation. MRR = 1.0 means the relevant doc is always ranked first.
- **Faithfulness** ≠ factual correctness. Faithfulness = grounded in the provided context. An answer can be faithful to wrong context.
- **G-Eval scores vary between runs.** Average over 3+ runs. Criteria quality directly determines score quality.
- **DeepEval's "faithfulness"** means faithful to provided context, NOT factually correct. "Hallucination" means the output contains info NOT in the context.
