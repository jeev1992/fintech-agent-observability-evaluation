# Evaluation with LangSmith + DeepEval
## A Complete Guide to Measuring Multi-Agent System Quality

---

## Table of Contents

1. [Why Evaluation Is Harder for Multi-Agent Systems](#1-why-evaluation-is-harder-for-multi-agent-systems)
2. [Evaluation Datasets: The Foundation](#2-evaluation-datasets-the-foundation)
3. [Custom Evaluators in LangSmith](#3-custom-evaluators-in-langsmith)
4. [LLM-as-Judge: Faithfulness and Correctness](#4-llm-as-judge-faithfulness-and-correctness)
5. [Experiments and A/B Comparison](#5-experiments-and-ab-comparison)
6. [MRR: Measuring Retrieval Quality](#6-mrr-measuring-retrieval-quality)
7. [DeepEval: Open-Source Metrics for CI/CD](#7-deepeval-open-source-metrics-for-cicd)
8. [G-Eval: Custom Criteria in Plain English](#8-g-eval-custom-criteria-in-plain-english)
9. [The Full Evaluation Loop](#9-the-full-evaluation-loop)
10. [Common Misconceptions](#10-common-misconceptions)
11. [How Our FinTech Agent Uses Evaluation](#11-how-our-fintech-agent-uses-evaluation)
12. [Integrating Evaluation into CI/CD Pipelines](#12-integrating-evaluation-into-cicd-pipelines)

---

## 1. Why Evaluation Is Harder for Multi-Agent Systems

A single-chain RAG app has one thing to evaluate: "Is the answer correct?" A multi-agent system has **five layers** of potential failure:

```
Layer 1: ROUTING
  Did the supervisor classify the intent correctly?
  "What is the overdraft fee?" → should be "policy", not "account_status"

Layer 2: RETRIEVAL
  Did the retriever return the relevant documents?
  Query about overdraft fees → should retrieve account_fees.md, not loan_policy.md

Layer 3: RETRIEVAL RANKING
  Was the relevant document ranked high enough?
  account_fees.md at position #1 (good) vs position #5 (buried in noise)

Layer 4: GENERATION FAITHFULNESS
  Is the answer grounded in the context, or hallucinated?
  Context says "$35" but the model says "$25" → hallucination

Layer 5: GENERATION CORRECTNESS
  Does the final answer match the expected answer?
  The overall output, end-to-end, compared to ground truth
```

Each layer needs different evaluators. You can't use a single "is the answer good?" metric.

### The Cascading Failure Problem

```
Wrong routing → wrong agent → wrong answer (regardless of agent quality)
     │
     └─ If the supervisor sends "What is the overdraft fee?" to the
        escalation_agent, you'll get an empathetic response about
        fees instead of the actual fee amount. The escalation agent
        did its job perfectly — the supervisor failed.
```

This is why **routing accuracy is the most critical metric**. Fix routing first, then retrieval, then generation.

### Comprehensive Evaluator Map

Here is the full set of evaluators that apply to a multi-agent FinTech system, mapped to the layer they target:

| Evaluator | Layer | What it measures |
|---|---|---|
| `routing_evaluator` | Supervisor | Did the intent classifier pick the right agent? |
| `keyword_correctness` | All agents | Do key numbers/amounts appear in the response? |
| `faithfulness_evaluator` | Policy agent | Is the answer grounded in the retrieved context? |
| `correctness_evaluator` | Account agent | Do account details match the ground truth? |
| `mrr_evaluator` | Retriever | Is the relevant document ranked near the top? |
| `hallucination_evaluator` | End-to-end | Does the response contain made-up information? |
| `answer_relevancy_evaluator` | End-to-end | Does the response actually address the question? |
| `empathy_evaluator` (G-Eval) | Escalation agent | Is the tone warm, empathetic, and professional? |
| `pii_leakage_evaluator` | All agents | Does the response leak SSNs or sensitive data? |
| `latency_evaluator` | All agents | Did the agent respond within acceptable time? |

**In the demo (`demo.py`)**, we use only two of these — `routing_evaluator` and `keyword_correctness` — to keep the A/B comparison focused. The exercise and solution implement the broader set (faithfulness, correctness, MRR, DeepEval metrics, G-Eval empathy).

**Why these two for the demo?**
- `routing_evaluator` is the most critical metric — if routing is wrong, nothing else matters.
- `keyword_correctness` is simple, interpretable, and visibly improves when we change `top_k` — making it ideal for demonstrating the hill-climbing loop.

---

## 2. Evaluation Datasets: The Foundation

### Why Labeled Data Is Critical

Without labeled data, you can't:
- Compare prompt version A vs version B
- Set quality alerts ("notify me if faithfulness drops below 0.8")
- Catch regressions ("last week's deploy broke escalation routing")
- Measure improvement ("did increasing k from 3 to 5 help retrieval?")

### Dataset Format

Each example is a triplet: **input**, **expected output**, and **expected intent**:

```python
{
    "inputs": {"question": "What is the overdraft fee?"},
    "outputs": {
        "answer": "The overdraft fee is $35 per transaction, with a maximum of 3 per day ($105).",
        "intent": "policy"
    }
}
```

### What Makes a Good Dataset

```
GOOD DATASET:                           BAD DATASET:
─────────────────────                    ─────────────────────
✅ Covers all agent paths               ❌ Only tests policy questions
   (policy, account, escalation)            (misses account and escalation)

✅ Tests routing boundaries              ❌ Only uses obvious queries
   ("I hate these overdraft fees!" →        ("What is the overdraft fee?"
    is this policy or escalation?)           always routes correctly)

✅ Includes edge cases                   ❌ Only happy paths
   (non-existent account ACC-99999,         (always valid account,
    out-of-scope questions)                  always in-scope questions)

✅ Has "I don't know" cases              ❌ Every question has an answer
   ("What stock should I invest in?"         (never tests refusal behavior)
    → should say "I don't have info")

✅ 12-15 examples minimum               ❌ 3-4 examples
   (grow to 100+ over time)                  (not enough to catch patterns)
```

### Coverage Requirements for Our FinTech Agent

| Agent Path | Minimum Examples | What They Test |
|------------|-----------------|----------------|
| Policy (account fees) | 3 | Overdraft fee, monthly fee, ATM fee |
| Policy (loans) | 2 | Credit score requirement, prepayment penalty |
| Policy (transfers) | 2 | Wire transfer cost, cancellation policy |
| Policy (fraud) | 1 | Reporting timeline |
| Account status | 3 | Active account, frozen account, non-existent account |
| Escalation | 2 | Frustrated customer, manager request |
| Out-of-scope | 1 | Investment advice → refusal |
| **Total** | **14+** | |

### Mock Accounts Available for Testing

```
ACC-12345: Alice Johnson, Premium Checking, $12,450.75, active
ACC-67890: Bob Smith, Basic Checking, $234.50, active
ACC-11111: Carol Davis, High-Yield Savings, $85,320.00, FROZEN (fraud review)
```

---

## 3. Custom Evaluators in LangSmith

### The Evaluator Pattern

Every LangSmith evaluator is a function that takes a `run` (what the agent produced) and an `example` (what we expected), and returns a score:

```python
def my_evaluator(run, example):
    # run.outputs  → what the agent actually returned
    # example.outputs → what we expected (ground truth)
    # example.inputs  → the original question

    predicted = run.outputs.get("intent", "")
    expected = example.outputs.get("intent", "")
    score = 1.0 if predicted == expected else 0.0

    return {"key": "routing_accuracy", "score": score}
```

### Routing Accuracy Evaluator

The simplest but most critical evaluator:

```python
def routing_evaluator(run, example):
    """Did the supervisor route to the correct agent?"""
    predicted = run.outputs.get("intent", "")
    expected = example.outputs.get("intent", "")
    score = 1.0 if predicted == expected else 0.0
    return {"key": "routing_accuracy", "score": score}
```

If routing accuracy < 95%, fix the supervisor before touching anything else. Wrong routing makes all other metrics meaningless.

### Running Evaluation

```python
from langsmith.evaluation import evaluate

results = evaluate(
    run_agent,                    # function that takes inputs, returns outputs
    data="fintech-demo-eval",    # each file gets its own dataset
    evaluators=[routing_evaluator, faithfulness_evaluator, correctness_evaluator],
    experiment_prefix="v1-baseline",
    metadata={"model": "gpt-4o-mini", "k": 3},
)
```

This runs the agent on every example in the dataset, applies each evaluator, and stores the results in LangSmith for viewing.

---

## 4. LLM-as-Judge: Faithfulness and Correctness

### Why Use a Judge LLM?

The generating model is biased — it thinks its own output is good. A separate LLM applies consistent evaluation criteria.

```
GENERATING MODEL:               JUDGE MODEL:
─────────────────                ─────────────────
"I answered well!"               "Let me compare the answer
                                  to the context and score
                                  objectively on a 0-1 scale."
```

**Important:** LLM-as-judge costs tokens. Each evaluation example = ~1 LLM call. Budget for this.

### Faithfulness Evaluator

**Question:** Is the answer grounded in the context (or was it hallucinated)?

```python
FAITHFULNESS_PROMPT = """
You are an expert evaluator. Assess whether the answer is faithful
to the provided context.

Score 1.0 = fully faithful: every claim is supported by the context
Score 0.5 = partially faithful: some claims are unsupported
Score 0.0 = not faithful: contains claims contradicting or absent from context

If the context is empty (escalation response), score 1.0 if the response
is a general empathetic handoff without specific policy claims.

Respond ONLY with JSON: {"score": <float>, "reason": "<one sentence>"}
"""
```

**Different agents need different faithfulness criteria:**

```
Policy Agent:    Every claim must appear in the retrieved policy documents
                 "$35 per transaction" must be in account_fees.md

Account Agent:   Account details must match the mock database
                 Balance $12,450.75 must match ACC-12345's actual balance

Escalation Agent: Should NOT contain specific policy claims
                  "I sincerely apologize" is good
                  "The overdraft fee is $35" is bad — escalation shouldn't cite policy
```

### Correctness Evaluator

**Question:** Does the answer match the expected reference answer?

```python
CORRECTNESS_PROMPT = """
Compare the AI's answer to the expected answer.
Score 1.0 = all key facts correct
Score 0.5 = partially correct
Score 0.0 = key facts wrong or missing

Focus on factual accuracy, not exact wording.
"""
```

### Parsing Judge Responses

The judge LLM returns JSON. Parse it carefully:

```python
response = judge_llm.invoke(messages).content.strip()
try:
    start = response.find("{")
    end = response.rfind("}") + 1
    parsed = json.loads(response[start:end])
    score = float(parsed.get("score", 0.5))
    reason = parsed.get("reason", "")
    return {"key": "faithfulness", "score": score, "comment": reason}
except (json.JSONDecodeError, ValueError):
    return {"key": "faithfulness", "score": 0.5}  # fallback on parse failure
```

---

## 5. Experiments and A/B Comparison

### The Iteration Workflow

```
OBSERVE (Module A)
    │  Find failing traces in LangSmith
    ▼
EVALUATE (Module B)
    │  Run evaluation dataset, get baseline scores
    ▼
COMPARE
    │  Change prompt/model/config, re-run evaluation
    │  Compare experiments side-by-side in LangSmith
    ▼
IMPROVE
    │  Pick the better version, deploy it
    ▼
REPEAT
    │  New traces → new failing examples → grow dataset → re-evaluate
    └──────────────────────────────────────────────────┘
```

### Running Two Experiments

```python
# Experiment A: baseline with chunk_size=200 (fragmented retrieval)
agent_v1 = build_support_agent(chunk_size=200, chunk_overlap=20)

results_a = evaluate(
    run_agent_v1,
    data="fintech-demo-eval",
    experiment_prefix="demo-v1-baseline",
    num_repetitions=3,  # run each example 3 times, average scores
    metadata={"chunk_size": 200},
)

# Experiment B: improved chunking with chunk_size=1500
agent_v2 = build_support_agent(chunk_size=1500, chunk_overlap=100)

results_b = evaluate(
    run_agent_v2,
    data="fintech-demo-eval",
    experiment_prefix="demo-v2-improved",
    num_repetitions=3,
    metadata={"chunk_size": 1500},
)
```

In LangSmith: Datasets → fintech-demo-eval → select both experiments → click **Compare** at the bottom.

### Hill Climbing Through A/B Testing

The goal is measurable, incremental improvement — **one variable at a time**.

```
ITERATION 1: Observe
  Run v1-baseline (chunk_size=200), check scores
  keyword_correctness = 0.55–0.65    ← low
  routing_accuracy    = 1.00         ← fine

ITERATION 2: Hypothesize
  "keyword_correctness is low because chunk_size=200 fragments
   policy documents into tiny pieces. A sentence like '$35 per
   transaction, maximum 3 per day ($105)' gets split across
   chunks, so the LLM only sees partial information."

ITERATION 3: Change ONE variable
  Change chunk_size from 200 → 1500 (keep full sections intact)
  Same model, same prompt, same top_k — isolate the effect of chunking

ITERATION 4: Re-evaluate
  Run v2-improved (chunk_size=1500), check scores
  keyword_correctness = 0.80–0.90    ← improved!
  routing_accuracy    = 1.00         ← unchanged (as expected)

ITERATION 5: Confirm
  Compare v1 vs v2 side-by-side in LangSmith
  The improvement is real and attributable to one change
```

**Why one variable matters:** If you change the prompt AND the retrieval config AND the model at the same time, you can't tell which change helped (or hurt). Changing one variable per experiment gives you a clean signal.

This is the **evaluation hill-climbing loop**: observe a low score → hypothesize a cause → change one thing → re-evaluate → confirm improvement → repeat.

### Why Repetitions Matter (`num_repetitions`)

LLM outputs are **non-deterministic** — even with `temperature=0`, the same query can produce slightly different responses across runs. A single evaluation run can be noisy:

```
Run 1: keyword_correctness = 0.72
Run 2: keyword_correctness = 0.58    ← same agent, same dataset!
Run 3: keyword_correctness = 0.68
```

If you compare v1 (single run: 0.72) vs v2 (single run: 0.68), you'd wrongly conclude v1 is better. With `num_repetitions=3`, LangSmith runs each example 3 times and averages:

```
v1 average: (0.72 + 0.58 + 0.68) / 3 = 0.66
v2 average: (0.82 + 0.88 + 0.85) / 3 = 0.85    ← clearly better
```

**Rule of thumb:**
- `num_repetitions=1` — fast but noisy (good for quick iteration)
- `num_repetitions=3` — good balance of speed and reliability (what we use in the demo)
- `num_repetitions=5+` — more robust, but slower and costs more tokens

The demo uses 3 repetitions: 15 examples × 3 reps × 2 experiments = **90 total runs**.

### Hill Climbing Across Multiple Dimensions

In practice, a multi-agent system has many **tunable factors** and many **evaluators**. Hill climbing means making incremental changes — one factor at a time — while watching how all evaluators respond.

#### The Factors You Can Change

```
RETRIEVAL FACTORS            GENERATION FACTORS          ARCHITECTURE FACTORS
─────────────────            ──────────────────          ────────────────────
chunk_size (200→1500)        system prompt wording       model choice (mini→4o)
chunk_overlap (20→200)       temperature (0→0.3)         top_k (1→5→10)
embedding model              few-shot examples           routing prompt
splitting strategy           output format instructions  agent graph structure
```

Each factor can be tuned independently. **Never change two at once** — you lose the ability to attribute improvement.

#### The Evaluators That Score Each Change

Every change affects evaluators differently. Some improve one metric while hurting another:

```
CHANGE                    routing_accuracy  keyword_correctness  faithfulness  latency  cost
──────────────────────    ────────────────  ───────────────────  ────────────  ───────  ────
chunk_size 200→1500           same               ↑ better          ↑ better    same    same
top_k 3→10                    same               ↑ better          ↑ better    ↑ slower ↑ more
temperature 0→0.3             same               ↓ worse           ↓ worse     same    same
model mini→4o                 ↑ better           ↑ better          ↑ better    ↑ slower ↑↑ 10x
add few-shot examples         same               ↑ better          same        same    ↑ more
rewrite routing prompt        ↑ better           may change        same        same    same
```

**Key insight:** There is no single "best" config. Hill climbing is about finding the best *tradeoff* for your constraints (quality vs cost vs latency).

#### The Hill-Climbing Workflow (Multi-Step)

```
STEP 1: Establish baseline
  Run all evaluators on the current agent
  ┌──────────────────────────────────────────────────┐
  │ routing_accuracy:    1.00  ← good, don't touch   │
  │ keyword_correctness: 0.55  ← low, fix this first │
  │ faithfulness:        0.70  ← could improve        │
  │ empathy (G-Eval):   0.85  ← acceptable           │
  └──────────────────────────────────────────────────┘

STEP 2: Fix the worst metric first
  keyword_correctness is lowest → hypothesize: chunk_size too small
  Change: chunk_size 200 → 1500
  ┌──────────────────────────────────────────────────┐
  │ routing_accuracy:    1.00  ← unchanged           │
  │ keyword_correctness: 0.85  ← ↑ fixed!            │
  │ faithfulness:        0.78  ← slight improvement   │
  │ empathy (G-Eval):   0.85  ← unchanged            │
  └──────────────────────────────────────────────────┘

STEP 3: Fix the next worst metric
  faithfulness is now lowest → hypothesize: prompt doesn't enforce grounding
  Change: add "cite only from provided context" to system prompt
  ┌──────────────────────────────────────────────────┐
  │ routing_accuracy:    1.00  ← unchanged           │
  │ keyword_correctness: 0.87  ← stable              │
  │ faithfulness:        0.91  ← ↑ fixed!            │
  │ empathy (G-Eval):   0.83  ← slight dip, monitor  │
  └──────────────────────────────────────────────────┘

STEP 4: Watch for regressions
  empathy dipped 0.85 → 0.83 — small enough to accept,
  but if it dropped to 0.60, you'd need to investigate.
  Every change can have side effects on other metrics.

STEP 5: Repeat until all metrics meet your thresholds
```

**The principle:** Fix the worst metric first, one change at a time, and always check that your other metrics didn't regress. This is greedy hill climbing — not globally optimal, but practical and debuggable.

### What This Is NOT

```
This is OFFLINE evaluation on curated datasets.
This is NOT production A/B testing.

Production A/B testing requires:
  ✗ Traffic splitting (50% users see v1, 50% see v2)
  ✗ Statistical significance testing
  ✗ Weeks of data collection

Offline evaluation gives you:
  ✓ Fast feedback (minutes, not weeks)
  ✓ Deterministic comparison (same dataset)
  ✓ Reproducible results
```

---

## 6. MRR: Measuring Retrieval Quality

### What MRR Measures

MRR (Mean Reciprocal Rank) answers: **How quickly does the retriever surface the correct document?**

This is a **retrieval metric**, not a generation metric. It evaluates the vector store + retriever, not the LLM.

### The Formula

$$MRR = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$

Where $\text{rank}_i$ is the position (1-based) of the first relevant document for query $i$.

### Worked Example

```
Query 1: "What is the overdraft fee?"
  Retrieved: [account_fees.md, loan_policy.md, transfer_policy.md]
  Relevant:   account_fees.md
  Rank:       1 (first position)
  RR:         1/1 = 1.0

Query 2: "How much does a wire transfer cost?"
  Retrieved: [fraud_policy.md, transfer_policy.md, account_fees.md]
  Relevant:   transfer_policy.md
  Rank:       2 (second position)
  RR:         1/2 = 0.5

Query 3: "What is the late payment fee for loans?"
  Retrieved: [loan_policy.md, account_fees.md, fraud_policy.md]
  Relevant:   loan_policy.md
  Rank:       1 (first position)
  RR:         1/1 = 1.0

MRR = (1.0 + 0.5 + 1.0) / 3 = 0.833
```

### Interpreting MRR

```
MRR = 1.0    Perfect — relevant doc is always ranked #1
MRR > 0.8    Good — relevant doc is usually in top 2
MRR 0.5-0.8  Okay — relevant doc is sometimes buried
MRR < 0.5    Poor — retriever is unreliable

For production RAG: aim for MRR > 0.8
```

### Computing MRR in Code

```python
reciprocal_ranks = []

for item in mrr_queries:
    docs = retriever.invoke(item["query"])
    rank = 0
    for i, doc in enumerate(docs, 1):
        if doc.metadata.get("source") == item["relevant_source"]:
            rank = i
            break

    rr = 1.0 / rank if rank > 0 else 0.0
    reciprocal_ranks.append(rr)

mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
```

### MRR vs Precision vs Recall

```
MRR:       "How quickly do I find the right doc?"     → Position matters
Precision: "Of what I retrieved, how much is relevant?" → Noise matters
Recall:    "Of what's relevant, how much did I find?"   → Completeness matters
```

MRR is most useful when you primarily care about the **top result** being correct (which is the case for our support agent — the top-ranked document drives the answer).

---

## 7. DeepEval: Open-Source Metrics for CI/CD

### What Is DeepEval?

DeepEval is an open-source evaluation framework that's **pytest-native** — you can run evaluations as unit tests in CI/CD.

### LangSmith vs DeepEval

```
┌──────────────────┬─────────────────────┬─────────────────────┐
│ Feature          │ LangSmith           │ DeepEval            │
├──────────────────┼─────────────────────┼─────────────────────┤
│ Tracing          │ ✅ Excellent         │ ❌ Not included      │
│ Web UI           │ ✅ Dashboard + viewer│ ❌ CLI only          │
│ Built-in metrics │ ⚠️ Custom only       │ ✅ 50+ pre-built     │
│ CI/CD native     │ ⚠️ Via SDK           │ ✅ pytest-native     │
│ Cost             │ Free tier (5K traces)│ Free (open-source)  │
│ Best for         │ Observability + eval │ Automated test suite │
└──────────────────┴─────────────────────┴─────────────────────┘
```

**They complement each other.** Use LangSmith for observability and interactive evaluation. Use DeepEval for automated metric scoring in CI/CD.

### Key DeepEval Metrics

#### Faithfulness

"Is the answer faithful to the provided context?"

```python
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric

test_case = LLMTestCase(
    input="What is the overdraft fee?",
    actual_output="The overdraft fee is $35 per transaction.",
    retrieval_context=["...account_fees.md content..."]
)

metric = FaithfulnessMetric(threshold=0.7)
metric.measure(test_case)
print(metric.score)  # 0.0 to 1.0
```

**Critical distinction:** DeepEval's "faithfulness" = faithful to provided **context**, NOT factually correct. If the context says "$25" and the answer says "$25", faithfulness is 1.0 — even though the real answer is "$35".

#### Hallucination

"Does the output contain information NOT in the context?"

```python
from deepeval.metrics import HallucinationMetric

metric = HallucinationMetric(threshold=0.7)
metric.measure(test_case)
# score = 1.0 means NO hallucination (good)
# score = 0.0 means heavy hallucination (bad)
```

#### Answer Relevancy

"Does the answer address the question that was asked?"

```python
from deepeval.metrics import AnswerRelevancyMetric

metric = AnswerRelevancyMetric(threshold=0.7)
metric.measure(test_case)
```

### Running DeepEval as Tests

```python
from deepeval import assert_test

# This throws an assertion error if score < threshold
assert_test(test_case, [faithfulness_metric, hallucination_metric])
```

In CI/CD: `pytest test_evaluation.py` — fails the build if quality drops.

---

## 8. G-Eval: Custom Criteria in Plain English

### What Is G-Eval?

G-Eval lets you define evaluation criteria in **natural language**. The LLM scores based on your description. No code needed for the criteria themselves.

### Example: Empathy Scoring for Escalation Responses

```python
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams

empathy_metric = GEval(
    name="Empathy",
    criteria=(
        "Evaluate whether the response shows genuine empathy and concern "
        "for the customer's situation. A highly empathetic response should: "
        "1) Acknowledge the customer's frustration or distress, "
        "2) Validate their feelings without being dismissive, "
        "3) Offer a clear next step (escalation, contact info), "
        "4) Use warm, professional language. "
        "Score 0 if the response is robotic or dismissive. "
        "Score 1 if the response is genuinely empathetic."
    ),
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.7,
)
```

### When to Use G-Eval vs Built-in Metrics

```
Built-in metrics (FaithfulnessMetric, etc.):
  ✅ Well-tested, consistent behavior
  ✅ No criteria design needed
  ❌ Limited to what DeepEval provides

G-Eval:
  ✅ Define ANY criteria in plain English
  ✅ Domain-specific quality scoring
  ❌ Criteria quality determines score quality
  ❌ Scores vary between runs (non-deterministic)
```

### Best Practices for G-Eval Criteria

```
GOOD CRITERIA:                           BAD CRITERIA:
─────────────────                        ─────────────────
"1) Acknowledge frustration              "Be empathetic"
 2) Validate feelings                    (too vague — what does
 3) Offer next steps                      "empathetic" mean to the LLM?)
 4) Use warm, professional language"

Specific, numbered, actionable           Single-word or vague adjective
```

**Always average G-Eval scores over 3+ runs.** A single run score is noisy.

---

## 9. The Full Evaluation Loop

This is the core workflow that connects observability (Module A) to evaluation:

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  1. OBSERVE          Find failing traces in LangSmith           │
│     │                "This query returned the wrong fee"        │
│     │                                                           │
│     ▼                                                           │
│  2. CURATE           Add failing example to eval dataset        │
│     │                inputs: "What is the overdraft fee?"       │
│     │                outputs: {answer: "$35/txn", intent: "policy"} │
│     │                                                           │
│     ▼                                                           │
│  3. EVALUATE         Run dataset against current agent          │
│     │                Routing: 92%, Faithfulness: 0.78           │
│     │                                                           │
│     ▼                                                           │
│  4. DIAGNOSE         Identify root cause from scores            │
│     │                "Faithfulness is low on fee questions"      │
│     │                                                           │
│     ▼                                                           │
│  5. FIX              Change prompt/config/retrieval             │
│     │                Add: "Quote exact dollar amounts from docs" │
│     │                                                           │
│     ▼                                                           │
│  6. RE-EVALUATE      Run same dataset against fixed agent       │
│     │                Routing: 92%, Faithfulness: 0.91 ← better  │
│     │                                                           │
│     ▼                                                           │
│  7. COMPARE          Side-by-side in LangSmith experiments      │
│     │                                                           │
│     └─────────────── REPEAT ────────────────────────────────────┘
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**This loop is NOT automatic.** Tooling enables it; engineering discipline drives it.

---

## 10. Common Misconceptions

### ❌ Misconception 1: "One metric is enough"

**Reality:** You need different metrics for different layers. Routing accuracy for the supervisor, MRR for the retriever, faithfulness for the generator, empathy for the escalation agent. A single "correctness" score hides where failures actually occur.

### ❌ Misconception 2: "Faithfulness = factual correctness"

**Reality:** Faithfulness means "grounded in the provided context." If the retriever returns the wrong document and the LLM faithfully summarizes it, faithfulness will be high but the answer will be wrong. You need both faithfulness AND correctness.

### ❌ Misconception 3: "More evaluation examples = better"

**Reality:** 15 well-chosen examples covering all agent paths, edge cases, and failure modes are far more valuable than 200 examples that only test happy-path policy queries. Coverage and diversity matter more than volume.

### ❌ Misconception 4: "LLM-as-judge is deterministic"

**Reality:** LLM-as-judge scores vary between runs. The same input can get scored 0.7 one time and 0.8 the next. Use `num_repetitions=3` or run evaluations multiple times and average. Never make decisions based on a single evaluation run.

### ❌ Misconception 5: "DeepEval replaces LangSmith (or vice versa)"

**Reality:** They complement each other. LangSmith excels at observability, trace-based debugging, and interactive evaluation. DeepEval excels at pre-built metrics and CI/CD integration. Use both.

### ❌ Misconception 6: "Evaluation is a one-time event"

**Reality:** Evaluation is a continuous loop. Every prompt change, model update, or retrieval config change requires re-evaluation. Wire evaluation into CI/CD to catch regressions automatically.

---

## 11. How Our FinTech Agent Uses Evaluation

```
EVALUATION LAYER         METRIC              WHAT IT CATCHES
─────────────────────────────────────────────────────────────────────────
Supervisor routing       routing_accuracy     "Overdraft fee" sent to
                                              account_agent instead of
                                              policy_agent

Policy retrieval         MRR                  account_fees.md ranked #4
                                              instead of #1 for fee query

Policy generation        faithfulness         LLM says "$25" when context
                                              says "$35" (hallucination)

Account generation       correctness          LLM says balance is $12,000
                                              when actual is $12,450.75

Escalation quality       G-Eval (empathy)     Response is robotic instead
                                              of warm and empathetic

End-to-end               DeepEval (hallu.)    Agent makes up a fee waiver
                                              policy that doesn't exist

A/B comparison           LangSmith exp.       v2 prompt scores higher on
                                              faithfulness than v1 prompt
```

---

## 12. Integrating Evaluation into CI/CD Pipelines

### Why CI/CD Evaluation Matters

Every prompt change, model update, or retrieval config change can silently break your agent. Without automated evaluation, regressions slip into production unnoticed. CI/CD evaluation catches them before they reach users.

```
WITHOUT CI/CD EVALUATION:                WITH CI/CD EVALUATION:
──────────────────────                    ─────────────────────
1. Engineer changes prompt                1. Engineer changes prompt
2. Manually tests 2-3 queries            2. Pushes to branch
3. "Looks good to me"                    3. CI runs 15+ evaluation examples
4. Pushes to production                  4. Routing: 1.00 ✅ Faithfulness: 0.45 ❌
5. Users report wrong answers            5. PR blocked — faithfulness regression
6. Fire drill to debug                   6. Engineer fixes prompt before merge
```

### LangSmith Evaluation in GitHub Actions

The LangSmith SDK's `evaluate()` returns results you can assert on programmatically:

```python
# tests/test_agent_quality.py
import os
from langsmith.evaluation import evaluate
from langsmith import Client

def test_agent_quality():
    """Fail the build if agent quality drops below thresholds."""
    from project.fintech_support_agent import build_support_agent, ask

    agent = build_support_agent(collection_name="ci_eval")
    app = agent["app"]

    def run_agent(inputs):
        result = ask(app, inputs["question"])
        return {
            "answer": result["response"],
            "intent": result["intent"],
        }

    def routing_evaluator(run, example):
        predicted = run.outputs.get("intent", "")
        expected = example.outputs.get("intent", "")
        return {"key": "routing_accuracy", "score": 1.0 if predicted == expected else 0.0}

    results = evaluate(
        run_agent,
        data="fintech-ci-eval",        # persistent dataset in LangSmith
        evaluators=[routing_evaluator],
        experiment_prefix=f"ci-{os.environ.get('GITHUB_SHA', 'local')[:8]}",
    )

    # Assert minimum quality thresholds
    scores = [r.evaluation_results for r in results]
    # Check that all routing scores are 1.0
    for result in results:
        for eval_result in result.evaluation_results:
            if eval_result.key == "routing_accuracy":
                assert eval_result.score >= 0.95, (
                    f"Routing accuracy dropped to {eval_result.score}"
                )
```

### GitHub Actions Workflow

```yaml
# .github/workflows/eval.yml
name: Agent Evaluation

on:
  pull_request:
    paths:
      - 'project/**'
      - 'prompts/**'

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Run evaluation suite
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          LANGCHAIN_API_KEY: ${{ secrets.LANGCHAIN_API_KEY }}
          LANGCHAIN_TRACING_V2: "true"
        run: pytest tests/test_agent_quality.py -v

      - name: Run DeepEval metrics
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: deepeval test run tests/test_deepeval.py
```

### DeepEval's `deepeval test run` for CI

DeepEval is pytest-native, which makes CI integration straightforward:

```python
# tests/test_deepeval.py
import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, HallucinationMetric

@pytest.fixture
def agent():
    from project.fintech_support_agent import build_support_agent, ask
    agent = build_support_agent(collection_name="ci_deepeval")
    return agent["app"], ask

EVAL_CASES = [
    ("What is the overdraft fee?", "policy"),
    ("What is the balance on ACC-12345?", "account_status"),
    ("Someone stole money from my account!", "escalation"),
]

@pytest.mark.parametrize("query,expected_intent", EVAL_CASES)
def test_faithfulness(agent, query, expected_intent):
    app, ask_fn = agent
    result = ask_fn(app, query)
    test_case = LLMTestCase(
        input=query,
        actual_output=result["response"],
        retrieval_context=[result["context"]] if result["context"] else ["N/A"],
    )
    assert_test(test_case, [FaithfulnessMetric(threshold=0.7)])
```

Run in CI with:
```bash
deepeval test run tests/test_deepeval.py
```

DeepEval prints a table of pass/fail results and returns a non-zero exit code if any metric fails — perfect for gating PRs.

### The CI/CD Evaluation Architecture

```
Developer pushes code
    │
    ▼
┌──────────────────────────────────────────────────┐
│  CI PIPELINE                                     │
│                                                  │
│  Step 1: LangSmith evaluate()                    │
│    Dataset: fintech-ci-eval (persistent)         │
│    Evaluators: routing, faithfulness, correctness│
│    Prefix: ci-<commit-sha>                       │
│    Gate: fail if routing < 0.95                  │
│                                                  │
│  Step 2: DeepEval test run                       │
│    Metrics: faithfulness, hallucination           │
│    Gate: fail if any metric < 0.7                │
│                                                  │
│  Step 3: Compare to baseline (optional)          │
│    LangSmith: compare ci-<sha> vs ci-main        │
│    Alert if any metric regresses > 5%            │
│                                                  │
│  All gates pass → PR can merge                   │
│  Any gate fails → PR blocked                     │
│                                                  │
└──────────────────────────────────────────────────┘
```

### Key Patterns for CI/CD Evaluation

**1. Persistent dataset**: The evaluation dataset lives in LangSmith permanently. CI runs experiments against it — never recreates or deletes it.

**2. Commit-stamped experiments**: Use the Git SHA as the experiment prefix (`ci-a1b2c3d4`). This creates a history of quality over time, viewable in LangSmith.

**3. Minimum quality gates**: Set thresholds that block merges: routing >= 0.95, faithfulness >= 0.7, no hallucinations on critical queries.

**4. Cost awareness**: Each CI run = evaluation examples × evaluators × LLM calls. A 15-example dataset with 3 evaluators = ~45 LLM calls per CI run. At ~$0.001/call with gpt-4o-mini, that's ~$0.05 per PR — trivial compared to the cost of shipping a broken agent.

**5. Separate CI dataset**: Use a dedicated `fintech-ci-eval` dataset. Don't reuse demo/exercise datasets — CI datasets should be stable and not changed by workshop runs.

---

## Summary

| Concept | Key Takeaway |
|---------|-------------|
| **Multi-agent evaluation** | Each agent needs different metrics — one "correctness" score is insufficient |
| **Evaluation datasets** | 12-15 examples minimum, covering all agent paths + edge cases |
| **Routing accuracy** | Most critical metric — wrong agent = wrong answer |
| **MRR** | Measures retrieval quality (position of first relevant doc), not generation |
| **Faithfulness** | Grounded in context ≠ factually correct — different things |
| **Correctness** | Compares final answer to reference answer using LLM-as-judge |
| **DeepEval** | Open-source, pytest-native, 50+ metrics, great for CI/CD |
| **G-Eval** | Custom criteria in plain English — average over 3+ runs |
| **Experiments** | Offline A/B comparison, not production traffic splitting |
| **The loop** | Observe → curate → evaluate → diagnose → fix → re-evaluate → repeat |

---

*Next: [Module C — Output Guardrails →](../module_c_guardrails/notes.md)*

## Why Evaluation Matters for Multi-Agent Systems

You can't improve what you can't measure. You can't compare prompt versions, set quality alerts, or catch regressions without labeled evaluation data.

**Multi-agent systems are harder to evaluate** because each agent has different success criteria:

| Component | What to Evaluate | Key Metrics |
|-----------|-----------------|-------------|
| **Supervisor** | Did it route correctly? | Routing accuracy |
| **Policy Agent (RAG)** | Did it retrieve right docs? Is the answer faithful? | MRR, Precision, Recall, Faithfulness |
| **Account Agent** | Are account details correct? | Correctness (exact match) |
| **Escalation Agent** | Is it empathetic? Does it avoid policy claims? | G-Eval (empathy), Faithfulness (should NOT contain specifics) |
| **End-to-end** | Is the final answer correct? | Correctness, Hallucination |

---

## Segment 5: Evaluation Datasets

### Why Labeled Data Is Critical

- Can't compare prompt A vs prompt B without test cases
- Can't set quality alerts without baselines
- Can't catch regressions without regression tests

### Dataset Format

Each example includes:
```python
{
    "inputs": {"question": "What is the overdraft fee?"},
    "outputs": {
        "answer": "The overdraft fee is $35 per transaction.",
        "intent": "policy"
    }
}
```

### Good Dataset Criteria

- Covers all agent paths (policy, account, escalation)
- Tests routing boundaries (ambiguous queries)
- Includes edge cases (non-existent accounts, out-of-scope questions)
- Has "I don't know" cases
- **Minimum 12–15 examples** to start; grow to 100+ over time

### Mock Accounts Available

```
ACC-12345: Alice Johnson, Premium Checking, $12,450.75, active
ACC-67890: Bob Smith, Basic Checking, $234.50, active
ACC-11111: Carol Davis, High-Yield Savings, $85,320.00, frozen (fraud review)
```

---

## Segment 6: LangSmith Evaluators

### Custom Evaluators

Evaluators are functions that grade agent outputs:

```python
def routing_evaluator(run, example):
    predicted = run.outputs.get("intent", "")
    expected = example.outputs.get("intent", "")
    score = 1.0 if predicted == expected else 0.0
    return {"key": "routing_accuracy", "score": score}
```

### LLM-as-Judge

Use a separate LLM to score quality on a 0–1 scale:

```python
# Judge prompt: "Score faithfulness given context and answer"
# Why separate judge? The generating model is biased; a judge applies consistent criteria.
```

**Key insight:** Evaluators grade the FULL SYSTEM (prompt + model + tools), not the model alone. LLM-as-judge costs tokens — each eval example ≈ 1 LLM call.

---

## Segment 7: Experiments & A/B Comparison

Run the same dataset against two agent configs:

```python
# Experiment A: original prompt
evaluate(run_agent_v1, data="fintech-demo-eval", experiment_prefix="demo-v1-baseline")

# Experiment B: improved prompt
evaluate(run_agent_v2, data="fintech-demo-eval", experiment_prefix="demo-v2-improved")
```

Compare in LangSmith's side-by-side view. This is the iteration workflow:

```
observe → evaluate → compare → improve → repeat
```

**Note:** This is offline evaluation on curated datasets, not production A/B testing (which requires traffic splitting + statistical significance).

---

## Segment 8: MRR (Mean Reciprocal Rank)

MRR measures how quickly the retriever surfaces the correct document.

### Formula

$$MRR = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$

Where $\text{rank}_i$ is the position of the first relevant document for query $i$.

### Example

| Query | First Relevant Doc Rank | Reciprocal Rank |
|-------|------------------------|-----------------|
| "Overdraft fee?" | 1 (account_fees.md) | 1.0 |
| "Wire transfer cost?" | 2 (transfer_policy.md) | 0.5 |
| "Loan interest rate?" | 1 (loan_policy.md) | 1.0 |

$$MRR = \frac{1.0 + 0.5 + 1.0}{3} = 0.833$$

- MRR = 1.0 is perfect (relevant doc always ranked first)
- MRR = 0.5–0.8 is typical in production RAG
- MRR evaluates **RETRIEVAL**, not **GENERATION**

---

## Segment 9: DeepEval

DeepEval is an open-source, pytest-native evaluation framework with 50+ built-in metrics.

### Where LangSmith Excels vs DeepEval

| Feature | LangSmith | DeepEval |
|---------|-----------|----------|
| Tracing & observability | ✅ Excellent | ❌ Not included |
| UI & dashboard | ✅ Web UI | ❌ CLI only |
| Built-in metrics | ⚠️ Custom only | ✅ 50+ pre-built |
| CI/CD integration | ⚠️ Via SDK | ✅ pytest-native |
| Cost | Free tier (5K traces) | Free (open-source) |

### Key Metrics

- **Faithfulness** — Is the answer faithful to the provided context? (NOT factual correctness — faithful to context)
- **Answer Relevancy** — Does the answer address the question asked?
- **Hallucination** — Does the output contain information NOT in the context?

Scores are 0–1 continuous. You set the passing threshold.

```python
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric, HallucinationMetric
from deepeval import assert_test
```

---

## Segment 10: G-Eval

G-Eval lets you define evaluation criteria in plain English:

```python
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams

empathy_metric = GEval(
    name="Empathy",
    criteria="Evaluate whether the response shows genuine empathy...",
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.7,
)
```

### The Full Loop

```
Find bad trace → Add to dataset → Evaluate → Fix prompt → Re-evaluate → Improved score
```

This loop is NOT automatic — it requires engineering discipline. Tooling enables it; humans drive it.

**Important:** G-Eval scores vary between runs. Average over 3+ runs. Criteria quality directly determines score quality.

---

## Segment Breakdown

| Segment | Duration | Content |
|---------|----------|---------|
| 5. Evaluation Datasets | 15 min | Create dataset, upload examples |
| 6. LangSmith Evaluators | 15 min | Custom + LLM-as-judge evaluators |
| 7. A/B Experiments | 15 min | Compare two agent configs |
| 8. MRR | 15 min | Retrieval quality metric |
| 9. DeepEval | 15 min | Faithfulness, hallucination, relevancy |
| 10. G-Eval | 15 min | Custom criteria (empathy), full loop |
