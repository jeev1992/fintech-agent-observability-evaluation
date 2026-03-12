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
    data="fintech-agent-eval",    # dataset name in LangSmith
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
# Experiment A: baseline prompt
results_a = evaluate(
    run_agent_v1,
    data="fintech-agent-eval",
    experiment_prefix="v1-baseline",
)

# Experiment B: improved prompt (e.g., added "quote exact numbers")
results_b = evaluate(
    run_agent_v2,
    data="fintech-agent-eval",
    experiment_prefix="v2-improved-prompt",
)
```

In LangSmith: Datasets → fintech-agent-eval → Compare Experiments → select both → see side-by-side scores.

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
evaluate(run_agent_v1, data="fintech-agent-eval", experiment_prefix="v1-baseline")

# Experiment B: improved prompt
evaluate(run_agent_v2, data="fintech-agent-eval", experiment_prefix="v2-improved")
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
