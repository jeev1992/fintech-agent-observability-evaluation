"""
Module B Demo: Evaluation with LangSmith
------------------------------------------------------
Demonstrates evaluation dataset creation and A/B experiment comparison
for a multi-agent system using LangSmith.

What this demo covers:
  - Creating a labeled evaluation dataset (15 examples, all agent paths)
  - Two custom evaluators (routing_accuracy, keyword_correctness)
  - A/B experiment: chunk_size=100 (v1) vs chunk_size=1500 (v2)
  - num_repetitions=3 for statistically meaningful results
  - Hill-climbing loop: observe low score → change one variable → re-evaluate

What this demo does NOT cover (see exercise.py / solution.py):
  - LLM-as-judge evaluators (faithfulness, correctness)
  - MRR (Mean Reciprocal Rank) for retrieval quality
  - DeepEval metrics (faithfulness, hallucination, answer relevancy)
  - G-Eval custom criteria (empathy scoring)
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from langsmith import Client
from langsmith.evaluation import evaluate

from eval_dataset import DEMO_DATASET_NAME, EVAL_EXAMPLES

load_dotenv()

os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")

sys.path.insert(0, str(Path(__file__).parent.parent / "project"))
from fintech_support_agent import build_support_agent, ask

# ---------------------------------------------------------------------------
# 1. Connect to LangSmith
# ---------------------------------------------------------------------------
client = Client()
print("Connected to LangSmith.")

# ---------------------------------------------------------------------------
# 2. SEGMENT 5: Create the evaluation dataset
# ---------------------------------------------------------------------------
# Dataset examples are defined in eval_dataset.py (shared by exercise/solution).
# Demo always force-recreates to ensure a clean slate.
existing = list(client.list_datasets(dataset_name=DEMO_DATASET_NAME))
if existing:
    print(f"Dataset '{DEMO_DATASET_NAME}' already exists. Deleting and recreating...")
    client.delete_dataset(dataset_id=existing[0].id)

dataset = client.create_dataset(
    dataset_name=DEMO_DATASET_NAME,
    description=(
        "Labeled evaluation examples for the FinTech multi-agent support system. "
        "Covers policy questions, account lookups, and escalation scenarios."
    ),
)
client.create_examples(
    inputs=[e["inputs"] for e in EVAL_EXAMPLES],
    outputs=[e["outputs"] for e in EVAL_EXAMPLES],
    dataset_id=dataset.id,
)
print(f"Created dataset '{DEMO_DATASET_NAME}' with {len(EVAL_EXAMPLES)} examples.\n")

# ---------------------------------------------------------------------------
# 4. Build the agent pipeline (v1 — small chunks)
# ---------------------------------------------------------------------------
# v1 uses chunk_size=100 with zero overlap: policy documents get shredded
# into tiny fragments. Key details like "$35 per transaction, maximum 3 per
# day ($105)" get split across chunks, so the LLM only sees partial info.
print("Building FinTech support agent (v1 — chunk_size=100, shredded)...")
agent = build_support_agent(collection_name="eval_demo_v1", chunk_size=100, chunk_overlap=0)
app = agent["app"]
print("Pipeline ready.\n")

# ---------------------------------------------------------------------------
# 5. Target function for evaluation
# ---------------------------------------------------------------------------
def run_agent(inputs):
    """Run the multi-agent graph and return outputs for evaluation."""
    result = ask(app, inputs["question"])
    return {
        "answer": result["response"],
        "intent": result["intent"],
        "retrieved_sources": result["retrieved_sources"],
        "context": result["context"],
    }

# ---------------------------------------------------------------------------
# 6. Evaluators
# ---------------------------------------------------------------------------
# COMPREHENSIVE EVALUATOR MAP for a multi-agent FinTech system:
#
#   Evaluator                  | Layer          | What it measures
#   ---------------------------+----------------+------------------------------------------
#   routing_evaluator          | Supervisor     | Did the intent classifier pick the right agent?
#   keyword_correctness        | All agents     | Do key numbers/amounts appear in the response?
#   faithfulness_evaluator     | Policy agent   | Is the answer grounded in retrieved context?
#   correctness_evaluator      | Account agent  | Do account details match the ground truth?
#   mrr_evaluator              | Retriever      | Is the relevant doc ranked near the top?
#   hallucination_evaluator    | End-to-end     | Does the response contain made-up info?
#   answer_relevancy_evaluator | End-to-end     | Does the response actually address the question?
#   empathy_evaluator (G-Eval) | Escalation     | Is the tone warm, empathetic, and professional?
#   pii_leakage_evaluator      | All agents     | Does the response leak SSNs or sensitive data?
#   latency_evaluator          | All agents     | Did the agent respond within acceptable time?
#
# For this demo, we use only TWO to keep it focused:
#   1. routing_evaluator     — the most critical metric (wrong agent = wrong answer)
#   2. keyword_correctness   — a simple, interpretable metric that visibly improves
#                              when we increase top_k (the hill-climbing variable)
#
# The exercise (exercise.py) and solution (solution.py) implement the full set.
# ---------------------------------------------------------------------------
def routing_evaluator(run, example):
    """Check if the supervisor routed to the correct agent."""
    predicted = run.outputs.get("intent", "")
    expected = example.outputs.get("intent", "")
    score = 1.0 if predicted == expected else 0.0
    print(f"  [Routing] expected={expected}, predicted={predicted}, score={score}")
    return {"key": "routing_accuracy", "score": score}


def keyword_correctness(run, example):
    """Simple keyword overlap check between actual and expected answers."""
    actual = run.outputs.get("answer", "").lower()
    expected = example.outputs.get("answer", "").lower()

    # Extract key terms from expected (numbers, amounts, names)
    import re
    key_terms = re.findall(r"\$[\d,.]+|\d+(?:\.\d+)?|acc-\d+", expected)
    if not key_terms:
        return {"key": "keyword_correctness", "score": 0.5}

    matches = sum(1 for term in key_terms if term in actual)
    score = matches / len(key_terms)
    return {"key": "keyword_correctness", "score": score}


# ---------------------------------------------------------------------------
# 7. SEGMENT 7: Run Experiment A (baseline)
# ---------------------------------------------------------------------------
# num_repetitions=3: run each example 3 times and average the scores.
# LLM outputs are non-deterministic — a single run can be noisy.
# 3 repetitions gives statistically meaningful averages.
print("Running Experiment A (baseline, 3 repetitions)...")
results_a = evaluate(
    run_agent,
    data=DEMO_DATASET_NAME,
    evaluators=[routing_evaluator, keyword_correctness],
    experiment_prefix="demo-v1-baseline",
    num_repetitions=3,
    metadata={"model": "gpt-4o-mini", "version": "baseline", "chunk_size": 100},
)

print("\n>>> Experiment A complete. View results in LangSmith.\n")

# ---------------------------------------------------------------------------
# 8. SEGMENT 7: Run Experiment B — better chunking (one change: chunk_size)
# ---------------------------------------------------------------------------
# A proper A/B test changes ONE variable. Here we only increase chunk_size:
#   v1: chunk_size=100  (tiny fragments, numbers split from context)
#   v2: chunk_size=1500 (full sections intact, all details preserved)
# Same model, same prompt, same top_k — only chunking strategy changes.

print("Building improved agent (v2 — chunk_size=1500)...")
agent_v2 = build_support_agent(collection_name="eval_demo_v2", chunk_size=1500, chunk_overlap=100)
app_v2 = agent_v2["app"]


def run_agent_v2(inputs):
    """Run the improved agent and return outputs for evaluation."""
    result = ask(app_v2, inputs["question"])
    return {
        "answer": result["response"],
        "intent": result["intent"],
        "retrieved_sources": result["retrieved_sources"],
        "context": result["context"],
    }


print("Running Experiment B (chunk_size=1500, 3 repetitions)...")
results_b = evaluate(
    run_agent_v2,
    data=DEMO_DATASET_NAME,
    evaluators=[routing_evaluator, keyword_correctness],
    experiment_prefix="demo-v2-improved",
    num_repetitions=3,
    metadata={"model": "gpt-4o-mini", "version": "improved-chunking", "chunk_size": 1500},
)

print("\n>>> Experiment B complete.")
print(">>> To compare side-by-side in LangSmith:")
print(">>>   1. Open LangSmith → Datasets & Experiments → fintech-demo-eval")
print(">>>   2. On the Experiments tab, check the boxes next to both experiments")
print(">>>   3. Click the 'Compare' button at the bottom of the page")
print(">>>")
print(">>> One change: chunk_size=100 → chunk_size=1500")
print(">>> Watch keyword_correctness improve from v1 → v2.")
