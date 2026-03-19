"""
Module B Exercise: Evaluation with LangSmith + DeepEval
---------------------------------------------------------
Implement evaluators, compute MRR, use DeepEval metrics, build
a G-Eval custom metric, enhance the dataset, and hill-climb.

Segments covered:
  6.  LangSmith evaluators (custom + LLM-as-judge)
  8.  MRR (Mean Reciprocal Rank)
  9.  DeepEval (faithfulness, hallucination, answer relevancy)
  10. G-Eval (custom criteria — empathy)
  11. Dataset enhancement (add edge-case examples)
  12. Hill climbing (top_k=1 vs top_k=5)
"""

import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langsmith.evaluation import evaluate
from langsmith import Client

from eval_dataset import DATASET_NAME, EVAL_EXAMPLES, ensure_dataset_exists
from eval_dataset import HILL_CLIMB_DATASET_NAME, ensure_hill_climb_dataset_exists

load_dotenv()

os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")

sys.path.insert(0, str(Path(__file__).parent.parent / "project"))
from fintech_support_agent import build_support_agent, ask

# --- Build pipeline (provided) ---
print("Building FinTech support agent...")
agent = build_support_agent(collection_name="eval_exercise")
app = agent["app"]
retriever = agent["retriever"]
judge_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
print("Pipeline ready.\n")

# --- Ensure evaluation dataset exists in LangSmith ---
ensure_dataset_exists()

# ===================================================================
# SEGMENT 6: LangSmith Evaluators
# ===================================================================

# ---------------------------------------------------------------------------
# TODO 1: Implement run_agent
#
# Run the multi-agent graph and return:
#   {"answer": ..., "intent": ..., "context": ..., "retrieved_sources": [...]}
# ---------------------------------------------------------------------------
def run_agent(inputs):
    question = inputs["question"]
    # YOUR CODE HERE
    pass


# ---------------------------------------------------------------------------
# TODO 2: Implement routing accuracy evaluator
#
# Compare run.outputs["intent"] to example.outputs["intent"]
# Score 1.0 if match, 0.0 if not.
# Return: {"key": "routing_accuracy", "score": float}
# ---------------------------------------------------------------------------
def routing_evaluator(run, example):
    # YOUR CODE HERE
    pass


# ---------------------------------------------------------------------------
# TODO 3: Implement LLM-as-judge faithfulness evaluator
#
# Use judge_llm to assess whether the answer is faithful to the context.
# Score: 1.0 = fully faithful, 0.5 = partial, 0.0 = not faithful
# For escalation responses (empty context), score 1.0 if it's a general
# empathetic handoff without specific policy claims.
#
# Return: {"key": "faithfulness", "score": float}
#
# Hint: Build a ChatPromptTemplate, call judge_llm, parse JSON response.
# ---------------------------------------------------------------------------
def faithfulness_evaluator(run, example):
    answer = run.outputs.get("answer", "")
    context = run.outputs.get("context", "")
    question = example.inputs.get("question", "")

    # YOUR CODE HERE — build prompt, call judge_llm, parse JSON, return score
    pass


# ---------------------------------------------------------------------------
# TODO 4: Implement LLM-as-judge correctness evaluator
#
# Compare actual answer to expected answer using judge_llm.
# Focus on factual accuracy, not exact wording.
# Score: 1.0 = all key facts correct, 0.5 = partial, 0.0 = wrong
#
# Return: {"key": "correctness", "score": float}
# ---------------------------------------------------------------------------
def correctness_evaluator(run, example):
    actual = run.outputs.get("answer", "")
    expected = example.outputs.get("answer", "")
    question = example.inputs.get("question", "")

    # YOUR CODE HERE
    pass


# ---------------------------------------------------------------------------
# TODO 5: Run evaluate() with all evaluators
# Use data=DATASET_NAME, experiment_prefix="fintech-eval-student"
# ---------------------------------------------------------------------------
# YOUR CODE HERE
print("Complete TODOs 1-4, then run evaluate() here.\n")


# ===================================================================
# SEGMENT 8: MRR (Mean Reciprocal Rank)
# ===================================================================

# ---------------------------------------------------------------------------
# TODO 6: Compute MRR for the retriever
#
# For each query below, run it through the retriever and find
# the rank (1-based) of the first relevant document.
#
# Expected relevant sources are provided.
# If no relevant doc is found in results, reciprocal rank = 0.
#
# MRR = average of all reciprocal ranks
# ---------------------------------------------------------------------------
mrr_queries = [
    # Easy — clearly maps to one document
    {"query": "What credit score do I need for a personal loan?", "relevant_source": "loan_policy.md"},
    {"query": "How do I report identity theft?", "relevant_source": "fraud_policy.md"},
    # Ambiguous — wire transfer fees appear in BOTH account_fees.md and transfer_policy.md
    {"query": "How much does an international wire transfer cost?", "relevant_source": "transfer_policy.md"},
    {"query": "What are the wire transfer fees?", "relevant_source": "account_fees.md"},
    # Cross-domain — "interest rate" matches savings APY AND loan APR
    {"query": "What interest rate will I get?", "relevant_source": "account_fees.md"},
    {"query": "What is the APR on a used car loan?", "relevant_source": "loan_policy.md"},
    # Confusing — "late fee" could match overdraft fee OR loan late payment fee
    {"query": "What happens if I'm late on a payment?", "relevant_source": "loan_policy.md"},
    # Overlapping term — "card replacement" is in account_fees.md AND fraud_policy.md
    {"query": "How much does a replacement debit card cost?", "relevant_source": "account_fees.md"},
    # Vague — "daily limit" is only in transfer_policy.md but could match account_fees.md
    {"query": "What are the daily transaction limits?", "relevant_source": "transfer_policy.md"},
    # Specific but tricky — "two-factor authentication" in both fraud_policy.md and transfer_policy.md
    {"query": "When is two-factor authentication required?", "relevant_source": "fraud_policy.md"},
]

print("=" * 60)
print("SEGMENT 8: MRR COMPUTATION")
print("=" * 60)

# YOUR CODE HERE
# For each query:
#   1. docs = retriever.invoke(query["query"])
#   2. Find rank of first doc where metadata["source"] == query["relevant_source"]
#   3. reciprocal_rank = 1/rank if found, else 0
#   4. Print query, rank, reciprocal rank
# Then compute MRR = mean of all reciprocal ranks

print("Complete TODO 6 to compute MRR.\n")


# ===================================================================
# SEGMENT 9: DeepEval
# ===================================================================

# ---------------------------------------------------------------------------
# TODO 7: Run DeepEval metrics
#
# pip install deepeval
#
# Create 5 test cases from the agent's actual outputs.
# Run: FaithfulnessMetric, AnswerRelevancyMetric, HallucinationMetric
#
# Example:
#   from deepeval.test_case import LLMTestCase
#   from deepeval.metrics import FaithfulnessMetric
#   from deepeval import assert_test
#
#   test_case = LLMTestCase(
#       input="What is the overdraft fee?",
#       actual_output="The overdraft fee is $35.",
#       retrieval_context=["...retrieved doc content..."]
#   )
#   faithfulness = FaithfulnessMetric(threshold=0.7)
#   assert_test(test_case, [faithfulness])
# ---------------------------------------------------------------------------
print("=" * 60)
print("SEGMENT 9: DEEPEVAL METRICS")
print("=" * 60)

# YOUR CODE HERE
# Step 1: Run 5 queries through the agent to get actual outputs + context
# Step 2: Create LLMTestCase objects
# Step 3: Run FaithfulnessMetric, AnswerRelevancyMetric, HallucinationMetric
# Step 4: Print results

print("Complete TODO 7 to run DeepEval metrics.\n")


# ===================================================================
# SEGMENT 10: G-Eval
# ===================================================================

# ---------------------------------------------------------------------------
# TODO 8: Build a G-Eval metric for empathy in escalation responses
#
# from deepeval.metrics import GEval
# from deepeval.test_case import LLMTestCaseParams
#
# Define a GEval metric with:
#   name: "Empathy"
#   criteria: describe what empathetic support looks like
#   evaluation_params: [LLMTestCaseParams.ACTUAL_OUTPUT]
#   threshold: 0.7
#
# Run on escalation test cases (frustrated customer queries).
# ---------------------------------------------------------------------------
print("=" * 60)
print("SEGMENT 10: G-EVAL (EMPATHY)")
print("=" * 60)

escalation_queries = [
    "This is ridiculous! Someone withdrew $15,000 from my savings without my permission!",
    "I've been waiting 3 weeks for my fraud dispute to be resolved! This is unacceptable!",
    "Your bank charged me $105 in overdraft fees in one day! I want to speak to a manager!",
]

# YOUR CODE HERE
# Step 1: Run escalation queries through agent
# Step 2: Create LLMTestCase objects
# Step 3: Create GEval metric for empathy
# Step 4: Evaluate and print results

print("Complete TODO 8 to run G-Eval empathy metric.")


# ===================================================================
# SEGMENT 11: Dataset Enhancement
# ===================================================================

# ---------------------------------------------------------------------------
# TODO 9: Add new evaluation examples to improve dataset coverage
#
# The base dataset (EVAL_EXAMPLES) has 15 examples. Real-world evaluation
# datasets grow over time as you discover failure modes.
#
# Add 3 new examples that test edge cases the current dataset misses:
#   a) A multi-part question (asks two things at once)
#   b) A question with a wrong/misspelled account number
#   c) A boundary-case policy question (e.g., exact threshold amounts)
#
# Steps:
#   1. Define the 3 new examples in the same format as EVAL_EXAMPLES
#   2. Upload them to the existing LangSmith dataset using client.create_examples()
#   3. Print confirmation
#
# Hint: Each example needs {"inputs": {"question": ...}, "outputs": {"answer": ..., "intent": ...}}
# ---------------------------------------------------------------------------
print("=" * 60)
print("SEGMENT 11: DATASET ENHANCEMENT")
print("=" * 60)

client = Client()

# YOUR CODE HERE
# new_examples = [
#     {"inputs": {"question": "..."}, "outputs": {"answer": "...", "intent": "..."}},
#     ...
# ]
# existing = list(client.list_datasets(dataset_name=DATASET_NAME))
# if existing:
#     client.create_examples(
#         inputs=[e["inputs"] for e in new_examples],
#         outputs=[e["outputs"] for e in new_examples],
#         dataset_id=existing[0].id,
#     )

print("Complete TODO 9 to add new examples to the dataset.\n")


# ===================================================================
# SEGMENT 12: Hill Climbing
# ===================================================================

# ---------------------------------------------------------------------------
# TODO 10: Hill climbing — improve correctness by changing ONE variable
#
# The demo showed hill climbing on keyword_correctness by changing chunk_size.
# Now you'll hill-climb on a CORRECTNESS evaluator by changing top_k.
#
# This uses a separate LangSmith dataset (fintech-hill-climb-eval) with
# 8 policy questions that require precise factual answers.
#
# IMPORTANT: Both agents use chunk_size=200 (tiny fragments) so that
# top_k actually matters. With large chunks, even top_k=1 has enough info.
# With tiny chunks, top_k=1 gets one incomplete fragment while top_k=5
# assembles a more complete picture.
#
# Steps:
#   1. Create the hill climbing dataset (provided — just call the helper)
#   2. Implement a correctness_evaluator (LLM-as-judge): compare the agent's
#      answer to the expected answer. Score 1.0 = all key facts correct,
#      0.5 = partial, 0.0 = wrong/missing facts.
#      Return: {"key": "correctness", "score": float}
#      Hint: Use judge_llm with a ChatPromptTemplate, parse JSON response.
#   3. Build a baseline agent with chunk_size=200, top_k=1
#   4. Run evaluate() with [routing_evaluator, keyword_correctness,
#      correctness_evaluator] on the hill-climb dataset.
#      Use experiment_prefix="hill-climb-topk1".
#   5. Build an improved agent with chunk_size=200, top_k=5
#   6. Run evaluate() with the same evaluators.
#      Use experiment_prefix="hill-climb-topk5".
#   7. Compare in LangSmith: Datasets → fintech-hill-climb-eval → Compare
#
# Why this works:
#   chunk_size=200: documents are split into tiny fragments (incomplete facts)
#   top_k=1 + small chunks: only ONE tiny fragment → LLM misses key details
#   top_k=5 + small chunks: FIVE fragments → more context → better correctness
# ---------------------------------------------------------------------------
print("=" * 60)
print("SEGMENT 12: HILL CLIMBING")
print("=" * 60)

# --- Create the hill climbing dataset ---
ensure_hill_climb_dataset_exists()

# --- Evaluators from the demo (provided) ---
def routing_evaluator_hc(run, example):
    predicted = run.outputs.get("intent", "")
    expected = example.outputs.get("intent", "")
    return {"key": "routing_accuracy", "score": 1.0 if predicted == expected else 0.0}

def keyword_correctness_hc(run, example):
    import re
    actual = run.outputs.get("answer", "").lower()
    expected = example.outputs.get("answer", "").lower()
    key_terms = re.findall(r"\$[\d,.]+|\d+(?:\.\d+)?%?|acc-\d+", expected)
    if not key_terms:
        return {"key": "keyword_correctness", "score": 0.5}
    matches = sum(1 for term in key_terms if term in actual)
    return {"key": "keyword_correctness", "score": matches / len(key_terms)}

# --- YOUR CODE: Implement correctness_evaluator ---
def correctness_evaluator_hc(run, example):
    actual = run.outputs.get("answer", "")
    expected = example.outputs.get("answer", "")
    question = example.inputs.get("question", "")

    # YOUR CODE HERE
    # 1. Build a ChatPromptTemplate that asks judge_llm to compare actual vs expected
    # 2. Score: 1.0 = all key facts correct, 0.5 = partial, 0.0 = wrong
    # 3. Parse JSON response: {"score": <float>, "reason": "<one sentence>"}
    # 4. Return: {"key": "correctness", "score": <float>}
    pass

# --- YOUR CODE: Build agents and run experiments ---
# Step 3: Build baseline (chunk_size=200, top_k=1)
# agent_v1 = build_support_agent(collection_name="hill_climb_v1", chunk_size=200, chunk_overlap=20, top_k=1)
# app_v1 = agent_v1["app"]
#
# def run_agent_v1(inputs):
#     result = ask(app_v1, inputs["question"])
#     return {"answer": result["response"], "intent": result["intent"],
#             "context": result["context"], "retrieved_sources": result["retrieved_sources"]}
#
# Step 4: Run baseline evaluate()
# evaluate(run_agent_v1, data=HILL_CLIMB_DATASET_NAME,
#          evaluators=[routing_evaluator_hc, keyword_correctness_hc, correctness_evaluator_hc],
#          experiment_prefix="hill-climb-topk1",
#          metadata={"chunk_size": 200, "top_k": 1})
#
# Step 5: Build improved (chunk_size=200, top_k=5)
# agent_v2 = build_support_agent(collection_name="hill_climb_v2", chunk_size=200, chunk_overlap=20, top_k=5)
# app_v2 = agent_v2["app"]
#
# def run_agent_v2(inputs): ...
#
# Step 6: Run improved evaluate()
# evaluate(run_agent_v2, data=HILL_CLIMB_DATASET_NAME,
#          evaluators=[routing_evaluator_hc, keyword_correctness_hc, correctness_evaluator_hc],
#          experiment_prefix="hill-climb-topk5",
#          metadata={"chunk_size": 200, "top_k": 5})

print("Complete TODO 10 to run hill climbing experiment.")
