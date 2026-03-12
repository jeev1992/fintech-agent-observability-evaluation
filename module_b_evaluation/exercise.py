"""
Module B Exercise: Evaluation with LangSmith + DeepEval
---------------------------------------------------------
Implement evaluators, compute MRR, use DeepEval metrics, and build
a G-Eval custom metric for the FinTech multi-agent system.

Segments covered:
  6.  LangSmith evaluators (custom + LLM-as-judge)
  8.  MRR (Mean Reciprocal Rank)
  9.  DeepEval (faithfulness, hallucination, answer relevancy)
  10. G-Eval (custom criteria — empathy)
"""

import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langsmith.evaluation import evaluate

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
# Use data="fintech-agent-eval", experiment_prefix="fintech-eval-student"
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
    {"query": "What is the overdraft fee?", "relevant_source": "account_fees.md"},
    {"query": "What credit score do I need for a personal loan?", "relevant_source": "loan_policy.md"},
    {"query": "How much does a domestic wire transfer cost?", "relevant_source": "transfer_policy.md"},
    {"query": "How long do I have to report fraud?", "relevant_source": "fraud_policy.md"},
    {"query": "What is the interest rate on a High-Yield Savings account?", "relevant_source": "account_fees.md"},
    {"query": "What is the late payment fee for loans?", "relevant_source": "loan_policy.md"},
    {"query": "Can I reverse a wire transfer?", "relevant_source": "transfer_policy.md"},
    {"query": "What is the maximum overdraft fees per day?", "relevant_source": "account_fees.md"},
    {"query": "What is the APR range for auto loans?", "relevant_source": "loan_policy.md"},
    {"query": "How do I report identity theft?", "relevant_source": "fraud_policy.md"},
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
