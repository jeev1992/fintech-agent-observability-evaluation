"""
Module B Solution: Evaluation with LangSmith + DeepEval
---------------------------------------------------------
Full working solution: evaluators, MRR, DeepEval metrics, G-Eval.
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

# --- Build pipeline ---
print("Building FinTech support agent...")
agent = build_support_agent(collection_name="eval_solution")
app = agent["app"]
retriever = agent["retriever"]
judge_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
print("Pipeline ready.\n")

# ===================================================================
# SEGMENT 6: LangSmith Evaluators
# ===================================================================

# --- SOLUTION 1: run_agent ---
def run_agent(inputs):
    result = ask(app, inputs["question"])
    return {
        "answer": result["response"],
        "intent": result["intent"],
        "context": result["context"],
        "retrieved_sources": result["retrieved_sources"],
    }


# --- SOLUTION 2: Routing evaluator ---
def routing_evaluator(run, example):
    predicted = run.outputs.get("intent", "")
    expected = example.outputs.get("intent", "")
    score = 1.0 if predicted == expected else 0.0
    return {"key": "routing_accuracy", "score": score}


# --- SOLUTION 3: Faithfulness evaluator ---
FAITHFULNESS_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are an expert evaluator. Assess whether the answer is faithful "
     "to the provided context.\n\n"
     "Score 1.0 = fully faithful: every claim is supported by the context\n"
     "Score 0.5 = partially faithful: some claims are unsupported\n"
     "Score 0.0 = not faithful: contains claims contradicting or absent from context\n\n"
     "If the context is empty (escalation response), score 1.0 if the response "
     "is a general empathetic handoff without specific policy claims, else 0.5.\n\n"
     'Respond ONLY with JSON: {{"score": <float>, "reason": "<one sentence>"}}'),
    ("human",
     "Context:\n{context}\n\nQuestion: {question}\n\nAnswer to evaluate:\n{answer}"),
])


def faithfulness_evaluator(run, example):
    answer = run.outputs.get("answer", "")
    context = run.outputs.get("context", "")
    question = example.inputs.get("question", "")

    if not answer:
        return {"key": "faithfulness", "score": 0.0}

    messages = FAITHFULNESS_PROMPT.format_messages(
        context=context[:2000] if context else "(no context — escalation response)",
        question=question,
        answer=answer,
    )
    response = judge_llm.invoke(messages).content.strip()

    try:
        start, end = response.find("{"), response.rfind("}") + 1
        parsed = json.loads(response[start:end])
        score = float(parsed.get("score", 0.5))
        reason = parsed.get("reason", "")
        return {"key": "faithfulness", "score": score, "comment": reason}
    except (json.JSONDecodeError, ValueError):
        return {"key": "faithfulness", "score": 0.5}


# --- SOLUTION 4: Correctness evaluator ---
CORRECTNESS_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are an expert evaluator. Compare the AI's answer to the expected answer.\n\n"
     "Score 1.0 = all key facts correct\n"
     "Score 0.5 = partially correct\n"
     "Score 0.0 = key facts wrong or missing\n\n"
     "Focus on factual accuracy, not exact wording. "
     "For escalation responses, check that empathy and contact info are present.\n\n"
     'Respond ONLY with JSON: {{"score": <float>, "reason": "<one sentence>"}}'),
    ("human",
     "Question: {question}\n\nExpected: {expected}\n\nActual: {actual}"),
])


def correctness_evaluator(run, example):
    actual = run.outputs.get("answer", "")
    expected = example.outputs.get("answer", "")
    question = example.inputs.get("question", "")

    if not actual or not expected:
        return {"key": "correctness", "score": 0.0}

    messages = CORRECTNESS_PROMPT.format_messages(
        question=question, expected=expected, actual=actual
    )
    response = judge_llm.invoke(messages).content.strip()

    try:
        start, end = response.find("{"), response.rfind("}") + 1
        parsed = json.loads(response[start:end])
        score = float(parsed.get("score", 0.5))
        reason = parsed.get("reason", "")
        return {"key": "correctness", "score": score, "comment": reason}
    except (json.JSONDecodeError, ValueError):
        return {"key": "correctness", "score": 0.5}


# --- SOLUTION 5: Run evaluation ---
print("Running evaluation with all evaluators...")

results = evaluate(
    run_agent,
    data="fintech-agent-eval",
    evaluators=[routing_evaluator, faithfulness_evaluator, correctness_evaluator],
    experiment_prefix="fintech-eval-solution",
    metadata={"model": "gpt-4o-mini"},
)

print("\nEvaluation complete. View results in LangSmith.\n")


# ===================================================================
# SEGMENT 8: MRR
# ===================================================================

print("=" * 60)
print("SEGMENT 8: MRR COMPUTATION")
print("=" * 60)

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
    print(f"  Query: {item['query'][:50]:50s} | Rank: {rank} | RR: {rr:.2f}")

mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
print(f"\n  MRR = {mrr:.4f}")
print(f"  Interpretation: {'Excellent' if mrr > 0.9 else 'Good' if mrr > 0.7 else 'Needs improvement'}")


# ===================================================================
# SEGMENT 9: DeepEval
# ===================================================================

print("\n" + "=" * 60)
print("SEGMENT 9: DEEPEVAL METRICS")
print("=" * 60)

try:
    from deepeval.test_case import LLMTestCase
    from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric, HallucinationMetric
    from deepeval import assert_test

    # Run 5 queries to get actual outputs
    eval_queries = [
        "What is the overdraft fee?",
        "How much does a domestic wire transfer cost?",
        "What credit score do I need for a personal loan?",
        "How long do I have to report unauthorized transactions?",
        "What is the monthly fee for a Premium Checking account?",
    ]

    test_cases = []
    for query in eval_queries:
        result = ask(app, query)
        test_cases.append(LLMTestCase(
            input=query,
            actual_output=result["response"],
            retrieval_context=[result["context"]] if result["context"] else ["No context retrieved."],
        ))

    # Run metrics
    faithfulness = FaithfulnessMetric(threshold=0.7)
    relevancy = AnswerRelevancyMetric(threshold=0.7)
    hallucination = HallucinationMetric(threshold=0.7)

    for i, tc in enumerate(test_cases):
        print(f"\n  Test case {i+1}: {tc.input[:50]}...")
        for metric in [faithfulness, relevancy, hallucination]:
            metric.measure(tc)
            print(f"    {metric.__class__.__name__}: {metric.score:.2f} "
                  f"({'PASS' if metric.is_successful() else 'FAIL'})")

except ImportError:
    print("  DeepEval not installed. Run: pip install deepeval")
    print("  Skipping DeepEval metrics.")


# ===================================================================
# SEGMENT 10: G-Eval (Empathy)
# ===================================================================

print("\n" + "=" * 60)
print("SEGMENT 10: G-EVAL (EMPATHY)")
print("=" * 60)

try:
    from deepeval.metrics import GEval
    from deepeval.test_case import LLMTestCaseParams, LLMTestCase

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

    escalation_queries = [
        "This is ridiculous! Someone withdrew $15,000 from my savings without my permission!",
        "I've been waiting 3 weeks for my fraud dispute to be resolved! This is unacceptable!",
        "Your bank charged me $105 in overdraft fees in one day! I want to speak to a manager!",
    ]

    for query in escalation_queries:
        result = ask(app, query)
        tc = LLMTestCase(
            input=query,
            actual_output=result["response"],
        )
        empathy_metric.measure(tc)
        print(f"\n  Query: {query[:60]}...")
        print(f"  Empathy Score: {empathy_metric.score:.2f} "
              f"({'PASS' if empathy_metric.is_successful() else 'FAIL'})")
        if hasattr(empathy_metric, 'reason') and empathy_metric.reason:
            print(f"  Reason: {empathy_metric.reason[:100]}")

except ImportError:
    print("  DeepEval not installed. Run: pip install deepeval")
    print("  Skipping G-Eval empathy metric.")

print("\n>>> All evaluation segments complete.")
print(">>> View results in LangSmith: https://smith.langchain.com")
