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
from langsmith import Client

from eval_dataset import DATASET_NAME, EVAL_EXAMPLES, recreate_dataset
from eval_dataset import HILL_CLIMB_DATASET_NAME, recreate_hill_climb_dataset

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

# --- Recreate evaluation dataset (clean slate each run) ---
recreate_dataset()

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
    data=DATASET_NAME,
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

    # Run 3 queries — 2 clean passes + 1 that should fail
    eval_queries = [
        # Clean pass — straightforward single-doc lookup
        "What is the overdraft fee?",
        # Clean pass — clear policy question
        "What credit score do I need for a personal loan?",
        # Likely fail — vague query, retriever may pull wrong doc, agent may
        # hallucinate or deny having info despite context containing rates
        "What interest rate will I get?",
    ]

    test_cases = []
    for query in eval_queries:
        result = ask(app, query)
        ctx = [result["context"]] if result["context"] else ["No context retrieved."]
        test_cases.append(LLMTestCase(
            input=query,
            actual_output=result["response"],
            retrieval_context=ctx,
            context=ctx,
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
            if hasattr(metric, 'reason') and metric.reason:
                print(f"      Reason: {metric.reason[:120]}")

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


# ===================================================================
# SEGMENT 11: Dataset Enhancement
# ===================================================================

print("\n" + "=" * 60)
print("SEGMENT 11: DATASET ENHANCEMENT")
print("=" * 60)

# --- SOLUTION 9: Add edge-case examples ---
new_examples = [
    # Multi-part question (asks two things at once)
    {
        "inputs": {
            "question": "What is the overdraft fee and is there a daily limit?"
        },
        "outputs": {
            "answer": "The overdraft fee is $35 per transaction, with a maximum of 3 overdraft fees per day ($105 maximum).",
            "intent": "policy",
        },
    },
    # Misspelled / wrong account number
    {
        "inputs": {"question": "What is the balance on ACC-00000?"},
        "outputs": {
            "answer": "I couldn't find account ACC-00000 in our system.",
            "intent": "account_status",
        },
    },
    # Boundary-case policy question (exact threshold)
    {
        "inputs": {
            "question": "If I keep exactly $1,500 in my Premium Checking, do I still pay the monthly fee?"
        },
        "outputs": {
            "answer": "No, the monthly fee of $12.99 is waived if the daily balance stays above $1,500.",
            "intent": "policy",
        },
    },
]

client = Client()
existing = list(client.list_datasets(dataset_name=DATASET_NAME))
if existing:
    client.create_examples(
        inputs=[e["inputs"] for e in new_examples],
        outputs=[e["outputs"] for e in new_examples],
        dataset_id=existing[0].id,
    )
    print(f"Added {len(new_examples)} new edge-case examples to '{DATASET_NAME}'.")
else:
    print(f"Dataset '{DATASET_NAME}' not found — this shouldn't happen.")


# ===================================================================
# SEGMENT 12: Hill Climbing
# ===================================================================

print("\n" + "=" * 60)
print("SEGMENT 12: HILL CLIMBING (top_k=1 vs top_k=5)")
print("=" * 60)

# --- Recreate the hill climbing dataset (clean slate) ---
recreate_hill_climb_dataset()

# --- Evaluators from demo (provided) ---
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


# --- SOLUTION 10: Correctness evaluator ---
HC_CORRECTNESS_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are an expert evaluator. Compare the AI's answer to the expected answer.\n\n"
     "Score 1.0 = all key facts correct\n"
     "Score 0.5 = partially correct (some facts right, some missing or wrong)\n"
     "Score 0.0 = key facts wrong or missing\n\n"
     "Focus on factual accuracy: numbers, amounts, percentages, thresholds.\n"
     "Exact wording does not matter — only factual content.\n\n"
     'Respond ONLY with JSON: {{"score": <float>, "reason": "<one sentence>"}}'),
    ("human",
     "Question: {question}\n\nExpected answer: {expected}\n\nActual answer: {actual}"),
])


def correctness_evaluator_hc(run, example):
    actual = run.outputs.get("answer", "")
    expected = example.outputs.get("answer", "")
    question = example.inputs.get("question", "")

    if not actual or not expected:
        return {"key": "correctness", "score": 0.0}

    messages = HC_CORRECTNESS_PROMPT.format_messages(
        question=question, expected=expected, actual=actual
    )
    response = judge_llm.invoke(messages).content.strip()

    try:
        start, end = response.find("{"), response.rfind("}") + 1
        parsed = json.loads(response[start:end])
        score = float(parsed.get("score", 0.5))
        return {"key": "correctness", "score": score}
    except (json.JSONDecodeError, ValueError):
        return {"key": "correctness", "score": 0.5}


hc_evaluators = [routing_evaluator_hc, keyword_correctness_hc, correctness_evaluator_hc]

# --- SOLUTION 10a: Baseline (top_k=1, small chunks) ---
# With chunk_size=200, each chunk is a tiny fragment. top_k=1 returns only
# one fragment — often incomplete. top_k=5 assembles more context.
print("\nBuilding baseline agent (chunk_size=200, top_k=1)...")
agent_v1 = build_support_agent(collection_name="hill_climb_v1", chunk_size=200, chunk_overlap=20, top_k=1)
app_v1 = agent_v1["app"]


def run_agent_v1(inputs):
    result = ask(app_v1, inputs["question"])
    return {
        "answer": result["response"],
        "intent": result["intent"],
        "context": result["context"],
        "retrieved_sources": result["retrieved_sources"],
    }


print("Running baseline evaluation (chunk_size=200, top_k=1)...")
results_v1 = evaluate(
    run_agent_v1,
    data=HILL_CLIMB_DATASET_NAME,
    evaluators=hc_evaluators,
    experiment_prefix="hill-climb-topk1",
    metadata={"model": "gpt-4o-mini", "chunk_size": 200, "top_k": 1},
)

# --- SOLUTION 10b: Improved (top_k=5, same small chunks) ---
print("\nBuilding improved agent (chunk_size=200, top_k=5)...")
agent_v2 = build_support_agent(collection_name="hill_climb_v2", chunk_size=200, chunk_overlap=20, top_k=5)
app_v2 = agent_v2["app"]


def run_agent_v2(inputs):
    result = ask(app_v2, inputs["question"])
    return {
        "answer": result["response"],
        "intent": result["intent"],
        "context": result["context"],
        "retrieved_sources": result["retrieved_sources"],
    }


print("Running improved evaluation (chunk_size=200, top_k=5)...")
results_v2 = evaluate(
    run_agent_v2,
    data=HILL_CLIMB_DATASET_NAME,
    evaluators=hc_evaluators,
    experiment_prefix="hill-climb-topk5",
    metadata={"model": "gpt-4o-mini", "chunk_size": 200, "top_k": 5},
)

print("\n>>> Hill climbing complete.")
print(">>> Compare in LangSmith: Datasets → fintech-hill-climb-eval → select both → Compare")
print(">>> Watch the 'correctness' evaluator improve from top_k=1 → top_k=5.")
print(">>> The demo changed chunk_size; you just changed top_k on a new dataset.")

print("\n>>> All evaluation segments complete.")
print(">>> View results in LangSmith: https://smith.langchain.com")
