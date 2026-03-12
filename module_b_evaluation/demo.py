"""
Module B Demo: Evaluation with LangSmith + DeepEval
------------------------------------------------------
Demonstrates the full evaluation pipeline for a multi-agent system:
dataset creation, custom evaluators, A/B comparison, MRR, DeepEval, G-Eval.

Segments covered:
  5. Dataset creation & upload
  7. A/B experiment comparison
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from langsmith import Client
from langsmith.evaluation import evaluate

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
DATASET_NAME = "fintech-agent-eval"

existing = list(client.list_datasets(dataset_name=DATASET_NAME))
if existing:
    print(f"Dataset '{DATASET_NAME}' already exists. Deleting and recreating...")
    client.delete_dataset(dataset_id=existing[0].id)

dataset = client.create_dataset(
    dataset_name=DATASET_NAME,
    description=(
        "Labeled evaluation examples for the FinTech multi-agent support system. "
        "Covers policy questions, account lookups, and escalation scenarios."
    ),
)
print(f"Created dataset: '{DATASET_NAME}' (id: {dataset.id})")

# ---------------------------------------------------------------------------
# 3. Define evaluation examples
# ---------------------------------------------------------------------------
examples = [
    # --- Policy: Account Fees ---
    {
        "inputs": {"question": "What is the overdraft fee?"},
        "outputs": {
            "answer": "The overdraft fee is $35 per transaction, with a maximum of 3 overdraft fees per day ($105 maximum).",
            "intent": "policy",
        },
    },
    {
        "inputs": {"question": "What is the monthly fee for a Premium Checking account?"},
        "outputs": {
            "answer": "The Premium Checking account has a monthly fee of $12.99, which is waived if the daily balance stays above $1,500 or with a direct deposit of $500 or more per month.",
            "intent": "policy",
        },
    },
    {
        "inputs": {"question": "How much does it cost to use an out-of-network ATM?"},
        "outputs": {
            "answer": "Out-of-network ATM transactions cost $3.00 per transaction. The ATM owner may also charge an additional fee.",
            "intent": "policy",
        },
    },
    # --- Policy: Loans ---
    {
        "inputs": {"question": "What credit score do I need for a personal loan?"},
        "outputs": {
            "answer": "You need a credit score of 620 or higher to qualify for a personal loan.",
            "intent": "policy",
        },
    },
    {
        "inputs": {"question": "Is there a prepayment penalty on personal loans?"},
        "outputs": {
            "answer": "No, there is no prepayment penalty on personal loans at SecureBank.",
            "intent": "policy",
        },
    },
    # --- Policy: Transfers ---
    {
        "inputs": {"question": "How much does a domestic wire transfer cost?"},
        "outputs": {
            "answer": "A domestic outgoing wire transfer costs $25. Incoming domestic wires are free.",
            "intent": "policy",
        },
    },
    {
        "inputs": {"question": "Can I cancel a wire transfer?"},
        "outputs": {
            "answer": "Wire transfers cannot be reversed once sent. Contact us immediately if sent in error; recall requests are not guaranteed and may take 2 to 4 weeks.",
            "intent": "policy",
        },
    },
    # --- Policy: Fraud ---
    {
        "inputs": {"question": "How long do I have to report unauthorized transactions?"},
        "outputs": {
            "answer": "You should report unauthorized transactions within 60 days of the statement date. Reporting within 2 business days limits your liability to $50.",
            "intent": "policy",
        },
    },
    # --- Account Status ---
    {
        "inputs": {"question": "What is the balance on ACC-12345?"},
        "outputs": {
            "answer": "Account ACC-12345 (Premium Checking) has a balance of $12,450.75 and is active.",
            "intent": "account_status",
        },
    },
    {
        "inputs": {"question": "Show me recent transactions for ACC-67890."},
        "outputs": {
            "answer": "Account ACC-67890 recent transactions include a debit card purchase at a grocery store for $67.30 on March 14, a direct deposit of $1,500 on March 11, and a bill pay of $145.00 on March 10.",
            "intent": "account_status",
        },
    },
    {
        "inputs": {"question": "What's the status of ACC-11111?"},
        "outputs": {
            "answer": "Account ACC-11111 is currently frozen due to suspected unauthorized activity and is under fraud review.",
            "intent": "account_status",
        },
    },
    {
        "inputs": {"question": "What is the balance on ACC-99999?"},
        "outputs": {
            "answer": "I couldn't find account ACC-99999 in our system.",
            "intent": "account_status",
        },
    },
    # --- Escalation ---
    {
        "inputs": {
            "question": "This is ridiculous! Someone withdrew $15,000 from my savings without my permission!"
        },
        "outputs": {
            "answer": "I sincerely apologize for this alarming situation. A senior fraud specialist will follow up. Contact fraud@securebank.com or 1-800-555-0199 option 1.",
            "intent": "escalation",
        },
    },
    {
        "inputs": {"question": "I want to speak to a manager. Your fees are outrageous!"},
        "outputs": {
            "answer": "I'm sorry for the frustration. I'm escalating this to a senior specialist who will reach out shortly.",
            "intent": "escalation",
        },
    },
    # --- Out of scope ---
    {
        "inputs": {"question": "What stock should I invest in?"},
        "outputs": {
            "answer": "I'm sorry, I don't have information about that in our current policies. Please contact our support team at support@securebank.com for further assistance.",
            "intent": "policy",
        },
    },
]

# Upload examples
client.create_examples(
    inputs=[e["inputs"] for e in examples],
    outputs=[e["outputs"] for e in examples],
    dataset_id=dataset.id,
)
print(f"Uploaded {len(examples)} examples to '{DATASET_NAME}'.\n")

# ---------------------------------------------------------------------------
# 4. Build the agent pipeline
# ---------------------------------------------------------------------------
print("Building FinTech support agent...")
agent = build_support_agent(collection_name="eval_demo")
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
print("Running Experiment A (baseline)...")
results_a = evaluate(
    run_agent,
    data=DATASET_NAME,
    evaluators=[routing_evaluator, keyword_correctness],
    experiment_prefix="fintech-v1-baseline",
    metadata={"model": "gpt-4o-mini", "version": "baseline"},
)

print("\n>>> Experiment A complete. View results in LangSmith.\n")

# ---------------------------------------------------------------------------
# 8. SEGMENT 7: Run Experiment B (same agent — in practice you'd change
#    the prompt, model, or retrieval config)
# ---------------------------------------------------------------------------
# In a real A/B test, you'd modify the agent. Here we demonstrate the workflow.
print("Running Experiment B (same agent — compare in LangSmith)...")
results_b = evaluate(
    run_agent,
    data=DATASET_NAME,
    evaluators=[routing_evaluator, keyword_correctness],
    experiment_prefix="fintech-v2-comparison",
    metadata={"model": "gpt-4o-mini", "version": "comparison"},
)

print("\n>>> Experiment B complete.")
print(">>> Open LangSmith → Datasets → fintech-agent-eval → Compare Experiments")
print(">>> See side-by-side scores for v1-baseline vs v2-comparison.")
