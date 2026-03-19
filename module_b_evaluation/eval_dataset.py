"""
Shared evaluation dataset for Module B.
Used by demo.py, exercise.py, and solution.py.

Each file gets its own LangSmith dataset so experiments never collide:
  - Demo:     fintech-demo-eval     + fintech-demo-hill-climb
  - Exercise: fintech-exercise-eval + fintech-exercise-hill-climb
  - Solution: fintech-solution-eval + fintech-solution-hill-climb
"""

from langsmith import Client

# ---------------------------------------------------------------------------
# Dataset names — each file has its own so experiments stay isolated
# ---------------------------------------------------------------------------
DEMO_DATASET_NAME = "fintech-demo-eval"
EXERCISE_DATASET_NAME = "fintech-exercise-eval"
SOLUTION_DATASET_NAME = "fintech-solution-eval"

DEMO_HC_DATASET_NAME = "fintech-demo-hill-climb"
EXERCISE_HC_DATASET_NAME = "fintech-exercise-hill-climb"
SOLUTION_HC_DATASET_NAME = "fintech-solution-hill-climb"

EVAL_EXAMPLES = [
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


def _ensure_dataset(dataset_name, examples, description, client=None):
    """Create a LangSmith dataset if it doesn't already exist."""
    if client is None:
        client = Client()
    existing = list(client.list_datasets(dataset_name=dataset_name))
    if existing:
        print(f"Dataset '{dataset_name}' already exists in LangSmith.")
        return existing[0]
    dataset = client.create_dataset(dataset_name=dataset_name, description=description)
    client.create_examples(
        inputs=[e["inputs"] for e in examples],
        outputs=[e["outputs"] for e in examples],
        dataset_id=dataset.id,
    )
    print(f"Created dataset '{dataset_name}' with {len(examples)} examples.")
    return dataset


_EVAL_DESC = (
    "Labeled evaluation examples for the FinTech multi-agent support system. "
    "Covers policy questions, account lookups, and escalation scenarios."
)
_HC_DESC = (
    "Policy-focused evaluation examples for hill climbing experiments. "
    "Questions require precise factual answers sensitive to retrieval quality."
)


def ensure_demo_dataset(client=None):
    return _ensure_dataset(DEMO_DATASET_NAME, EVAL_EXAMPLES, f"{_EVAL_DESC} (demo)", client)


def ensure_exercise_dataset(client=None):
    return _ensure_dataset(EXERCISE_DATASET_NAME, EVAL_EXAMPLES, f"{_EVAL_DESC} (exercise)", client)


def ensure_solution_dataset(client=None):
    return _ensure_dataset(SOLUTION_DATASET_NAME, EVAL_EXAMPLES, f"{_EVAL_DESC} (solution)", client)


def ensure_demo_hc_dataset(client=None):
    return _ensure_dataset(DEMO_HC_DATASET_NAME, HILL_CLIMB_EXAMPLES, f"{_HC_DESC} (demo)", client)


def ensure_exercise_hc_dataset(client=None):
    return _ensure_dataset(EXERCISE_HC_DATASET_NAME, HILL_CLIMB_EXAMPLES, f"{_HC_DESC} (exercise)", client)


def ensure_solution_hc_dataset(client=None):
    return _ensure_dataset(SOLUTION_HC_DATASET_NAME, HILL_CLIMB_EXAMPLES, f"{_HC_DESC} (solution)", client)


# ---------------------------------------------------------------------------
# Hill Climbing examples — used by exercise and solution hill climb datasets
# ---------------------------------------------------------------------------

HILL_CLIMB_EXAMPLES = [
    # Policy questions where factual correctness is sensitive to retrieval quality
    {
        "inputs": {"question": "What is the overdraft fee and the daily maximum?"},
        "outputs": {
            "answer": "The overdraft fee is $35 per transaction, with a maximum of 3 overdraft fees per day ($105 maximum).",
            "intent": "policy",
        },
    },
    {
        "inputs": {"question": "What are the conditions to waive the Premium Checking monthly fee?"},
        "outputs": {
            "answer": "The $12.99 monthly fee is waived if the daily balance stays above $1,500 or with a direct deposit of $500 or more per month.",
            "intent": "policy",
        },
    },
    {
        "inputs": {"question": "What is the cost of a domestic wire transfer?"},
        "outputs": {
            "answer": "A domestic outgoing wire transfer costs $25. Incoming domestic wires are free.",
            "intent": "policy",
        },
    },
    {
        "inputs": {"question": "What is the APR range for auto loans?"},
        "outputs": {
            "answer": "Auto loan APR ranges from 4.49% to 12.99% for new vehicles and 5.49% to 14.99% for used vehicles, depending on the term length and credit score.",
            "intent": "policy",
        },
    },
    {
        "inputs": {"question": "What credit score do I need for a personal loan and what are the APR ranges?"},
        "outputs": {
            "answer": "You need a credit score of 620 or higher. Personal loan APR ranges from 6.99% to 24.99% depending on creditworthiness.",
            "intent": "policy",
        },
    },
    {
        "inputs": {"question": "How long do I have to report unauthorized transactions and what is my liability?"},
        "outputs": {
            "answer": "You should report unauthorized transactions within 60 days of the statement date. Reporting within 2 business days limits your liability to $50.",
            "intent": "policy",
        },
    },
    {
        "inputs": {"question": "What is the international wire transfer fee?"},
        "outputs": {
            "answer": "An international outgoing wire transfer costs $45. Incoming international wires cost $15.",
            "intent": "policy",
        },
    },
    {
        "inputs": {"question": "What is the late payment fee for loans?"},
        "outputs": {
            "answer": "The late payment fee is $39 or 5% of the payment amount, whichever is greater, charged after a 15-day grace period.",
            "intent": "policy",
        },
    },
]
