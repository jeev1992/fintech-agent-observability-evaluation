"""
Shared evaluation dataset for Module B.
Used by demo.py, exercise.py, and solution.py.

Provides:
  - DATASET_NAME: the LangSmith dataset name
  - EVAL_EXAMPLES: the 15 labeled evaluation examples
  - ensure_dataset_exists(client): create the dataset in LangSmith if it doesn't exist
"""

from langsmith import Client

DATASET_NAME = "fintech-agent-eval"

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


def ensure_dataset_exists(client=None):
    """Create the evaluation dataset in LangSmith if it doesn't already exist.

    Returns the dataset object.
    """
    if client is None:
        client = Client()

    existing = list(client.list_datasets(dataset_name=DATASET_NAME))
    if existing:
        print(f"Dataset '{DATASET_NAME}' already exists in LangSmith.")
        return existing[0]

    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
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
    print(f"Created dataset '{DATASET_NAME}' with {len(EVAL_EXAMPLES)} examples.")
    return dataset


def recreate_dataset(client=None):
    """Delete and recreate the evaluation dataset. Returns the dataset object."""
    if client is None:
        client = Client()

    existing = list(client.list_datasets(dataset_name=DATASET_NAME))
    if existing:
        client.delete_dataset(dataset_id=existing[0].id)
        print(f"Deleted existing dataset '{DATASET_NAME}'.")

    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
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
    print(f"Created dataset '{DATASET_NAME}' with {len(EVAL_EXAMPLES)} examples.")
    return dataset


# ---------------------------------------------------------------------------
# Hill Climbing dataset — separate from the main eval dataset so exercise/
# solution experiments don't pollute the demo's dataset.
# ---------------------------------------------------------------------------
HILL_CLIMB_DATASET_NAME = "fintech-hill-climb-eval"

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
            "answer": "Auto loan APR ranges from 3.49% to 7.99% depending on the term length and credit score.",
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


def ensure_hill_climb_dataset_exists(client=None):
    """Create the hill climbing dataset in LangSmith if it doesn't already exist."""
    if client is None:
        client = Client()

    existing = list(client.list_datasets(dataset_name=HILL_CLIMB_DATASET_NAME))
    if existing:
        print(f"Dataset '{HILL_CLIMB_DATASET_NAME}' already exists in LangSmith.")
        return existing[0]

    dataset = client.create_dataset(
        dataset_name=HILL_CLIMB_DATASET_NAME,
        description=(
            "Policy-focused evaluation examples for hill climbing experiments. "
            "Questions require precise factual answers sensitive to retrieval quality."
        ),
    )
    client.create_examples(
        inputs=[e["inputs"] for e in HILL_CLIMB_EXAMPLES],
        outputs=[e["outputs"] for e in HILL_CLIMB_EXAMPLES],
        dataset_id=dataset.id,
    )
    print(f"Created dataset '{HILL_CLIMB_DATASET_NAME}' with {len(HILL_CLIMB_EXAMPLES)} examples.")
    return dataset


def recreate_hill_climb_dataset(client=None):
    """Delete and recreate the hill climbing dataset. Returns the dataset object."""
    if client is None:
        client = Client()

    existing = list(client.list_datasets(dataset_name=HILL_CLIMB_DATASET_NAME))
    if existing:
        client.delete_dataset(dataset_id=existing[0].id)
        print(f"Deleted existing dataset '{HILL_CLIMB_DATASET_NAME}'.")

    dataset = client.create_dataset(
        dataset_name=HILL_CLIMB_DATASET_NAME,
        description=(
            "Policy-focused evaluation examples for hill climbing experiments. "
            "Questions require precise factual answers sensitive to retrieval quality."
        ),
    )
    client.create_examples(
        inputs=[e["inputs"] for e in HILL_CLIMB_EXAMPLES],
        outputs=[e["outputs"] for e in HILL_CLIMB_EXAMPLES],
        dataset_id=dataset.id,
    )
    print(f"Created dataset '{HILL_CLIMB_DATASET_NAME}' with {len(HILL_CLIMB_EXAMPLES)} examples.")
    return dataset
