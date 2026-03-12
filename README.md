# Agent Observability, Evaluation & Safety Workshop
## Learn LangSmith · DeepEval · Guardrails AI · Presidio

A **4-hour hands-on workshop** teaching production-grade AI agent hardening through a concrete, runnable project: a multi-agent FinTech customer support system for SecureBank.

---

## What You'll Learn

| Concept | What It Is | How We Use It |
|---|---|---|
| **LangSmith** | LLM observability & evaluation platform | Trace every LLM call, debug failures, run A/B experiments |
| **DeepEval** | Open-source evaluation framework (pytest-native) | Faithfulness, hallucination, answer relevancy, G-Eval metrics |
| **Guardrails AI** | Output validation framework with 50+ validators | RegexMatch (SSN), ToxicLanguage, CompetitorCheck |
| **Presidio** | Microsoft's PII detection & redaction engine | Detect and redact SSNs, names, emails, credit cards |
| **MRR** | Mean Reciprocal Rank — retrieval quality metric | Measure how quickly the retriever surfaces the right document |
| **tiktoken** | OpenAI's token counting library | Count tokens locally, measure cost per query |

---

## The Scenario

**Company**: SecureBank (FinTech customer support)

A customer sends a question. The multi-agent system classifies intent, routes to the right specialist agent, and responds — all traced, evaluated, guarded, and cost-optimized.

| Agent | Role | Data Source | Example Query |
|---|---|---|---|
| **Supervisor** | Classifies intent → routes query | LLM classification | *(internal — runs on every query)* |
| **Policy Agent** | Answers policy questions via RAG | 4 policy documents (Chroma) | "What is the overdraft fee?" |
| **Account Agent** | Looks up account details | Mock database (3 accounts) | "What is the balance on ACC-12345?" |
| **Escalation Agent** | Empathetic handoff for complaints | LLM generation (no context) | "I'm furious! Nobody is helping!" |

### Mock Accounts

```
ACC-12345: Alice Johnson, Premium Checking, $12,450.75, active
ACC-67890: Bob Smith, Basic Checking, $234.50, active
ACC-11111: Carol Davis, High-Yield Savings, $85,320.00, FROZEN (fraud review)
```

---

## Project Structure

Modules are lettered in teaching order. Each module introduces new production hardening techniques applied to the same FinTech agent.

```
Week 8/
│
├── project/                               # THE AGENT — pre-built, shared by all modules
│   ├── fintech_support_agent.py           # Multi-agent system (supervisor + 3 agents)
│   └── documents/                         # Policy knowledge base (RAG source)
│       ├── account_fees.md                # Account types, overdraft fees, ATM fees
│       ├── loan_policy.md                 # Personal/auto loans, rates, eligibility
│       ├── fraud_policy.md                # Fraud detection, disputes, liability
│       └── transfer_policy.md             # Wire transfers, ACH, P2P, limits
│
├── module_a_observability/                # MODULE A — See what's happening
│   ├── README.md                          # Module guide for learners
│   ├── notes.md                           # Reference: observability, traces, monitoring
│   ├── demo.py                            # Silent failure demo + tagging runs
│   ├── exercise.py                        # Setup tracing, inspect traces, debug errors
│   └── solution.py                        # Complete solution
│
├── module_b_evaluation/                   # MODULE B — Measure quality
│   ├── README.md                          # Module guide for learners
│   ├── notes.md                           # Reference: datasets, evaluators, MRR, DeepEval, G-Eval
│   ├── demo.py                            # Dataset creation + A/B experiments
│   ├── exercise.py                        # Evaluators, MRR, DeepEval metrics, G-Eval empathy
│   └── solution.py                        # Complete solution
│
├── module_c_guardrails/                   # MODULE C — Prevent bad outputs
│   ├── README.md                          # Module guide for learners
│   ├── notes.md                           # Reference: guardrails spectrum, Presidio, compliance
│   ├── demo.py                            # Before/after guardrail demo
│   ├── exercise.py                        # Guardrails AI validators + Presidio PII redaction
│   └── solution.py                        # Complete solution
│
├── module_d_cost_optimization/            # MODULE D — Optimize cost
│   ├── README.md                          # Module guide for learners
│   ├── notes.md                           # Reference: token economics, optimization patterns
│   ├── demo.py                            # Before/after cost comparison
│   ├── exercise.py                        # Token counting, cost measurement, comparison table
│   └── solution.py                        # Complete solution
│
├── .env.example                           # Copy to .env and add your API keys
└── requirements.txt                       # All Python dependencies
```

If module files feel overwhelming, start with the README inside each module folder.

### Notes live inside each module

Each module has a `notes.md` file with detailed reference documentation for that module's concepts.

| Module | Notes | Topics Covered |
|---|---|---|
| `module_a_observability/notes.md` | Observability deep dive | Traces, runs, run trees, debugging walkthrough, monitoring, sampling |
| `module_b_evaluation/notes.md` | Evaluation deep dive | Datasets, evaluators, MRR formula, DeepEval vs LangSmith, G-Eval criteria |
| `module_c_guardrails/notes.md` | Guardrails deep dive | Regex→Schema→LLM spectrum, Guardrails AI, Presidio, GDPR/HIPAA patterns |
| `module_d_cost_optimization/notes.md` | Cost deep dive | Token economics, cost structure, before/after methodology, 4 optimization patterns |

---

## Quick Start

### 1. Prerequisites

- Python 3.10+
- An OpenAI API key
- A free LangSmith account

**Verify installation:**
```bash
python --version  # should be 3.10+
```

### 2. Clone or open this repo

```bash
git clone <your-repo-url>
cd "Week 8"
```

### 3. Create a virtual environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
```

**macOS/Linux:**
```bash
python3 -m venv .venv
```

### 4. Activate the virtual environment

**Windows (PowerShell):**
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass; .\.venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```bat
.venv\Scripts\activate.bat
```

**macOS/Linux:**
```bash
source .venv/bin/activate
```

> **Tip**: Your prompt will change to show `(.venv)` when the environment is active.

### 5. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 6. Configure API keys

**Windows (PowerShell):**
```powershell
Copy-Item .env.example .env
```

**macOS/Linux:**
```bash
cp .env.example .env
```

Then edit `.env` and set:
```env
OPENAI_API_KEY=sk-your-key-here
LANGCHAIN_API_KEY=lsv2_pt_your-langsmith-key-here
LANGCHAIN_TRACING_V2=true
```

Get your LangSmith API key at [smith.langchain.com](https://smith.langchain.com) (free Developer plan — 1 seat, 5K traces/month).

### 7. Install Guardrails Hub validators (Module C)

```bash
pip install guardrails-ai presidio-analyzer presidio-anonymizer
guardrails hub install hub://guardrails/regex_match
guardrails hub install hub://guardrails/toxic_language
guardrails hub install hub://guardrails/competitor_check
```

### 8. Run a smoke test

```bash
python module_a_observability/demo.py
```

If Module A runs and you see traces in LangSmith, your environment is ready.

### 9. Run the workshop modules in order

```bash
# MODULE A: Observability — trace the agent, debug silent failures
python module_a_observability/demo.py        # Instructor demo
python module_a_observability/exercise.py    # Your turn

# MODULE B: Evaluation — measure quality with LangSmith + DeepEval
python module_b_evaluation/demo.py           # Create dataset + A/B experiments
python module_b_evaluation/exercise.py       # Evaluators, MRR, DeepEval, G-Eval

# MODULE C: Guardrails — prevent PII leaks and unsafe outputs
python module_c_guardrails/demo.py           # Before/after demo
python module_c_guardrails/exercise.py       # Guardrails AI + Presidio

# MODULE D: Cost Optimization — measure and reduce token costs
python module_d_cost_optimization/demo.py    # Before/after cost comparison
python module_d_cost_optimization/exercise.py # Build your own comparison
```

---

## Architecture Deep Dive

### Multi-Agent Flow

```
python module_a_observability/demo.py
    │
    └── project/fintech_support_agent.py
          │
          ├── LangGraph: START → classify_intent → [policy | account | escalation] → END
          │
          ├── classify_intent (supervisor)
          │     ├── ChatOpenAI(model="gpt-4o-mini")
          │     ├── System prompt: "Classify into policy/account_status/escalation"
          │     └── Returns intent string → routes to specialist
          │
          ├── policy_agent (RAG)
          │     ├── Chroma retriever → top-k document chunks
          │     ├── format_docs → concatenate retrieved context
          │     └── ChatOpenAI → answer grounded in policy docs
          │
          ├── account_agent (DB lookup)
          │     ├── Regex extract ACC-XXXXX from query
          │     ├── MOCK_ACCOUNTS dict lookup
          │     └── ChatOpenAI → format account data as response
          │
          └── escalation_agent (empathetic handoff)
                ├── ChatOpenAI → empathetic response
                └── No context retrieval — just empathy + contact info
```

### Full Production Pipeline (After All Modules)

```
Customer Query
    │
    ▼
INPUT GUARDRAIL (Module C)
    ├── Presidio: redact PII from query
    └── Content safety: check for harmful input
    │
    ▼
MULTI-AGENT GRAPH (project/)     ──────►  LANGSMITH TRACING (Module A)
    ├── Supervisor: classify intent               │
    ├── Policy Agent: RAG generation              ├── Every run traced
    ├── Account Agent: DB lookup                  ├── Token counts captured
    └── Escalation Agent: handoff                 └── Latency measured
    │
    ▼
OUTPUT GUARDRAIL (Module C)
    ├── Guardrails AI: RegexMatch, ToxicLanguage, CompetitorCheck
    └── Presidio: redact any PII in response
    │
    ▼
Response to User

CONTINUOUS EVALUATION (Module B)          COST MONITORING (Module D)
    ├── Curate failing traces              ├── Token counting (tiktoken)
    ├── Evaluation datasets                ├── Before/after comparison
    ├── LangSmith evaluators               └── Projected savings
    ├── DeepEval (CI/CD)
    └── G-Eval (custom criteria)
```

### Cost Structure (Per Query)

```
POLICY QUERY: "What is the overdraft fee?"
  Supervisor:    ~100 prompt + ~3 completion tokens     ← cheap, runs every query
  Policy Agent:  ~800-1,500 prompt + ~50-150 completion ← expensive (RAG context)
  TOTAL:         ~$0.0003 (GPT-4o-mini)

ACCOUNT QUERY: "What is the balance on ACC-12345?"
  Supervisor:    ~100 prompt + ~3 completion tokens
  Account Agent: ~300 prompt + ~70 completion tokens
  TOTAL:         ~$0.0001 (GPT-4o-mini)

ESCALATION: "I'm furious! Nobody is helping!"
  Supervisor:       ~100 prompt + ~3 completion tokens
  Escalation Agent: ~120 prompt + ~100 completion tokens
  TOTAL:            ~$0.0001 (GPT-4o-mini)
```

---

## Workshop Schedule (4 Hours)

| Time | Module | Topic | Key Files |
|---|---|---|---|
| 0:00–0:10 | Intro | What we're building, setup check | `README.md`, `.env` |
| 0:10–1:15 | Module A | Agent observability with LangSmith | `module_a_observability/` |
| 1:15–1:20 | Break | | |
| 1:20–2:50 | Module B | Evaluation: datasets, MRR, DeepEval, G-Eval | `module_b_evaluation/` |
| 2:50–3:00 | Break | | |
| 3:00–3:50 | Module C | Guardrails AI + Presidio PII redaction | `module_c_guardrails/` |
| 3:50–4:00 | Module D | Cost optimization (before/after) + wrap-up | `module_d_cost_optimization/` |

---

## Key Files Reference

| File | Key Function | What It Does |
|---|---|---|
| `project/fintech_support_agent.py` | `build_support_agent()` | Builds LangGraph multi-agent system (supervisor + 3 agents) |
| `project/fintech_support_agent.py` | `ask(app, query)` | Helper to invoke the agent with a query |
| `project/fintech_support_agent.py` | `MOCK_ACCOUNTS` | 3 mock bank accounts (ACC-12345, ACC-67890, ACC-11111) |
| `project/fintech_support_agent.py` | `SupportState` | TypedDict defining state flowing through the graph |
| `project/documents/*.md` | — | 4 policy documents embedded into Chroma for RAG |
| `module_b_evaluation/demo.py` | `evaluate()` | Creates LangSmith dataset and runs A/B experiments |
| `module_c_guardrails/solution.py` | `guarded_pipeline()` | Full pipeline: PII redact → agent → guard → PII redact |
| `module_d_cost_optimization/demo.py` | `measure_cost()` | Runs queries and captures token usage for before/after |

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'langchain'`**
```bash
pip install -r requirements.txt
```

**`AuthenticationError` from OpenAI**
```bash
# Check your .env file has the correct key
cat .env  # or: type .env (Windows)
```

**LangSmith traces not appearing**
```bash
# Ensure these are set in your .env:
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_pt_your-key
```

**`ModuleNotFoundError: No module named 'deepeval'`**
```bash
pip install deepeval
```

**Guardrails Hub validators not found**
```bash
pip install guardrails-ai
guardrails hub install hub://guardrails/regex_match
guardrails hub install hub://guardrails/toxic_language
guardrails hub install hub://guardrails/competitor_check
```

**Presidio not installed**
```bash
pip install presidio-analyzer presidio-anonymizer
```

**PowerShell `UnauthorizedAccess` error activating venv**
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass; .\.venv\Scripts\Activate.ps1
```

**`FileNotFoundError` running module scripts**
```bash
# Run from the Week 8/ directory, not from inside a module folder
cd "Week 8"
python module_a_observability/demo.py
```

---

*Built for the AI Agent Systems Workshop — teaching production hardening through a FinTech multi-agent support system.*
