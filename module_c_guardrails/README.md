# Module C — Output Guardrails (`module_c_guardrails`)

This module is about enforcement, not suggestion. Prompts say "don't leak SSNs." Guardrails *guarantee* it.

The goal is to wrap the multi-agent system with input and output validation — PII redaction before the LLM sees it, content validation before the user sees it.

---

## What this module teaches

> "Prompts suggest. Guardrails enforce."

Our FinTech agent has access to sensitive data — SSNs, account balances, transaction histories. Without guardrails:

- A prompt injection could extract SSN data
- The LLM could hallucinate financial advice not in our policies
- The agent could mention competitor banks in its response

This module adds three layers of protection:

| Layer | Tool | What it catches |
|-------|------|----------------|
| PII detection & redaction | Microsoft Presidio | SSNs, credit cards, emails, names in input/output |
| Pattern matching | Guardrails AI `RegexMatch` | Known PII patterns (deterministic, fast) |
| Semantic validation | Guardrails AI `ToxicLanguage`, `CompetitorCheck` | Toxic content, competitor mentions |

---

## File breakdown

### `demo.py` — Before and after

Shows the same queries run **without** guardrails (potential PII leaks, competitor mentions) and **with** a simple regex-based output guard that blocks them.

**What to watch for when you run it:**
- Does the agent try to answer "What is the SSN for account ACC-12345?"
- Does it mention competitor banks when asked to compare?
- How does the "after" version handle the same queries?

### `exercise.py` — Build production guardrails

Five TODOs that build up from simple to full pipeline:

| TODO | What you do |
|------|-------------|
| 1 | Set up Guardrails AI `Guard` with `RegexMatch` for SSN patterns |
| 2 | Add `ToxicLanguage` and `CompetitorCheck` validators (Chase, Wells Fargo, Citi, etc.) |
| 3 | Integrate the guard into the agent pipeline (`safe_pipeline()`) |
| 4 | Set up Microsoft Presidio for PII detection — test on sample strings |
| 5 | Build full `guarded_pipeline()`: PII redaction on input → agent → guard validation → PII redaction on output |

### `solution.py` — Reference implementation

Complete working code with all 5 TODOs solved. Gracefully handles missing dependencies — if Guardrails AI or Presidio aren't installed, it prints install instructions and skips those sections.

### `notes.md` — Concepts

Covers: guardrails vs prompt instructions, complexity spectrum (regex → schema → LLM-based), Guardrails AI architecture, Presidio for broad PII detection, and GDPR/HIPAA engineering patterns (DPA, BAA, data minimization).

---

## How to run

You need Guardrails AI and Presidio installed:

```bash
# Install dependencies
pip install guardrails-ai presidio-analyzer presidio-anonymizer

# Install Guardrails Hub validators
guardrails hub install hub://guardrails/regex_match
guardrails hub install hub://guardrails/toxic_language
guardrails hub install hub://guardrails/competitor_check
```

```bash
# Run from the project root directory

# Part 1: Watch the before/after demo
python module_c_guardrails/demo.py

# Part 2: Build guardrails yourself
python module_c_guardrails/exercise.py

# Part 3: Check against the solution
python module_c_guardrails/solution.py
```

**What to expect from `demo.py`:**
- Runs 3 unsafe queries (injection, out-of-scope, competitor comparison)
- Shows raw agent responses (BEFORE)
- Shows guarded responses (AFTER) — unsafe ones are blocked with a fallback message

**What to expect from `solution.py`:**
- Tests RegexMatch on SSN patterns (pass/block)
- Tests CompetitorCheck on competitor mentions
- Runs Presidio on PII-laden strings (shows BEFORE/AFTER redaction)
- Runs the full guarded pipeline on 3 test queries

---

## Quick mental model

- Use the **lightest check that works**: regex (fast, free) → schema (structural) → LLM-based (flexible, costs tokens).
- **Presidio** catches broad PII (names, addresses, medical IDs). **RegexMatch** catches known patterns (SSN, credit card). Use both.
- Apply guards on **both sides**: redact PII from the input before the LLM sees it, validate the output before the user sees it.
- Sending PII to LLM APIs requires a **DPA** (GDPR) or **BAA** (HIPAA) with the provider. Most engineers don't realize this.
