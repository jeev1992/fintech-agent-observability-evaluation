# Module C — Input & Output Guardrails (`module_c_guardrails`)

This module is about enforcement, not suggestion. Prompts say "don't leak SSNs." Guardrails *guarantee* it.

Guardrails work at **two levels**:
- **Input**: Block dangerous queries BEFORE the LLM call (saves cost)
- **Output**: Validate/redact the response BEFORE the user sees it

Four implementation strategies, each catching what the previous can't:

| Strategy | Tool | Cost | Catches |
|----------|------|------|---------|
| 1. Regex | `re.search`, Guardrails AI `RegexMatch` | $0, ~1ms | SSN patterns, credit cards, competitor keywords, harmful content |
| 2. Moderation | OpenAI Moderation API | $0 (free), ~100ms | Violence, self-harm, hate, harassment — catches intent, not keywords |
| 3. ML/NER | Microsoft Presidio | $0 (local), ~10ms | Names, emails, addresses, phone numbers — things regex can't match |
| 4. LLM-based | GPT classifier, Guardrails AI `CompetitorCheck`, `ToxicLanguage` | ~$0.001/call | Rephrased injection attacks, semantic meaning, toxicity |

---

## What this module teaches

> "Prompts suggest. Guardrails enforce."

Our FinTech agent has access to sensitive data — SSNs, account balances, transaction histories. Without guardrails:

- A prompt injection could extract SSN data
- The LLM could hallucinate financial advice not in our policies
- The agent could mention competitor banks in its response
- A user could ask harmful questions like "How do I make a bomb?"

---

## File breakdown

### `demo.py` — Why guardrails matter (and how they save money)

Seven-part demo showing input + output guardrails across all four strategies:

1. **Part 1: BEFORE** — Run 4 dangerous queries (SSN extraction, crypto advice, competitor comparison, "how to make a bomb") through the raw agent. Every query costs 2+ LLM calls.

2. **Part 2: AFTER — Input Regex Guard** — Same queries hit a regex `input_guard()` first. All 4 blocked in <1ms for $0. The LLM never sees them. Shows LLM calls saved.

3. **Part 3: OpenAI Moderation API** (Strategy 2: free) — Tests 5 queries against the Moderation endpoint. Catches violence, self-harm, harassment by **intent** — much smarter than keyword regex.

4. **Part 4: Prompt Injection Detection** (Strategy 4: LLM-based) — Shows that regex catches "SSN" but misses "last 4 digits of the social security number". An LLM classifier catches the **intent** behind rephrased injection attacks.

5. **Part 5: Guardrails AI — Output Validation** — Two validators side by side:
   - `RegexMatch` (Strategy 1: regex, free) — blocks SSN patterns in output
   - `CompetitorCheck` (Strategy 4: LLM-based, ~$0.001) — understands meaning, catches "Unlike Chase..."

6. **Part 6: Presidio — Output PII Redaction** (Strategy 3: ML/NER) — Account queries are legitimate, but the model says "Hello Alice!" every time. Presidio redacts names regex can't catch. Shows WITHOUT/WITH side by side.

7. **Part 7: Full Guarded Pipeline** — All strategies combined: Moderation → regex → injection classifier → agent → Guardrails AI → Presidio.

**What to watch for when you run it:**
- Part 1 vs Part 2: LLM calls saved by input guard
- Part 3: Moderation API catches self-harm intent that regex keyword misses
- Part 4: LLM classifier catches rephrased injection attacks
- Part 5: RegexMatch is instant and free; CompetitorCheck makes an LLM call but catches meaning
- Part 6: "Alice" → `<PERSON>` — Presidio catches what regex never can

### `exercise.py` — Build production guardrails

Eight TODOs that build up from simple to full pipeline:

| TODO | What you do |
|------|-------------|
| 1 | Build an `input_guard()` with regex patterns for SSN, advice, competitors, harmful content |
| 2 | Set up Guardrails AI `Guard` with `RegexMatch` for SSN patterns |
| 3 | Add `ToxicLanguage` and `CompetitorCheck` validators (Chase, Wells Fargo, Citi, etc.) |
| 4 | Integrate the guard into the agent pipeline (`safe_pipeline()`) |
| 5 | Set up Microsoft Presidio for PII detection — test on sample strings |
| 6 | Build full `guarded_pipeline()`: input guard → agent → Guardrails AI → Presidio |
| 7 | OpenAI Moderation API — free content safety classifier for violence/hate/self-harm |
| 8 | Prompt injection detection — LLM-based classifier that catches rephrased attacks |

### `solution.py` — Reference implementation

Complete working code with all 8 TODOs solved. Gracefully handles missing dependencies — if Guardrails AI or Presidio aren't installed, it prints install instructions and skips those sections.

### `notes.md` — Concepts

Covers: guardrails vs prompt instructions, four strategies (regex → Moderation API → ML/NER → LLM-based), OpenAI Moderation API, prompt injection detection, Guardrails AI architecture, Presidio for broad PII detection, and GDPR/HIPAA engineering patterns.

---

## How to run

You need Guardrails AI and Presidio installed:

```bash
# Install dependencies
pip install guardrails-ai presidio-analyzer presidio-anonymizer
python -m spacy download en_core_web_lg
```

Before installing hub validators, you need a free Guardrails Hub token:

1. Go to https://hub.guardrailsai.com/keys and sign up / log in
2. Copy your token
3. Run `guardrails configure` and paste the token when prompted

Then install the validators:

```bash
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
- Part 1: 4 dangerous queries hit the raw agent — every one costs 2+ LLM calls
- Part 2: Same queries blocked by regex in <1ms — 0 LLM calls, $0 cost
- Part 3: OpenAI Moderation API catches violence/self-harm/hate by intent (free)
- Part 4: LLM classifier catches rephrased prompt injections regex misses
- Part 5: Guardrails AI RegexMatch (SSN) + CompetitorCheck (semantic) on output
- Part 6: Presidio redacts names from account responses ("Alice" → `<PERSON>`)
- Part 7: Full pipeline with timing — shows actual ms per guardrail step

**What to expect from `solution.py`:**
- Tests input guard regex on 5 queries (block/pass)
- Tests Guardrails AI: RegexMatch blocks SSN patterns, CompetitorCheck blocks "Chase"
- Tests safe_pipeline: input guard + agent + Guardrails AI
- Tests Presidio on 4 PII-laden strings (BEFORE/AFTER redaction)
- Tests OpenAI Moderation API on 5 queries (flags violence/self-harm/harassment)
- Tests LLM injection classifier on 6 queries (catches rephrased attacks)
- Runs full guarded pipeline on 7 test queries with all 4 strategies + error handling

---

## Quick mental model

- Use the **lightest check that works**: regex (free, ~1ms) → Moderation API (free, ~100ms) → ML/NER (local, ~10ms) → LLM-based (semantic, ~$0.001).
- **Presidio** catches broad PII (names, addresses, medical IDs). **RegexMatch** catches known patterns (SSN, credit card). Use both.
- **OpenAI Moderation API** is free and catches intent (violence, self-harm) — not just keywords.
- Apply guards on **both sides**: block/redact PII from input before the LLM sees it, validate/redact the output before the user sees it.
- **Error handling matters**: Moderation API and injection classifier should fail-open (other layers still catch); output validation should fail-closed.
- Sending PII to LLM APIs requires a **DPA** (GDPR) or **BAA** (HIPAA) with the provider. Most engineers don't realize this.
