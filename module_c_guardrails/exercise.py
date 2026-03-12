"""
Module C Exercise: Output Guardrails
---------------------------------------
Implement guardrails using Guardrails AI validators and
Microsoft Presidio for PII detection/redaction.

Segments covered:
  12. Guardrails AI — RegexMatch, ToxicLanguage, CompetitorCheck
  13. Presidio PII redaction, compliance patterns

Prerequisites:
  pip install guardrails-ai presidio-analyzer presidio-anonymizer
  guardrails hub install hub://guardrails/regex_match
  guardrails hub install hub://guardrails/toxic_language
  guardrails hub install hub://guardrails/competitor_check
"""

import os
import re
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")

sys.path.insert(0, str(Path(__file__).parent.parent / "project"))
from fintech_support_agent import build_support_agent, ask

# --- Build pipeline (provided) ---
print("Building FinTech support agent...")
agent = build_support_agent(collection_name="guardrails_exercise")
app = agent["app"]
print("Pipeline ready.\n")

SAFE_FALLBACK = (
    "I'm sorry, I can only answer questions about SecureBank's account fees, "
    "loans, transfers, and fraud policies. Please contact support@securebank.com "
    "or call 1-800-555-0199 for further assistance."
)


# ===================================================================
# SEGMENT 12: Guardrails AI
# ===================================================================

# ---------------------------------------------------------------------------
# TODO 1: Set up a Guardrails AI Guard with RegexMatch
#
# Create a Guard that blocks SSN patterns (###-##-####) in output.
#
# from guardrails import Guard
# from guardrails.hub import RegexMatch
#
# guard = Guard().use(
#     RegexMatch(regex="...", match_type="search", on_fail="exception")
# )
#
# Test with: guard.validate("Your SSN is 123-45-6789")
# ---------------------------------------------------------------------------
print("=" * 60)
print("SEGMENT 12: GUARDRAILS AI")
print("=" * 60)

# YOUR CODE HERE — create guard with RegexMatch for SSN pattern
guard = None

# Test it:
test_strings = [
    "The overdraft fee is $35 per transaction.",           # Should pass
    "Your SSN on file is 123-45-6789.",                    # Should be blocked
    "Account ending in 6789 is active.",                   # Should pass
]

if guard is not None:
    for text in test_strings:
        try:
            result = guard.validate(text)
            print(f"  PASS: {text[:60]}")
        except Exception as e:
            print(f"  BLOCKED: {text[:60]} — {e}")
else:
    print("  Complete TODO 1 to test RegexMatch guard.")


# ---------------------------------------------------------------------------
# TODO 2: Add ToxicLanguage and CompetitorCheck validators
#
# Extend your guard to also check for:
#   - Toxic language (using ToxicLanguage validator)
#   - Competitor mentions (Chase, Wells Fargo, Citi, Bank of America, Capital One)
#
# from guardrails.hub import ToxicLanguage, CompetitorCheck
#
# guard = Guard().use_many(
#     RegexMatch(...),
#     ToxicLanguage(on_fail="exception"),
#     CompetitorCheck(competitors=[...], on_fail="exception"),
# )
# ---------------------------------------------------------------------------
# YOUR CODE HERE — create full guard with all 3 validators
full_guard = None

# Test with competitor mention:
if full_guard is not None:
    competitor_test = "Unlike Chase Bank, we offer free incoming wires."
    try:
        full_guard.validate(competitor_test)
        print(f"\n  PASS: {competitor_test}")
    except Exception as e:
        print(f"\n  BLOCKED: {competitor_test[:60]} — {e}")
else:
    print("\n  Complete TODO 2 to test full guard.")


# ---------------------------------------------------------------------------
# TODO 3: Integrate guard into the agent pipeline
#
# Create a safe_pipeline(query) function that:
#   1. Runs the multi-agent graph: result = ask(app, query)
#   2. Validates the response with full_guard
#   3. If validation fails, return SAFE_FALLBACK
#   4. Otherwise return the agent's response
# ---------------------------------------------------------------------------
def safe_pipeline(query: str) -> str:
    # YOUR CODE HERE
    pass


# Test the pipeline:
pipeline_tests = [
    "What is the overdraft fee?",
    "How does SecureBank compare to Chase?",
    "What is the balance on ACC-12345?",
]

if safe_pipeline("test") is not None:
    for query in pipeline_tests:
        print(f"\n  Query: {query}")
        response = safe_pipeline(query)
        print(f"  Response: {response[:150]}...")
else:
    print("\n  Complete TODO 3 to test safe pipeline.")


# ===================================================================
# SEGMENT 13: Presidio PII Redaction
# ===================================================================

print("\n" + "=" * 60)
print("SEGMENT 13: PRESIDIO PII REDACTION")
print("=" * 60)

# ---------------------------------------------------------------------------
# TODO 4: Set up Presidio for PII detection
#
# from presidio_analyzer import AnalyzerEngine
# from presidio_anonymizer import AnonymizerEngine
#
# analyzer = AnalyzerEngine()
# anonymizer = AnonymizerEngine()
#
# Detect PII in a sample string:
#   results = analyzer.analyze(text="...", language="en")
#   anonymized = anonymizer.anonymize(text="...", analyzer_results=results)
# ---------------------------------------------------------------------------
# YOUR CODE HERE — set up Presidio engines
analyzer = None
anonymizer = None

# Test PII detection:
pii_samples = [
    "My name is Alice Johnson and my SSN is 123-45-6789.",
    "Please email me at alice@example.com or call 555-123-4567.",
    "My credit card number is 4111-1111-1111-1111.",
    "What is the overdraft fee?",  # No PII — should pass through unchanged
]

if analyzer is not None and anonymizer is not None:
    for text in pii_samples:
        results = analyzer.analyze(text=text, language="en")
        if results:
            anonymized = anonymizer.anonymize(text=text, analyzer_results=results)
            print(f"\n  BEFORE: {text}")
            print(f"  AFTER:  {anonymized.text}")
            print(f"  Found:  {[r.entity_type for r in results]}")
        else:
            print(f"\n  CLEAN:  {text}")
else:
    print("  Complete TODO 4 to test Presidio.")


# ---------------------------------------------------------------------------
# TODO 5: Build a full guarded pipeline with PII redaction
#
# Create guarded_pipeline(query) that:
#   1. Redact PII from the INPUT query using Presidio
#   2. Run the redacted query through the multi-agent graph
#   3. Validate the OUTPUT with Guardrails AI
#   4. Redact any remaining PII from the OUTPUT
#   5. Return the safe response
# ---------------------------------------------------------------------------
def guarded_pipeline(query: str) -> str:
    # YOUR CODE HERE
    pass


# Test the full pipeline:
guarded_tests = [
    "What is the overdraft fee?",
    "My SSN is 123-45-6789, can you check my account?",
    "What is the balance on ACC-12345?",
]

if guarded_pipeline("test") is not None:
    print("\n--- Full Guarded Pipeline ---")
    for query in guarded_tests:
        print(f"\n  Query: {query}")
        response = guarded_pipeline(query)
        print(f"  Response: {response[:150]}...")
else:
    print("\n  Complete TODO 5 to test full guarded pipeline.")
