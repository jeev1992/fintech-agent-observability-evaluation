"""
Module C Solution: Output Guardrails
---------------------------------------
Full working solution with Guardrails AI validators and
Microsoft Presidio PII detection/redaction.

Prerequisites:
  pip install guardrails-ai presidio-analyzer presidio-anonymizer
  guardrails hub install hub://guardrails/regex_match
  guardrails hub install hub://guardrails/toxic_language
  guardrails hub install hub://guardrails/competitor_check
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")

sys.path.insert(0, str(Path(__file__).parent.parent / "project"))
from fintech_support_agent import build_support_agent, ask

# --- Build pipeline ---
print("Building FinTech support agent...")
agent = build_support_agent(collection_name="guardrails_solution")
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

print("=" * 60)
print("SEGMENT 12: GUARDRAILS AI")
print("=" * 60)

try:
    from guardrails import Guard
    from guardrails.hub import RegexMatch, ToxicLanguage, CompetitorCheck

    # --- SOLUTION 1: RegexMatch for SSN ---
    ssn_guard = Guard().use(
        RegexMatch(
            regex=r"\b\d{3}-\d{2}-\d{4}\b",
            match_type="search",
            on_fail="exception",
        )
    )

    # Test RegexMatch
    test_strings = [
        "The overdraft fee is $35 per transaction.",
        "Your SSN on file is 123-45-6789.",
        "Account ending in 6789 is active.",
    ]

    for text in test_strings:
        try:
            ssn_guard.validate(text)
            print(f"  PASS: {text[:60]}")
        except Exception as e:
            print(f"  BLOCKED: {text[:60]}")

    # --- SOLUTION 2: Full guard with all validators ---
    full_guard = Guard().use_many(
        RegexMatch(
            regex=r"\b\d{3}-\d{2}-\d{4}\b",
            match_type="search",
            on_fail="exception",
        ),
        ToxicLanguage(
            on_fail="exception",
        ),
        CompetitorCheck(
            competitors=["Chase", "Wells Fargo", "Citi", "Bank of America", "Capital One"],
            on_fail="exception",
        ),
    )

    # Test competitor check
    competitor_test = "Unlike Chase Bank, we offer free incoming wires."
    try:
        full_guard.validate(competitor_test)
        print(f"\n  PASS: {competitor_test}")
    except Exception:
        print(f"\n  BLOCKED: {competitor_test[:60]}")

    guardrails_available = True

except ImportError:
    print("  Guardrails AI not installed. Run:")
    print("    pip install guardrails-ai")
    print("    guardrails hub install hub://guardrails/regex_match")
    print("    guardrails hub install hub://guardrails/toxic_language")
    print("    guardrails hub install hub://guardrails/competitor_check")
    guardrails_available = False
    full_guard = None


# --- SOLUTION 3: Safe pipeline with guard ---
def safe_pipeline(query: str) -> str:
    """Run agent with output guardrail validation."""
    result = ask(app, query)
    answer = result["response"]

    if guardrails_available and full_guard is not None:
        try:
            full_guard.validate(answer)
        except Exception:
            return SAFE_FALLBACK

    return answer


# Test pipeline
pipeline_tests = [
    "What is the overdraft fee?",
    "How does SecureBank compare to Chase?",
    "What is the balance on ACC-12345?",
]

print("\n--- Safe Pipeline Tests ---")
for query in pipeline_tests:
    print(f"\n  Query: {query}")
    response = safe_pipeline(query)
    print(f"  Response: {response[:150]}...")


# ===================================================================
# SEGMENT 13: Presidio PII Redaction
# ===================================================================

print("\n" + "=" * 60)
print("SEGMENT 13: PRESIDIO PII REDACTION")
print("=" * 60)

try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine

    # --- SOLUTION 4: Presidio setup ---
    analyzer = AnalyzerEngine()
    anonymizer = AnonymizerEngine()

    # Test PII detection
    pii_samples = [
        "My name is Alice Johnson and my SSN is 123-45-6789.",
        "Please email me at alice@example.com or call 555-123-4567.",
        "My credit card number is 4111-1111-1111-1111.",
        "What is the overdraft fee?",
    ]

    for text in pii_samples:
        results = analyzer.analyze(text=text, language="en")
        if results:
            anonymized = anonymizer.anonymize(text=text, analyzer_results=results)
            print(f"\n  BEFORE: {text}")
            print(f"  AFTER:  {anonymized.text}")
            print(f"  Found:  {[r.entity_type for r in results]}")
        else:
            print(f"\n  CLEAN:  {text}")

    presidio_available = True

except ImportError:
    print("  Presidio not installed. Run:")
    print("    pip install presidio-analyzer presidio-anonymizer")
    presidio_available = False
    analyzer = None
    anonymizer = None


# --- SOLUTION 5: Full guarded pipeline ---
def guarded_pipeline(query: str) -> str:
    """Full pipeline: PII redaction → agent → guard validation → PII redaction."""
    # Step 1: Redact PII from input
    clean_query = query
    if presidio_available and analyzer and anonymizer:
        input_pii = analyzer.analyze(text=query, language="en")
        if input_pii:
            clean_query = anonymizer.anonymize(text=query, analyzer_results=input_pii).text
            print(f"    [INPUT PII] Redacted: {query[:50]} → {clean_query[:50]}")

    # Step 2: Run agent
    result = ask(app, clean_query)
    answer = result["response"]

    # Step 3: Validate with Guardrails AI
    if guardrails_available and full_guard is not None:
        try:
            full_guard.validate(answer)
        except Exception:
            return SAFE_FALLBACK

    # Step 4: Redact PII from output
    if presidio_available and analyzer and anonymizer:
        output_pii = analyzer.analyze(text=answer, language="en")
        if output_pii:
            answer = anonymizer.anonymize(text=answer, analyzer_results=output_pii).text
            print(f"    [OUTPUT PII] Redacted sensitive data from response")

    return answer


# Test full guarded pipeline
print("\n--- Full Guarded Pipeline ---")
guarded_tests = [
    "What is the overdraft fee?",
    "My SSN is 123-45-6789, can you check my account?",
    "What is the balance on ACC-12345?",
]

for query in guarded_tests:
    print(f"\n  Query: {query}")
    response = guarded_pipeline(query)
    print(f"  Response: {response[:150]}...")

print("\n" + "=" * 60)
print("KEY TAKEAWAYS")
print("=" * 60)
print("""
1. Guardrails AI provides pre-built validators (regex, toxicity, competitor)
2. Presidio detects broad PII categories (names, SSNs, emails, credit cards)
3. Layer both: Presidio for PII detection, Guardrails AI for content validation
4. Apply guards on BOTH input and output
5. Sending PII to LLM APIs requires DPA (GDPR) or BAA (HIPAA) with provider
""")
