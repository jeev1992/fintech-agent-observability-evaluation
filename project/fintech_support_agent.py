"""
Support Agent: Multi-Agent FinTech Customer Support System
------------------------------------------------------------
Shared module that builds the multi-agent support system for SecureBank.
Used by Modules A–D for observability, evaluation, guardrails, and cost monitoring.

Architecture:
  Query → Supervisor (intent classifier)
        → Policy Agent (RAG over banking docs)
        → Account Agent (mock database lookup)
        → Escalation Agent (empathetic handoff)
"""

# ── Standard library imports ──
import os
import re       # Used to extract account IDs (e.g. ACC-12345) from user queries
import json     # Used to serialize mock account data into LLM-readable context
from pathlib import Path
from typing import TypedDict, Literal

# ── LangChain / LangGraph imports ──
# RecursiveCharacterTextSplitter: splits policy docs into overlapping chunks for embedding
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Document: lightweight wrapper holding page_content + metadata (source filename)
from langchain_core.documents import Document
# ChatOpenAI: thin wrapper around the OpenAI Chat Completions API
# OpenAIEmbeddings: calls the text-embedding-3-small model to vectorize chunks
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# Chroma: in-process vector store used for semantic retrieval (RAG)
from langchain_chroma import Chroma
# ChatPromptTemplate: builds system + human message pairs for each agent
from langchain.prompts import ChatPromptTemplate
# StrOutputParser: extracts the raw string content from an LLM response message
from langchain.schema.output_parser import StrOutputParser
# RunnablePassthrough: passes the input unchanged — used in the RAG chain to
#   forward the user question while the retriever branch fetches context
from langchain.schema.runnable import RunnablePassthrough
# StateGraph: declarative graph builder for multi-agent orchestration
# END: sentinel node that terminates graph execution
from langgraph.graph import StateGraph, END

# Absolute path to the /project/documents/ folder containing the four policy
# markdown files that the Policy Agent uses for Retrieval-Augmented Generation.
DOCUMENTS_DIR = Path(__file__).parent / "documents"

# ── Mock account database ──
# In production this would be a real database call.  Three accounts are provided
# to demonstrate active accounts, frozen/fraud-review accounts, and varying
# balance tiers.  The Account Agent queries this dict by account ID.
MOCK_ACCOUNTS = {
    "ACC-12345": {
        "account_id": "ACC-12345",
        "name": "Alice Johnson",
        "ssn_last4": "6789",
        "account_type": "Premium Checking",
        "balance": 12450.75,
        "status": "active",
        "recent_transactions": [
            {"date": "2026-03-15", "description": "Direct Deposit - Employer", "amount": 3200.00},
            {"date": "2026-03-14", "description": "Wire Transfer Out", "amount": -500.00},
            {"date": "2026-03-12", "description": "ATM Withdrawal", "amount": -200.00},
            {"date": "2026-03-10", "description": "Online Purchase - Amazon", "amount": -89.99},
        ],
        "overdraft_protection": True,
        "monthly_fee_waived": True,
    },
    "ACC-67890": {
        "account_id": "ACC-67890",
        "name": "Bob Smith",
        "ssn_last4": "4321",
        "account_type": "Basic Checking",
        "balance": 234.50,
        "status": "active",
        "recent_transactions": [
            {"date": "2026-03-14", "description": "Debit Card - Grocery Store", "amount": -67.30},
            {"date": "2026-03-11", "description": "Direct Deposit - Employer", "amount": 1500.00},
            {"date": "2026-03-10", "description": "Bill Pay - Electric Company", "amount": -145.00},
        ],
        "overdraft_protection": False,
        "monthly_fee_waived": False,
    },
    "ACC-11111": {
        "account_id": "ACC-11111",
        "name": "Carol Davis",
        "ssn_last4": "9876",
        "account_type": "High-Yield Savings",
        "balance": 85320.00,
        "status": "frozen",
        "recent_transactions": [
            {"date": "2026-03-13", "description": "Suspicious Transfer Out", "amount": -15000.00},
            {"date": "2026-03-01", "description": "Interest Payment", "amount": 301.45},
        ],
        "overdraft_protection": False,
        "monthly_fee_waived": True,
        "freeze_reason": "Suspected unauthorized activity — under fraud review",
    },
}


# ── LangGraph shared state ──
# Every node in the graph reads from and writes to this typed dictionary.
# LangGraph merges each node's returned dict back into this state automatically.
class SupportState(TypedDict):
    query: str                    # The original customer question
    intent: str                   # Classified intent: "policy" | "account_status" | "escalation"
    response: str                 # Final answer returned to the customer
    context: str                  # Retrieved context (policy chunks or account JSON)
    retrieved_sources: list[str]  # Source filenames of retrieved docs (for traceability)


# Default system prompt for the Policy Agent.  Key design choices:
#   - "based ONLY on the provided policy documents" → prevents hallucination
#   - Explicit fallback phrasing → gives a graceful "I don't know" answer
#   - PII guardrail instruction → defense-in-depth even before Module C adds
#     programmatic guardrails
# Callers can override this via the `policy_system_prompt` parameter in
# build_support_agent() to test different prompting strategies (e.g. Module B
# A/B experiments).
DEFAULT_POLICY_SYSTEM_PROMPT = (
    "You are a helpful customer support agent for SecureBank.\n"
    "Answer the customer's question based ONLY on the provided policy documents.\n"
    "If the answer is not found in the provided context, say:\n"
    "\"I'm sorry, I don't have information about that in our current policies. "
    "Please contact our support team at support@securebank.com for further assistance.\"\n"
    "Do not make up information. Be concise, friendly, and professional.\n"
    "NEVER disclose sensitive account data (SSN, full account numbers) in policy responses."
)


def build_support_agent(
    collection_name="support_docs_multi",
    chunk_size=1000,
    chunk_overlap=100,
    top_k=3,
    model="gpt-4o-mini",
    policy_system_prompt=None,
    enable_reranking=False,
    rerank_fetch_k=None,
):
    """
    Build and return the multi-agent FinTech support system.

    This is the main factory function.  It:
      1. Loads the four policy markdown files from disk
      2. Chunks them with RecursiveCharacterTextSplitter
      3. Embeds chunks into an in-memory Chroma vector store
      4. Wires up three specialist agents + a supervisor into a LangGraph

    Args:
        collection_name : Chroma collection name (change to avoid collisions
                          when running multiple tests in the same process).
        chunk_size      : Maximum characters per chunk.
        chunk_overlap   : Overlap between consecutive chunks (helps preserve
                          context at chunk boundaries).
        top_k           : Number of chunks the retriever returns per query.
        model           : OpenAI model name for all LLM calls.
        policy_system_prompt : Override the default Policy Agent system prompt
                               (used in Module B for A/B prompt experiments).
        enable_reranking: If True, fetch more docs and re-sort by relevance
                          score before trimming to top_k (Module B exercise).
        rerank_fetch_k  : How many docs to fetch before reranking
                          (defaults to top_k * 2 when enable_reranking=True).

    Returns:
        dict with keys:
            app         : Compiled LangGraph application (invoke with ask())
            retriever   : The Chroma vector store retriever
            format_docs : Helper function that formats docs into a string
            llm         : The ChatOpenAI language model instance
            rag_chain   : The standalone Policy RAG chain (retriever → prompt → LLM)
            vectorstore : The underlying Chroma vector store
    """
    # ─── Step 1: Load policy documents from disk ───
    # Each file becomes a single LangChain Document with its filename stored in
    # metadata["source"] so we can trace which document a retrieved chunk came from.
    document_files = [
        "account_fees.md", "loan_policy.md",
        "fraud_policy.md", "transfer_policy.md",
    ]
    all_documents = []
    for filename in document_files:
        content = (DOCUMENTS_DIR / filename).read_text(encoding="utf-8")
        all_documents.append(Document(page_content=content, metadata={"source": filename}))

    # ─── Step 2: Split documents into chunks ───
    # RecursiveCharacterTextSplitter tries split points in order:
    #   "\n\n" → "\n" → " " → "" — so it prefers paragraph boundaries.
    # chunk_overlap ensures that sentences straddling a boundary aren't lost.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(all_documents)

    # ─── Step 3: Embed chunks and store in Chroma ───
    # Uses OpenAI's text-embedding-3-small (1536-dim) for vectorization.
    # Chroma runs in-memory here — no persistence, rebuilt each time
    # build_support_agent() is called.
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(
        chunks, embeddings, collection_name=collection_name
    )
    # The retriever wraps the vector store with a top-k similarity search.
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

    def format_docs(docs):
        """Concatenate retrieved documents into a single string for the LLM context.

        Each chunk is prefixed with its source filename in brackets so the LLM
        (and human reviewers) can see which policy file the information came from.
        Chunks are separated by '---' dividers for readability.
        """
        return "\n\n---\n\n".join(
            f"[{doc.metadata.get('source', '')}]\n{doc.page_content}"
            for doc in docs
        )

    # temperature=0 for deterministic, reproducible answers — important for
    # evaluation (Module B) where we need consistent outputs to measure quality.
    llm = ChatOpenAI(model=model, temperature=0)

    # ─── Step 4: Build the Policy Agent RAG chain ───
    # This standalone chain is also exposed in the returned dict so Module B can
    # invoke it directly for evaluation without going through the full graph.
    _policy_sys = policy_system_prompt or DEFAULT_POLICY_SYSTEM_PROMPT
    policy_prompt = ChatPromptTemplate.from_messages([
        ("system", _policy_sys),
        ("human",
         "Context from our policy documents:\n\n{context}\n\n"
         "Customer question: {question}"),
    ])
    # The RAG chain works in two parallel branches via RunnableParallel:
    #   "context" branch: query → retriever → format_docs → string of chunks
    #   "question" branch: query passes through unchanged (RunnablePassthrough)
    # Both feed into the prompt template, then the LLM, then string extraction.
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | policy_prompt | llm | StrOutputParser()
    )

    # ─── Step 5: Define the Supervisor (intent classifier) ───
    # The supervisor is the entry-point node. It makes a single LLM call to
    # classify the customer query into one of three intents, which determines
    # which specialist agent handles the request.
    classification_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Classify the customer query into exactly one category:\n"
         "- \"policy\" — general questions about account fees, loans, transfers, "
         "fraud policies, or banking products\n"
         "- \"account_status\" — requests to check balance, view transactions, "
         "or look up a SPECIFIC account (usually contains an account number like ACC-XXXXX)\n"
         "- \"escalation\" — complaints, frustration, requests for a manager, "
         "fraud reports, or complex issues needing human attention\n\n"
         "Respond with ONLY the category name."),
        ("human", "{query}"),
    ])

    def classify_intent(state):
        """Supervisor node: classifies intent and writes it to state.

        Falls back to "policy" if the LLM returns an unexpected value — this
        ensures the graph always routes to a valid agent.
        """
        chain = classification_prompt | llm | StrOutputParser()
        intent = chain.invoke({"query": state["query"]}).strip().lower()
        # Guard against unexpected LLM output — default to the safest route
        if intent not in ("policy", "account_status", "escalation"):
            intent = "policy"
        return {"intent": intent}

    # ─── Step 6: Define the Policy Agent ───
    # Handles "policy" intent.  Retrieves relevant chunks from the vector store,
    # feeds them as context to the LLM, and returns a grounded answer.
    def policy_agent(state):
        """Policy Agent node: answers banking policy questions using RAG.

        When `enable_reranking` is True (Module B exercise), it over-fetches
        documents and re-sorts by relevance score to improve retrieval quality
        (measured via MRR — Mean Reciprocal Rank).
        """
        question = state["query"]
        if enable_reranking:
            # Over-fetch candidate docs, then use the LLM to re-score each
            # one against the query and keep only the top_k most relevant
            fetch_k = rerank_fetch_k or top_k * 2
            candidate_docs = vectorstore.similarity_search(question, k=fetch_k)

            # LLM-as-reranker: ask the model to score each doc's relevance 0-10
            rerank_prompt = ChatPromptTemplate.from_messages([
                ("system",
                 "You are a relevance scoring system. Given a query and a document, "
                 "rate how relevant the document is to answering the query on a scale "
                 "of 0 to 10. Respond with ONLY a single integer."),
                ("human",
                 "Query: {query}\n\nDocument:\n{document}\n\nRelevance score (0-10):"),
            ])
            rerank_chain = rerank_prompt | llm | StrOutputParser()

            scored = []
            for doc in candidate_docs:
                score_str = rerank_chain.invoke({
                    "query": question,
                    "document": doc.page_content,
                })
                try:
                    score = int(score_str.strip())
                except ValueError:
                    score = 0
                scored.append((doc, score))

            scored.sort(key=lambda x: x[1], reverse=True)
            retrieved_docs = [doc for doc, _ in scored[:top_k]]
        else:
            # Standard retrieval — returns top_k chunks by vector similarity
            retrieved_docs = retriever.invoke(question)
        context = format_docs(retrieved_docs)
        # Track which source files were used (for observability / evaluation)
        sources = [doc.metadata.get("source", "") for doc in retrieved_docs]
        # Note: we call the prompt→LLM chain directly (not rag_chain) because
        # we already fetched the docs above and need the sources list.
        chain = policy_prompt | llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": question})
        return {
            "response": answer,
            "context": context,
            "retrieved_sources": sources,
        }

    # ─── Step 7: Define the Account Agent ───
    # Handles "account_status" intent.  Looks up mock account data by ID and
    # uses the LLM to formulate a natural-language summary.
    account_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a customer support agent helping with account inquiries at SecureBank.\n"
         "Answer based ONLY on the account data provided. Be friendly and concise.\n"
         "IMPORTANT: Never reveal the customer's SSN (even last 4 digits) in your response.\n"
         "If the account is frozen, explain the status and advise contacting fraud support."),
        ("human",
         "Account data:\n{account_data}\n\nCustomer question: {question}"),
    ])

    def account_agent(state):
        """Account Agent node: looks up account details from the mock database.

        Flow:
          1. Extract account ID (ACC-XXXXX) from the query using regex
          2. Look up the account in MOCK_ACCOUNTS
          3. Strip the SSN before sending to the LLM (defense-in-depth PII protection)
          4. Ask the LLM to compose a friendly summary of the account data
        """
        query = state["query"]
        # Try to find an account number pattern in the user's question
        match = re.search(r"ACC-\d+", query, re.IGNORECASE)
        if not match:
            # No account number found — ask the user to provide one
            return {
                "response": (
                    "I'd be happy to help with your account! Could you please "
                    "provide your account number? It starts with 'ACC-' followed "
                    "by digits (e.g., ACC-12345)."
                ),
                "context": "",
                "retrieved_sources": [],
            }
        account_id = match.group(0).upper()
        account = MOCK_ACCOUNTS.get(account_id)
        if not account:
            # Account ID parsed but doesn't exist in our mock data
            return {
                "response": (
                    f"I couldn't find account {account_id} in our system. "
                    "Please double-check or contact support@securebank.com."
                ),
                "context": "",
                "retrieved_sources": [],
            }
        # NOTE: SSN (last-4) is intentionally NOT stripped here so that
        # Module C's guardrails demo can show the data leakage risk.
        # In production, you would strip PII before it reaches the LLM.
        context = json.dumps(account, indent=2)
        chain = account_prompt | llm | StrOutputParser()
        response = chain.invoke({"account_data": context, "question": query})
        return {"response": response, "context": context, "retrieved_sources": []}

    # ─── Step 8: Define the Escalation Agent ───
    # Handles "escalation" intent.  Unlike the other agents, this one does NOT
    # perform any retrieval — it simply generates an empathetic response and
    # directs the customer to human support channels.
    escalation_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a senior customer support agent at SecureBank handling an escalation.\n"
         "1. Acknowledge their concern with empathy\n"
         "2. Summarize their issue briefly\n"
         "3. Let them know a senior specialist will follow up\n"
         "4. Provide: support@securebank.com or 1-800-555-0199\n\n"
         "Be warm, professional, and concise. "
         "Do NOT try to solve the problem yourself.\n"
         "Do NOT make specific policy claims (e.g., exact fee amounts or timeframes)."),
        ("human", "{query}"),
    ])

    def escalation_agent(state):
        """Escalation Agent node: provides empathetic handoff to human support.

        No retrieval or database lookup — the LLM generates a compassionate
        acknowledgment and routes the customer to live support channels.
        Module B uses G-Eval to measure the empathy quality of these responses.
        """
        chain = escalation_prompt | llm | StrOutputParser()
        response = chain.invoke({"query": state["query"]})
        return {"response": response, "context": "", "retrieved_sources": []}

    # ─── Step 9: Define routing logic ───
    # Maps the intent string (set by the supervisor) to the corresponding agent
    # node name.  Falls back to "policy_agent" for any unknown intent.
    def route_by_intent(state) -> Literal[
        "policy_agent", "account_agent", "escalation_agent"
    ]:
        return {
            "policy": "policy_agent",
            "account_status": "account_agent",
            "escalation": "escalation_agent",
        }.get(state["intent"], "policy_agent")

    # ─── Step 10: Assemble the LangGraph ───
    # Graph structure:
    #
    #   START → classify_intent ─┬─ "policy"          → policy_agent     → END
    #                             ├─ "account_status"  → account_agent    → END
    #                             └─ "escalation"      → escalation_agent → END
    #
    graph = StateGraph(SupportState)
    graph.add_node("classify_intent", classify_intent)    # Supervisor
    graph.add_node("policy_agent", policy_agent)          # RAG over policy docs
    graph.add_node("account_agent", account_agent)        # Mock DB lookup
    graph.add_node("escalation_agent", escalation_agent)  # Empathetic handoff

    graph.set_entry_point("classify_intent")
    # Conditional edges: the supervisor's output determines which agent runs next
    graph.add_conditional_edges("classify_intent", route_by_intent)
    # Each agent terminates the graph after producing a response
    graph.add_edge("policy_agent", END)
    graph.add_edge("account_agent", END)
    graph.add_edge("escalation_agent", END)

    app = graph.compile()

    # Return all key components so modules can access them individually.
    # For example, Module B uses `retriever` and `vectorstore` for MRR evaluation,
    # and Module D uses `llm` for token counting.
    return {
        "app": app,              # The compiled LangGraph — pass to ask()
        "retriever": retriever,  # Chroma retriever for direct retrieval tests
        "format_docs": format_docs,  # Doc formatter (used in evaluation)
        "llm": llm,              # ChatOpenAI instance (shared across agents)
        "rag_chain": rag_chain,  # Standalone RAG chain (retriever → LLM)
        "vectorstore": vectorstore,  # Raw Chroma store (for similarity_search_with_relevance_scores)
    }


def ask(app, query: str) -> dict:
    """Convenience helper to invoke the multi-agent system with a customer query.

    Initializes all state fields to empty defaults and runs the full graph:
    classify_intent → specialist agent → END.

    Args:
        app   : The compiled LangGraph application (from build_support_agent()["app"]).
        query : The customer's natural-language question.

    Returns:
        The final SupportState dict containing intent, response, context, and sources.
    """
    return app.invoke({
        "query": query,
        "intent": "",
        "response": "",
        "context": "",
        "retrieved_sources": [],
    })
