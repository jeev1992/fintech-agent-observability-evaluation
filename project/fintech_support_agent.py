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

import os
import re
import json
from pathlib import Path
from typing import TypedDict, Literal

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langgraph.graph import StateGraph, END

DOCUMENTS_DIR = Path(__file__).parent / "documents"

# --- Mock account database ---
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


class SupportState(TypedDict):
    query: str
    intent: str
    response: str
    context: str
    retrieved_sources: list[str]


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

    Returns dict with keys:
        app:              compiled LangGraph application
        retriever:        the vector store retriever
        format_docs:      document formatting function
        llm:              the language model
        rag_chain:        the policy RAG chain
        vectorstore:      the Chroma vector store
    """
    # --- Load documents ---
    document_files = [
        "account_fees.md", "loan_policy.md",
        "fraud_policy.md", "transfer_policy.md",
    ]
    all_documents = []
    for filename in document_files:
        content = (DOCUMENTS_DIR / filename).read_text(encoding="utf-8")
        all_documents.append(Document(page_content=content, metadata={"source": filename}))

    # --- Chunk ---
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(all_documents)

    # --- Embed and store ---
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(
        chunks, embeddings, collection_name=collection_name
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

    def format_docs(docs):
        return "\n\n---\n\n".join(
            f"[{doc.metadata.get('source', '')}]\n{doc.page_content}"
            for doc in docs
        )

    llm = ChatOpenAI(model=model, temperature=0)

    # --- RAG chain for Policy Agent ---
    _policy_sys = policy_system_prompt or DEFAULT_POLICY_SYSTEM_PROMPT
    policy_prompt = ChatPromptTemplate.from_messages([
        ("system", _policy_sys),
        ("human",
         "Context from our policy documents:\n\n{context}\n\n"
         "Customer question: {question}"),
    ])
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | policy_prompt | llm | StrOutputParser()
    )

    # --- Supervisor: intent classifier ---
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
        chain = classification_prompt | llm | StrOutputParser()
        intent = chain.invoke({"query": state["query"]}).strip().lower()
        if intent not in ("policy", "account_status", "escalation"):
            intent = "policy"
        return {"intent": intent}

    # --- Policy Agent ---
    def policy_agent(state):
        question = state["query"]
        if enable_reranking:
            fetch_k = rerank_fetch_k or top_k * 2
            scored_docs = vectorstore.similarity_search_with_relevance_scores(
                question, k=fetch_k
            )
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            retrieved_docs = [doc for doc, _ in scored_docs[:top_k]]
        else:
            retrieved_docs = retriever.invoke(question)
        context = format_docs(retrieved_docs)
        sources = [doc.metadata.get("source", "") for doc in retrieved_docs]
        chain = policy_prompt | llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": question})
        return {
            "response": answer,
            "context": context,
            "retrieved_sources": sources,
        }

    # --- Account Agent ---
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
        query = state["query"]
        match = re.search(r"ACC-\d+", query, re.IGNORECASE)
        if not match:
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
            return {
                "response": (
                    f"I couldn't find account {account_id} in our system. "
                    "Please double-check or contact support@securebank.com."
                ),
                "context": "",
                "retrieved_sources": [],
            }
        # Remove SSN from context sent to LLM for safety
        safe_account = {k: v for k, v in account.items() if k != "ssn_last4"}
        context = json.dumps(safe_account, indent=2)
        chain = account_prompt | llm | StrOutputParser()
        response = chain.invoke({"account_data": context, "question": query})
        return {"response": response, "context": context, "retrieved_sources": []}

    # --- Escalation Agent ---
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
        chain = escalation_prompt | llm | StrOutputParser()
        response = chain.invoke({"query": state["query"]})
        return {"response": response, "context": "", "retrieved_sources": []}

    # --- Routing ---
    def route_by_intent(state) -> Literal[
        "policy_agent", "account_agent", "escalation_agent"
    ]:
        return {
            "policy": "policy_agent",
            "account_status": "account_agent",
            "escalation": "escalation_agent",
        }.get(state["intent"], "policy_agent")

    # --- Build graph ---
    graph = StateGraph(SupportState)
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("policy_agent", policy_agent)
    graph.add_node("account_agent", account_agent)
    graph.add_node("escalation_agent", escalation_agent)

    graph.set_entry_point("classify_intent")
    graph.add_conditional_edges("classify_intent", route_by_intent)
    graph.add_edge("policy_agent", END)
    graph.add_edge("account_agent", END)
    graph.add_edge("escalation_agent", END)

    app = graph.compile()

    return {
        "app": app,
        "retriever": retriever,
        "format_docs": format_docs,
        "llm": llm,
        "rag_chain": rag_chain,
        "vectorstore": vectorstore,
    }


def ask(app, query: str) -> dict:
    """Helper to invoke the support agent with a query."""
    return app.invoke({
        "query": query,
        "intent": "",
        "response": "",
        "context": "",
        "retrieved_sources": [],
    })
