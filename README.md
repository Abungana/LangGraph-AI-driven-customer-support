# LangGraph-AI-driven-customer-support
# 1. Install deps (once):
#    %pip install -qU langgraph langchain-openai openai faiss-cpu typing-extensions requests

import os
import json
import time
from typing_extensions import TypedDict, Annotated

import requests
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from langchain_openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS

# ------------------------------------------------------------------------------
# Placeholder API functions (replace with your real endpoints / SDKs)
# ------------------------------------------------------------------------------
def fetch_new_tickets(api_key: str, limit: int = 5) -> list[dict]:
    """
    Fetch the latest 'limit' tickets from your help-desk system.
    Returns a list of {"id": str, "text": str, "customer_id": str}.
    """
    # Example GET request:
    # resp = requests.get("https://helpdesk.example.com/api/tickets/new",
    #                     headers={"Authorization": f"Bearer {api_key}"}, params={"limit": limit})
    # return resp.json()
    return [
        {"id": f"TKT-{i}", "text": "I can’t log in; receiving error code 401", "customer_id": "CUST123"}
        for i in range(1, limit+1)
    ]

def send_reply(api_key: str, ticket_id: str, message: str) -> bool:
    """Send 'message' to the ticket thread. Returns True on success."""
    # resp = requests.post(f"https://helpdesk.example.com/api/tickets/{ticket_id}/reply",
    #                      headers={"Authorization": f"Bearer {api_key}"},
    #                      json={"message": message})
    # return resp.status_code == 200
    print(f"[API] Reply sent to {ticket_id}:\n{message}\n")
    return True

def escalate_to_human(api_key: str, ticket_id: str, notes: str) -> bool:
    """Flag ticket for manual review by human support."""
    # resp = requests.post(f"https://helpdesk.example.com/api/tickets/{ticket_id}/escalate",
    #                      headers={"Authorization": f"Bearer {api_key}"},
    #                      json={"notes": notes})
    # return resp.ok
    print(f"[API] Ticket {ticket_id} escalated. Notes: {notes}")
    return True

def schedule_followup_email(api_key: str, customer_id: str, delay_days: int = 3) -> bool:
    """
    Schedule a follow-up email to the customer after 'delay_days'.
    In production, this might enqueue a job in your mailing system.
    """
    send_time = time.time() + delay_days * 86400
    print(f"[API] Follow-up email for {customer_id} scheduled at {time.ctime(send_time)}")
    return True

# ------------------------------------------------------------------------------
# Chunking helper
# ------------------------------------------------------------------------------
def split_text(text: str, chunk_size: int = 1000) -> list[str]:
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

# ------------------------------------------------------------------------------
# 2. Define the shared state schema
# ------------------------------------------------------------------------------
class State(TypedDict):
    messages: Annotated[list, add_messages]       # logged events / for chat-like memory if needed
    tickets: list[dict]                           # list of fetched tickets
    current_ticket: dict                          # ticket under processing
    classification: dict                          # {"intent": str, "entities": {...}}
    faq_snippets: list[str]                       # top N FAQ entries
    draft_reply: str                              # generated reply
    confidence: float                             # model’s confidence score
    retries: int                                  # number of FAQ-retry attempts
    max_retries: int                              # retry limit (e.g., 2)
    escalation_flag: bool                         # whether to escalate
    followup_scheduled: bool                      # flag if follow-up was set
    survey_response: str                          # recorded customer feedback

# ------------------------------------------------------------------------------
# 3. Node implementations
# ------------------------------------------------------------------------------

def load_tickets(state: State) -> State:
    """Fetch new tickets and initialize retry params."""
    api_key = os.getenv("HELPDESK_API_KEY")
    state["tickets"] = fetch_new_tickets(api_key, limit=5)
    # Initialize processing of the first ticket
    state["current_ticket"] = state["tickets"].pop(0)
    state["retries"] = 0
    state["max_retries"] = 2
    state["escalation_flag"] = False
    state["followup_scheduled"] = False
    print(f"[Load] Ticket {state['current_ticket']['id']} fetched for processing.")
    return state

def classify_ticket(state: State) -> State:
    """Classify intent & extract entities via LLM."""
    text = state["current_ticket"]["text"]
    llm = OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), temperature=0)
    prompt = (
        "You are a support-ticket classifier. Output JSON with keys 'intent' and 'entities'.\n\n"
        f"Ticket Text: {text}"
    )
    resp = llm(prompt).strip()
    try:
        classification = json.loads(resp)
    except json.JSONDecodeError:
        classification = {"intent": "unknown", "entities": {}}
    state["classification"] = classification
    print(f"[Classify] {state['classification']}")
    return state

def retrieve_faq(state: State) -> State:
    """Fetch top-2 FAQ snippets matching the ticket intent or entity."""
    kb = [
        "If you get error 401, reset your password here: ...",
        "To troubleshoot login issues, clear cache and cookies.",
        "How to change your password securely.",
    ]
    emb = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    vect = FAISS.from_documents(kb, emb)
    query = state["classification"]["entities"].get("error_code", state["classification"]["intent"])
    docs = vect.similarity_search(query, k=2)
    state["faq_snippets"] = [doc.page_content for doc in docs]
    print(f"[FAQ] Retrieved snippets: {state['faq_snippets']}")
    return state

def draft_reply(state: State) -> State:
    """Generate a personalized reply using LLM + FAQ snippets."""
    llm = OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), temperature=0.3)
    ticket = state["current_ticket"]["text"]
    faq_text = "\n".join(state["faq_snippets"])
    prompt = (
        "Draft a concise, friendly support reply integrating the ticket context and FAQ knowledge.\n\n"
        f"Ticket: {ticket}\n\n"
        f"FAQ:\n{faq_text}\n\n"
        "Reply:"
    )
    state["draft_reply"] = llm(prompt)
    print(f"[Draft] Reply generated.")
    return state

def score_confidence(state: State) -> State:
    """Heuristic confidence: look for FAQ keywords; else mark low."""
    reply = state["draft_reply"].lower()
    state["confidence"] = 0.9 if "reset your password" in reply else 0.4
    state["escalation_flag"] = state["confidence"] < 0.7 and state["retries"] >= state["max_retries"]
    print(f"[Score] confidence={state['confidence']:.2f}, escalation={state['escalation_flag']}")
    return state

def retry_or_proceed(state: State) -> State:
    """
    If confidence low but retries left, increment retry and go back to FAQ.
    Otherwise proceed.
    """
    if state["confidence"] < 0.7 and state["retries"] < state["max_retries"]:
        state["retries"] += 1
        print(f"[Retry] Attempt #{state['retries']} – looping back to FAQ.")
        return state  # Graph edge will loop back
    return state  # Graph edge will proceed

def send_or_escalate(state: State) -> State:
    """Send auto-reply or escalate to human based on flag."""
    api_key = os.getenv("HELPDESK_API_KEY")
    tid = state["current_ticket"]["id"]
    if state["escalation_flag"]:
        escalate_to_human(api_key, tid, notes="Low model confidence after retries.")
    else:
        send_reply(api_key, tid, state["draft_reply"])
    return state

def schedule_followup(state: State) -> State:
    """If auto-replied, schedule a follow-up email in 3 days."""
    api_key = os.getenv("HELPDESK_API_KEY")
    if not state["escalation_flag"]:
        cid = state["current_ticket"]["customer_id"]
        schedule_followup_email(api_key, cid, delay_days=3)
        state["followup_scheduled"] = True
    return state

def ingest_survey(state: State) -> State:
    """Simulate ingesting the follow-up survey response."""
    # In real usage, fetch from API or webhook
    state["survey_response"] = "Very satisfied"
    print(f"[Survey] Response recorded: {state['survey_response']}")
    return state

# ------------------------------------------------------------------------------
# 4. Build & compile the graph with conditional edges
# ------------------------------------------------------------------------------
builder = StateGraph(State)

# Register nodes
builder.add_node(load_tickets)
builder.add_node(classify_ticket)
builder.add_node(retrieve_faq)
builder.add_node(draft_reply)
builder.add_node(score_confidence)
builder.add_node(retry_or_proceed)
builder.add_node(send_or_escalate)
builder.add_node(schedule_followup)
builder.add_node(ingest_survey)

# Wire up the core linear flow
builder.set_entry_point("load_tickets")
builder.add_edge(START, "load_tickets")
builder.add_edge("load_tickets", "classify_ticket")
builder.add_edge("classify_ticket", "retrieve_faq")
builder.add_edge("retrieve_faq", "draft_reply")
builder.add_edge("draft_reply", "score_confidence")

# Conditional loop: if under-confidence and retries remain → retry_or_proceed → retrieve_faq
builder.add_edge("score_confidence", "retry_or_proceed")
builder.add_edge("retry_or_proceed", "retrieve_faq", condition=lambda s: s["retries"] <= s["max_retries"] and s["confidence"] < 0.7)

# Proceed when either confidence OK, or max retries hit → send_or_escalate
builder.add_edge("retry_or_proceed", "send_or_escalate", condition=lambda s: not (s["confidence"] < 0.7 and s["retries"] <= s["max_retries"]))

# Final nodes
builder.add_edge("send_or_escalate", "schedule_followup")
builder.add_edge("schedule_followup", "ingest_survey")
builder.add_edge("ingest_survey", END)

graph = builder.compile()

# ------------------------------------------------------------------------------
# 5. Execute the graph
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    initial_state: State = {
        "messages": [],
        "tickets": [],
        "current_ticket": {},
        "classification": {},
        "faq_snippets": [],
        "draft_reply": "",
        "confidence": 0.0,
        "retries": 0,
        "max_retries": 0,
        "escalation_flag": False,
        "followup_scheduled": False,
        "survey_response": "",
    }
    graph.invoke(initial_state)

