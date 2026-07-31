"""
agent.py — SehatBot's brain (LangGraph)
Flow:  router node (LLM picks ONE tool name as plain text)
         → tool node (our code calls that tool with the user's message)
No bind_tools / no JSON tool-calling → nothing for Groq to reject.
"""
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from tools import ALL_TOOLS
from llm import safe_invoke
import guardrails
from typing import TypedDict

# Look-up tables built automatically from tools.py
TOOL_MAP = {t.name: t for t in ALL_TOOLS}                       # name -> tool
TOOL_ARG = {t.name: list(t.args.keys())[0] for t in ALL_TOOLS}  # name -> its one argument
TOOL_NAMES = list(TOOL_MAP.keys())

ROUTER_PROMPT = """You are the router for SehatBot, Pakistan's AI medical assistant.
Pick the ONE best tool for the user's message.

Tools:
- check_symptoms: user describes symptoms (fever, pain, bukhar, dard...)
- analyze_lab_report: user pastes lab/test report values
- check_drug_interaction: user asks about medicines / combining medicines
- read_prescription_ocr: user uploaded a prescription image
- assess_mental_health: stress, anxiety, sadness, pareshani, mental health
- get_diet_plan: diet / food advice for a condition
- get_emergency_guide: urgent emergencies (choking, bleeding, heart attack...)

Reply with ONLY the tool name. One word. No punctuation, no JSON, no explanation.

User message: {query}"""


class AgentState(TypedDict):
    user_message: str
    tool_name: str
    answer: str


def router_node(state: AgentState):
    """LLM makes ONE tiny decision: which tool? Plain text out — can't fail validation."""
    prompt = ROUTER_PROMPT.format(query=state["user_message"])
    response = safe_invoke([HumanMessage(content=prompt)])   # note: NO tools= binding
    text = str(response.content).strip().lower()
    # Robust parse: find a known tool name anywhere in the reply
    chosen = next((name for name in TOOL_NAMES if name in text), "check_symptoms")
    print(f"🧭 Router picked: {chosen}")
    return {"tool_name": chosen}


def tool_node(state: AgentState):
    """OUR code fills the argument (instructor's advice) and runs the tool."""
    name = state["tool_name"]
    tool = TOOL_MAP[name]
    value = state["user_message"]
    if name == "read_prescription_ocr":
        # message looks like "...: temp_photo.png" — extract just the path
        value = value.split(":")[-1].strip()
    answer = tool.invoke({TOOL_ARG[name]: value})
    return {"answer": str(answer)}


graph = StateGraph(AgentState)
graph.add_node("router", router_node)
graph.add_node("tool", tool_node)
graph.set_entry_point("router")
graph.add_edge("router", "tool")
graph.add_edge("tool", END)

sehatbot_agent = graph.compile()


def _llm_says_health(user_message: str) -> bool:
    """LLM classifier: is this a health/medical/wellbeing question?
    Used for any message the keyword fast-pass didn't already clear."""
    prompt = (
        "You are a classifier for a medical assistant called SehatBot. "
        "Decide if the user's message is a health, medical, mental-health, "
        "nutrition, or wellbeing question that SehatBot should answer.\n"
        "Reply with ONLY one word: HEALTH or OFFTOPIC.\n\n"
        "Examples:\n"
        "'I have a fever' -> HEALTH\n"
        "'is panadol safe' -> HEALTH\n"
        "'I feel anxious' -> HEALTH\n"
        "'diet for diabetes' -> HEALTH\n"
        "'what is the capital of France' -> OFFTOPIC\n"
        "'write python code' -> OFFTOPIC\n"
        "'who won the cricket match' -> OFFTOPIC\n"
        "'tell me a joke' -> OFFTOPIC\n"
        "'how do I style my hair' -> OFFTOPIC\n"
        "'how to fix my car' -> OFFTOPIC\n\n"
        f"Message: {user_message}\n"
        "Answer:"
    )
    try:
        ans = safe_invoke([HumanMessage(content=prompt)]).content.strip().lower()
        # Explicit decision; default to OFFTOPIC if the model is unclear.
        if "offtopic" in ans or "off-topic" in ans or ans.startswith("off"):
            return False
        if "health" in ans:
            return True
        return False   # unclear reply → treat as off-topic (safer for a medical bot)
    except Exception:
        # If the classifier itself fails, fall back to the keyword signal only.
        return guardrails.is_health_related(user_message)

def _strip_mode_hint(message: str) -> str:
    """The frontend prepends a mode hint like 'Symptom check: <question>'.
    Guardrails must judge the ACTUAL question, not the hint (which always
    contains health words). Strip a leading 'something check/advice/... :' prefix."""
    if ":" in message:
        prefix, rest = message.split(":", 1)
        # Only strip short label-like prefixes (the mode hints), not real content
        if len(prefix.split()) <= 5:
            return rest.strip()
    return message

def run_agent(user_message: str):
    # Judge guardrails on the real question, not the frontend's mode-hint wrapper.
    actual = _strip_mode_hint(user_message)

    # ── GUARDRAIL 1: crisis detection (runs FIRST, before anything else) ──
    if guardrails.is_crisis(actual):
        return guardrails.CRISIS_RESPONSE

    # ── GUARDRAIL 2: domain restriction (keep SehatBot medical-only) ──
    # Fast-pass obvious health messages by keyword (0 tokens). Everything else
    # is decided by the LLM classifier — so ANY off-topic question is refused,
    # not just ones on the keyword list.
    if not guardrails.is_health_related(actual):
        if not _llm_says_health(actual):
            return guardrails.OFF_TOPIC_RESPONSE

    # ── GUARDRAIL 3: redact PII before logging (protects private health data) ──
    print(f"[query] {guardrails.redact_pii(actual)}")

    try:
        result = sehatbot_agent.invoke({"user_message": user_message})
        return result["answer"]
    except Exception as e:
        err = str(e)
        if "429" in err or "rate_limit" in err:
            return ("⚠️ SehatBot is very busy right now (daily free limit reached). "
                    "Please try again after some time. | معذرت، روزانہ کی حد پوری ہو گئی ہے، کچھ دیر بعد کوشش کریں۔")
        return f"⚠️ Something went wrong: {err[:150]}"