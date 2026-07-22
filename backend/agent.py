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


def run_agent(user_message: str):
    try:
        initial_state: AgentState = {
            "user_message": user_message,
            "tool_name": "",
            "answer": "",
        }
        result = sehatbot_agent.invoke(initial_state)
        return result["answer"]
    except Exception as e:
        err = str(e)
        if "429" in err or "rate_limit" in err:
            return ("⚠️ SehatBot is very busy right now (daily free limit reached). "
                    "Please try again after some time. | معذرت، روزانہ کی حد پوری ہو گئی ہے، کچھ دیر بعد کوشش کریں۔")
        return f"⚠️ Something went wrong: {err[:150]}"