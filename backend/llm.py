"""
llm.py — One place that talks to the AI models.
Primary engine : Groq (fast, free)
Backup engine  : Google Gemini (kicks in ONLY when Groq's tokens run out)
"""
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from config import LLM_MODEL, BACKUP_MODEL

# max_retries=0 → if Groq is out of tokens, fail FAST and switch to backup
# (instead of silently retrying and burning time)
primary_llm = ChatGroq(model=LLM_MODEL, max_retries=0)
backup_llm  = ChatGoogleGenerativeAI(model=BACKUP_MODEL)


def _is_rate_limit(error) -> bool:
    text = str(error).lower()
    return "429" in text or "rate_limit" in text or "quota" in text


def _clean_for_gemini(messages):
    """Gemini rejects messages with empty content ("contents are required").
    Tool-call AI messages often have content="" — give them a space instead."""
    for m in messages:
        if hasattr(m, "content") and (m.content == "" or m.content is None):
            m.content = " "
    return messages


def safe_invoke(messages, tools=None):
    """Try Groq first. If its token tank is empty (429), use Gemini instead.
    Pass tools=ALL_TOOLS when the agent needs tool-calling."""
    primary = primary_llm.bind_tools(tools) if tools else primary_llm
    try:
        return primary.invoke(messages)
    except Exception as e:
        if _is_rate_limit(e):
            print("⚠️ Groq limit reached — switching to Gemini backup")
            backup = backup_llm.bind_tools(tools) if tools else backup_llm
            try:
                return backup.invoke(_clean_for_gemini(messages))
            except Exception:
                import traceback
                traceback.print_exc()   # show the FULL error in terminal
                raise
        raise  # a different error → let run_agent's try/except handle it