"""
memory.py — Conversation memory for SehatBot's Mental Health mode.

Why only mental health? The other 6 modes are one-shot lookups ("Is Panadol
safe with Brufen?") — a complete question with a complete answer, where past
turns add nothing but token cost. Mental health is the one genuinely
conversational mode, where "I still can't sleep" only makes sense if the
assistant remembers the earlier turns.

We keep a sliding window of the last N exchanges (like
ConversationBufferWindowMemory) so context stays useful but cheap.
"""

# How many recent (user, bot) exchanges to remember
WINDOW = 5

# In-memory store of recent mental-health turns for the current session.
# (Single-session app; resets when the backend restarts.)
_history = []


def add_turn(user_message: str, bot_reply: str):
    """Record one exchange, keeping only the last WINDOW turns."""
    _history.append({"user": user_message, "bot": bot_reply})
    if len(_history) > WINDOW:
        del _history[0]


def get_history_text() -> str:
    """Return the recent conversation as plain text for the prompt.
    Empty string on the first message (nothing to recall yet)."""
    if not _history:
        return ""
    lines = []
    for turn in _history:
        lines.append(f"User: {turn['user']}")
        lines.append(f"SehatBot: {turn['bot']}")
    return "\n".join(lines)


def clear():
    """Reset the conversation (e.g. when the user starts fresh)."""
    _history.clear()