"""
guardrails.py — SehatBot safety layer (lightweight, rule-based).

Two jobs, both critical for a medical app:

1. CRISIS DETECTION — if a user expresses self-harm or suicidal intent, we
   must NOT hand them a generic RAG answer. We immediately return Pakistan
   crisis helplines and urge them to reach a real person. This check runs on
   the INPUT, before anything else.

2. PII REDACTION — strip personal identifiers (phone, CNIC, email) from text
   before it is logged or processed, so health data stays private (the app
   promises "your health data stays on your device"). Applied to logs, not to
   the user's own on-screen answer.

Rule-based on purpose: no heavy dependencies, fast, predictable, and easy to
explain — exactly what a safety layer should be.
"""
import re

# ── 1. CRISIS DETECTION ──
# Phrases that signal self-harm / suicidal intent (English + Roman Urdu).
CRISIS_PATTERNS = [
    r"\bkill myself\b", r"\bsuicide\b", r"\bsuicidal\b", r"\bend my life\b",
    r"\bend it all\b", r"\bwant to die\b", r"\bdon'?t want to live\b",
    r"\bno reason to live\b", r"\bharm myself\b", r"\bhurt myself\b",
    r"\bself harm\b", r"\bcut myself\b",
    r"\bkhudkushi\b", r"\bmarna chahta\b", r"\bmarna chahti\b",
    r"\bjeena nahi chahta\b", r"\bjeena nahi chahti\b", r"\bapne aap ko\b.*\bnuqsan\b",
]

CRISIS_RESPONSE = (
    "**Please reach out for immediate help — you are not alone.**\n\n"
    "If you are having thoughts of harming yourself, please talk to someone right now:\n"
    "* **Umang Helpline (Pakistan): 0311-7786264** — free, confidential, 24/7\n"
    "* **Rozan Counselling: 0304-1111744**\n"
    "* Or call **Rescue 1122** in an emergency.\n\n"
    "Please also reach out to a trusted family member or friend. Talking to a real person "
    "who cares can make a big difference. You matter, and help is available."
)


def is_crisis(text: str) -> bool:
    """True if the message shows signs of self-harm or suicidal intent."""
    low = text.lower()
    return any(re.search(p, low) for p in CRISIS_PATTERNS)


# ── 2. PII REDACTION ──
PII_RULES = [
    (re.compile(r"\b\d{5}-?\d{7}-?\d\b"), "[CNIC]"),                 # Pakistani CNIC
    (re.compile(r"\b(?:\+92|0)3\d{2}[- ]?\d{7}\b"), "[PHONE]"),      # Pakistani mobile
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"), "[EMAIL]"),        # email
    (re.compile(r"\b\d{16}\b"), "[CARD]"),                          # 16-digit card
]


def redact_pii(text: str) -> str:
    """Replace personal identifiers with placeholders (for safe logging)."""
    for pattern, placeholder in PII_RULES:
        text = pattern.sub(placeholder, text)
    return text


def has_pii(text: str) -> bool:
    return any(p.search(text) for p, _ in PII_RULES)


# ── 3. DOMAIN RESTRICTION (keep SehatBot medical-only) ──
# SehatBot should politely refuse clearly off-topic questions (sports, coding,
# geography, jokes) instead of trying to answer them. We use a fast two-stage
# check: obvious medical keywords pass instantly (0 tokens); only genuinely
# uncertain messages fall back to a tiny yes/no LLM classifier.

HEALTH_KEYWORDS = [
    # symptoms & body
    "fever","pain","ache","headache","cough","cold","flu","vomit","nausea","dizzy",
    "tired","weak","rash","injury","wound","burn","bleed","swelling","infection",
    "blood","pressure","sugar","diabetes","cholesterol","heart","kidney","liver",
    "stomach","chest","breath","throat","skin","eye","ear","bone","joint",
    # medical topics
    "medicine","tablet","dose","drug","antibiotic","panadol","brufen","disprin",
    "symptom","disease","doctor","hospital","clinic","treatment","diagnosis",
    "lab","report","test","hemoglobin","platelet","prescription","vaccine",
    "diet","nutrition","food","weight","health","mental","anxiety","anxious","stress",
    "stressed","depression","depressed","sad","worried","lonely","panic","mood",
    "sleep","emergency","first aid","choking","cpr","pregnant",
    "pregnancy","period","allergy","asthma","cancer","bp","thyroid",
    # Roman Urdu common
    "bukhar","dard","dawa","bimari","sehat","pareshani","neend","ghabrahat",
    "khansi","zukam","pait","dil","khoon","ilaj",
]

OFF_TOPIC_RESPONSE = (
    "I'm SehatBot, a health assistant — I can only help with medical and health "
    "questions (symptoms, medicines, lab reports, diet, mental health, emergencies). "
    "Please ask me something health-related and I'll be glad to help. | "
    "میں صرف صحت سے متعلق سوالات میں مدد کر سکتا ہوں۔"
)


def is_health_related(text: str) -> bool:
    """Fast keyword check: True if the message clearly mentions a health topic."""
    low = text.lower()
    return any(kw in low for kw in HEALTH_KEYWORDS)