"""
auth.py — Login / Sign Up for SehatBot.

Security choices (interview-ready):
- Passwords are NEVER stored in plain text. We store salt + PBKDF2-HMAC-SHA256
  hash (100,000 iterations) — the same approach Django uses by default.
- Users live in a local users.json (add it to .gitignore — never commit user data).
- Zero external dependencies and zero LLM/API calls — pure Python stdlib.
"""
import json
import os
import hashlib
import secrets
from pathlib import Path

import streamlit as st

# Store users next to the frontend, outside version control
USERS_FILE = Path(__file__).resolve().parent.parent / "users.json"
ITERATIONS = 100_000


# ── password hashing (salted PBKDF2) ──
def _hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode(), bytes.fromhex(salt), ITERATIONS
    ).hex()


def _load_users() -> dict:
    if USERS_FILE.exists():
        try:
            return json.loads(USERS_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_users(users: dict) -> None:
    USERS_FILE.write_text(json.dumps(users, indent=2), encoding="utf-8")


def signup(username: str, password: str) -> str:
    """Create an account. Returns '' on success or an error message."""
    username = username.strip().lower()
    if len(username) < 3:
        return "Username must be at least 3 characters."
    if len(password) < 6:
        return "Password must be at least 6 characters."
    users = _load_users()
    if username in users:
        return "This username is already taken."
    salt = secrets.token_hex(16)
    users[username] = {"salt": salt, "hash": _hash_password(password, salt)}
    _save_users(users)
    return ""


def login(username: str, password: str) -> bool:
    """True if the username/password pair is correct."""
    username = username.strip().lower()
    users = _load_users()
    user = users.get(username)
    if not user:
        return False
    return secrets.compare_digest(
        _hash_password(password, user["salt"]), user["hash"]
    )


# ── Streamlit UI ──
def render_auth_page() -> None:
    """Full-page login/signup shown when the user is not authenticated."""
    st.markdown(
        "<h1 style='text-align:center'>🏥 SehatBot</h1>"
        "<p style='text-align:center;color:#64748b'>Pakistan's AI Health Assistant "
        "— please log in to continue</p>",
        unsafe_allow_html=True,
    )

    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        tab_login, tab_signup = st.tabs(["🔑 Login", "📝 Sign Up"])

        with tab_login:
            l_user = st.text_input("Username", key="login_user")
            l_pass = st.text_input("Password", type="password", key="login_pass")
            if st.button("Login", use_container_width=True, type="primary"):
                if login(l_user, l_pass):
                    st.session_state.authenticated = True
                    st.session_state.username = l_user.strip().lower()
                    st.rerun()
                else:
                    st.error("Invalid username or password.")

        with tab_signup:
            s_user = st.text_input("Choose a username", key="signup_user")
            s_pass = st.text_input("Choose a password", type="password", key="signup_pass")
            s_pass2 = st.text_input("Confirm password", type="password", key="signup_pass2")
            if st.button("Create account", use_container_width=True, type="primary"):
                if s_pass != s_pass2:
                    st.error("Passwords do not match.")
                else:
                    err = signup(s_user, s_pass)
                    if err:
                        st.error(err)
                    else:
                        st.success("Account created! Please log in.")


def require_login() -> bool:
    """Call at the top of app.py. Returns True if logged in; otherwise renders
    the auth page and returns False (app should st.stop())."""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if st.session_state.authenticated:
        return True
    render_auth_page()
    return False


def render_logout() -> None:
    """Small logout control for the sidebar."""
    user = st.session_state.get("username", "")
    st.sidebar.markdown(f"👤 Logged in as **{user}**")
    if st.sidebar.button("Logout"):
        st.session_state.authenticated = False
        st.session_state.username = ""
        st.session_state.messages = []
        st.rerun()
