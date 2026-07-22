import streamlit as st
import requests

API_URL = "http://localhost:8000"

MODE_HINTS = {
    "symptom"  : "Symptom check: {}",
    "lab"      : "Explain lab result: {}",
    "medicine" : "Check medicine safety: {}",
    "mental"   : "Mental health support: {}",
    "diet"     : "Diet advice for: {}",
    "emergency": "Emergency first aid for: {}",
}

def render_chat():
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)

    if not st.session_state.messages:
        mode    = st.session_state.active_mode
        st.markdown(f"""
        <div style="text-align:center;padding:3rem 1rem;color:#94a3b8">
            <p style="font-size:1rem;font-weight:500;color:#64748b">
                How can I help you today?
            </p>
            <p style="font-size:0.85rem">Type your question below in Urdu or English</p>
        </div>
        """, unsafe_allow_html=True)

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="user-bubble">{msg["content"]}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(f"""
            <div class="bot-bubble">
                <div class="bot-label">🏥 SehatBot</div>
                {msg["content"]}
            </div>""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    with st.form("chat_form", clear_on_submit=True):
        c1, c2 = st.columns([5, 1])
        with c1:
            user_input = st.text_input(
                "message",
                placeholder="Type your question here... (Urdu or English)",
                label_visibility="collapsed"
            )
        with c2:
            send = st.form_submit_button("Send →", use_container_width=True)

    if send and user_input.strip():
        st.session_state.messages.append({"role": "user", "content": user_input})
        mode   = st.session_state.active_mode
        prompt = MODE_HINTS.get(mode, "{}").format(user_input)

        with st.spinner("SehatBot is thinking..."):
            try:
                res    = requests.post(f"{API_URL}/ask", json={"question": prompt})
                answer = res.json()["answer"] if res.status_code == 200 else f"Error: {res.json().get('detail')}"
            except Exception as e:
                answer = f"Cannot connect to server. Make sure backend is running. ({e})"

        st.session_state.messages.append({"role": "bot", "content": answer})
        st.rerun()