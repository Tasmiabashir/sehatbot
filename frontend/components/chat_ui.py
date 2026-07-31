import streamlit as st
import requests
import re

API_URL = "http://localhost:8000"


def markdown_to_html(text):
    """Convert the LLM's Markdown into real HTML, since the chat bubble is
    inserted via unsafe_allow_html and browsers don't render raw Markdown."""
    lines = text.split('\n')
    html_lines = []
    in_list = False
    for line in lines:
        stripped = line.strip()

        # Headings: ### Heading  or  ## Heading  -> bold heading
        heading = re.match(r'^(#{1,6})\s*(.+)$', stripped)
        if heading:
            if in_list:
                html_lines.append('</ul>')
                in_list = False
            html_lines.append(f'<div style="font-weight:700;font-size:1.05rem;margin:0.5rem 0 0.25rem">{heading.group(2)}</div>')
            continue

        # Bullet lines: "* item" or "- item"
        if stripped.startswith('* ') or stripped.startswith('- '):
            if not in_list:
                html_lines.append('<ul>')
                in_list = True
            html_lines.append(f'<li>{stripped[2:].strip()}</li>')
        else:
            if in_list:
                html_lines.append('</ul>')
                in_list = False
            html_lines.append(line)

    if in_list:
        html_lines.append('</ul>')

    html = '\n'.join(html_lines)
    # **bold** -> <strong>bold</strong>  (done after line handling so it works everywhere)
    html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)
    return html

MODE_HINTS = {
    "symptom"  : "Symptom check: {}",
    "lab"      : "Explain lab result: {}",
    "medicine" : "Check medicine safety: {}",
    "mental"   : "Mental health support: {}",
    "diet"     : "Diet advice for: {}",
    "emergency": "Emergency first aid for: {}",
}

def render_chat():
    # Build ALL the inner HTML first, then render it in ONE st.markdown call.
    # (Splitting div-open / content / div-close across separate st.markdown
    #  calls creates an empty box, since each call is its own isolated block.)
    inner_html = ""

    if not st.session_state.messages:
        inner_html += """
        <div style="text-align:center;padding:3rem 1rem;color:#94a3b8">
            <p style="font-size:1rem;font-weight:500;color:#64748b">
                How can I help you today?
            </p>
            <p style="font-size:0.85rem">Type your question below in Urdu or English</p>
        </div>
        """

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            inner_html += f'<div class="user-bubble">{msg["content"]}</div>'
        else:
            formatted = markdown_to_html(msg["content"])
            inner_html += f"""
            <div class="bot-bubble">
                <div class="bot-label">🏥 SehatBot</div>
                {formatted}
            </div>"""

    st.markdown(f'<div class="chat-container">{inner_html}</div>', unsafe_allow_html=True)

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