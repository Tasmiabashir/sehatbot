import streamlit as st

MODES = [
    ("🩺", "Symptom Checker",  "symptom"),
    ("🧪", "Lab Report",       "lab"),
    ("💊", "Medicine Safety",  "medicine"),
    ("📷", "Prescription OCR", "ocr"),
    ("🧠", "Mental Health",    "mental"),
    ("🥗", "Diet Advisor",     "diet"),
    ("🚨", "Emergency Guide",  "emergency"),
]

QUICK_QUESTIONS = [
    "I have fever and headache",
    "Is Panadol safe with Brufen?",
    "I feel very anxious",
    "Diet plan for diabetes",    "Someone is choking",
]

def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-logo">
            <h2>🏥 SehatBot</h2>
            <p>Pakistan's AI Health Assistant</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Select Mode**")
        for icon, label, key in MODES:
            if st.button(f"{icon}  {label}", key=f"btn_{key}", use_container_width=True):
                st.session_state.active_mode = key
                st.session_state.messages    = []
                st.rerun()

        st.markdown("---")
        st.markdown("**Quick Questions**")
        for q in QUICK_QUESTIONS:
            if st.button(q, key=f"q_{q}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": q})
                st.rerun()

        st.markdown("---")
        st.markdown("""
        <div style="text-align:center;padding:0.5rem 0">
            <span class="status-badge">🟢 API Connected</span>
        </div>
        """, unsafe_allow_html=True)