import streamlit as st
import requests
import pathlib
import sys, os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from components.sidebar import render_sidebar
from components.chat_ui  import render_chat

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="SehatBot",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load external CSS file
css_path = pathlib.Path(__file__).parent / "components" / "styles.css"
st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

# Session state defaults
if "messages"    not in st.session_state: st.session_state.messages    = []
if "active_mode" not in st.session_state: st.session_state.active_mode = "symptom"

# Mode info
MODE_INFO = {
    "symptom"  : ("🩺 Symptom Checker",   "Describe your symptoms in Urdu or English"),
    "lab"      : ("🧪 Lab Report Analyzer","Upload your lab report PDF"),
    "medicine" : ("💊 Medicine Safety",    "Check drug interactions and side effects"),
    "ocr"      : ("📷 Prescription OCR",   "Upload a prescription photo"),
    "mental"   : ("🧠 Mental Health",      "Talk to a compassionate AI counselor"),
    "diet"     : ("🥗 Diet Advisor",       "Get a Pakistani diet plan for your condition"),
    "emergency": ("🚨 Emergency Guide",    "Get immediate first aid instructions"),
}

# Render sidebar
render_sidebar()

# Render header
title, subtitle = MODE_INFO[st.session_state.active_mode]
st.markdown(f"""
<div class="header-box">
    <h1>{title}</h1>
    <p>{subtitle}</p>
</div>
""", unsafe_allow_html=True)

# Layout
col_main, col_side = st.columns([2, 1])

with col_side:
    if st.session_state.active_mode == "lab":
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.markdown('<p class="upload-title">📄 Upload Lab Report</p>', unsafe_allow_html=True)
        lab_file = st.file_uploader("PDF only", type=["pdf"], label_visibility="collapsed")
        if lab_file and st.button("Analyze Report"):
            with st.spinner("Analyzing..."):
                res = requests.post(f"{API_URL}/upload-report", files={"file": lab_file})
                if res.status_code == 200:
                    st.session_state.messages.append({"role":"bot","content":res.json()["answer"]})
        st.markdown('</div>', unsafe_allow_html=True)

    elif st.session_state.active_mode == "ocr":
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.markdown('<p class="upload-title">📷 Upload Prescription</p>', unsafe_allow_html=True)
        img_file = st.file_uploader("JPG or PNG", type=["jpg","jpeg","png"], label_visibility="collapsed")
        cam_img  = st.camera_input("Or take a photo")
        upload   = img_file or cam_img
        if upload and st.button("Read Prescription"):
            with st.spinner("Reading..."):
                res = requests.post(f"{API_URL}/upload-prescription", files={"file": upload})
                if res.status_code == 200:
                    st.session_state.messages.append({"role":"bot","content":res.json()["answer"]})
        st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="info-card">
            <h4>🇵🇰 Pakistan-First AI</h4>
            <p>Trained on Pakistani diseases, medicines and diet culture</p>
        </div>
        <div class="info-card">
            <h4>🌐 Bilingual Support</h4>
            <p>Ask in Urdu or English — SehatBot understands both</p>
        </div>
        <div class="info-card">
            <h4>🔒 Private & Safe</h4>
            <p>Your health data stays on your device</p>
        </div>
        <div class="warning-card">
            <p>⚠️ SehatBot provides information only. Always consult a doctor.</p>
        </div>
        """, unsafe_allow_html=True)

with col_main:
    render_chat()