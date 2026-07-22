# 🏥 SehatBot — Pakistan's AI Health Assistant

A bilingual (Urdu/English) AI medical assistant built for Pakistan. SehatBot answers health questions using a RAG pipeline over Pakistan-specific medical knowledge, routes each question to one of 7 specialized tools via a LangGraph agent, and reads prescription images with OCR.

> ⚠️ **Disclaimer:** SehatBot provides information only. Always consult a doctor.

## ✨ Features

- 🩺 **Symptom Checker** — possible conditions, urgency level, and red-flag warnings
- 🧪 **Lab Report Analyzer** — explains test values in simple words
- 💊 **Medicine Safety** — checks drug–drug interactions
- 📷 **Prescription OCR** — reads prescription images (Tesseract) and extracts medicines, dosage, and frequency
- 🧠 **Mental Health Support** — coping tips + Pakistani helplines (Umang)
- 🥗 **Diet Advisor** — meal plans using Pakistani foods (daal, roti, sabzi)
- 🚨 **Emergency Guide** — first-aid steps + emergency numbers (Rescue 1122, Edhi 115)
- 🌐 **Bilingual** — ask in Urdu or English, get answers in the same language
- 🔁 **Provider Failover** — Groq primary → automatic Google Gemini backup on rate limits (no downtime on free tiers)

## 🛠 Tech Stack

- **Python 3.10+**
- **LangChain** — LLM orchestration and tool definitions
- **LangGraph** — agent StateGraph (router node → tool node)
- **ChromaDB** — persistent vector store for the medical knowledge base
- **Sentence-Transformers** (all-MiniLM-L6-v2) — embeddings (offline mode)
- **Groq** (llama-3.3-70b-versatile) — primary LLM
- **Google Gemini** (gemini-2.5-flash) — automatic backup LLM
- **FastAPI + Uvicorn** — backend REST API
- **Streamlit** — frontend chat UI
- **Tesseract (pytesseract)** — prescription image OCR
- **Pydantic** — request/response validation

## 🏗 Architecture Highlights

- **Plain-text tool routing:** instead of fragile native JSON tool-calling (which small open models often fail), the LLM answers one tiny question — *which tool?* — as plain text. The code then fills in the tool's parameters deterministically and executes it. More reliable and ~50% fewer tokens per query.
- **Graceful degradation:** every LLM call goes through a failover layer (`backend/llm.py`) — if Groq returns 429 (rate limit), the same request is retried on Gemini automatically, with message sanitization for Gemini's stricter format.
- **Offline embeddings:** HuggingFace offline mode so the app starts without internet access to the model hub.

## 📁 Project Structure

```
sehatbot/
├── README.md
├── requirements.txt
├── .gitignore
├── backend/
│   ├── main.py        # FastAPI app + routes
│   ├── agent.py       # LangGraph router → tool executor
│   ├── llm.py         # Groq → Gemini failover layer
│   ├── tools.py       # 7 @tool functions (RAG + OCR)
│   ├── rag.py         # load → chunk → embed → ChromaDB
│   ├── schemas.py     # Pydantic models
│   ├── config.py      # settings and model names
│   └── .env           # API keys (not committed)
├── frontend/
│   ├── app.py         # Streamlit entry point
│   └── components/
│       ├── sidebar.py
│       ├── chat_ui.py
│       └── styles.css
├── knowledge_base/    # Pakistan-specific medical PDFs
└── chroma_db/         # auto-generated vector store (not committed)
```

## 🚀 Setup

1. **Clone and create a virtual environment**
   ```bash
   git clone https://github.com/Tasmiabashir/sehatbot.git
   cd sehatbot
   python -m venv .venv
   .venv\Scripts\activate        # Windows
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Tesseract OCR** (for prescription reading)
   - Windows: install to the default path `C:\Program Files\Tesseract-OCR\`

4. **Add API keys** — create `backend/.env`:
   ```
   GROQ_API_KEY=your_groq_key
   GOOGLE_API_KEY=your_gemini_key
   ```

5. **Run the backend**
   ```bash
   cd backend
   python main.py        # FastAPI on http://localhost:8000
   ```

6. **Run the frontend** (new terminal)
   ```bash
   cd frontend
   streamlit run app.py
   ```

## 🔌 API Endpoints

| Method | Endpoint               | Description                                    |
|--------|------------------------|------------------------------------------------|
| GET    | `/`                    | Health check                                   |
| POST   | `/ask`                 | Ask a question (Urdu/English) → routed answer  |
| POST   | `/upload-report`       | Upload a lab report for analysis               |
| POST   | `/upload-prescription` | Upload a prescription image for OCR extraction |

## 📝 Notes

- OCR works best on **typed/printed** prescriptions; handwritten prescriptions are unreliable (a known limitation of Tesseract). Urdu OCR support is planned as future work.
- Free-tier LLM limits are handled gracefully: users see a friendly bilingual message instead of a frozen UI, and the Gemini backup keeps the app running when Groq's daily quota is exhausted.
- Planned next steps: conversation memory, guardrails with PII redaction, and RAGAS evaluation with a test-case suite.