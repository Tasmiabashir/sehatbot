import os
from dotenv import load_dotenv

# Load secret keys from .env file
load_dotenv()

# ── LLM (AI Brain) 
GROQ_API_KEY   = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")          # backup engine key
LLM_MODEL      = "llama-3.3-70b-versatile"            # primary (Groq)
BACKUP_MODEL   = "gemini-2.5-flash"                  # backup (Gemini)

# ── EMBEDDINGS (Text → Numbers)
EMBED_MODEL  = "all-MiniLM-L6-v2"         

# ── CHROMADB (Vector Database) 
CHROMA_PATH  = "chroma_db"                
# ── RAG SETTINGS 
CHUNK_SIZE    = 500    # each text piece = 500 characters
CHUNK_OVERLAP = 50     # overlap between pieces = 50 characters
TOP_K         = 3      # retrieve top 3 matching chunks

# ── KNOWLEDGE BASE FOLDERS 
KB_BASE = "knowledge_base"

MODE_FOLDERS = {
    "symptom_checker"   : f"{KB_BASE}/mode1_symptoms",
    "lab_report"        : f"{KB_BASE}/mode2_lab_report",
    "medicine_safety"   : f"{KB_BASE}/mode3_medicine_safety",
    "mental_health"     : f"{KB_BASE}/mode5_mental_health",
    "diet_advisor"      : f"{KB_BASE}/mode6_diet_advisor",
    "emergency"         : f"{KB_BASE}/mode7_emergency",
}

# ── APP SETTINGS 
APP_TITLE    = "SehatBot"
APP_SUBTITLE = "Pakistan's AI Medical Health Assistant"