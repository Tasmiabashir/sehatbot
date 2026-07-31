from langchain.tools import tool
from langchain_core.messages import HumanMessage
from rag import search
from llm import safe_invoke
import memory
import pytesseract
from PIL import Image
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def ask_llm(prompt):
    response = safe_invoke([HumanMessage(content=prompt)])
    return str(response.content)

@tool
def check_symptoms(symptoms: str) -> str:
    """Analyzes patient symptoms and returns possible conditions with urgency level.
    Input should be a plain text string describing the symptoms."""
    docs    = search("symptom_checker", symptoms)
    context = "\n".join([d.page_content for d in docs])
    prompt  = f"""You are a medical assistant for Pakistan.
Context: {context}
Patient symptoms: {symptoms}
Cover: possible conditions, urgency level, which doctor to see, first aid, and red-flag warning signs.
LANGUAGE RULE: Detect the language of the user message. If it is written in English letters and English words, reply ONLY in English. If it is written in Urdu script or Roman Urdu (Urdu words in English letters like "bukhar", "pareshani"), reply in that same Urdu style. Never switch to Urdu when the user wrote plain English. LENGTH RULE: Keep the answer short and to the point, about 5 to 6 lines for simple questions. Only go longer if the question truly needs more detail (like emergencies or multi-part questions). Use short headings and bullet points. Be warm and clear. Do NOT output JSON or code."""
    return ask_llm(prompt)

@tool
def analyze_lab_report(report_text: str) -> str:
    """Explains lab report values in simple Urdu/English."""
    docs    = search("lab_report", report_text)
    context = "\n".join([d.page_content for d in docs])
    prompt  = f"""You are a medical lab expert.
Context: {context}
Lab report: {report_text}
Explain each value in simple words, flag abnormal ones, and say what action to take.
LANGUAGE RULE: Detect the language of the user message. If it is written in English letters and English words, reply ONLY in English. If it is written in Urdu script or Roman Urdu (Urdu words in English letters like "bukhar", "pareshani"), reply in that same Urdu style. Never switch to Urdu when the user wrote plain English. LENGTH RULE: Keep the answer short and to the point, about 5 to 6 lines for simple questions. Only go longer if the question truly needs more detail (like emergencies or multi-part questions). Use short headings and bullet points. Be warm and clear. Do NOT output JSON or code."""
    return ask_llm(prompt)

@tool
def check_drug_interaction(medicines: str) -> str:
    """Checks safety of medicine combinations."""
    docs    = search("medicine_safety", medicines)
    context = "\n".join([d.page_content for d in docs])
    prompt  = f"""You are a pharmacist in Pakistan.
Context: {context}
Medicines: {medicines}
Say clearly if the combination is safe, the risk level, why, and what to do instead.
LANGUAGE RULE: Detect the language of the user message. If it is written in English letters and English words, reply ONLY in English. If it is written in Urdu script or Roman Urdu (Urdu words in English letters like "bukhar", "pareshani"), reply in that same Urdu style. Never switch to Urdu when the user wrote plain English. LENGTH RULE: Keep the answer short and to the point, about 5 to 6 lines for simple questions. Only go longer if the question truly needs more detail (like emergencies or multi-part questions). Use short headings and bullet points. Be warm and clear. Do NOT output JSON or code."""
    return ask_llm(prompt)

@tool
def read_prescription_ocr(image_path: str) -> str:
    """Reads prescription image using OCR and extracts medicine names."""
    try:
        image = Image.open(image_path)
        text  = pytesseract.image_to_string(image)
        prompt = f"""Extract medicine names, dosage and frequency from this prescription text:
{text}
List each medicine with its dosage and frequency.
LANGUAGE RULE: Detect the language of the user message. If it is written in English letters and English words, reply ONLY in English. If it is written in Urdu script or Roman Urdu (Urdu words in English letters like "bukhar", "pareshani"), reply in that same Urdu style. Never switch to Urdu when the user wrote plain English. LENGTH RULE: Keep the answer short and to the point, about 5 to 6 lines for simple questions. Only go longer if the question truly needs more detail (like emergencies or multi-part questions). Use short headings and bullet points. Be warm and clear. Do NOT output JSON or code."""
        return ask_llm(prompt)
    except Exception as e:
        return f"OCR Error: {str(e)}"

@tool
def assess_mental_health(message: str) -> str:
    """Provides mental health support in Urdu/English."""
    docs    = search("mental_health", message)
    context = "\n".join([d.page_content for d in docs])
    history = memory.get_history_text()
    history_block = f"Recent conversation (for context):\n{history}\n\n" if history else ""
    prompt  = f"""You are a compassionate mental health assistant for Pakistan.
{history_block}Context: {context}
Patient message: {message}
Gently share how serious this seems (mild/moderate/severe/crisis), practical coping tips, and helplines if needed (crisis: Umang 0317-4288665).
Use the recent conversation to stay consistent and remember what the person already told you.
LANGUAGE RULE: Detect the language of the user message. If it is written in English letters and English words, reply ONLY in English. If it is written in Urdu script or Roman Urdu (Urdu words in English letters like "bukhar", "pareshani"), reply in that same Urdu style. Never switch to Urdu when the user wrote plain English. LENGTH RULE: Keep the answer short and to the point, about 5 to 6 lines for simple questions. Only go longer if the question truly needs more detail (like emergencies or multi-part questions). Use short headings and bullet points. Be warm and clear. Do NOT output JSON or code."""
    reply = ask_llm(prompt)
    memory.add_turn(message, reply)   # remember this exchange for next time
    return reply

@tool
def get_diet_plan(condition: str) -> str:
    """Creates Pakistani diet plan for a health condition."""
    docs    = search("diet_advisor", condition)
    context = "\n".join([d.page_content for d in docs])
    prompt  = f"""You are a nutritionist in Pakistan.
Context: {context}
Condition: {condition}
Create a diet plan using Pakistani foods (daal, roti, sabzi, chawal).
Give: foods to eat, foods to limit, foods to avoid, and a simple daily meal plan.
LANGUAGE RULE: Detect the language of the user message. If it is written in English letters and English words, reply ONLY in English. If it is written in Urdu script or Roman Urdu (Urdu words in English letters like "bukhar", "pareshani"), reply in that same Urdu style. Never switch to Urdu when the user wrote plain English. LENGTH RULE: Keep the answer short and to the point, about 5 to 6 lines for simple questions. Only go longer if the question truly needs more detail (like emergencies or multi-part questions). Use short headings and bullet points. Be warm and clear. Do NOT output JSON or code."""
    return ask_llm(prompt)

@tool
def get_emergency_guide(emergency: str) -> str:
    """Returns immediate first aid steps for an emergency."""
    docs    = search("emergency", emergency)
    context = "\n".join([d.page_content for d in docs])
    prompt  = f"""You are an emergency first aid expert in Pakistan.
Context: {context}
Emergency: {emergency}
Give numbered first-aid steps and emergency numbers (Rescue 1122, Edhi 115). Keep it short and urgent.
LANGUAGE RULE: Detect the language of the user message. If it is written in English letters and English words, reply ONLY in English. If it is written in Urdu script or Roman Urdu (Urdu words in English letters like "bukhar", "pareshani"), reply in that same Urdu style. Never switch to Urdu when the user wrote plain English. LENGTH RULE: Keep the answer short and to the point, about 5 to 6 lines for simple questions. Only go longer if the question truly needs more detail (like emergencies or multi-part questions). Use short headings and bullet points. Be warm and clear. Do NOT output JSON or code."""
    return ask_llm(prompt)

ALL_TOOLS = [
    check_symptoms,
    analyze_lab_report,
    check_drug_interaction,
    read_prescription_ocr,
    assess_mental_health,
    get_diet_plan,
    get_emergency_guide
]