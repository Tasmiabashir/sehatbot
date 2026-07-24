from langchain.tools import tool
from langchain_core.messages import HumanMessage
from backend.rag import search
from backend.llm import safe_invoke
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
Reply directly to the patient in the SAME language they used (Urdu or English). Use short headings and bullet points. Be warm and clear. Do NOT output JSON or code."""
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
Reply directly to the patient in the SAME language they used (Urdu or English). Use short headings and bullet points. Be warm and clear. Do NOT output JSON or code."""
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
Reply directly to the patient in the SAME language they used (Urdu or English). Use short headings and bullet points. Be warm and clear. Do NOT output JSON or code."""
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
Reply directly to the patient in the SAME language they used (Urdu or English). Use short headings and bullet points. Be warm and clear. Do NOT output JSON or code."""
        return ask_llm(prompt)
    except Exception as e:
        return f"OCR Error: {str(e)}"

@tool
def assess_mental_health(message: str) -> str:
    """Provides mental health support in Urdu/English."""
    docs    = search("mental_health", message)
    context = "\n".join([d.page_content for d in docs])
    prompt  = f"""You are a compassionate mental health assistant for Pakistan.
Context: {context}
Patient message: {message}
Gently share how serious this seems (mild/moderate/severe/crisis), practical coping tips, and helplines if needed (crisis: Umang 0317-4288665).
Reply directly to the patient in the SAME language they used (Urdu or English). Use short headings and bullet points. Be warm and clear. Do NOT output JSON or code."""
    return ask_llm(prompt)

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
Reply directly to the patient in the SAME language they used (Urdu or English). Use short headings and bullet points. Be warm and clear. Do NOT output JSON or code."""
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
Reply directly to the patient in the SAME language they used (Urdu or English). Use short headings and bullet points. Be warm and clear. Do NOT output JSON or code."""
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