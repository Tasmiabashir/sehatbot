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
Reference data (background medical knowledge only — these are example patient records from a dataset, NOT information about the current patient): {context}
Patient symptoms (this is ALL that is actually known about this patient): {symptoms}
IMPORTANT: Do not assume the current patient has any condition, disease, or medical history (e.g. diabetes, hypertension, pregnancy) that they did not explicitly mention in their symptoms above, even if the reference data mentions such conditions. The reference data is only for identifying which diseases match the described symptoms — never treat it as facts about this specific patient.
Cover: possible conditions, urgency level, which doctor to see, first aid, and red-flag warning signs — based ONLY on the symptoms the patient actually described.
LANGUAGE RULE: Detect the language of the user message. If it is written in English letters and English words, reply ONLY in English. If it is written in Urdu script or Roman Urdu (Urdu words in English letters like "bukhar", "pareshani"), reply in that same Urdu style. When replying in Urdu/Roman Urdu, use pure Urdu vocabulary only — do NOT use Hindi-origin words. Examples of words to AVOID and their correct Urdu replacements: "shanti" -> use "sukoon" or "aaram", "swasthya" -> use "sehat", "anubhav" -> use "tajurba" or "mahsoos", "atirikt" -> use "izafi", "prayas" -> use "koshish", "dhyan" -> use "tawajjo" or "khayal", "labh" -> use "faida", "vishesh" -> use "khaas". If you are ever unsure whether a word is Urdu or Hindi, choose the more common, everyday Pakistani Urdu word instead. Never switch to Urdu when the user wrote plain English. LENGTH RULE: Keep the answer short and to the point, about 5 to 6 lines for simple questions. Only go longer if the question truly needs more detail (like emergencies or multi-part questions). Use short headings and bullet points. Be warm and clear. Do NOT output JSON or code."""
    return ask_llm(prompt)

@tool
def analyze_lab_report(report_text: str) -> str:
    """Explains lab report values in simple Urdu/English."""
    docs    = search("lab_report", report_text)
    context = "\n".join([d.page_content for d in docs])
    prompt  = f"""You are a medical lab expert.
Context (reference data only, not facts about this patient unless they said it): {context}
Lab report: {report_text}
Explain each value in simple words, flag abnormal ones, and say what action to take. Do not assume this patient has any condition not shown in their own report values.
LANGUAGE RULE: Detect the language of the user message. If it is written in English letters and English words, reply ONLY in English. If it is written in Urdu script or Roman Urdu (Urdu words in English letters like "bukhar", "pareshani"), reply in that same Urdu style. When replying in Urdu/Roman Urdu, use pure Urdu vocabulary only — do NOT use Hindi-origin words. Examples of words to AVOID and their correct Urdu replacements: "shanti" -> use "sukoon" or "aaram", "swasthya" -> use "sehat", "anubhav" -> use "tajurba" or "mahsoos", "atirikt" -> use "izafi", "prayas" -> use "koshish", "dhyan" -> use "tawajjo" or "khayal", "labh" -> use "faida", "vishesh" -> use "khaas". If you are ever unsure whether a word is Urdu or Hindi, choose the more common, everyday Pakistani Urdu word instead. Never switch to Urdu when the user wrote plain English. LENGTH RULE: Keep the answer short and to the point, about 5 to 6 lines for simple questions. Only go longer if the question truly needs more detail (like emergencies or multi-part questions). Use short headings and bullet points. Be warm and clear. Do NOT output JSON or code."""
    return ask_llm(prompt)

@tool
def check_drug_interaction(medicines: str) -> str:
    """Checks safety of medicine combinations."""
    docs    = search("medicine_safety", medicines)
    context = "\n".join([d.page_content for d in docs])
    prompt  = f"""You are a pharmacist in Pakistan.
Context (reference drug-safety data only, not facts about this patient unless they said it): {context}
Medicines: {medicines}
Say clearly if the combination is safe, the risk level, why, and what to do instead. Do not assume this patient has any medical condition they did not mention.
LANGUAGE RULE: Detect the language of the user message. If it is written in English letters and English words, reply ONLY in English. If it is written in Urdu script or Roman Urdu (Urdu words in English letters like "bukhar", "pareshani"), reply in that same Urdu style. When replying in Urdu/Roman Urdu, use pure Urdu vocabulary only — do NOT use Hindi-origin words. Examples of words to AVOID and their correct Urdu replacements: "shanti" -> use "sukoon" or "aaram", "swasthya" -> use "sehat", "anubhav" -> use "tajurba" or "mahsoos", "atirikt" -> use "izafi", "prayas" -> use "koshish", "dhyan" -> use "tawajjo" or "khayal", "labh" -> use "faida", "vishesh" -> use "khaas". If you are ever unsure whether a word is Urdu or Hindi, choose the more common, everyday Pakistani Urdu word instead. Never switch to Urdu when the user wrote plain English. LENGTH RULE: Keep the answer short and to the point, about 5 to 6 lines for simple questions. Only go longer if the question truly needs more detail (like emergencies or multi-part questions). Use short headings and bullet points. Be warm and clear. Do NOT output JSON or code."""
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
LANGUAGE RULE: Detect the language of the user message. If it is written in English letters and English words, reply ONLY in English. If it is written in Urdu script or Roman Urdu (Urdu words in English letters like "bukhar", "pareshani"), reply in that same Urdu style. When replying in Urdu/Roman Urdu, use pure Urdu vocabulary only — do NOT use Hindi-origin words. Examples of words to AVOID and their correct Urdu replacements: "shanti" -> use "sukoon" or "aaram", "swasthya" -> use "sehat", "anubhav" -> use "tajurba" or "mahsoos", "atirikt" -> use "izafi", "prayas" -> use "koshish", "dhyan" -> use "tawajjo" or "khayal", "labh" -> use "faida", "vishesh" -> use "khaas". If you are ever unsure whether a word is Urdu or Hindi, choose the more common, everyday Pakistani Urdu word instead. Never switch to Urdu when the user wrote plain English. LENGTH RULE: Keep the answer short and to the point, about 5 to 6 lines for simple questions. Only go longer if the question truly needs more detail (like emergencies or multi-part questions). Use short headings and bullet points. Be warm and clear. Do NOT output JSON or code."""
        return ask_llm(prompt)
    except Exception as e:
        return f"OCR Error: {str(e)}"

@tool
def assess_mental_health(message: str) -> str:
    """Provides mental health support in Urdu/English."""
    docs    = search("mental_health", message)
    context = "\n".join([d.page_content for d in docs])
    history = memory.get_history_text()
    history_block = f"Earlier in this conversation (for background only):\n{history}\n\n" if history else ""
    prompt  = f"""You are a compassionate mental health assistant for Pakistan.
{history_block}Reference material (background only): {context}
The patient's NEW message right now: {message}

IMPORTANT: Your answer must respond specifically to what the patient JUST said in their NEW message above — not a generic template, and not a repeat of your previous reply. Read the new message carefully and notice its specific content (for example: overthinking, sleep problems, work stress, sadness, fear — each is different and needs a different, tailored response). Do not reuse the same coping tips or wording you used earlier in this conversation unless the patient is repeating the same concern.
Gently share how serious this seems (mild/moderate/severe/crisis), 2-3 coping tips that specifically fit what they described, and a helpline if needed (crisis: Umang 0317-4288665).
LANGUAGE RULE: Detect the language of the user message. If it is written in English letters and English words, reply ONLY in English. If it is written in Urdu script or Roman Urdu (Urdu words in English letters like "bukhar", "pareshani"), reply in that same Urdu style. When replying in Urdu/Roman Urdu, use pure Urdu vocabulary only — do NOT use Hindi-origin words. Examples of words to AVOID and their correct Urdu replacements: "shanti" -> use "sukoon" or "aaram", "swasthya" -> use "sehat", "anubhav" -> use "tajurba" or "mahsoos", "atirikt" -> use "izafi", "prayas" -> use "koshish", "dhyan" -> use "tawajjo" or "khayal", "labh" -> use "faida", "vishesh" -> use "khaas". If you are ever unsure whether a word is Urdu or Hindi, choose the more common, everyday Pakistani Urdu word instead. Never switch to Urdu when the user wrote plain English. LENGTH RULE: Keep the answer short and to the point, about 5 to 6 lines for simple questions. Only go longer if the question truly needs more detail (like emergencies or multi-part questions). Use short headings and bullet points. Be warm and clear. Do NOT output JSON or code."""
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
LANGUAGE RULE: Detect the language of the user message. If it is written in English letters and English words, reply ONLY in English. If it is written in Urdu script or Roman Urdu (Urdu words in English letters like "bukhar", "pareshani"), reply in that same Urdu style. When replying in Urdu/Roman Urdu, use pure Urdu vocabulary only — do NOT use Hindi-origin words. Examples of words to AVOID and their correct Urdu replacements: "shanti" -> use "sukoon" or "aaram", "swasthya" -> use "sehat", "anubhav" -> use "tajurba" or "mahsoos", "atirikt" -> use "izafi", "prayas" -> use "koshish", "dhyan" -> use "tawajjo" or "khayal", "labh" -> use "faida", "vishesh" -> use "khaas". If you are ever unsure whether a word is Urdu or Hindi, choose the more common, everyday Pakistani Urdu word instead. Never switch to Urdu when the user wrote plain English. LENGTH RULE: Keep the answer short and to the point, about 5 to 6 lines for simple questions. Only go longer if the question truly needs more detail (like emergencies or multi-part questions). Use short headings and bullet points. Be warm and clear. Do NOT output JSON or code."""
    return ask_llm(prompt)

@tool
def get_emergency_guide(emergency: str) -> str:
    """Returns immediate first aid steps for an emergency."""
    docs    = search("emergency", emergency)
    context = "\n".join([d.page_content for d in docs])
    prompt  = f"""You are an emergency first aid expert in Pakistan.
Context (retrieved from our first-aid dataset): {context}
Emergency described by the user: {emergency}

HOW TO DECIDE YOUR ANSWER:
1. First check: does the Context above genuinely match the specific emergency the user described (same injury/situation, not just similar words)?
2. If YES — base your answer primarily on that context, adding Pakistan-specific details (Rescue 1122, Edhi 115) as needed.
3. If NO — the context is irrelevant or only loosely related. In that case, IGNORE the context completely and instead answer using your own general medical first-aid knowledge for the SPECIFIC emergency the user actually described (whether it's an accident, heart attack, snake bite, drowning, poisoning, burns, fracture, choking, seizure, or anything else). Give medically standard, safe first-aid steps for that exact situation.
4. NEVER answer with an unrelated specific topic just because it appeared in the context (for example, never answer a road accident question with dental/teeth advice, or a burn question with sprain advice). The answer must always match what the user actually described.
5. For ANY vague/general emergency (like "accident", "someone is hurt", "emergency help"), give universal first-aid basics: check the scene is safe, check breathing and consciousness, do not move the person if a neck/back/head injury is possible, control bleeding with firm pressure, call for help immediately, and keep the person calm and warm.

Give numbered first-aid steps and emergency numbers (Rescue 1122, Edhi 115). Keep it short and urgent.
LANGUAGE RULE: Detect the language of the user message. If it is written in English letters and English words, reply ONLY in English. If it is written in Urdu script or Roman Urdu (Urdu words in English letters like "bukhar", "pareshani"), reply in that same Urdu style. When replying in Urdu/Roman Urdu, use pure Urdu vocabulary only — do NOT use Hindi-origin words. Examples of words to AVOID and their correct Urdu replacements: "shanti" -> use "sukoon" or "aaram", "swasthya" -> use "sehat", "anubhav" -> use "tajurba" or "mahsoos", "atirikt" -> use "izafi", "prayas" -> use "koshish", "dhyan" -> use "tawajjo" or "khayal", "labh" -> use "faida", "vishesh" -> use "khaas". If you are ever unsure whether a word is Urdu or Hindi, choose the more common, everyday Pakistani Urdu word instead. Never switch to Urdu when the user wrote plain English. LENGTH RULE: Keep the answer short and to the point, about 5 to 6 lines for simple questions. Only go longer if the question truly needs more detail (like emergencies or multi-part questions). Use short headings and bullet points. Be warm and clear. Do NOT output JSON or code."""
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