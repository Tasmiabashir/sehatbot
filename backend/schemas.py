from pydantic import BaseModel
from typing import List, Optional

# ── REQUESTS (what user sends IN) 

class AskRequest(BaseModel):
    """User sends a question with an optional mode"""
    question : str
    mode     : str = "auto" 

class SymptomRequest(BaseModel):
    """User describes symptoms"""
    symptoms : str
    age      : Optional[int] = None
    duration : Optional[str] = None 

class MedicineRequest(BaseModel):
    """User enters medicine names to check"""
    medicines : List[str]           

class DietRequest(BaseModel):
    """User describes their condition and food habits"""
    condition    : str                
    current_diet : Optional[str] = None  

class EmergencyRequest(BaseModel):
    """User describes an emergency"""
    situation : str                   

# RESPONSES 

class SehatBotResponse(BaseModel):
    """Standard response for all modes"""
    status  : str             
    mode    : str              
    answer  : str              
    details : Optional[dict] = None  

class SymptomResponse(BaseModel):
    """Structured response for symptom checker"""
    status             : str
    possible_conditions: List[str]
    urgency_level      : str    
    recommended_doctor : str
    first_aid          : str
    red_flags          : List[str]

class MedicineResponse(BaseModel):
    """Structured response for medicine safety"""
    status      : str
    safe        : bool
    risk_level  : str      
    explanation : str
    suggestion  : str