# core/llm_services.py

import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

llm_fast_api = ChatGroq(
    model_name="llama-3.1-8b-instant", 
    temperature=0.0
)


llm_powerful_api = ChatGroq(
    model_name="llama-3.3-70b-versatile", 
    temperature=0.7
)

print("LLM Services Initialized (Groq-only - Standard Models).")