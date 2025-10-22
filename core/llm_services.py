# core/llm_services.py

import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()

# --- 1. The Fast LLM (for Analysis, Staging) ---
# CORRECTED to the standard Llama 3 8B instruct model
llm_fast_api = ChatGroq(
    model_name="llama-3.1-8b-instant",  # <-- Standard name
    temperature=0.0
)

# --- 2. The Powerful LLM (for Generation, Feature Engineering) ---
# CORRECTED to the standard Llama 3 70B instruct model
llm_powerful_api = ChatGroq(
    model_name="llama-3.3-70b-versatile", # <-- Standard name
    temperature=0.7
)

print("LLM Services Initialized (Groq-only - Standard Models).")