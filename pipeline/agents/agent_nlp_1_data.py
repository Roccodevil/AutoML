import pandas as pd
import os
import pdfplumber
import docx
from core.llm_services import llm_powerful_api
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

class NLPDataAgent:
    def __init__(self):
        self.kv_extraction_prompt = ChatPromptTemplate.from_template(
            """Extract structured text data from this document chunk.
            Return a JSON List: [ {{"text": "extracted text segment..."}}, ... ]
            Ignore headers/footers.
            Text: {text_data}"""
        )
        self.chain = self.kv_extraction_prompt | llm_powerful_api | JsonOutputParser()

    def _extract_from_doc(self, filepath, ext):
        print(f"   [Agent 1] Extracting from {ext}...")
        text_content = ""
        try:
            if ext == '.pdf':
                with pdfplumber.open(filepath) as pdf:
                    text_content = "\n".join([p.extract_text() or "" for p in pdf.pages])
            elif ext == '.docx':
                doc = docx.Document(filepath)
                text_content = "\n".join([p.text for p in doc.paragraphs])
            elif ext == '.txt':
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    text_content = f.read()
        except Exception as e:
            raise ValueError(f"Read error: {e}")

        # Chunking Strategy for Long Docs
        if len(text_content) > 5000:
            print("   [Agent 1] Document is large. Using LLM Extraction...")
            chunks = [text_content[i:i+4000] for i in range(0, len(text_content), 4000)]
            results = []
            for chunk in chunks:
                try:
                    res = self.chain.invoke({"text_data": chunk})
                    if isinstance(res, list): results.extend(res)
                except: pass
            
            if results: return pd.DataFrame(results)
        
        return pd.DataFrame({'text': [text_content]})

    def run(self, state):
        print("-> NLP Agent 1: Data Acquisition...")
        mode = state.get('acquisition_mode', 'input_text')
        inp = state.get('acquisition_input')
        df = None

        if mode == "upload":
            if not os.path.exists(inp): raise FileNotFoundError("File not found")
            ext = os.path.splitext(inp)[1].lower()
            if ext == '.csv': df = pd.read_csv(inp)
            else: df = self._extract_from_doc(inp, ext)
        elif mode == "input_text":
            df = pd.DataFrame({'text': [inp]})

        if df is None or df.empty: raise ValueError("Dataset empty.")

        # Identify Text Column
        text_col = None
        max_len = 0
        for col in df.columns:
            avg_len = df[col].astype(str).str.len().mean()
            if avg_len > max_len:
                max_len = avg_len
                text_col = col
        
        state['raw_df'] = df
        state['text_column'] = text_col or df.columns[0]
        try: state['data_preview_html'] = df.head(5).to_html(classes='table')
        except: pass
        
        print(f"   Loaded {df.shape}. Text Column: {state['text_column']}")
        return state