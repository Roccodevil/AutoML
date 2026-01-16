import pandas as pd
import io
import os
import json
import math
import time
import pdfplumber
import docx
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from core.llm_services import llm_powerful_api
from huggingface_hub import HfApi
from datasets import load_dataset
from langchain_community.tools.tavily_search import TavilySearchResults

# --- SAFE IMPORT: Kaggle ---
try: 
    from kaggle.api.kaggle_api_extended import KaggleApi
except ImportError: 
    KaggleApi = None

class DataAgent:
    def __init__(self):
        self.kv_extraction_prompt = ChatPromptTemplate.from_template(
            """You are an Expert Data Extraction AI.
            Your task is to convert unstructured text chunks into a structured JSON list.
            ANALYSIS STEPS:
            1. Scan the text for repeating record patterns.
            2. Identify common fields.
            3. Extract EVERY record found.
            CRITICAL OUTPUT RULES:
            - Return a **JSON LIST** of objects.
            - Do NOT nest the list.
            - If a record is cut off, IGNORE it.
            Raw Text Chunk:
            {text_data}
            """
        )
        self.kv_chain = self.kv_extraction_prompt | llm_powerful_api | JsonOutputParser()

        self.gen_prompt = ChatPromptTemplate.from_template(
            "Generate a synthetic dataset for: {description}. Return ONLY a valid JSON list of objects."
        )
        self.gen_chain = self.gen_prompt | llm_powerful_api | JsonOutputParser()

        self.hf_api = HfApi()
        
        self.kaggle_api = None
        k_user = os.environ.get('KAGGLE_USERNAME')
        k_key = os.environ.get('KAGGLE_KEY')
        local_creds = os.path.expanduser("~/.kaggle/kaggle.json")
        if KaggleApi and ((k_user and k_key) or os.path.exists(local_creds)):
            try:
                self.kaggle_api = KaggleApi()
                self.kaggle_api.authenticate()
            except: pass

        try: self.web_search = TavilySearchResults()
        except: self.web_search = None

    def _load_structured_file(self, filepath, ext):
        try:
            if ext in ['.xlsx', '.xls']: return pd.read_excel(filepath)
            elif ext == '.json':
                try: return pd.read_json(filepath)
                except: 
                    with open(filepath) as f: return pd.json_normalize(json.load(f))
            elif ext == '.xml': return pd.read_xml(filepath)
            elif ext == '.parquet': return pd.read_parquet(filepath)
        except: pass
        return None

    def _extract_from_doc(self, filepath, ext):
        text_content = ""
        tables = []

        if ext == '.pdf':
            with pdfplumber.open(filepath) as pdf:
                for page in pdf.pages:
                    extracted = page.extract_table()
                    if extracted: tables.extend(extracted)
                    text_content += (page.extract_text() or "") + "\n"
        elif ext == '.docx':
            doc = docx.Document(filepath)
            for table in doc.tables:
                data = [[cell.text for cell in row.cells] for row in table.rows]
                tables.extend(data)
            text_content += "\n".join([p.text for p in doc.paragraphs])
        elif ext == '.txt':
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                text_content = f.read()

        if len(tables) > 5: 
            headers = tables[0]
            headers = [f"{h}_{i}" if headers.count(h) > 1 else h for i, h in enumerate(headers)]
            return pd.DataFrame(tables[1:], columns=headers)

        CHUNK_SIZE = 3000
        OVERLAP = 200 
        total_length = len(text_content)
        extracted_dfs = []
        current_pos = 0
        chunk_counter = 0
        
        while current_pos < total_length:
            chunk_counter += 1
            end_pos = min(current_pos + CHUNK_SIZE, total_length)
            chunk = text_content[current_pos:end_pos]
            try:
                if chunk_counter > 1: time.sleep(0.5)
                json_data = self.kv_chain.invoke({"text_data": chunk})
                
                chunk_df = None
                if isinstance(json_data, list):
                    chunk_df = pd.DataFrame(json_data)
                elif isinstance(json_data, dict):
                    if all(isinstance(v, list) for v in json_data.values()):
                        chunk_df = pd.DataFrame(json_data)
                    else:
                        chunk_df = pd.DataFrame([json_data])
                
                if chunk_df is not None and not chunk_df.empty:
                    extracted_dfs.append(chunk_df)
            except: pass
            current_pos += (CHUNK_SIZE - OVERLAP)

        if len(extracted_dfs) > 0:
            full_df = pd.concat(extracted_dfs, ignore_index=True)
            full_df = full_df.drop_duplicates()
            return full_df
        
        raise ValueError("Could not extract data.")

    def search_online(self, query, max_results=5):
        results = []
        if self.kaggle_api:
            try:
                k_res = self.kaggle_api.dataset_list(search=query, sort_by='votes', file_type='csv')
                for d in k_res[:max_results]:
                    results.append({'source': 'kaggle', 'id': d.ref, 'display': f"Kaggle: {d.title}"})
            except: pass
        try:
            hf_res = list(self.hf_api.list_datasets(search=query, full=True, limit=50))
            for info in sorted(hf_res, key=lambda x: x.downloads or 0, reverse=True)[:max_results]:
                 if info.downloads > 10: results.append({'source': 'huggingface', 'id': info.id, 'display': f"HF: {info.id}"})
        except: pass
        return results

    def download_selected(self, source, dataset_id, download_path):
        os.makedirs(download_path, exist_ok=True)
        if source == 'kaggle':
            self.kaggle_api.dataset_download_files(dataset_id, path=download_path, unzip=True)
            csvs = [f for f in os.listdir(download_path) if f.endswith('.csv')]
            return pd.read_csv(os.path.join(download_path, max(csvs, key=lambda f: os.path.getsize(os.path.join(download_path, f)))))
        elif source == 'huggingface':
            ds = load_dataset(dataset_id)
            return ds[list(ds.keys())[0]].to_pandas()
        return None

    def run(self, state):
        print("-> Agent 1: Data Acquisition...")
        mode = state['acquisition_mode']
        inp = state['acquisition_input']
        df = None

        if mode == "upload":
            if not os.path.exists(inp): raise FileNotFoundError(f"File missing: {inp}")
            ext = os.path.splitext(inp)[1].lower()
            if ext == '.csv': df = pd.read_csv(inp)
            elif ext in ['.xlsx', '.xls', '.json', '.xml', '.parquet']: df = self._load_structured_file(inp, ext)
            elif ext in ['.pdf', '.docx', '.txt']: df = self._extract_from_doc(inp, ext)
            else: raise ValueError(f"Unsupported format: {ext}")

        elif mode == "search":
            state['search_results'] = self.search_online(inp)
            return state
        elif mode == "generate":
            json_data = self.gen_chain.invoke({"description": inp})
            df = pd.DataFrame(json_data)
        elif mode == "download_selected":
             dl_dir = os.path.join(state.get('results_dir', '.'), "downloaded_data")
             df = self.download_selected(inp['source'], inp['id'], dl_dir)

        if df is not None:
            # 1. Drop Empties
            df.dropna(how='all', inplace=True) 
            df.dropna(axis=1, how='all', inplace=True) 
            
            # 2. Heal Header
            if all(str(c).isdigit() for c in df.columns):
                new_header = df.iloc[0]
                df = df[1:]
                df.columns = new_header
            
            # 3. Shuffle
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)

            # 4. FIX: Force Numeric Conversion
            # This turns "5.1" (string) into 5.1 (float) so VizAgent can plot it
            print("   [Agent 1] Auto-converting numeric columns...")
            df = df.apply(pd.to_numeric, errors='ignore')

            print(f"   Data Loaded. Shape: {df.shape}")
            state['raw_df'] = df
            state['data_preview_html'] = df.head(50).to_html(classes='table', border=0, index=False)
            state['data_shape'] = str(df.shape)
            
        return state