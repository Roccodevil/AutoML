import pandas as pd
import io
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from core.llm_services import llm_powerful_api
from huggingface_hub import HfApi
from datasets import load_dataset
from langchain_community.tools.tavily_search import TavilySearchResults

# --- SAFE IMPORT: Prevents crash if 'kaggle' library is missing ---
try:
    from kaggle.api.kaggle_api_extended import KaggleApi
except ImportError:
    KaggleApi = None

class DataAgent:
    def __init__(self):
        # 1. LLM Chain (For Synthetic Data)
        self.llm_chain = (
            ChatPromptTemplate.from_template(
                """You are a data generation bot. Generate a synthetic dataset based on the user's description. Respond ONLY with CSV data.
                Description: "{description}"
                Start the CSV data now:"""
            )
            | llm_powerful_api | StrOutputParser()
        )
        
        # 2. Hugging Face API (Always available)
        self.hf_api = HfApi()
        
        # 3. Kaggle API (Conditional Load)
        self.kaggle_api = None
        
        # Check environment variables
        k_user = os.environ.get('KAGGLE_USERNAME')
        k_key = os.environ.get('KAGGLE_KEY')
        # Check local file
        local_creds = os.path.expanduser("~/.kaggle/kaggle.json")
        has_creds = (k_user and k_key) or os.path.exists(local_creds)

        if KaggleApi and has_creds:
            try:
                self.kaggle_api = KaggleApi()
                self.kaggle_api.authenticate()
                print("   [Agent 1] ✅ Kaggle API authenticated.")
            except Exception as e:
                print(f"   [Agent 1] ⚠️ Kaggle Auth failed: {e}")
        else:
            if not KaggleApi:
                print("   [Agent 1] ℹ️ 'kaggle' library not installed. Skipping Kaggle features.")
            else:
                print("   [Agent 1] ℹ️ No Kaggle credentials found. Skipping Kaggle features.")

        # 4. Web Search
        try: 
            self.web_search = TavilySearchResults()
        except: 
            self.web_search = None

    def search_online(self, query, max_results=5):
        print(f"   Searching for: '{query}'...")
        results = []
        
        # A. Kaggle Search (Only runs if authenticated)
        if self.kaggle_api:
            try:
                k_res = self.kaggle_api.dataset_list(search=query, sort_by='votes', file_type='csv')
                for d in k_res[:max_results]:
                    # Estimate size in MB
                    size_bytes = getattr(d, 'totalBytes', 0)
                    size_mb = round(size_bytes / (1024*1024), 2) if size_bytes else 0
                    results.append({
                        'source': 'kaggle', 
                        'id': d.ref, 
                        'display': f"Kaggle: {d.title} ({size_mb}MB)"
                    })
            except Exception as e: 
                print(f"   [Agent 1] Kaggle search error: {e}")

        # B. Hugging Face Search (Always runs)
        try:
            hf_res = list(self.hf_api.list_datasets(search=query, full=True, limit=50))
            count = 0
            # Sort by downloads to show popular datasets
            for info in sorted(hf_res, key=lambda x: x.downloads or 0, reverse=True):
                if count >= max_results: break
                if info.downloads > 10: # Filter out empty/unused datasets
                     results.append({
                         'source': 'huggingface', 
                         'id': info.id, 
                         'display': f"HF: {info.id} (DLs: {info.downloads})"
                     })
                     count += 1
        except Exception as e: 
            print(f"   [Agent 1] HF search error: {e}")

        if not results: 
            # Only raise error if BOTH failed to find anything
            raise ValueError("No datasets found. Try a different query or upload a file.")
        return results

    def download_selected(self, source, dataset_id, download_path="temp_data"):
        print(f"   Downloading {source}: {dataset_id}")
        os.makedirs(download_path, exist_ok=True)

        if source == 'kaggle':
            if not self.kaggle_api:
                raise ValueError("Kaggle API not active. Check credentials or library installation.")
            try:
                # Clear previous downloads to avoid confusion
                for f in os.listdir(download_path): 
                    try: os.remove(os.path.join(download_path, f))
                    except: pass
                
                self.kaggle_api.dataset_download_files(dataset_id, path=download_path, unzip=True)
                
                # Find the largest CSV file
                csvs = [f for f in os.listdir(download_path) if f.lower().endswith('.csv')]
                if not csvs: raise FileNotFoundError("No CSV files found in this Kaggle dataset.")
                
                largest_csv = max(csvs, key=lambda f: os.path.getsize(os.path.join(download_path, f)))
                return pd.read_csv(os.path.join(download_path, largest_csv))
            except Exception as e: 
                print(f"   [Agent 1] Kaggle download failed: {e}")
                raise

        elif source == 'huggingface':
            try:
                dataset = load_dataset(dataset_id)
                # Auto-detect split (train/test/validation)
                split = 'train' if 'train' in dataset else list(dataset.keys())[0]
                return dataset[split].to_pandas()
            except Exception as e: 
                print(f"   [Agent 1] HF download failed: {e}")
                raise
        
        raise ValueError(f"Unknown source: {source}")

    def generate_synthetic(self, description):
        print(f"   Generating synthetic data for: {description}")
        try:
            csv_str = self.llm_chain.invoke({"description": description}).strip()
            # Remove markdown blocks if LLM adds them
            csv_str = csv_str.replace("```csv", "").replace("```", "")
            
            # Simple header validation
            if '\n' in csv_str:
                first_line = csv_str.split('\n')[0]
                if ',' not in first_line: 
                    # Skip preamble text if LLM chatters before CSV
                    first_n = csv_str.find('\n')
                    csv_str = csv_str[first_n+1:]
            
            return pd.read_csv(io.StringIO(csv_str))
        except Exception as e: 
            raise ValueError(f"Generation failed: {e}")

    def run(self, state):
        print("-> Agent 1: Acquiring data...")
        mode = state['acquisition_mode']
        inp = state['acquisition_input']

        # 1. UPLOAD MODE (Always Safe)
        if mode == "upload": 
            if not os.path.exists(inp): raise FileNotFoundError(f"Uploaded file missing: {inp}")
            state['raw_df'] = pd.read_csv(inp)
            
        # 2. SEARCH MODE (Safe fallback to HF if Kaggle missing)
        elif mode == "search":
            state['search_results'] = self.search_online(inp)
            return state # Return early to show results in UI
            
        # 3. DOWNLOAD MODE
        elif mode == "download_selected":
            dl_dir = os.path.join(state.get('results_dir', '.'), "downloaded_data")
            state['raw_df'] = self.download_selected(inp['source'], inp['id'], dl_dir)
            
        # 4. GENERATE MODE (Always Safe)
        elif mode == "generate": 
            state['raw_df'] = self.generate_synthetic(inp)
            
        # Common Post-Processing
        if 'raw_df' in state:
            df = state['raw_df']
            print(f"   Data acquired. Shape: {df.shape}")
            # Generate Preview HTML for Frontend
            state['data_preview_html'] = df.head(50).to_html(classes='table', border=0, index=False)
            state['data_shape'] = str(df.shape)
            
        return state