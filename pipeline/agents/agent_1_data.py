import pandas as pd
import io
import re
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from core.llm_services import llm_powerful_api
from huggingface_hub import HfApi
from datasets import load_dataset
from kaggle.api.kaggle_api_extended import KaggleApi
from langchain_community.tools.tavily_search import TavilySearchResults

class DataAgent:
    def __init__(self):
        self.llm_chain = (
            ChatPromptTemplate.from_template(
                """You are a data generation bot. Generate a synthetic dataset based on the user's description.
                Respond ONLY with the data in CSV format.
                Description: "{description}"
                Start the CSV data now:"""
            )
            | llm_powerful_api
            | StrOutputParser()
        )
        self.hf_api = HfApi()
        try:
            self.kaggle_api = KaggleApi()
            self.kaggle_api.authenticate()
        except Exception:
            self.kaggle_api = None
            
        try:
            self.web_search = TavilySearchResults()
        except:
            self.web_search = None

    def search_online(self, query, max_results=5):
        print(f"   Searching for: '{query}'...")
        results = []
        
        # Kaggle Search
        if self.kaggle_api:
            try:
                k_res = self.kaggle_api.dataset_list(search=query, sort_by='votes', file_type='csv')
                for d in k_res[:max_results]:
                    results.append({'source': 'kaggle', 'id': d.ref, 'display': f"Kaggle: {d.title} ({d.ref})"})
            except Exception as e: print(f"Kaggle error: {e}")

        # HF Search
        try:
            hf_res = list(self.hf_api.list_datasets(search=query, full=True, limit=50))
            count = 0
            for info in sorted(hf_res, key=lambda x: x.downloads or 0, reverse=True):
                if count >= max_results: break
                if info.downloads > 50:
                     results.append({'source': 'huggingface', 'id': info.id, 'display': f"HF: {info.id} (DLs: {info.downloads})"})
                     count += 1
        except Exception as e: print(f"HF error: {e}")

        if not results: raise ValueError("Online dataset search returned no results.")
        return results

    def download_selected(self, source, dataset_id, download_path="temp_data"):
        print(f"   Downloading {source}: {dataset_id}")
        os.makedirs(download_path, exist_ok=True)

        if source == 'kaggle':
            try:
                # Clear dir
                for f in os.listdir(download_path): os.remove(os.path.join(download_path, f))
                self.kaggle_api.dataset_download_files(dataset_id, path=download_path, unzip=True)
                # Find csv
                csvs = [f for f in os.listdir(download_path) if f.lower().endswith('.csv')]
                if not csvs: raise FileNotFoundError("No CSV in Kaggle dataset.")
                return pd.read_csv(os.path.join(download_path, max(csvs, key=lambda f: os.path.getsize(os.path.join(download_path, f)))))
            except Exception as e: print(f"Kaggle DL fail: {e}"); raise

        elif source == 'huggingface':
            try:
                dataset = load_dataset(dataset_id)
                split = 'train' if 'train' in dataset else list(dataset.keys())[0]
                return dataset[split].to_pandas()
            except Exception as e: print(f"HF DL fail: {e}"); raise
        
        raise ValueError(f"Unknown source: {source}")

    def generate_synthetic(self, description):
        print(f"   Generating data for: {description}")
        try:
            csv_str = self.llm_chain.invoke({"description": description}).strip()
            # Basic cleanup
            if '\n' in csv_str:
                first_n = csv_str.find('\n')
                if ',' not in csv_str[:first_n]: csv_str = csv_str[first_n+1:]
            return pd.read_csv(io.StringIO(csv_str))
        except Exception as e: raise ValueError(f"Generation failed: {e}")

    def run(self, state):
        print("-> Agent 1: Acquiring data...")
        mode = state['acquisition_mode']
        inp = state['acquisition_input']

        if mode == "upload":
            state['raw_df'] = pd.read_csv(inp)
        elif mode == "search":
            state['search_results'] = self.search_online(inp)
            return state # Pause for selection
        elif mode == "download_selected":
            dl_dir = os.path.join(state.get('results_dir', '.'), "downloaded_data")
            state['raw_df'] = self.download_selected(inp['source'], inp['id'], dl_dir)
        elif mode == "generate":
            state['raw_df'] = self.generate_synthetic(inp)
            
        print(f"   Data acquired. Shape: {state['raw_df'].shape}")
        return state