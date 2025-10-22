# agents/agent_1_data.py

import pandas as pd
import io
import re
import time
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from core.llm_services import llm_powerful_api

class DataAgent: 
    def __init__(self):
        self.llm_chain = (
            ChatPromptTemplate.from_template(
                """You are a data generation bot. Generate a synthetic dataset based on the user's description.
                Respond ONLY with the data in CSV format. Do not include any other text, explanation, or markdown.
                
                Description: "{description}"
                
                Start the CSV data now:"""
            )
            | llm_powerful_api
            | StrOutputParser()
        )

    def search_online(self, query):
        print(f"   Searching online for: '{query}'...")
        time.sleep(1)
        # Placeholder: Return a well-known dataset
        try:
            df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
            print("   Found and loaded Titanic dataset as a placeholder.")
            return df
        except Exception as e:
            print(f"   Failed to load placeholder dataset: {e}")
            raise ValueError("Online search (placeholder) failed.")

    def generate_synthetic(self, description):
        print(f"   Generating synthetic data with LLM based on: '{description}'...")
        
        try:
            csv_string = self.llm_chain.invoke({"description": description})
            
            # Clean the output to get just the CSV
            csv_lines = []
            for line in csv_string.strip().split('\n'):
                if re.search(r'[a-zA-Z]', line) and ',' in line:
                    csv_lines.append(line)
                elif csv_lines: 
                    csv_lines.append(line)
            
            csv_data = "\n".join(csv_lines).strip()
            if not csv_data:
                raise ValueError("LLM did not return data in a recognizable CSV format.")
            
            df = pd.read_csv(io.StringIO(csv_data))
            print(f"   LLM generated synthetic data. Shape: {df.shape}")
            return df

        except Exception as e:
            print(f"   LLM data generation failed: {e}. Raising error.")
            raise ValueError(f"LLM data generation failed: {e}")

    def run(self, state):
        print("-> Agent 1: Acquiring data...")
        mode = state['acquisition_mode']
        input_data = state['acquisition_input']
        
        df = pd.DataFrame()
        
        if mode == "upload":
            print(f"   Reading dataset from: {input_data}")
            df = pd.read_csv(input_data)
        
        elif mode == "search":
            df = self.search_online(input_data)
            
        elif mode == "generate":
            df = self.generate_synthetic(input_data)

        if df.empty:
            raise ValueError("Data Acquisition failed. The resulting DataFrame is empty.")
            
        print(f"   Dataset acquired. Shape: {df.shape}")
        state['raw_df'] = df
        return state