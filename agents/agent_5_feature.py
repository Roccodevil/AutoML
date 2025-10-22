# agents/agent_5_feature.py

import pandas as pd
import numpy as np
import re
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from core.llm_services import llm_powerful_api

class FeatureAgent:

    def __init__(self):
        self.parser = JsonOutputParser()
        # Prompt remains the same, asking for code based on original columns
        self.prompt = ChatPromptTemplate.from_template(
            """You are a data scientist. Your task is to suggest new features based on the provided columns.
            The problem is: "{problem}"
            The {problem_type} target is: "{target}"
            Consider these key columns: {columns_subset}

            Suggest 3 new features relevant to the problem. For each, provide "name" (string), "reason" (string),
            and "code" (string, a single line of Python pandas code using 'df' to create the new column from the ORIGINAL columns).
            Ensure the code is valid pandas and handles potential errors (e.g., use pd.to_numeric for conversions, np.log1p for log).

            Respond ONLY with a valid JSON list.
            {format_instructions}
            """
        )
        self.chain = self.prompt | llm_powerful_api | self.parser

    def run(self, state):
        print("-> Agent 5: Engineering new features with LLM suggestions...")
        cleaned_df = state['cleaned_df'] # This is the df we need to ADD features to
        raw_df = state['raw_df'].copy() # Use a copy of raw_df to GENERATE features safely
        analysis = state['analysis']
        target = analysis['target_variable']

        # --- Select subset of original columns for the prompt ---
        original_numeric = raw_df.select_dtypes(include=np.number).columns.tolist()
        original_categorical = raw_df.select_dtypes(include=['object', 'category']).columns.tolist()
        if target in original_numeric: original_numeric.remove(target)
        if target in original_categorical: original_categorical.remove(target)
        columns_for_prompt = original_numeric + original_categorical
        MAX_COLS_FOR_PROMPT = 50
        if len(columns_for_prompt) > MAX_COLS_FOR_PROMPT:
             print(f"   Warning: Too many original columns ({len(columns_for_prompt)}). Sending only the first {MAX_COLS_FOR_PROMPT} to LLM.")
             columns_for_prompt = columns_for_prompt[:MAX_COLS_FOR_PROMPT]
        # --- End of Column Selection Logic ---

        try:
            new_features_info = self.chain.invoke({
                "problem": state['problem_description'],
                "problem_type": analysis['problem_type'],
                "target": analysis['target_variable'],
                "columns_subset": columns_for_prompt,
                "format_instructions": self.parser.get_format_instructions()
            })

            print("   LLM suggested new features based on original columns:")
            newly_created_features = pd.DataFrame(index=raw_df.index) # DataFrame to store new features

            for feature in new_features_info:
                try:
                    # --- KEY CHANGE: Execute code on raw_df ---
                    code = feature['code']
                    # Ensure TotalCharges is numeric if needed
                    if 'TotalCharges' in code and 'TotalCharges' in raw_df.columns:
                        raw_df['TotalCharges'] = pd.to_numeric(raw_df['TotalCharges'], errors='coerce')
                        raw_df['TotalCharges'].fillna(0, inplace=True) # Basic imputation

                    exec_globals = {'df': raw_df, 'np': np, 'pd': pd}
                    exec(code, exec_globals)

                    # --- KEY CHANGE: Store the new feature ---
                    if feature['name'] in raw_df.columns:
                         newly_created_features[feature['name']] = raw_df[feature['name']]
                         print(f"      - Generated '{feature['name']}': {feature['reason']}")
                    else:
                         print(f"      - Code executed but feature '{feature['name']}' not found in raw_df. Skipping.")

                except Exception as e:
                    print(f"      - Failed to generate feature '{feature['name']}' on raw_df: {e}")

            # --- KEY CHANGE: Merge new features into cleaned_df ---
            if not newly_created_features.empty:
                print(f"   Merging {len(newly_created_features.columns)} new features into the processed dataset...")
                # Ensure indices match before merging
                final_df = cleaned_df.merge(newly_created_features, left_index=True, right_index=True, how='left')
                state['featured_df'] = final_df
                print(f"   Feature engineering complete. New shape: {final_df.shape}")
            else:
                print("   No new features were successfully generated.")
                state['featured_df'] = cleaned_df # Pass cleaned_df if no features added

            return state

        except Exception as e:
            print(f"   LLM feature engineering failed entirely: {e}. Using original cleaned data.")
            state['featured_df'] = state['cleaned_df'] # Pass through the original cleaned_df
            return state