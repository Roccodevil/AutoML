import pandas as pd
import numpy as np
import re
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from core.llm_services import llm_powerful_api
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

class FeatureAgent:

    def __init__(self):
        self.parser = JsonOutputParser()
        self.prompt = ChatPromptTemplate.from_template(
            """You are a data scientist. Your task is to suggest new features based on the provided ORIGINAL columns.
            The problem is: "{problem}"
            The {problem_type} target is: "{target}"
            Consider these key original columns: {columns_subset}

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
        cleaned_df = state['cleaned_df'] 
        raw_df = state['raw_df'].copy() 
        analysis = state['analysis']
        target = analysis['target_variable']

        # --- FIX: Select a subset of ORIGINAL columns for the prompt ---
        original_features = [col for col in raw_df.columns if col != target]
        MAX_COLS_FOR_PROMPT = 50 
        if len(original_features) > MAX_COLS_FOR_PROMPT:
             print(f"   Warning: Too many original columns ({len(original_features)}). Sending a subset to LLM.")
             columns_for_prompt = original_features[:MAX_COLS_FOR_PROMPT]
        else:
             columns_for_prompt = original_features

        try:
            new_features_info = self.chain.invoke({
                "problem": state['problem_description'],
                "problem_type": analysis['problem_type'],
                "target": analysis['target_variable'],
                "columns_subset": columns_for_prompt,
                "format_instructions": self.parser.get_format_instructions()
            })

            print("   LLM suggested new features based on original columns:")
            newly_created_features = pd.DataFrame(index=raw_df.index)

            for feature in new_features_info:
                try:
                    # --- FIX: Execute code on raw_df ---
                    code = feature['code']
                    # Proactively fix common data type issues
                    if 'TotalCharges' in code and 'TotalCharges' in raw_df.columns:
                        raw_df['TotalCharges'] = pd.to_numeric(raw_df['TotalCharges'].replace(r'\s+', '', regex=True), errors='coerce')
                    
                    exec_globals = {'df': raw_df, 'np': np, 'pd': pd}
                    exec(code, exec_globals)

                    if feature['name'] in raw_df.columns:
                         newly_created_features[feature['name']] = raw_df[feature['name']]
                         print(f"      - Generated '{feature['name']}': {feature['reason']}")
                    else:
                         print(f"      - Code executed but feature '{feature['name']}' not found in raw_df. Skipping.")

                except Exception as e:
                    print(f"      - Failed to generate feature '{feature['name']}' on raw_df: {e}")

            # --- FIX: Preprocess and Merge new features ---
            if not newly_created_features.empty:
                print(f"   Merging {len(newly_created_features.columns)} new features into the processed dataset...")
                
                # Preprocess the new features just like the others
                numeric_new = newly_created_features.select_dtypes(include=np.number)
                if not numeric_new.empty:
                    imputer = SimpleImputer(strategy='mean')
                    scaler = StandardScaler()
                    newly_created_features[numeric_new.columns] = imputer.fit_transform(numeric_new)
                    newly_created_features[numeric_new.columns] = scaler.fit_transform(newly_created_features[numeric_new.columns])
                
                # We need to drop the target from cleaned_df before merge, then re-add
                y = cleaned_df[target]
                X_cleaned = cleaned_df.drop(columns=[target])
                
                final_X = X_cleaned.merge(newly_created_features, left_index=True, right_index=True, how='left')
                final_df = pd.concat([final_X, y], axis=1)

                state['featured_df'] = final_df
                print(f"   Feature engineering complete. New shape: {final_df.shape}")
            else:
                print("   No new features were successfully generated.")
                state['featured_df'] = cleaned_df # Pass cleaned_df if no features added

            return state

        except Exception as e:
            print(f"   LLM feature engineering failed entirely: {e}. Using original cleaned data.")
            state['featured_df'] = state['cleaned_df']
            return state