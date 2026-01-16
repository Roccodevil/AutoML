import pandas as pd
from sklearn.model_selection import train_test_split
from core.llm_services import llm_powerful_api
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

class StagingAgent:
    def __init__(self):
        self.chain = (
            ChatPromptTemplate.from_template(
                """Suggest a train-test split strategy.
                Data Shape: {shape}. Target: {target}. Problem Type: {problem}.
                Return JSON: {{"test_size": 0.2, "stratify": true/false}}"""
            )
            | llm_powerful_api | JsonOutputParser()
        )
    
    def run(self, state):
        print("-> Agent 6: Staging data...")
        
        # 1. Get Data
        df = state.get('featured_df')
        if df is None: df = state.get('cleaned_df')
        if df is None: raise ValueError("No data available for staging.")
        
        target = state['analysis'].get('target_variable')
        problem_type = state['analysis'].get('problem_type', 'classification').lower()
        
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in dataset.")

        # --- 2. CRITICAL FIX: Clean Target Variable ---
        # Remove rows where target is NaN/None to prevent sorting errors
        initial_len = len(df)
        df = df.dropna(subset=[target])
        
        if len(df) < initial_len:
            print(f"   [Agent 6] Dropped {initial_len - len(df)} rows with missing target values.")
        
        if len(df) == 0:
            raise ValueError("All rows have missing target values. Cannot train.")

        # Separate X and y
        X = df.drop(columns=[target])
        y = df[target]
        
        # 3. Determine Strategy
        test_size = 0.2
        stratify = None
        
        # Stratify only if we have enough samples per class
        if 'classification' in problem_type:
            # Check class counts
            class_counts = y.value_counts()
            # We can only stratify if every class has at least 2 samples
            if (class_counts >= 2).all():
                stratify = y
                print("   Using Stratified Split (Balanced classes)")
            else:
                print("   Skipping stratification (some classes have < 2 samples)")

        # 4. Split
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, stratify=stratify, random_state=42
            )
        except ValueError as e:
            # Fallback if stratification fails (e.g., extremely rare classes)
            print(f"   Stratification failed ({e}). Falling back to random split.")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, stratify=None, random_state=42
            )
        
        state['X_train'] = X_train
        state['X_test'] = X_test
        state['y_train'] = y_train
        state['y_test'] = y_test
        
        print(f"   Data split complete. Train: {X_train.shape}, Test: {X_test.shape}")
        return state