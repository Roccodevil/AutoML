import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class StagingAgent:
    def run(self, state):
        print("-> Agent 6: Staging Data (Professional Split)...")
        
        # 1. Get Unified Data Stream
        # Priority: current_data > featured_df > cleaned_df
        df = state.get('current_data')
        if df is None: df = state.get('featured_df')
        if df is None: df = state.get('cleaned_df')
        
        if df is None: 
            raise ValueError("Agent 6 Error: No data stream available. Please run previous steps.")
        
        # 2. Get Analysis Context
        analysis = state.get('analysis', {})
        target = analysis.get('target_variable')
        problem_type = analysis.get('problem_type', 'Classification')
        
        # 3. Target Validation
        if not target:
             raise ValueError("Agent 6 Error: Target variable is undefined. Run Agent 2 (Analysis) first.")
        
        if target not in df.columns:
            raise ValueError(f"CRITICAL: Target '{target}' is missing from the dataset columns: {list(df.columns[:5])}...")
            
        # 4. Clean Target (Drop NaNs)
        # We cannot train on rows where the answer (target) is missing
        initial_len = len(df)
        df = df.dropna(subset=[target])
        if len(df) < initial_len:
            print(f"   [Agent 6] Dropped {initial_len - len(df)} rows with missing target values.")

        if df.empty: 
            raise ValueError("Dataset is empty after dropping missing targets. Check your data quality.")

        # 5. Separate Features & Target
        X = df.drop(columns=[target])
        y = df[target]
        
        # 6. Determine Split Strategy
        test_size = 0.2
        stratify = None
        is_time_series = False
        
        # Check for Time Series (Heuristic: 'Date' in columns or sequential index)
        # If user explicitly requested time series in config, we'd respect that here.
        # For now, we assume standard tabular unless sorted index suggests otherwise.
        
        if problem_type == 'Classification':
            # Check class balance for Stratification
            class_counts = y.value_counts()
            min_class = class_counts.min()
            
            if min_class < 2:
                print(f"   [Agent 6] Warning: Rare class detected (count={min_class}). Stratification disabled.")
                stratify = None
            else:
                stratify = y # Enable Stratified Split to maintain class ratios
                
        # 7. Perform Split
        try:
            if is_time_series:
                # Temporal Split (No Shuffle)
                print("   [Agent 6] Performing Temporal Split (No Shuffle).")
                split_idx = int(len(X) * (1 - test_size))
                X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
                y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            else:
                # Standard Random Split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, 
                    test_size=test_size, 
                    stratify=stratify, 
                    random_state=42
                )
        except ValueError as e:
            # Fallback if stratify fails unexpectedly
            print(f"   [Agent 6] Stratification failed ({e}). Falling back to simple random split.")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, stratify=None, random_state=42
            )
        
        # 8. Save State (Pandas Objects for Agent 7)
        state['X_train'] = X_train
        state['X_test'] = X_test
        state['y_train'] = y_train
        state['y_test'] = y_test
        
        print(f"   Split Complete: Train={len(X_train)}, Test={len(X_test)}")
        
        # 9. Standalone Output Message
        split_type = "Temporal" if is_time_series else ("Stratified" if stratify is not None else "Random")
        
        state['final_message'] = (
            f"✅ Data Staging Complete.\n"
            f"• Strategy: {split_type} Split (80/20)\n"
            f"• Training Set: {X_train.shape[0]} rows\n"
            f"• Testing Set: {X_test.shape[0]} rows\n"
            f"• Features: {X_train.shape[1]} columns"
        )
        
        return state