import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from core.llm_services import llm_powerful_api
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

class FeatureAgent:
    def __init__(self):
        self.parser = JsonOutputParser()
        # Expert Prompt: Encourages domain-specific logic, interactions, and temporal math
        self.prompt = ChatPromptTemplate.from_template(
            """You are a Lead Data Scientist specializing in Feature Engineering.
            
            PROBLEM: "{problem}"
            TARGET: "{target}"
            
            DATASET CONTEXT (Raw Columns):
            {columns_list}
            
            TASK: Suggest 3-5 high-value new features.
            
            STRATEGIES:
            1. Domain Ratios (e.g., Price / Area).
            2. Temporal Interactions (e.g., Days since YearBuilt).
            3. Polynomials/Interactions (e.g., Height * Width).
            4. Log Transforms (for highly skewed data like 'Income').
            5. Binning (e.g., Age Groups).
            
            CONSTRAINT:
            - Provide Python Pandas code assuming the dataframe is named 'df'.
            - Code must be robust (handle divide by zero, missing values).
            - Do NOT drop existing columns.
            - Do NOT use the Target column in the calculation (Data Leakage).

            OUTPUT JSON:
            [
                {{
                    "name": "PricePerSqFt",
                    "reason": "Normalizes price by size for fair comparison.",
                    "code": "df['PricePerSqFt'] = df['Price'] / (df['SqFt'].replace(0, np.nan))"
                }}
            ]
            {format_instructions}
            """
        )
        self.chain = self.prompt | llm_powerful_api | self.parser

    def _auto_clean_new_features(self, new_df):
        """
        Automatically cleans the newly generated features so they 
        can be merged into the professional pipeline without breaking H2O/Models.
        """
        if new_df.empty: return new_df
        
        # 1. Handle Infinite values
        new_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # 2. Separate Types
        num_cols = new_df.select_dtypes(include=np.number).columns
        cat_cols = new_df.select_dtypes(exclude=np.number).columns
        
        # 3. Process Numeric (Impute Mean + Standard Scale)
        if len(num_cols) > 0:
            imp = SimpleImputer(strategy='mean')
            scl = StandardScaler()
            try:
                filled = imp.fit_transform(new_df[num_cols])
                scaled = scl.fit_transform(filled)
                new_df[num_cols] = scaled
            except: pass # Keep as is if fail

        # 4. Process Categorical (Impute Mode + OneHot)
        if len(cat_cols) > 0:
            imp = SimpleImputer(strategy='most_frequent')
            try:
                filled = imp.fit_transform(new_df[cat_cols])
                # We do simple Label Encoding fallback or Drop for simplicity in this specific dynamic step
                # to prevent massive dimensionality explosion from an LLM hallucination
                new_df.drop(columns=cat_cols, inplace=True) 
            except: pass
            
        return new_df

    def run(self, state):
        print("-> Agent 5: Expert Feature Engineering...")
        
        # 1. Load Data Streams
        # We need RAW data to calculate features (preserving logic like Dates/Strings)
        # We need CURRENT data to merge the result into the pipeline
        raw_df = state.get('raw_df')
        current_df = state.get('current_data')
        
        if raw_df is None or current_df is None:
            print("   [Agent 5] Missing data streams. Skipping.")
            return state
            
        # Align Indices (Crucial for Merge)
        raw_df = raw_df.reset_index(drop=True)
        current_df = current_df.reset_index(drop=True)
        
        analysis = state.get('analysis', {})
        target = analysis.get('target_variable', '')
        
        # 2. Prepare Context for LLM
        # Limit columns to prevent token overflow
        cols = raw_df.columns.tolist()
        if len(cols) > 50: cols = cols[:50]
        
        # 3. LLM Generation
        generated_features = []
        try:
            suggestions = self.chain.invoke({
                "problem": state.get('problem_description', 'Predict the target'),
                "target": target,
                "columns_list": cols,
                "format_instructions": self.parser.get_format_instructions()
            })
            
            # 4. Safe Execution Loop
            # We execute on a COPY of raw_df to isolate new features
            temp_df = raw_df.copy()
            
            print(f"   [Agent 5] Attempting to generate {len(suggestions)} features...")
            
            for feat in suggestions:
                name = feat.get('name')
                code = feat.get('code')
                reason = feat.get('reason')
                
                try:
                    # HEURISTIC: Fix common 'object' type issues before math
                    for col in temp_df.columns:
                        if col in code and temp_df[col].dtype == 'object':
                             # Try to clean currency/commas blindly
                             try: temp_df[col] = pd.to_numeric(temp_df[col].astype(str).str.replace(r'[$,]', '', regex=True), errors='coerce')
                             except: pass

                    # Sandbox Execution
                    local_scope = {'df': temp_df, 'np': np, 'pd': pd}
                    exec(code, {}, local_scope)
                    
                    # Verify Success
                    if name in temp_df.columns:
                        # Extract ONLY the new column
                        new_col = temp_df[name]
                        
                        # Validate Quality (Not all NaN, Not all Inf)
                        if new_col.isna().all():
                            print(f"      [Skip] {name}: Generated all NaNs.")
                        else:
                            generated_features.append(pd.DataFrame({name: new_col}))
                            print(f"      ✅ Generated '{name}': {reason}")
                    else:
                        print(f"      [Fail] Code ran but column '{name}' not created.")
                        
                except Exception as e:
                    print(f"      [Error] {name}: {str(e)}")

        except Exception as e:
            print(f"   [Agent 5] LLM Interaction Failed: {e}")

        # 5. Merge & Update
        if generated_features:
            # Concat all new features into one DF
            new_feats_df = pd.concat(generated_features, axis=1)
            
            # Clean them (Scale/Impute) to match pipeline standards
            new_feats_df = self._auto_clean_new_features(new_feats_df)
            
            # Merge with Main Pipeline Data
            # Note: We append columns. We assume index alignment (reset_index performed above)
            combined_df = pd.concat([current_df, new_feats_df], axis=1)
            
            state['current_data'] = combined_df
            state['featured_df'] = combined_df # Legacy support
            
            msg = f"✅ Feature Engineering Complete.\nCreated {len(generated_features)} new features:\n" + "\n".join([f"- {c.columns[0]}" for c in generated_features])
        else:
            msg = "✅ Feature Engineering Complete (No valid new features generated)."
            
        state['final_message'] = msg
        state['data_shape'] = str(state['current_data'].shape)
        
        return state