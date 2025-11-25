from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np

class PreprocessAgent: 
    
    def run(self, state):
        print("-> Agent 3: Cleaning and preprocessing data...")
        df = state['raw_df']
        target = state['analysis']['target_variable']
        
        # Separate features and target
        # Drop target from X if it exists
        if target in df.columns:
            X = df.drop(columns=[target])
            y = df[target]
        else:
            # Fallback: assume target is separate or already handled
            X = df
            y = pd.Series() # Empty series if target not found (shouldn't happen in this flow)

        # --- Simplified Feature Type Identification ---
        # 1. Identify Numeric Columns
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # 2. Identify Categorical Columns (Object/Category)
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # 3. Handle potential "numeric-as-string" columns
        # If a categorical column has few unique values, keep it categorical.
        # If it has many unique values and looks numeric, try to convert it.
        for col in categorical_features[:]: # Iterate over a copy
            try:
                # Try converting to numeric to check
                pd.to_numeric(X[col])
                # If successful and high cardinality, move to numeric
                # (Low cardinality likely means it's a categorical code, like zip code)
                if X[col].nunique() > 20: 
                    X[col] = pd.to_numeric(X[col])
                    numeric_features.append(col)
                    categorical_features.remove(col)
            except (ValueError, TypeError):
                pass # Not numeric

        print(f"   Identified {len(numeric_features)} numeric features.")
        print(f"   Identified {len(categorical_features)} categorical features.")

        # --- Create Transformers ---
        transformers = []
        
        # Only add numeric transformer if there are numeric features
        if numeric_features:
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ])
            transformers.append(('num', numeric_transformer, numeric_features))

        # Only add categorical transformer if there are categorical features
        if categorical_features:
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', max_categories=20, sparse_output=False))
            ])
            transformers.append(('cat', categorical_transformer, categorical_features))

        # Create the main column transformer
        # If NO features found (rare), use 'passthrough' to avoid crash
        if not transformers:
             print("   Warning: No features identified for preprocessing. Passing through.")
             preprocessor = ColumnTransformer(transformers=[('pass', 'passthrough', X.columns)], remainder='drop')
        else:
             preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')

        print("   Fitting preprocessor...")
        X_processed = preprocessor.fit_transform(X)
        
        # --- Reconstruct DataFrame with Column Names ---
        try:
            feature_names = preprocessor.get_feature_names_out()
        except Exception as e:
            # Fallback for older sklearn or if get_feature_names_out fails
            feature_names = [f"feat_{i}" for i in range(X_processed.shape[1])]
        
        X_processed_df = pd.DataFrame(X_processed, columns=feature_names, index=X.index)
        
        # Re-combine with target
        cleaned_df = pd.concat([X_processed_df, y.reset_index(drop=True)], axis=1)

        print(f"   Preprocessing complete. New shape: {cleaned_df.shape}")
        state['cleaned_df'] = cleaned_df
        return state