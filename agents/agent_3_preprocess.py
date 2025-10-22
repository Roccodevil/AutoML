# agents/agent_3_preprocess.py

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
        
        # Ensure target is not in the features list
        features = [col for col in df.columns if col != target]
        X = df[features]
        y = df[target]

        # Identify feature types
        numeric_features = X.select_dtypes(include=np.number).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

        # Create preprocessing pipelines
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        # Create the main column transformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='passthrough'
        )

        print("   Fitting preprocessor...")
        X_processed = preprocessor.fit_transform(X)
        
        # Get feature names after transformation
        try:
            ohe_cols = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
        except:
            ohe_cols = [] # Handle case with no categorical features
            
        processed_cols = numeric_features + list(ohe_cols)
        
        # Handle remainder columns
        remainder_indices = preprocessor.remainder_indices_ if hasattr(preprocessor, 'remainder_indices_') else [i for i, (name, trans, cols) in enumerate(preprocessor.transformers_) if name == 'remainder']
        if isinstance(remainder_indices, list) and remainder_indices:
             # This logic might need adjustment based on scikit-learn version
             # Simplified: just get columns not in num or cat
             processed_col_set = set(numeric_features + categorical_features)
             remainder_cols = [col for col in X.columns if col not in processed_col_set]
        else:
             remainder_cols = []
             
        processed_cols.extend(remainder_cols)

        X_processed_df = pd.DataFrame(X_processed, columns=processed_cols, index=X.index)
        
        # Re-combine with target
        cleaned_df = pd.concat([X_processed_df, y.reset_index(drop=True)], axis=1)

        print(f"   Preprocessing complete. New shape: {cleaned_df.shape}")
        state['cleaned_df'] = cleaned_df
        return state