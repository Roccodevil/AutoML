from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

class PreprocessAgent: 
    
    def run(self, state):
        print("-> Agent 3: Cleaning and preprocessing data...")
        df = state['raw_df']
        target = state['analysis']['target_variable']
        
        # --- Read Configuration ---
        config = state.get('node_configs', {}).get('agent_3_preprocess', {})
        mode = config.get('mode', 'default')
        
        use_impute = True; use_scale = True; use_ohe = True
        use_outliers = False; use_pca = False
        
        if mode == 'custom':
            use_impute = config.get('impute', False)
            use_scale = config.get('scale', False)
            use_ohe = config.get('encode', False)
            use_outliers = config.get('outliers', False)
            use_pca = config.get('pca', False)
            print(f"   Custom Config: Impute={use_impute}, Scale={use_scale}, Encode={use_ohe}, Outliers={use_outliers}, PCA={use_pca}")

        if target in df.columns:
            X = df.drop(columns=[target])
            y = df[target]
        else:
            X = df; y = pd.Series()

        # --- 1. Outlier Removal (Simple IQR) ---
        if use_outliers:
            print("   Removing outliers (IQR method)...")
            numeric_cols = X.select_dtypes(include=np.number).columns
            if not numeric_cols.empty:
                Q1 = X[numeric_cols].quantile(0.25)
                Q3 = X[numeric_cols].quantile(0.75)
                IQR = Q3 - Q1
                # Simple filter: remove rows where ANY column is an outlier
                # This can be aggressive, so we use a relaxed threshold (3.0 instead of 1.5)
                mask = ~((X[numeric_cols] < (Q1 - 3.0 * IQR)) | (X[numeric_cols] > (Q3 + 3.0 * IQR))).any(axis=1)
                X = X[mask]
                y = y[mask]
                print(f"   Rows remaining after outlier removal: {len(X)}")

        # Identify types (simplified for robustness)
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        transformers = []
        
        if numeric_features:
            steps = []
            if use_impute: steps.append(('imputer', SimpleImputer(strategy='mean')))
            if use_scale: steps.append(('scaler', StandardScaler()))
            if use_pca: steps.append(('pca', PCA(n_components=0.95))) # Retain 95% variance
            
            if steps: transformers.append(('num', Pipeline(steps=steps), numeric_features))
            else: transformers.append(('num', 'passthrough', numeric_features))

        if categorical_features:
            steps = []
            if use_impute: steps.append(('imputer', SimpleImputer(strategy='most_frequent')))
            if use_ohe: steps.append(('onehot', OneHotEncoder(handle_unknown='ignore', max_categories=20, sparse_output=False)))
            
            if steps: transformers.append(('cat', Pipeline(steps=steps), categorical_features))
            
        if not transformers: preprocessor = ColumnTransformer(transformers=[('pass', 'passthrough', X.columns)], remainder='drop')
        else: preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')

        print("   Fitting preprocessor...")
        X_processed = preprocessor.fit_transform(X)
        
        # Feature names logic (PCA breaks named features, so fallback)
        feature_names = [f"feat_{i}" for i in range(X_processed.shape[1])]
        if not use_pca:
            try: feature_names = preprocessor.get_feature_names_out()
            except: pass
        
        X_processed_df = pd.DataFrame(X_processed, columns=feature_names, index=X.index)
        cleaned_df = pd.concat([X_processed_df, y.reset_index(drop=True)], axis=1)

        print(f"   Preprocessing complete. New shape: {cleaned_df.shape}")
        state['cleaned_df'] = cleaned_df
        return state