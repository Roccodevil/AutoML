import pandas as pd
import numpy as np
import re
import json
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, Normalizer, Binarizer,
    OneHotEncoder, OrdinalEncoder, PowerTransformer, QuantileTransformer,
    FunctionTransformer, KBinsDiscretizer, PolynomialFeatures
)
from sklearn.decomposition import PCA, TruncatedSVD
from core.llm_services import llm_powerful_api
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

def _clip_outliers(X):
    data = X.values if hasattr(X, "values") else X
    try:
        lower = np.nanpercentile(data, 1, axis=0)
        upper = np.nanpercentile(data, 99, axis=0)
        return np.clip(data, lower, upper)
    except: return X

class PreprocessAgent:
    def __init__(self):
        self.strategy_prompt = ChatPromptTemplate.from_template(
            """You are a Senior Data Engineer. Design a Scikit-Learn preprocessing pipeline.
            DATA PROFILE: {data_profile}
            USER COMMAND: "{user_request}"
            DECISION RULES:
            1. Skewed > 1.0 -> 'yeo-johnson' or 'log'.
            2. Outliers -> 'robust' scaling or 'clip'.
            3. High Cardinality -> 'ordinal'.
            4. ID Columns -> DROP.
            
            OUTPUT JSON:
            {{
                "drop_columns": ["col_name"],
                "numeric_impute": "mean|median|knn|zero",
                "numeric_scale": "standard|minmax|robust|none",
                "numeric_transform": "none|log|yeo-johnson|box-cox|quantile_normal",
                "outlier_handling": "none|clip",
                "numeric_binning": "none|uniform|quantile|kmeans",
                "dimensionality_reduction": "none|pca",
                "categorical_encode": "onehot|ordinal"
            }}"""
        )
        self.chain = self.strategy_prompt | llm_powerful_api | JsonOutputParser()

    def _auto_fix_types(self, df):
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    cleaned = df[col].astype(str).str.replace(r'[$,\s]', '', regex=True)
                    if pd.to_numeric(cleaned, errors='coerce').notna().mean() > 0.7:
                        print(f"   [Agent 3] Auto-converting dirty numeric col: {col}")
                        df[col] = pd.to_numeric(cleaned, errors='coerce')
                except: pass
        return df

    def _generate_profile(self, df):
        summary = [f"Rows: {df.shape[0]}, Cols: {df.shape[1]}"]
        for col in df.columns[:15]: 
            dtype = df[col].dtype
            missing = df[col].isna().mean()
            if np.issubdtype(dtype, np.number):
                summary.append(f"- {col} (Num): {missing:.1%} missing")
            else:
                summary.append(f"- {col} (Cat): {missing:.1%} missing, {df[col].nunique()} unique")
        return "\n".join(summary)

    def run(self, state):
        print("-> Agent 3: Smart Preprocessing...")
        
        df = state.get('current_data')
        if df is None: raise ValueError("No data received.")
        df = df.copy()

        # Config & Mode
        config = state.get('node_configs', {}).get('agent_3_preprocess', {})
        mode = config.get('mode', 'default')
        
        # Target Management
        target = state.get('analysis', {}).get('target_variable')
        y = None
        if target and target in df.columns:
            y = df[target]
            df = df.drop(columns=[target])

        # Universal Cleaning
        df = self._auto_fix_types(df)
        
        # Determine Strategy
        if mode == 'custom':
            strategy = {
                "numeric_impute": config.get('numeric_impute', 'mean'),
                "numeric_scale": config.get('numeric_scale', 'standard'),
                "numeric_transform": config.get('numeric_transform', 'none'),
                "numeric_binning": config.get('numeric_binning', 'none'),
                "categorical_encode": config.get('categorical_encode', 'onehot'),
                "outlier_handling": config.get('outlier_handling', 'none'),
                "dimensionality_reduction": config.get('dimensionality_reduction', 'none'),
                "drop_columns": []
            }
        else:
            try:
                profile_text = self._generate_profile(df)
                strategy = self.chain.invoke({
                    "data_profile": profile_text,
                    "user_request": config.get('user_request', '')
                })
            except:
                strategy = {"numeric_impute": "mean", "numeric_scale": "standard", "categorical_encode": "ordinal"}

        # Apply Drops
        drops = strategy.get('drop_columns', [])
        for c in df.select_dtypes(exclude=np.number).columns:
            if df[c].nunique() == len(df): drops.append(c) # Auto-drop IDs
            
        real_drops = [c for c in drops if c in df.columns]
        if real_drops: df = df.drop(columns=real_drops)

        # Build Transformers
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
        transformers = []

        if num_cols:
            steps = []
            imp = strategy.get('numeric_impute', 'mean')
            if imp == 'median': steps.append(('imp', SimpleImputer(strategy='median')))
            elif imp == 'knn': steps.append(('imp', KNNImputer(n_neighbors=5)))
            elif imp == 'zero': steps.append(('imp', SimpleImputer(strategy='constant', fill_value=0)))
            else: steps.append(('imp', SimpleImputer(strategy='mean')))
            
            if strategy.get('outlier_handling') == 'clip':
                steps.append(('clip', FunctionTransformer(_clip_outliers, validate=False)))

            trans = strategy.get('numeric_transform', 'none')
            if trans == 'log': steps.append(('log', FunctionTransformer(np.log1p, validate=True)))
            elif trans == 'yeo-johnson': steps.append(('pwr', PowerTransformer(method='yeo-johnson')))
            elif trans == 'box-cox': steps.append(('box', PowerTransformer(method='box-cox')))
            elif trans == 'quantile_normal': steps.append(('qtl_n', QuantileTransformer(output_distribution='normal')))
            elif trans == 'binarize': steps.append(('bin', Binarizer(threshold=0.0)))

            binn = strategy.get('numeric_binning', 'none')
            if binn in ['uniform', 'quantile', 'kmeans']:
                steps.append(('kbins', KBinsDiscretizer(n_bins=5, encode='ordinal', strategy=binn)))

            scl = strategy.get('numeric_scale', 'standard')
            if scl == 'minmax': steps.append(('scl', MinMaxScaler()))
            elif scl == 'robust': steps.append(('scl', RobustScaler()))
            elif scl == 'maxabs': steps.append(('scl', MaxAbsScaler()))
            elif scl == 'normalizer': steps.append(('scl', Normalizer()))
            elif scl == 'none': pass
            else: steps.append(('scl', StandardScaler()))

            if strategy.get('dimensionality_reduction') == 'pca':
                steps.append(('pca', PCA(n_components=0.95)))

            transformers.append(('num', Pipeline(steps), list(num_cols)))

        if cat_cols:
            steps = []
            steps.append(('imp', SimpleImputer(strategy='most_frequent')))
            
            enc = strategy.get('categorical_encode', 'ordinal')
            if enc == 'onehot':
                steps.append(('ohe', OneHotEncoder(sparse_output=False, handle_unknown='ignore', max_categories=15)))
            else:
                steps.append(('ord', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)))
            
            transformers.append(('cat', Pipeline(steps), list(cat_cols)))

        # Execution
        try:
            if not num_cols and not cat_cols: raise ValueError("No features left.")

            preprocessor = ColumnTransformer(transformers, verbose_feature_names_out=False, remainder='passthrough')
            
            if isinstance(df, pd.Series): df = df.to_frame()
            X_processed = preprocessor.fit_transform(df)
            
            # Smart Renaming
            try: new_cols = preprocessor.get_feature_names_out()
            except: 
                new_cols = []
                if num_cols: new_cols.extend(num_cols)
                if cat_cols: new_cols.extend(cat_cols)
                while len(new_cols) < X_processed.shape[1]: new_cols.append(f"feat_{len(new_cols)}")
            
            df_processed = pd.DataFrame(X_processed, columns=new_cols)
            
            if y is not None:
                df_processed.reset_index(drop=True, inplace=True)
                y.reset_index(drop=True, inplace=True)
                df_processed[target] = y

            state['current_data'] = df_processed
            
            state['data_preview_html'] = df_processed.head(50).to_html(classes='table table-striped', border=0, index=False)
            state['final_message'] = f"✅ Preprocessing Complete. Shape: {df_processed.shape}"

        except Exception as e:
            print(f"   [Agent 3 Error] {e}")
            state['final_message'] = f"❌ Preprocessing Failed: {str(e)}"
            if y is not None: df[target] = y
            state['current_data'] = df 
            
        return state