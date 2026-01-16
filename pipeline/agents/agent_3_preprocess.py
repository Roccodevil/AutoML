import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, 
    OneHotEncoder, OrdinalEncoder, PowerTransformer, QuantileTransformer,
    FunctionTransformer, KBinsDiscretizer, PolynomialFeatures
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from core.llm_services import llm_powerful_api
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

class PreprocessAgent:
    def __init__(self):
        self.strategy_prompt = ChatPromptTemplate.from_template(
            """You are a Lead Data Scientist. Build a preprocessing strategy.

            DATA PROFILE:
            - Rows: {rows}, Cols: {cols}
            - Numerical Cols: {num_cols}
            - Text/Categorical Cols: {cat_cols}
            - Missing: {missing_info}
            - User Command: "{user_request}" (PRIORITIZE THIS!)

            CAPABILITIES:
            1. IMPUTATION: 'mean', 'median', 'knn', 'zero'.
            2. SCALING: 'standard', 'minmax', 'robust'.
            3. TRANSFORMATION: 'yeo-johnson', 'quantile', 'log'.
            4. DISCRETIZATION (Binning): 'kmeans', 'uniform', 'quantile'.
            5. INTERACTIONS: 'poly' (PolynomialFeatures).
            6. TEXT: 'tfidf' (for long text columns).
            7. REDUCTION: 'pca'.

            OUTPUT JSON:
            {{
                "numeric_impute": "mean",
                "numeric_scale": "standard",
                "numeric_transform": "none",
                "outlier_handling": "none" | "clip",
                "dimensionality_reduction": "none" | "pca",
                "feature_interaction": "none" | "poly",
                "binned_columns": [], 
                "text_columns": [],
                "categorical_impute": "mode",
                "categorical_encode": "onehot",
                "drop_columns": []
            }}
            """
        )
        self.llm_chain = self.strategy_prompt | llm_powerful_api | JsonOutputParser()

    def _handle_dates(self, df):
        """Expands datetime columns"""
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    sample = df[col].dropna().head(10)
                    if len(sample) > 0 and pd.to_datetime(sample, errors='coerce').notna().all():
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                except: pass

        date_cols = df.select_dtypes(include=['datetime64']).columns
        for col in date_cols:
            df[f"{col}_year"] = df[col].dt.year
            df[f"{col}_month"] = df[col].dt.month
            df[f"{col}_day"] = df[col].dt.day
            df = df.drop(columns=[col])
        return df

    def _get_transformers(self, config):
        # 1. Numeric
        num_steps = []
        imp = config.get('numeric_impute', 'mean')
        if imp == 'knn': num_steps.append(('imputer', KNNImputer(n_neighbors=5)))
        elif imp == 'median': num_steps.append(('imputer', SimpleImputer(strategy='median')))
        elif imp == 'zero': num_steps.append(('imputer', SimpleImputer(strategy='constant', fill_value=0)))
        else: num_steps.append(('imputer', SimpleImputer(strategy='mean')))
        
        if config.get('outlier_handling') == 'clip':
            def clip_outliers(X):
                lower = np.percentile(X, 1, axis=0)
                upper = np.percentile(X, 99, axis=0)
                return np.clip(X, lower, upper)
            num_steps.append(('outlier_clip', FunctionTransformer(clip_outliers, validate=False)))

        trans = config.get('numeric_transform', 'none')
        if trans == 'yeo-johnson': num_steps.append(('transform', PowerTransformer(method='yeo-johnson')))
        elif trans == 'quantile': num_steps.append(('transform', QuantileTransformer(output_distribution='normal')))
        elif trans == 'log': num_steps.append(('log', FunctionTransformer(np.log1p, validate=False)))
        
        if config.get('feature_interaction') == 'poly':
            num_steps.append(('poly', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)))

        scale = config.get('numeric_scale', 'standard')
        if scale == 'minmax': num_steps.append(('scaler', MinMaxScaler()))
        elif scale == 'robust': num_steps.append(('scaler', RobustScaler()))
        elif scale != 'none': num_steps.append(('scaler', StandardScaler()))

        if config.get('dimensionality_reduction') == 'pca':
            num_steps.append(('pca', PCA(n_components=0.95)))

        # 2. Categorical
        cat_steps = []
        cat_imp = config.get('categorical_impute', 'mode')
        if cat_imp == 'constant': cat_steps.append(('imputer', SimpleImputer(strategy='constant', fill_value='missing')))
        else: cat_steps.append(('imputer', SimpleImputer(strategy='most_frequent')))
        
        enc = config.get('categorical_encode', 'onehot')
        if enc == 'ordinal': 
            cat_steps.append(('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)))
        else: 
            cat_steps.append(('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)))

        # 3. Binning
        bin_steps = [
            ('imputer', SimpleImputer(strategy='mean')),
            ('bin', KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform'))
        ]

        # 4. Text
        text_steps = [
            ('imputer', SimpleImputer(strategy='constant', fill_value='')),
            ('reshape', FunctionTransformer(lambda x: x.reshape(-1), validate=False)),
            ('tfidf', TfidfVectorizer(max_features=50, stop_words='english'))
        ]

        return Pipeline(num_steps), Pipeline(cat_steps), Pipeline(bin_steps), Pipeline(text_steps)

    def run(self, state):
        print("-> Agent 3: Advanced Preprocessing...")
        
        df = state.get('cleaned_df')
        if df is None: df = state.get('raw_df')
        if df is None: raise ValueError("No data found.")
        
        # 1. Expand Dates
        df = self._handle_dates(df.copy())

        # --- CRITICAL: Protect Target Variable ---
        target_col = state.get('analysis', {}).get('target_variable')
        y_series = None
        if target_col and target_col in df.columns:
            print(f"   [Agent 3] Detaching target '{target_col}' for safety.")
            y_series = df[target_col].copy()
            df = df.drop(columns=[target_col])

        # 2. Identify Types
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        object_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        
        # 3. Get LLM Strategy
        user_config = state.get('node_configs', {}).get('agent_3_preprocess', {})
        user_request = user_config.get('user_request', '')
        
        stats = {
            "rows": len(df), "cols": len(df.columns),
            "num_cols": numeric_cols, "cat_cols": object_cols,
            "missing_info": df.isnull().sum().to_dict(),
            "user_request": user_request 
        }

        try:
            config = self.llm_chain.invoke(stats)
            print(f"   LLM Strategy: {config}")
        except:
            config = {"numeric_impute": "mean", "numeric_scale": "standard"}

        # 4. Separate Columns for Pipelines
        binned_cols = [c for c in config.get('binned_columns', []) if c in numeric_cols]
        # Remove binned cols from standard numeric list so they don't get processed twice
        numeric_cols = [c for c in numeric_cols if c not in binned_cols]
        
        text_cols = [c for c in config.get('text_columns', []) if c in object_cols]
        # Remove text cols from standard categorical list
        object_cols = [c for c in object_cols if c not in text_cols]

        # 5. Build Transformers
        num_pipe, cat_pipe, bin_pipe, text_pipe = self._get_transformers(config)

        transformers_list = [
            ('num', num_pipe, numeric_cols),
            ('cat', cat_pipe, object_cols)
        ]
        if binned_cols: transformers_list.append(('bin', bin_pipe, binned_cols))
        if text_cols: 
             # TF-IDF usually handles 1 column at a time
             for tc in text_cols:
                 transformers_list.append((f'text_{tc}', text_pipe, tc))

        preprocessor = ColumnTransformer(
            transformers=transformers_list,
            verbose_feature_names_out=False,
            remainder='passthrough' 
        )
        
        print("   Applying Transformations...")
        try:
            processed_data = preprocessor.fit_transform(df)
            
            # Generate Names
            new_cols = []
            if hasattr(preprocessor, 'get_feature_names_out'):
                try: new_cols = preprocessor.get_feature_names_out()
                except: pass
            
            if len(new_cols) != processed_data.shape[1]:
                new_cols = [f"feat_{i}" for i in range(processed_data.shape[1])]

            processed_df = pd.DataFrame(processed_data, columns=new_cols)
            
            # --- CRITICAL: Re-Attach Target ---
            if y_series is not None:
                print(f"   [Agent 3] Re-attaching target '{target_col}'.")
                processed_df.reset_index(drop=True, inplace=True)
                y_series.reset_index(drop=True, inplace=True)
                processed_df[target_col] = y_series

            state['cleaned_df'] = processed_df
            state['data_preview_html'] = processed_df.head(50).to_html(classes='table', border=0, index=False)
            state['data_shape'] = str(processed_df.shape)
            state['analysis']['preprocessing_log'] = config
            
            return state

        except Exception as e:
            print(f"   Pipeline Failed: {e}")
            raise e