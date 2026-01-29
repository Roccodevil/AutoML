import h2o
import os
import joblib
import shutil
import zipfile
import pandas as pd
import numpy as np
import datetime
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from core.llm_services import llm_powerful_api

class ExportAgent: 
    def run(self, state):
        print("-> Agent 8: Packaging Artifacts & Generating App...")
        
        results_dir = state['results_dir']
        model = state.get('best_model')
        models_dir = os.path.join(results_dir, "models")
        os.makedirs(models_dir, exist_ok=True)

        # 1. REPORT
        report_path = os.path.join(results_dir, "final_report.txt")
        self._write_report(state, report_path)
        with open(report_path, "r", encoding='utf-8') as f: state['report_content'] = f.read()

        # 2. SAVE MODEL BINARY
        model_filename = "model_not_trained.txt"
        model_type = "none"
        
        if model:
            try:
                # Check for H2O
                if hasattr(model, 'download_mojo'):
                    model_type = "h2o"
                    # Try saving MOJO (Best for deployment)
                    try:
                        mojo_path = model.download_mojo(path=models_dir, get_genmodel_jar=True)
                        model_filename = os.path.basename(mojo_path)
                    except:
                        # Fallback to Binary
                        model_filename = "model.bin"
                        h2o.save_model(model, path=models_dir, force=True, filename=model_filename)
                else:
                    model_type = "sklearn"
                    model_filename = "model.pkl"
                    joblib.dump(model, os.path.join(models_dir, model_filename))
            except Exception as e:
                print(f"   [Agent 8] Model Save Failed: {e}")

        # 3. ZIP MODEL FILES (Standalone Download)
        model_zip_path = os.path.join(results_dir, "model_artifacts.zip")
        with zipfile.ZipFile(model_zip_path, 'w') as zf:
            for f in os.listdir(models_dir):
                # Don't zip the app files yet, just the model files
                if f not in ["app.py", "requirements.txt", "run_app.bat", "README.txt"]:
                    zf.write(os.path.join(models_dir, f), f)
        state['dl_model'] = "/api/download/model_only"

        # 4. GENERATE APP (Deterministic Template)
        if model and model_type != "none":
            # Extract Schema for Inputs
            df_train = state.get('X_train')
            if df_train is None: df_train = state.get('current_data')
            schema = self._get_data_schema(df_train)

            # Generate Robust Code
            app_code = self._generate_robust_app_code(model_filename, model_type, schema)

            # Write Files
            with open(os.path.join(models_dir, "app.py"), "w", encoding='utf-8') as f:
                f.write(app_code)
                
            # Clean Requirements (No unnecessary libs)
            reqs = "streamlit\npandas\nnumpy\n"
            if model_type == "h2o": reqs += "h2o\n"
            else: reqs += "scikit-learn\njoblib\n"
            
            with open(os.path.join(models_dir, "requirements.txt"), "w") as f: f.write(reqs)
            
            # Windows Runner
            with open(os.path.join(models_dir, "run_app.bat"), "w") as f:
                f.write("@echo off\ncall pip install -r requirements.txt\nstreamlit run app.py\npause")

            # Instructions
            self._write_instructions(models_dir)

            # Zip App Bundle
            app_zip_path = os.path.join(results_dir, "deployment_app.zip")
            with zipfile.ZipFile(app_zip_path, 'w') as zf:
                zf.write(os.path.join(models_dir, "app.py"), "app.py")
                zf.write(os.path.join(models_dir, "requirements.txt"), "requirements.txt")
                zf.write(os.path.join(models_dir, "run_app.bat"), "run_app.bat")
                zf.write(os.path.join(models_dir, "README.txt"), "README.txt")
                zf.write(os.path.join(models_dir, model_filename), model_filename)
                
                # If H2O, include the jar if it exists
                if model_type == "h2o":
                    for f in os.listdir(models_dir):
                        if f.endswith(".jar"): zf.write(os.path.join(models_dir, f), f)

            state['dl_app'] = "/api/download/deployment_app.zip"

        state['final_message'] = state['report_content']
        return state

    def _get_data_schema(self, df):
        if df is None: return {}
        schema = {}
        for col in df.columns[:20]: # Limit inputs
            is_num = pd.api.types.is_numeric_dtype(df[col])
            info = {
                "type": "number" if is_num else "text",
                "default": float(df[col].mean()) if is_num else str(df[col].mode()[0]),
                "options": df[col].unique().tolist()[:10] if not is_num and df[col].nunique() < 10 else None
            }
            schema[col] = info
        return schema

    def _write_instructions(self, folder):
        text = """=== AI APP DEPLOYMENT GUIDE ===

1. INSTALLATION
   - Ensure you have Python 3.9+ installed.
   - If using an H2O model (default), you MUST have Java installed.
     Check by running: java -version

2. RUNNING THE APP (Windows)
   - Double-click 'run_app.bat'.
   - This will automatically install dependencies and launch the browser.

3. RUNNING THE APP (Mac/Linux)
   Open terminal in this folder and run:
   pip install -r requirements.txt
   streamlit run app.py

4. USING THE APP
   - Tab 1 (Single): Use the sidebar to input values.
   - Tab 2 (Batch): Upload a CSV file to predict on thousands of rows at once.

5. TROUBLESHOOTING
   - "H2O Server Error": Install Java (JDK 8 or 11 recommended).
   - "Module not found": Run 'pip install -r requirements.txt' manually.
"""
        with open(os.path.join(folder, "README.txt"), "w") as f: f.write(text)

    def _write_report(self, state, path):
        with open(path, "w", encoding='utf-8') as f:
            f.write(f"JOB ID: {state.get('job_id')}\n")
            f.write(f"MODEL: {state.get('best_model_id')}\n")

    def _generate_robust_app_code(self, model_filename, model_type, schema):
        """
        Generates a 100% valid Streamlit app using a deterministic template.
        Solves the 'H2OAutoML' import error and 'Magic Number' zip error.
        """
        
        # 1. Input Widgets Code
        inputs_code = ""
        for col, info in schema.items():
            if info['type'] == 'number':
                inputs_code += f"    {col} = st.sidebar.number_input('{col}', value={info['default']})\n"
            elif info['options']:
                safe_opts = [str(x) for x in info['options']]
                inputs_code += f"    {col} = st.sidebar.selectbox('{col}', {safe_opts})\n"
            else:
                inputs_code += f"    {col} = st.sidebar.text_input('{col}', value='{info['default']}')\n"
            inputs_code += f"    input_data['{col}'] = {col}\n"

        # 2. H2O Loader Logic (The Fix)
        if model_type == 'h2o':
            loader_logic = f"""
import h2o
@st.cache_resource
def load_engine():
    try:
        h2o.init(nthreads=-1, max_mem_size="2G")
    except:
        st.error("H2O failed to start. Please install Java.")
        return None
    
    path = os.path.join(os.getcwd(), "{model_filename}")
    
    # CRITICAL FIX: Distinguish between MOJO (zip) and BINARY
    if path.endswith(".zip"):
        return h2o.import_mojo(path)
    else:
        return h2o.load_model(path)

def make_prediction(model, df):
    hf = h2o.H2OFrame(df)
    preds = model.predict(hf).as_data_frame()
    return preds
"""
        else:
            # Sklearn Loader
            loader_logic = f"""
import joblib
@st.cache_resource
def load_engine():
    path = os.path.join(os.getcwd(), "{model_filename}")
    return joblib.load(path)

def make_prediction(model, df):
    return model.predict(df)
"""

        # 3. Full App Template
        return f"""
import streamlit as st
import pandas as pd
import numpy as np
import os

st.set_page_config(page_title="AI Model Deployment", layout="wide")
st.title("🤖 AI Prediction System")

# --- MODEL ENGINE ---
{loader_logic}

model = load_engine()
if not model:
    st.stop()

# --- TABS ---
tab1, tab2 = st.tabs(["⚡ Single Prediction", "📂 Batch Prediction"])

# --- TAB 1: SIDEBAR INPUTS ---
with tab1:
    st.sidebar.header("Feature Inputs")
    input_data = {{}}
{inputs_code}
    
    st.subheader("Input Data")
    df_single = pd.DataFrame([input_data])
    st.dataframe(df_single)
    
    if st.button("Predict Single", type="primary"):
        try:
            res = make_prediction(model, df_single)
            st.success("Prediction Complete")
            st.write(res)
        except Exception as e:
            st.error(f"Error: {{e}}")

# --- TAB 2: BATCH CSV ---
with tab2:
    st.header("Batch Processing")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    if uploaded_file:
        df_batch = pd.read_csv(uploaded_file)
        st.write(f"Loaded {{len(df_batch)}} rows.")
        
        if st.button("Run Batch Prediction"):
            try:
                with st.spinner("Processing..."):
                    preds = make_prediction(model, df_batch)
                    
                    # Merge results if possible
                    if isinstance(preds, pd.DataFrame):
                        final_df = pd.concat([df_batch.reset_index(drop=True), preds.reset_index(drop=True)], axis=1)
                    else:
                        df_batch['Prediction'] = preds
                        final_df = df_batch
                        
                    st.dataframe(final_df.head())
                    
                    # Download
                    csv = final_df.to_csv(index=False).encode('utf-8')
                    st.download_button("Download Results", data=csv, file_name="predictions.csv", mime="text/csv")
            except Exception as e:
                st.error(f"Batch Error: {{e}}")
"""