import h2o
import os
import joblib
import shutil
import zipfile
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from core.llm_services import llm_powerful_api

class ExportAgent: 
    def run(self, state):
        print("-> Agent 8: Preparing granular download artifacts...")
        model = state['best_model']
        results_dir = state['results_dir']
        models_dir = os.path.join(results_dir, "models")
        charts_dir = os.path.join(results_dir, "charts")
        os.makedirs(models_dir, exist_ok=True)
        
        final_model_path = ""
        model_type = "sklearn" 
        h2o_version = "3.46.0.8" # Default fallback

        # --- 1. SAVE RAW MODEL ---
        if "h2o" in str(type(model)).lower():
            model_type = "h2o"
            h2o_version = h2o.__version__ # Capture EXACT server version
            try:
                # Try MOJO first (Version independent)
                final_model_path = model.download_mojo(path=models_dir, get_genmodel_jar=False)
            except:
                # Fallback to Binary (Version locked)
                final_model_path = h2o.save_model(model=model, path=models_dir, force=True)
        else:
            final_model_path = os.path.join(models_dir, "model.pkl")
            joblib.dump(model, final_model_path)

        state['final_model_path'] = final_model_path

        # --- 2. BUNDLE CHARTS ---
        # Explicitly zip charts here to ensure the file exists for the route
        if os.path.exists(charts_dir) and len(os.listdir(charts_dir)) > 0:
            zip_path = os.path.join(results_dir, "charts_bundle.zip")
            with zipfile.ZipFile(zip_path, 'w') as zf:
                for f in os.listdir(charts_dir):
                    if f.endswith('.png'):
                        zf.write(os.path.join(charts_dir, f), f)
            state['charts_zip_path'] = zip_path # Notify state

        # --- 3. GENERATE APP ---
        # CRITICAL FIX: Pin exact H2O version
        reqs = "pandas\nstreamlit\nwatchdog\n"
        if model_type == "h2o": 
            reqs += f"h2o=={h2o_version}\n" # <--- LOCK VERSION
        else: 
            reqs += "scikit-learn\njoblib\n"
            
        with open(os.path.join(models_dir, "requirements.txt"), "w") as f: f.write(reqs)

        # AI-Generated Form
        df_sample = state.get('X_train')
        if df_sample is None: df_sample = state.get('cleaned_df')
        
        print("   Generating AI Dashboard code...")
        single_pred_code = self._generate_single_pred_code(df_sample, state['problem_description'])
        
        model_filename = os.path.basename(final_model_path)
        app_content = self._generate_streamlit_app(model_type, model_filename, single_pred_code)
        with open(os.path.join(models_dir, "app.py"), "w") as f: f.write(app_content)

        # Scripts
        win_script = "@echo off\necho Installing specific H2O version match...\npip install -r requirements.txt\ncls\nstreamlit run app.py\npause"
        with open(os.path.join(models_dir, "run_windows.bat"), "w") as f: f.write(win_script)
        linux_script = "#!/bin/bash\npip3 install -r requirements.txt\nstreamlit run app.py"
        with open(os.path.join(models_dir, "run_linux.sh"), "w") as f: f.write(linux_script)

        # Zip App Package
        zip_path = os.path.join(results_dir, "deployment_app.zip")
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            zipf.write(os.path.join(models_dir, "app.py"), "app.py")
            zipf.write(final_model_path, model_filename)
            zipf.write(os.path.join(models_dir, "run_windows.bat"), "run_windows.bat")
            zipf.write(os.path.join(models_dir, "run_linux.sh"), "run_linux.sh")
            zipf.write(os.path.join(models_dir, "requirements.txt"), "requirements.txt")
        
        state['deployment_zip'] = zip_path

        # --- 4. FINAL REPORT ---
        report_path = os.path.join(results_dir, "final_report.txt")
        with open(report_path, "w") as f:
            f.write(f"Task: {state['problem_description']}\n")
            f.write(f"Best Model: {state['best_model_id']}\n")
            f.write(f"H2O Version Used: {h2o_version}\n\n")
            f.write("--- ARTIFACTS GENERATED ---\n")
            f.write("1. Interactive Dashboard (deployment_app.zip)\n")
            f.write("2. Raw Model File\n")
            f.write("3. Charts Bundle\n")
            
        with open(report_path, "r") as f: state['report_content'] = f.read()
        return state

    def _generate_single_pred_code(self, df, problem_desc):
        if df is None: return "st.warning('No data for form generation.')"
        schema_info = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            min_val, max_val = "N/A", "N/A"
            if 'int' in dtype or 'float' in dtype:
                min_val = float(df[col].min()); max_val = float(df[col].max())
            schema_info.append(f"- {col} ({dtype}) Min:{min_val} Max:{max_val}")
        
        schema_text = "\n".join(schema_info[:25]) 
        prompt = ChatPromptTemplate.from_template(
            """Write Python Streamlit code for sidebar inputs based on this schema: {schema}.
            - Use `st.sidebar`.
            - Create a dictionary `input_data`.
            - Convert to `input_df = pd.DataFrame([input_data])`.
            - Return ONLY code."""
        )
        try: 
            return (prompt | llm_powerful_api | StrOutputParser()).invoke({"schema": schema_text}).replace("```python","").replace("```","")
        except: 
            return "st.error('AI Gen Failed')"

    def _generate_streamlit_app(self, model_type, model_filename, single_pred_code):
        indented_ai_code = "\n".join(["    " + line for line in single_pred_code.split("\n")])
        
        code = f"""import streamlit as st
import pandas as pd
import os
import sys
st.set_page_config(page_title="AI Dashboard", layout="wide")
st.title("🤖 AI Prediction Dashboard")

# Model Loader
"""
        if model_type == "h2o":
            code += f"""
import h2o
@st.cache_resource
def load_model():
    try: h2o.init(verbose=False)
    except: st.error("Java Required!"); st.stop()
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "{model_filename}")
    return h2o.import_mojo(path) if path.endswith(".zip") else h2o.load_model(path)
"""
        else:
            code += f"""
import joblib
@st.cache_resource
def load_model(): return joblib.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "{model_filename}"))
"""
        code += """
try: model = load_model(); st.sidebar.success("Model Active")
except Exception as e: st.error(str(e)); st.stop()

tab1, tab2 = st.tabs(["📂 Batch Upload", "✍️ Single Entry"])
with tab1:
    up = st.file_uploader("Upload CSV", type="csv")
    if up:
        try:
            df = pd.read_csv(up)
            st.dataframe(df.head())
            if st.button("Predict Batch"):
                """
        if model_type == "h2o": code += "preds = model.predict(h2o.H2OFrame(df)).as_data_frame()"
        else: code += "preds = pd.DataFrame(model.predict(df), columns=['Prediction'])"
        code += """
                st.dataframe(pd.concat([df, preds], axis=1))
        except Exception as e: st.error(str(e))

with tab2:
    st.header("Manual Input")
"""
        code += indented_ai_code
        code += """
    if 'input_df' in locals():
        st.write("Preview:")
        st.dataframe(input_df)
        if st.button("Predict Single"):
            try:
                """
        if model_type == "h2o": code += "res = model.predict(h2o.H2OFrame(input_df)).as_data_frame().iloc[0,0]"
        else: code += "res = model.predict(input_df)[0]"
        code += """
                st.metric("Prediction", str(res))
            except Exception as e: st.error(str(e))
"""
        return code