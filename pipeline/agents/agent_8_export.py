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
        print("-> Agent 8: Preparing download artifacts...")
        model = state['best_model']
        results_dir = state['results_dir']
        models_dir = os.path.join(results_dir, "models")
        charts_dir = os.path.join(results_dir, "charts")
        os.makedirs(models_dir, exist_ok=True)
        
        final_model_path = ""
        model_type = "sklearn" 
        h2o_version = ""

        # --- 1. Save Raw Model ---
        if "h2o" in str(type(model)).lower():
            model_type = "h2o"
            h2o_version = h2o.__version__
            try:
                final_model_path = model.download_mojo(path=models_dir, get_genmodel_jar=False)
            except:
                final_model_path = h2o.save_model(model=model, path=models_dir, force=True)
        else:
            final_model_path = os.path.join(models_dir, "model.pkl")
            joblib.dump(model, final_model_path)

        state['final_model_path'] = final_model_path

        # --- 2. Bundle Charts ---
        if os.path.exists(charts_dir) and len(os.listdir(charts_dir)) > 0:
            shutil.make_archive(os.path.join(results_dir, "charts_bundle"), 'zip', charts_dir)
            state['charts_zip_path'] = os.path.join(results_dir, "charts_bundle.zip")

        # --- 3. Generate App ---
        reqs = "pandas\nstreamlit\nwatchdog\n"
        if model_type == "h2o": reqs += f"h2o=={h2o_version}\n" 
        else: reqs += "scikit-learn\njoblib\n"
        with open(os.path.join(models_dir, "requirements.txt"), "w") as f: f.write(reqs)

        df_sample = state.get('X_train')
        if df_sample is None: df_sample = state.get('cleaned_df')
        single_pred_code = self._generate_single_pred_code(df_sample, state['problem_description'])
        
        model_filename = os.path.basename(final_model_path)
        app_content = self._generate_streamlit_app(model_type, model_filename, single_pred_code)
        with open(os.path.join(models_dir, "app.py"), "w") as f: f.write(app_content)

        win_script = "@echo off\npip install -r requirements.txt\ncls\nstreamlit run app.py\npause"
        with open(os.path.join(models_dir, "run_windows.bat"), "w") as f: f.write(win_script)
        linux_script = "#!/bin/bash\npip3 install -r requirements.txt\nstreamlit run app.py"
        with open(os.path.join(models_dir, "run_linux.sh"), "w") as f: f.write(linux_script)

        zip_path = os.path.join(results_dir, "deployment_app.zip")
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            zipf.write(os.path.join(models_dir, "app.py"), "app.py")
            zipf.write(final_model_path, model_filename)
            zipf.write(os.path.join(models_dir, "run_windows.bat"), "run_windows.bat")
            zipf.write(os.path.join(models_dir, "run_linux.sh"), "run_linux.sh")
            zipf.write(os.path.join(models_dir, "requirements.txt"), "requirements.txt")
        
        state['deployment_zip'] = zip_path

        # --- 4. Final Report ---
        report_path = os.path.join(results_dir, "final_report.txt")
        with open(report_path, "w") as f:
            f.write(f"Task: {state['problem_description']}\n")
            f.write(f"Best Model: {state['best_model_id']}\n")
            
        with open(report_path, "r") as f: state['report_content'] = f.read()
        return state

    # ... (Keep helper methods _generate_single_pred_code and _generate_streamlit_app same as before) ...
    def _generate_single_pred_code(self, df, problem_desc):
        if df is None: return "st.warning('No data.')"
        schema_info = []
        for col in df.columns:
            schema_info.append(f"- {col} ({df[col].dtype})")
        schema_text = "\n".join(schema_info[:20])
        prompt = ChatPromptTemplate.from_template("""Write Streamlit sidebar input code for: {schema}. Return ONLY code.""")
        try: return (prompt | llm_powerful_api | StrOutputParser()).invoke({"schema": schema_text}).replace("```python","").replace("```","")
        except: return ""

    def _generate_streamlit_app(self, model_type, model_filename, single_pred_code):
        indented = "\n".join(["    " + line for line in single_pred_code.split("\n")])
        code = f"""import streamlit as st
import pandas as pd
import os
"""
        if model_type == "h2o":
            code += f"""import h2o
@st.cache_resource
def load_model():
    h2o.init(verbose=False)
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "{model_filename}")
    return h2o.import_mojo(path) if path.endswith(".zip") else h2o.load_model(path)
"""
        else:
            code += f"""import joblib
@st.cache_resource
def load_model(): return joblib.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "{model_filename}"))
"""
        code += f"""
st.title("AI Dashboard")
model = load_model()
tab1, tab2 = st.tabs(["Batch", "Single"])
with tab1:
    up = st.file_uploader("CSV", type="csv")
    if up and st.button("Predict"):
        df = pd.read_csv(up)
        # Prediction logic...
        st.write("Done")
with tab2:
    st.header("Inputs")
{indented}
    if st.button("Predict"):
        # Prediction logic...
        st.write("Result")
"""
        return code