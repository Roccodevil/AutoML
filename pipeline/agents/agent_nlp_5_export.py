import os
import joblib
import shutil
import zipfile

class NLPExportAgent:
    def run(self, state):
        print("-> NLP Agent 5: Exporting...")
        res = state['results_dir']
        pkg_dir = os.path.join(res, "nlp_model_pkg")
        os.makedirs(pkg_dir, exist_ok=True)
        
        # Save Model
        joblib.dump(state['model'], os.path.join(pkg_dir, "model.pkl"))
        
        # Generate App
        code = """
import streamlit as st
import joblib
st.title("NLP Classifier")
model = joblib.load("model.pkl")
txt = st.text_area("Text")
if st.button("Predict"):
    st.write(model.predict([txt])[0])
"""
        with open(os.path.join(pkg_dir, "app.py"), "w") as f: f.write(code)
        
        # Zip
        shutil.make_archive(os.path.join(res, "nlp_app"), 'zip', pkg_dir)
        state['dl_app'] = "/api/download/nlp_app.zip"
        
        return state