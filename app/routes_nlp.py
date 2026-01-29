import datetime
import os
import json
import zipfile
import threading
import uuid
from flask import render_template, jsonify, Blueprint, request, send_from_directory
from app import mongo
from pipeline.orchestrator_nlp import run_nlp_pipeline

# Force CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

nlp_bp = Blueprint('nlp', __name__)
NLP_JOBS = {}

def get_db():
    if mongo.db is None: raise Exception("MongoDB not connected.")
    return mongo.db

# --- CORRECT GRAPH DEFINITION (Matches Agent Filenames) ---
NLP_GRAPH = {
    "nodes": [
        {"id": "agent_nlp_1_data", "type": "agent_nlp_1_data", "position": {"x": 50, "y": 100}, "data": {"label": "1. Text Input"}},
        {"id": "agent_nlp_2_task", "type": "agent_nlp_2_task", "position": {"x": 300, "y": 100}, "data": {"label": "2. Task Router"}},
        {"id": "agent_nlp_3_preprocess", "type": "agent_nlp_3_preprocess", "position": {"x": 550, "y": 100}, "data": {"label": "3. Cleaning"}},
        {"id": "agent_nlp_4_model", "type": "agent_nlp_4_model", "position": {"x": 50, "y": 250}, "data": {"label": "4. Modeling"}},
        {"id": "agent_nlp_5_export", "type": "agent_nlp_5_export", "position": {"x": 300, "y": 250}, "data": {"label": "5. Export"}}
    ],
    "edges": [
        {"id": "e1", "source": "agent_nlp_1_data", "target": "agent_nlp_2_task", "animated": True},
        {"id": "e2", "source": "agent_nlp_2_task", "target": "agent_nlp_3_preprocess", "animated": True},
        {"id": "e3", "source": "agent_nlp_3_preprocess", "target": "agent_nlp_4_model", "animated": True},
        {"id": "e4", "source": "agent_nlp_4_model", "target": "agent_nlp_5_export", "animated": True}
    ]
}

def nlp_background_worker(job_id, initial_state, graph_layout, target_node_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    try:
        def on_progress(msg):
            if job_id in NLP_JOBS:
                NLP_JOBS[job_id]["logs"].append(msg)
                NLP_JOBS[job_id]["status"] = "running"

        final_result = run_nlp_pipeline(initial_state, graph_layout, target_node_id, status_callback=on_progress)
        
        results_dir = initial_state['results_dir']
        has_model = os.path.exists(os.path.join(results_dir, "nlp_model_pkg"))

        run_data = {
            "status": "success", 
            "final_message": final_result.get('final_message', "Task Complete"),
            "summary_result": final_result.get('summary_result'),
            "processed_preview": final_result.get('data_preview_html'),
            "dl_app": final_result.get('dl_app') or ("/api/download/nlp_app.zip" if has_model else ""),
            "dl_report": "/results/final_report.txt"
        }
        
        NLP_JOBS[job_id]["status"] = "completed"
        NLP_JOBS[job_id]["result"] = run_data
        
        try: get_db().runs.insert_one({**run_data, "run_at": datetime.datetime.utcnow(), "job_id": job_id, "type": "nlp"})
        except: pass

    except Exception as e:
        import traceback; traceback.print_exc()
        NLP_JOBS[job_id]["status"] = "error"
        NLP_JOBS[job_id]["message"] = str(e)

@nlp_bp.route('/autonlp')
def nlp_interface():
    db = get_db()
    # --- CRITICAL FIX: Upsert (Update if exists, Insert if new) ---
    # This forces the DB to use the new NLP_GRAPH structure defined in code
    db.projects.update_one(
        {"name": "NLP Default"},
        {"$set": {"graph_json": json.dumps(NLP_GRAPH)}},
        upsert=True
    )
    
    project = db.projects.find_one({"name": "NLP Default"})
    project['_id'] = str(project['_id'])
    return render_template('autonlp.html', project=project)

@nlp_bp.route('/api/run_nlp', methods=['POST'])
def api_run_nlp():
    acq_mode = request.form.get('acquisition_mode', 'input_text')
    # Use the fresh graph definition
    graph_str = request.form.get('graph_json', json.dumps(NLP_GRAPH))
    node_configs = json.loads(request.form.get('node_configs', '{}'))
    
    acq_input = None
    if acq_mode == 'upload':
        if 'file' in request.files and request.files['file'].filename != '':
            f = request.files['file']
            path = os.path.join(os.getcwd(), "data", f"nlp_{f.filename}")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            f.save(path)
            acq_input = path
    elif acq_mode == 'input_text':
        acq_input = request.form.get('input_text')

    initial_state = {
        "project_id": request.form.get('project_id'),
        "acquisition_mode": acq_mode,
        "acquisition_input": acq_input,
        "problem_description": request.form.get('problem_description', ''),
        "node_configs": node_configs,
        "results_dir": os.path.join(os.getcwd(), "results")
    }

    job_id = str(uuid.uuid4())
    NLP_JOBS[job_id] = {"status": "starting", "logs": ["Initializing..."], "result": None}
    
    thread = threading.Thread(
        target=nlp_background_worker, 
        args=(job_id, initial_state, json.loads(graph_str), None)
    )
    thread.start()
    return jsonify({"status": "started", "job_id": job_id})

@nlp_bp.route('/api/status_nlp/<job_id>', methods=['GET'])
def api_status_nlp(job_id):
    job = NLP_JOBS.get(job_id)
    if not job: return jsonify({"status": "error"}), 404
    return jsonify(job)