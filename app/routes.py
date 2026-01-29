import datetime
import os
import json
import zipfile
import shutil
import threading
import uuid
import pandas as pd
import traceback
from flask import render_template, jsonify, Blueprint, request, send_from_directory
from app import mongo
from bson.objectid import ObjectId
from pipeline.orchestrator import run_pipeline_from_graph

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
bp = Blueprint('main', __name__)
JOBS = {}

def get_db():
    if mongo.db is None: raise Exception("MongoDB not connected.")
    return mongo.db

def background_worker(job_id, initial_state, graph_layout, target_node_id, pipeline_type):
    try:
        def on_progress(msg):
            if job_id in JOBS:
                JOBS[job_id]["logs"].append(msg)
                JOBS[job_id]["status"] = "running"

        final_result = {}
        # Run Pipeline
        if pipeline_type == 'autonlp':
             from pipeline.orchestrator_nlp import run_nlp_pipeline
             final_result = run_nlp_pipeline(initial_state, graph_layout, target_node_id, status_callback=on_progress)
        else:
             final_result = run_pipeline_from_graph(initial_state, graph_layout, target_node_id, status_callback=on_progress)
        
        # --- FORCE SAVE ACTIVE DATA ---
        results_dir = initial_state['results_dir']
        df = final_result.get('current_data')
        
        if df is None: df = final_result.get('featured_df')
        if df is None: df = final_result.get('cleaned_df')
        if df is None: df = final_result.get('raw_df')
        
        if df is not None and not df.empty:
            csv_path = os.path.join(results_dir, "active_data.csv")
            df.to_csv(csv_path, index=False)
            on_progress("Finalizing: Active Data Saved to Disk.")
        
        # Prepare Visuals
        chart_files = []
        c_dir = os.path.join(results_dir, "charts")
        if os.path.exists(c_dir): 
            chart_files = [f for f in os.listdir(c_dir) if f.endswith('.png')]
        
        # Construct Response
        run_data = {
            "status": "success", 
            "best_model": final_result.get('best_model_id', "Unknown"), 
            "final_message": final_result.get('final_message', "Process Complete"),
            "job_id": job_id,
            "chart_urls": [f"/results/charts/{f}" for f in chart_files],
            "data_preview_html": final_result.get('data_preview_html', ''),
            "leaderboard_html": final_result.get('leaderboard_html'),
            "dl_app": final_result.get('dl_app'),
            "dl_model": final_result.get('dl_model'),
            "dl_charts": final_result.get('dl_charts'),
            "dl_report": final_result.get('dl_report', f"/results/{job_id}/final_report.txt")
        }
        
        JOBS[job_id]["status"] = "completed"
        JOBS[job_id]["result"] = run_data
        
        # DB Log
        try:
            get_db().runs.insert_one({**run_data, "run_at": datetime.datetime.utcnow(), "project_id": initial_state.get('project_id')})
        except: pass

    except Exception as e:
        traceback.print_exc()
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["message"] = str(e)

@bp.route('/')
def index(): return render_template('index.html')

@bp.route('/automl')
def automl_interface():
    db = get_db()
    project = db.projects.find_one({"name": "Default Project"})
    if not project:
        DEFAULT_GRAPH = {"nodes":[], "edges":[]}
        res = db.projects.insert_one({"name": "Default Project", "graph_json": json.dumps(DEFAULT_GRAPH)})
        project = db.projects.find_one({"_id": res.inserted_id})
    project['_id'] = str(project['_id'])
    return render_template('automl.html', project=project)

@bp.route('/api/run_pipeline', methods=['POST'])
def api_run_pipeline():
    acq_mode = request.form.get('acquisition_mode', 'upload')
    prob_desc = request.form.get('problem_description', 'Analyze')
    graph_str = request.form.get('graph_json')
    node_configs = json.loads(request.form.get('node_configs', '{}'))
    target_node_id = request.form.get('target_node_id')
    pipeline_type = request.form.get('pipeline_type', 'automl')
    
    acq_input = None
    if acq_mode == 'upload' and 'file' in request.files:
        file = request.files['file']
        unique_name = f"{uuid.uuid4().hex[:8]}_{file.filename}"
        save_path = os.path.join(os.getcwd(), "data", unique_name)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        file.save(save_path)
        acq_input = save_path
    elif acq_mode == 'search': acq_input = request.form.get('search_query')
    elif acq_mode == 'generate': acq_input = request.form.get('gen_description')
    elif acq_mode == 'download_selected':
        acq_input = {'source': request.form.get('source'), 'id': request.form.get('dataset_id')}

    job_id = str(uuid.uuid4())
    results_dir = os.path.join(os.getcwd(), "results", job_id)
    os.makedirs(results_dir, exist_ok=True)

    initial_state = {
        "job_id": job_id,
        "project_id": request.form.get('project_id'), 
        "acquisition_mode": acq_mode, 
        "acquisition_input": acq_input,
        "problem_description": prob_desc, 
        "node_configs": node_configs,
        "results_dir": results_dir,
        "current_data": None
    }

    JOBS[job_id] = {"status": "starting", "logs": [], "result": None}
    
    thread = threading.Thread(
        target=background_worker, 
        args=(job_id, initial_state, json.loads(graph_str), target_node_id, pipeline_type)
    )
    thread.start()
    return jsonify({"status": "started", "job_id": job_id})

@bp.route('/api/status/<job_id>', methods=['GET'])
def api_job_status(job_id):
    return jsonify(JOBS.get(job_id, {"status": "error", "message": "Job not found"}))

@bp.route('/results/charts/<path:filename>')
def serve_chart(filename): 
    base = os.path.join(os.getcwd(), "results")
    for root, dirs, files in os.walk(base):
        if filename in files:
            return send_from_directory(root, filename)
    return "Chart not found", 404

# --- DOWNLOAD HANDLER (FIXED) ---
@bp.route('/api/download/<file_type>')
def download_file(file_type):
    # Safety Check: Clean URL params if they got stuck
    if '&' in file_type:
        file_type = file_type.split('&')[0]
        
    base_results = os.path.join(os.getcwd(), "results")
    job_id = request.args.get('job_id')
    
    target_dir = None
    
    # 1. Exact Match via Job ID
    if job_id:
        path = os.path.join(base_results, job_id)
        if os.path.exists(path): target_dir = path
        
    # 2. Fallback to Latest
    if not target_dir:
        try:
            dirs = [os.path.join(base_results, d) for d in os.listdir(base_results) if os.path.isdir(os.path.join(base_results, d))]
            if dirs: target_dir = max(dirs, key=os.path.getmtime)
        except: pass
        
    if not target_dir: return "Results not found", 404

    # 3. Mappings
    file_map = {
        'data.csv': 'active_data.csv',
        'deployment_app.zip': 'deployment_app.zip',
        'charts.zip': 'charts_bundle.zip',
        'model_only': 'model_artifacts.zip' 
    }
    
    filename = file_map.get(file_type)
    if not filename: return "Invalid file type", 400
    
    full_path = os.path.join(target_dir, filename)
    if os.path.exists(full_path):
        return send_from_directory(target_dir, filename, as_attachment=True)
    
    return f"File generation pending or failed: {filename}", 404

@bp.route('/api/save_graph', methods=['POST'])
def api_save_graph():
    get_db().projects.update_one(
        {"_id": ObjectId(request.json.get('project_id'))}, 
        {"$set": {"graph_json": request.json.get('graph_json')}}
    )
    return jsonify({"status": "success", "message": "Graph saved."})