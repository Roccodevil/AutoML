import datetime
import os
import json
import zipfile
import shutil
from flask import render_template, jsonify, Blueprint, request, send_from_directory
from app import mongo
from bson.objectid import ObjectId
from pipeline.orchestrator import run_pipeline_from_graph

bp = Blueprint('main', __name__)

def get_db():
    if mongo.db is None: raise Exception("MongoDB not connected.")
    return mongo.db

# ... (DEFAULT_GRAPH and index route remain standard) ...
DEFAULT_GRAPH = { "nodes": [ {"id": "1", "type": "agent_1_data", "position": {"x": 50, "y": 100}, "data": {"label": "1. Data Acquisition"}}, {"id": "2", "type": "agent_2_analysis", "position": {"x": 300, "y": 100}, "data": {"label": "2. Problem Analysis"}}, {"id": "3", "type": "agent_3_preprocess", "position": {"x": 550, "y": 100}, "data": {"label": "3. Preprocess Data"}}, {"id": "4", "type": "agent_4_viz", "position": {"x": 800, "y": 100}, "data": {"label": "4. Visualize"}}, {"id": "5", "type": "agent_5_feature", "position": {"x": 50, "y": 250}, "data": {"label": "5. Feature Engineer"}}, {"id": "6", "type": "agent_6_staging", "position": {"x": 300, "y": 250}, "data": {"label": "6. Stage Data"}}, {"id": "7", "type": "agent_7_automl", "position": {"x": 550, "y": 250}, "data": {"label": "7. Train (H2O)"}}, {"id": "8", "type": "agent_8_export", "position": {"x": 800, "y": 250}, "data": {"label": "8. Export"}} ], "edges": [ {"id": "e1-2", "source": "1", "target": "2"}, {"id": "e2-3", "source": "2", "target": "3"}, {"id": "e3-4", "source": "3", "target": "4"}, {"id": "e4-5", "source": "4", "target": "5"}, {"id": "e5-6", "source": "5", "target": "6"}, {"id": "e6-7", "source": "6", "target": "7"}, {"id": "e7-8", "source": "7", "target": "8"} ] }

@bp.route('/')
def index():
    db = get_db(); project = db.projects.find_one()
    if not project:
        result = db.projects.insert_one({ "name": "Default Project", "created_at": datetime.datetime.utcnow(), "graph_json": json.dumps(DEFAULT_GRAPH) })
        project = db.projects.find_one({"_id": result.inserted_id})
    current_graph = json.loads(project.get("graph_json", "{}") or "{}")
    if not current_graph.get("nodes") or len(current_graph["nodes"]) < 8:
        db.projects.update_one({"_id": project["_id"]}, {"$set": {"graph_json": json.dumps(DEFAULT_GRAPH)}})
        project["graph_json"] = json.dumps(DEFAULT_GRAPH)
    project['_id'] = str(project['_id'])
    return render_template('index.html', project=project)

@bp.route('/results/charts/<path:filename>')
def serve_chart(filename): return send_from_directory(os.path.join(os.getcwd(), "results", "charts"), filename)

@bp.route('/results/models/<path:filename>')
def serve_model(filename): return send_from_directory(os.path.join(os.getcwd(), "results", "models"), filename)

@bp.route('/results/<path:filename>')
def serve_result_file(filename): return send_from_directory(os.path.join(os.getcwd(), "results"), filename)

@bp.route('/api/download/<file_type>')
def download_file(file_type):
    results_dir = os.path.join(os.getcwd(), "results")
    
    if file_type == 'data.csv':
        if os.path.exists(os.path.join(results_dir, "active_data.csv")): return send_from_directory(results_dir, "active_data.csv", as_attachment=True)
    
    elif file_type == 'deployment_app.zip':
        if os.path.exists(os.path.join(results_dir, "deployment_app.zip")): return send_from_directory(results_dir, "deployment_app.zip", as_attachment=True)
    
    elif file_type == 'charts.zip':
        # Re-zip if needed
        charts_dir = os.path.join(results_dir, "charts")
        if os.path.exists(charts_dir) and len(os.listdir(charts_dir)) > 0:
            zip_path = os.path.join(results_dir, "charts_bundle.zip")
            with zipfile.ZipFile(zip_path, 'w') as zf:
                for f in os.listdir(charts_dir):
                    if f.endswith('.png'): zf.write(os.path.join(charts_dir, f), f)
            return send_from_directory(results_dir, "charts_bundle.zip", as_attachment=True)
        
    elif file_type == 'model_only':
        models_dir = os.path.join(results_dir, "models")
        if os.path.exists(models_dir):
            files = [f for f in os.listdir(models_dir) if not f.startswith('.')]
            if len(files) == 1: return send_from_directory(models_dir, files[0], as_attachment=True)
            elif len(files) > 1:
                # Zip if multiple (e.g. H2O parts)
                zip_path = os.path.join(results_dir, "model_files.zip")
                with zipfile.ZipFile(zip_path, 'w') as zf:
                    for f in files: zf.write(os.path.join(models_dir, f), f)
                return send_from_directory(results_dir, "model_files.zip", as_attachment=True)

    return "File not found.", 404

@bp.route('/api/run_pipeline', methods=['POST'])
def api_run_pipeline():
    db = get_db()
    acq_mode = request.form.get('acquisition_mode', 'upload')
    prob_desc = request.form.get('problem_description', 'Analyze')
    graph_str = request.form.get('graph_json')
    node_configs_str = request.form.get('node_configs', '{}')
    target_node_id = request.form.get('target_node_id')
    
    acq_input = None
    
    if acq_mode == 'upload':
        if 'file' in request.files and request.files['file'].filename != '':
            file = request.files['file']
            filename = file.filename
            ext = os.path.splitext(filename)[1].lower()
            save_path = os.path.join(os.getcwd(), "data", f"uploaded_data{ext}")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            file.save(save_path)
            acq_input = save_path
        else:
            data_dir = os.path.join(os.getcwd(), "data")
            if os.path.exists(data_dir):
                files = os.listdir(data_dir)
                existing = [f for f in files if f.startswith('uploaded_data')]
                if existing: acq_input = os.path.join(data_dir, existing[0])
                else: return jsonify({"status": "error", "message": "No file uploaded."}), 400
    elif acq_mode == 'search': acq_input = request.form.get('search_query')
    elif acq_mode == 'download_selected': acq_input = {'source': request.form.get('source'), 'id': request.form.get('dataset_id')}
    elif acq_mode == 'generate': acq_input = request.form.get('gen_description')

    initial_state = {
        "project_id": request.form.get('project_id'), "acquisition_mode": acq_mode, "acquisition_input": acq_input,
        "problem_description": prob_desc, "node_configs": json.loads(node_configs_str),
        "results_dir": os.path.join(os.getcwd(), "results")
    }
    
    try:
        final_result = run_pipeline_from_graph(initial_state, json.loads(graph_str), target_node_id)
        
        # --- FILESYSTEM CHECK FOR DOWNLOADS ---
        # This guarantees buttons appear if files exist
        chart_files = []
        c_dir = os.path.join(os.getcwd(), "results", "charts")
        if os.path.exists(c_dir): chart_files = [f for f in os.listdir(c_dir) if f.endswith('.png')]
        
        m_dir = os.path.join(os.getcwd(), "results", "models")
        has_model = os.path.exists(m_dir) and len(os.listdir(m_dir)) > 0
        
        run_data = {
            "status": "success", 
            "best_model": final_result.get('best_model_id', "Unknown"), 
            "final_message": final_result.get('final_message'),
            "chart_urls": [f"/results/charts/{f}" for f in chart_files],
            "search_results": final_result.get('search_results'),
            "suggest_generate": final_result.get('suggest_generate'),
            "data_preview_html": final_result.get('data_preview_html', ''),
            "data_shape": final_result.get('data_shape', ''),
            "analysis": final_result.get('analysis'),
            "leaderboard_html": final_result.get('leaderboard_html'),
            # Force URLs if files exist
            "dl_app": "/api/download/deployment_app.zip" if has_model else "", 
            "dl_charts": "/api/download/charts.zip" if len(chart_files) > 0 else "", 
            "dl_model": "/api/download/model_only" if has_model else "", 
            "dl_report": "/results/final_report.txt"
        }
        db.runs.insert_one({**run_data, "run_at": datetime.datetime.utcnow()})
        return jsonify(run_data)
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

@bp.route('/api/save_graph', methods=['POST'])
def api_save_graph():
    get_db().projects.update_one({"_id": ObjectId(request.json.get('project_id'))}, {"$set": {"graph_json": request.json.get('graph_json')}})
    return jsonify({"status": "success", "message": "Graph saved."})

@bp.route('/api/get_node_info', methods=['GET'])
def api_get_node_info(): return jsonify({"node_type": request.args.get('node_type'), "description": "Node Info"})