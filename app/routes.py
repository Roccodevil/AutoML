import datetime
import os
import json
from flask import render_template, jsonify, Blueprint, request, send_from_directory
from app import mongo
from bson.objectid import ObjectId
from pipeline.orchestrator import run_pipeline_from_graph

bp = Blueprint('main', __name__)

def get_db():
    if mongo.db is None:
        raise Exception("MongoDB not connected. Check MONGO_URI in .env")
    return mongo.db

# --- STANDARD 8-STEP PIPELINE LAYOUT ---
DEFAULT_GRAPH = {
    "nodes": [
        {"id": "1", "type": "agent_1_data", "position": {"x": 50, "y": 100}, "data": {"label": "1. Data Acquisition"}},
        {"id": "2", "type": "agent_2_analysis", "position": {"x": 300, "y": 100}, "data": {"label": "2. Problem Analysis"}},
        {"id": "3", "type": "agent_3_preprocess", "position": {"x": 550, "y": 100}, "data": {"label": "3. Preprocess"}},
        {"id": "4", "type": "agent_4_viz", "position": {"x": 800, "y": 100}, "data": {"label": "4. Visualize"}},
        # Start second row
        {"id": "5", "type": "agent_5_feature", "position": {"x": 50, "y": 250}, "data": {"label": "5. Feature Eng."}},
        {"id": "6", "type": "agent_6_staging", "position": {"x": 300, "y": 250}, "data": {"label": "6. Stage Data"}},
        {"id": "7", "type": "agent_7_automl", "position": {"x": 550, "y": 250}, "data": {"label": "7. Train (H2O)"}},
        {"id": "8", "type": "agent_8_export", "position": {"x": 800, "y": 250}, "data": {"label": "8. Export Report"}}
    ],
    "edges": [
        {"id": "e1-2", "source": "1", "target": "2"},
        {"id": "e2-3", "source": "2", "target": "3"},
        {"id": "e3-4", "source": "3", "target": "4"},
        # Connect end of row 1 to start of row 2
        {"id": "e4-5", "source": "4", "target": "5"},
        {"id": "e5-6", "source": "5", "target": "6"},
        {"id": "e6-7", "source": "6", "target": "7"},
        {"id": "e7-8", "source": "7", "target": "8"}
    ]
}

@bp.route('/')
def index():
    db = get_db()
    project = db.projects.find_one()
    
    if not project:
        # Create new project
        project_data = { 
            "name": "Default Project", 
            "created_at": datetime.datetime.utcnow(), 
            "graph_json": json.dumps(DEFAULT_GRAPH) 
        }
        result = db.projects.insert_one(project_data)
        project = db.projects.find_one({"_id": result.inserted_id})
    
    # --- FORCE RESET LOGIC ---
    # Load the current graph from DB
    try:
        current_graph_str = project.get("graph_json", "{}")
        if not current_graph_str or current_graph_str == "null":
             current_graph = {}
        else:
             current_graph = json.loads(current_graph_str)
             
        # CHECK: If we have fewer than 8 nodes, it's the old graph. Reset it.
        if not current_graph.get("nodes") or len(current_graph["nodes"]) < 8:
            print("Detected incomplete graph (old version). Resetting to 8-Node Pipeline...")
            new_json = json.dumps(DEFAULT_GRAPH)
            db.projects.update_one({"_id": project["_id"]}, {"$set": {"graph_json": new_json}})
            project["graph_json"] = new_json
            
    except Exception as e:
        print(f"Error parsing graph, resetting to default: {e}")
        new_json = json.dumps(DEFAULT_GRAPH)
        db.projects.update_one({"_id": project["_id"]}, {"$set": {"graph_json": new_json}})
        project["graph_json"] = new_json

    project['_id'] = str(project['_id'])
    return render_template('index.html', project=project)

@bp.route('/results/charts/<path:filename>')
def serve_chart(filename):
    results_dir = os.path.join(os.getcwd(), "results", "charts")
    return send_from_directory(results_dir, filename)

@bp.route('/api/run_pipeline', methods=['POST'])
def api_run_pipeline():
    db = get_db()
    print("API: /api/run_pipeline called")
    
    # Form Inputs
    acquisition_mode = request.form.get('acquisition_mode', 'upload')
    problem_description = request.form.get('problem_description', 'Analyze this data')
    graph_str = request.form.get('graph_json')
    target_node_id = request.form.get('target_node_id') 
    
    acquisition_input = None
    
    if acquisition_mode == 'upload':
        if 'file' in request.files and request.files['file'].filename != '':
            file = request.files['file']
            upload_folder = os.path.join(os.getcwd(), "data")
            os.makedirs(upload_folder, exist_ok=True)
            filepath = os.path.join(upload_folder, "uploaded_data.csv")
            file.save(filepath)
            acquisition_input = filepath
        elif os.path.exists(os.path.join(os.getcwd(), "data", "uploaded_data.csv")):
             acquisition_input = os.path.join(os.getcwd(), "data", "uploaded_data.csv")
        else:
             return jsonify({"status": "error", "message": "No file uploaded."}), 400
             
    elif acquisition_mode == 'search':
        acquisition_input = request.form.get('search_query')
        
    elif acquisition_mode == 'download_selected':
        acquisition_input = {
            'source': request.form.get('source'),
            'id': request.form.get('dataset_id')
        }
        
    elif acquisition_mode == 'generate':
        acquisition_input = request.form.get('gen_description')

    graph_layout = json.loads(graph_str)
    initial_state = {
        "project_id": request.form.get('project_id'),
        "acquisition_mode": acquisition_mode,
        "acquisition_input": acquisition_input,
        "problem_description": problem_description,
        "results_dir": os.path.join(os.getcwd(), "results")
    }
    
    try:
        final_result = run_pipeline_from_graph(initial_state, graph_layout)
        
        chart_files = []
        charts_dir = os.path.join(os.getcwd(), "results", "charts")
        if os.path.exists(charts_dir):
            chart_files = [f for f in os.listdir(charts_dir) if f.endswith('.png')]
        
        run_data = {
            "status": "success",
            "best_model": final_result.get('best_model_id', "Unknown"),
            "model_path": final_result.get('final_model_path'),
            "final_message": final_result.get('final_message'),
            "chart_urls": [f"/results/charts/{f}" for f in chart_files],
            "search_results": final_result.get('search_results'),
            "suggest_generate": final_result.get('suggest_generate')
        }
        db.runs.insert_one({**run_data, "run_at": datetime.datetime.utcnow()})
        
        return jsonify(run_data)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

@bp.route('/api/save_graph', methods=['POST'])
def api_save_graph():
    db = get_db()
    data = request.json
    db.projects.update_one(
        {"_id": ObjectId(data.get('project_id'))},
        {"$set": {"graph_json": data.get('graph_json')}}
    )
    return jsonify({"status": "success", "message": "Graph saved."})

@bp.route('/api/get_node_info', methods=['GET'])
def api_get_node_info():
    node_type = request.args.get('node_type')
    node_docs = {
        "agent_1_data": "Data Acquisition. Click to Configure: Choose Upload, Search (Kaggle/HF), or Generate.",
        "agent_2_analysis": "Problem Analysis. AI analyzes your data structure.",
        "agent_3_preprocess": "Preprocess Data. Cleans and scales data.",
        "agent_4_viz": "Visualize. Generates charts based on data analysis.",
        "agent_5_feature": "Feature Engineer. Creates new features.",
        "agent_6_staging": "Stage Data. Splits data.",
        "agent_7_automl": "Train (H2O). Runs AutoML to find the best model.",
        "agent_7_optuna": "Train (Optuna). Tunes models.",
        "agent_8_export": "Export Report. Saves results."
    }
    return jsonify({"node_type": node_type, "description": node_docs.get(node_type, "")})