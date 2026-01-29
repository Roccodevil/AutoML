import os
import h2o
import pandas as pd
import traceback
from typing import TypedDict, Dict, Any, List, Optional
from langgraph.graph import StateGraph, END
from app import mongo

# --- IMPORT AGENTS ---
from pipeline.agents.agent_1_data import DataAgent
from pipeline.agents.agent_2_analysis import AnalysisAgent
from pipeline.agents.agent_3_preprocess import PreprocessAgent
from pipeline.agents.agent_4_viz import VizAgent
from pipeline.agents.agent_5_feature import FeatureAgent
from pipeline.agents.agent_6_staging import StagingAgent
from pipeline.agents.agent_7_automl import AutoMLAgent
from pipeline.agents.agent_8_export import ExportAgent

# Placeholder
class OptunaAgentPlaceholder:
    def run(self, state): return state

# --- STATE DEFINITION ---
class AgentState(TypedDict):
    job_id: str
    project_id: str
    results_dir: str
    acquisition_mode: str
    acquisition_input: Any
    problem_description: str
    node_configs: Dict[str, Any]
    current_data: Optional[pd.DataFrame] 
    data_shape: str
    data_preview_html: str
    analysis: Dict[str, Any]
    chart_images: List[str]
    X_train: Optional[pd.DataFrame]
    X_test: Optional[pd.DataFrame]
    y_train: Optional[pd.Series]
    y_test: Optional[pd.Series]
    best_model: Any
    best_model_id: str
    leaderboard_html: str
    search_results: List[Dict[str, Any]]
    suggest_generate: bool
    final_message: str
    report_content: str
    dl_app: str
    dl_charts: str
    dl_model: str
    dl_report: str
    final_model_path: str
    charts_zip_path: str
    deployment_zip: str
    raw_df: Optional[pd.DataFrame]
    cleaned_df: Optional[pd.DataFrame]
    featured_df: Optional[pd.DataFrame]

AGENT_NODE_MAP = {
    "agent_1_data": DataAgent,
    "agent_2_analysis": AnalysisAgent,
    "agent_3_preprocess": PreprocessAgent,
    "agent_4_viz": VizAgent,
    "agent_5_feature": FeatureAgent,
    "agent_6_staging": StagingAgent,
    "agent_7_automl": AutoMLAgent,
    "agent_7_optuna": OptunaAgentPlaceholder,
    "agent_8_export": ExportAgent,
}

# --- PERSISTENCE HELPER ---
def save_intermediate(state, node_id):
    try:
        df = state.get('current_data')
        results_dir = state.get('results_dir')
        
        if results_dir:
            if df is not None and not df.empty:
                step_file = f"{node_id}_data.csv"
                df.to_csv(os.path.join(results_dir, step_file), index=False)
                
                # Update 'Active Data' only if this agent modifies data
                if node_id in ['agent_1_data', 'agent_3_preprocess', 'agent_5_feature']:
                    active_path = os.path.join(results_dir, "active_data.csv")
                    df.to_csv(active_path, index=False)
                    state['data_shape'] = str(df.shape)
                    try:
                        state['data_preview_html'] = df.head(50).to_html(classes='table table-striped', border=0, index=False)
                    except: pass
            else:
                print(f"   [System] {node_id} produced no new data. Keeping previous.")

    except Exception as e:
        print(f"   [System Warning] Snapshot failed: {e}")
    return state

# --- NODE FACTORY ---
def create_agent_node(agent_class, node_id, status_callback=None):
    def agent_node(state: AgentState):
        try:
            if status_callback: status_callback(f"START:{node_id}")
            print(f"--- EXECUTING {node_id} ---")
            
            agent_instance = agent_class()
            new_state = agent_instance.run(state)
            
            new_state = save_intermediate(new_state, node_id)
            
            if status_callback: status_callback(f"FINISH:{node_id}")
            return new_state
        except Exception as e:
            err_msg = f"Error in {node_id}: {str(e)}"
            print(f"!!! {err_msg}")
            if status_callback: status_callback(f"ERROR:{node_id}")
            state['final_message'] = err_msg
            raise e
    return agent_node

# --- GRAPH BUILDER ---
def build_graph(nodes_from_gui, edges_from_gui, status_callback=None):
    workflow = StateGraph(AgentState)
    entry_point = ""
    added_node_ids = set()

    for node in nodes_from_gui:
        node_type = node['type']
        node_id = node['id']
        if node_type in AGENT_NODE_MAP:
            node_func = create_agent_node(AGENT_NODE_MAP[node_type], node_id, status_callback)
            workflow.add_node(node_id, node_func)
            added_node_ids.add(node_id)
            if node_type == "agent_1_data": entry_point = node_id

    if not entry_point: raise ValueError("No Data Agent (Agent 1) found.")
    workflow.set_entry_point(entry_point)

    for edge in edges_from_gui:
        source = edge['source']
        target = edge['target']
        if source in added_node_ids and target in added_node_ids:
            if source == entry_point:
                def decide_next(state):
                    if state.get("search_results") or state.get("suggest_generate"): return "pause"
                    if state.get("current_data") is not None: return "continue"
                    return "error"
                workflow.add_conditional_edges(entry_point, decide_next, {"pause": END, "continue": target, "error": END})
            else:
                workflow.add_edge(source, target)

    sources = {e['source'] for e in edges_from_gui}
    for node_id in added_node_ids:
        if node_id not in sources and (node_id != entry_point or len(added_node_ids) == 1):
            workflow.add_edge(node_id, END)

    return workflow.compile()

# --- MAIN RUNNER ---
def run_pipeline_from_graph(initial_state, graph_layout, target_node_id=None, status_callback=None):
    try:
        if h2o.connection() is None: 
            h2o.init(nthreads=-1, max_mem_size="2G", verbose=False)
    except:
        try: h2o.init(nthreads=-1, verbose=False)
        except: print("Warning: H2O Init Failed.")

    app_graph = build_graph(graph_layout['nodes'], graph_layout['edges'], status_callback)
    
    results_dir = initial_state['results_dir']
    os.makedirs(os.path.join(results_dir, "charts"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "models"), exist_ok=True)
    
    final_state = initial_state
    
    try:
        if status_callback: status_callback("PIPELINE_START")
        
        for s in app_graph.stream(initial_state):
            step_name = list(s.keys())[0]
            final_state = s.get(step_name)
            
            if target_node_id and step_name == target_node_id:
                if not final_state.get('final_message'):
                    final_state['final_message'] = f"✅ Run stopped at user target: {target_node_id}"
                return final_state

            if final_state.get('search_results') or final_state.get('suggest_generate'):
                return final_state
        
        if 'report_content' in final_state: 
            final_state['final_message'] = final_state['report_content']
        elif 'best_model_id' in final_state:
            final_state['final_message'] = f"✅ Pipeline Completed. Best Model: {final_state['best_model_id']}"
        elif not final_state.get('final_message'):
            final_state['final_message'] = "✅ Pipeline Finished Successfully."
             
        return final_state

    except Exception as e:
        print(f"PIPELINE CRASHED: {e}")
        traceback.print_exc()
        final_state['final_message'] = f"Critical Error: {str(e)}"
        return final_state