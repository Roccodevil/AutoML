import os
import h2o
from typing import TypedDict, Dict, Any, List
from langgraph.graph import StateGraph, END
import pandas as pd
from app import mongo

# Import agents
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

class AgentState(TypedDict):
    project_id: str
    acquisition_mode: str
    acquisition_input: Any
    problem_description: str
    results_dir: str
    final_message: str
    node_configs: Dict[str, Any]
    data_preview_html: str
    data_shape: str
    raw_df: pd.DataFrame
    analysis: Dict[str, Any]
    cleaned_df: pd.DataFrame
    chart_images: List[Any]
    featured_df: pd.DataFrame
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    best_model: Any
    best_model_id: str
    final_model_path: str
    leaderboard: Any
    leaderboard_html: str
    report_content: str
    search_results: List[Dict[str, Any]]
    suggest_generate: bool
    charts_zip_path: str
    deployment_zip: str
    dl_app: str
    dl_charts: str
    dl_model: str
    dl_report: str

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

def create_agent_node(agent_class):
    def agent_node(state: AgentState):
        try:
            print(f"--- EXECUTING {agent_class.__name__} ---")
            return agent_class().run(state)
        except Exception as e:
            print(f"!!! ERROR IN {agent_class.__name__}: {e}")
            raise e
    return agent_node

def build_graph(nodes_from_gui: List[Dict], edges_from_gui: List[Dict]):
    workflow = StateGraph(AgentState)
    entry_point = ""
    added_node_ids = set()

    for node in nodes_from_gui:
        node_type = node['type']
        node_id = node['id']
        if node_type in AGENT_NODE_MAP:
            workflow.add_node(node_id, create_agent_node(AGENT_NODE_MAP[node_type]))
            added_node_ids.add(node_id)
            if node_type == "agent_1_data":
                entry_point = node_id

    if not entry_point:
        raise ValueError("Pipeline Error: No 'Data Acquisition' (Agent 1) node found.")
        
    workflow.set_entry_point(entry_point)

    for edge in edges_from_gui:
        source = edge['source']
        target = edge['target']
        if source in added_node_ids and target in added_node_ids:
            if source == entry_point:
                def decide(state):
                    if state.get("search_results") or state.get("suggest_generate"): return "pause"
                    if "raw_df" in state and not state['raw_df'].empty: return "continue"
                    return "error"
                workflow.add_conditional_edges(entry_point, decide, {"pause": END, "continue": target, "error": END})
            else:
                workflow.add_edge(source, target)

    sources = {e['source'] for e in edges_from_gui}
    for node_id in added_node_ids:
        if node_id not in sources and (node_id != entry_point or len(added_node_ids) == 1):
            workflow.add_edge(node_id, END)

    return workflow.compile()

def run_pipeline_from_graph(initial_state: dict, graph_layout: dict, target_node_id=None):
    # H2O Init
    try:
        if h2o.connection() is None: h2o.init(nthreads=-1)
    except:
        try: h2o.init()
        except: print("Warning: H2O failed to init.")

    app_graph = build_graph(graph_layout['nodes'], graph_layout['edges'])
    if not app_graph: raise ValueError("Invalid Graph")

    results_dir = initial_state['results_dir']
    os.makedirs(os.path.join(results_dir, "charts"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "downloaded_data"), exist_ok=True)
    
    final_state = {}
    
    try:
        if 'node_configs' not in initial_state: initial_state['node_configs'] = {}
        
        print(f"--- Pipeline Started (Target: {target_node_id or 'End'}) ---")
        
        for s in app_graph.stream(initial_state):
            step_name = list(s.keys())[0]
            print(f"--- Finished Step: {step_name} ---")
            
            final_state = s.get(step_name)
            
            # Save Active Data for Preview/Download
            active_df = None
            if 'featured_df' in final_state: active_df = final_state['featured_df']
            elif 'cleaned_df' in final_state: active_df = final_state['cleaned_df']
            elif 'raw_df' in final_state: active_df = final_state['raw_df']
            
            if active_df is not None:
                final_state['data_shape'] = str(active_df.shape)
                active_path = os.path.join(results_dir, "active_data.csv")
                active_df.to_csv(active_path, index=False)
                # Max 50 rows for HTML preview to keep UI fast
                final_state['data_preview_html'] = active_df.head(50).to_html(classes='table', border=0, index=False)

            # --- STOP LOGIC: If we hit the target node, we exit successfully ---
            if target_node_id and step_name == target_node_id:
                final_state['final_message'] = f"✅ Run successful up to step: {target_node_id}"
                return final_state

            # --- PAUSE LOGIC: For Search/Generate actions ---
            if final_state.get('search_results') or final_state.get('suggest_generate'):
                return final_state
        
        # Complete Run Logic
        if 'report_content' in final_state: 
            final_state['final_message'] = final_state['report_content']
        else: 
            final_state['final_message'] = f"✅ Pipeline complete!\nModel: {final_state.get('best_model_id', 'N/A')}"
             
        return final_state

    except Exception as e:
        print(f"PIPELINE FAILED: {e}")
        raise e