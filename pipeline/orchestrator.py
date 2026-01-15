import os
import h2o
from typing import TypedDict, Dict, Any, List
from langgraph.graph import StateGraph, END
import pandas as pd
from app import mongo

# Import all your agent classes
from pipeline.agents.agent_1_data import DataAgent
from pipeline.agents.agent_2_analysis import AnalysisAgent
from pipeline.agents.agent_3_preprocess import PreprocessAgent
from pipeline.agents.agent_4_viz import VizAgent
from pipeline.agents.agent_5_feature import FeatureAgent
from pipeline.agents.agent_6_staging import StagingAgent
from pipeline.agents.agent_7_automl import AutoMLAgent
from pipeline.agents.agent_8_export import ExportAgent

# --- Placeholder Class for Optuna (Prevents Crash) ---
class OptunaAgentPlaceholder:
    def run(self, state):
        print("-> Agent 7 (Optuna): Placeholder executing.")
        return state

# --- 1. Define the State ---
class AgentState(TypedDict):
    project_id: str
    acquisition_mode: str
    acquisition_input: Any
    problem_description: str
    results_dir: str
    final_message: str
    
    # Configuration & GUI Feedback
    node_configs: Dict[str, Any]
    data_preview_html: str
    data_shape: str
    
    # Data Flow
    raw_df: pd.DataFrame
    analysis: Dict[str, Any]
    cleaned_df: pd.DataFrame
    chart_images: List[Any]
    featured_df: pd.DataFrame
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    
    # Model & Outputs
    best_model: Any
    best_model_id: str
    final_model_path: str
    leaderboard: Any
    leaderboard_html: str
    report_content: str  # Stores the full text report
    
    # Branching/Control
    search_results: List[Dict[str, Any]]
    suggest_generate: bool

# --- 2. Create Agent "Nodes" ---
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
        print(f"--- EXECUTING {agent_class.__name__} ---")
        return agent_class().run(state)
    return agent_node

# --- 3. Define the Graph Edges & Logic ---
def build_graph(nodes_from_gui: List[Dict], edges_from_gui: List[Dict]):
    workflow = StateGraph(AgentState)
    entry_point = ""
    added_node_ids = set()

    # Add Nodes
    for node in nodes_from_gui:
        node_type = node['type']
        node_id = node['id']
        if node_type in AGENT_NODE_MAP:
            workflow.add_node(node_id, create_agent_node(AGENT_NODE_MAP[node_type]))
            added_node_ids.add(node_id)
            if node_type == "agent_1_data":
                entry_point = node_id

    if not entry_point:
        return None
    
    workflow.set_entry_point(entry_point)

    # Add Edges
    for edge in edges_from_gui:
        source = edge['source']
        target = edge['target']
        
        if source in added_node_ids and target in added_node_ids:
            # Agent 1 has special conditional logic
            if source == entry_point:
                def decide(state):
                    if state.get("search_results") or state.get("suggest_generate"): return "pause"
                    if "raw_df" in state and not state['raw_df'].empty: return "continue"
                    return "error"
                
                workflow.add_conditional_edges(
                    entry_point, 
                    decide, 
                    {"pause": END, "continue": target, "error": END}
                )
            else:
                # Standard connection
                workflow.add_edge(source, target)

    # Connect terminal nodes to END
    sources = {e['source'] for e in edges_from_gui}
    for node_id in added_node_ids:
        if node_id not in sources and (node_id != entry_point or len(added_node_ids) == 1):
            workflow.add_edge(node_id, END)

    return workflow.compile()

# --- 4. The Main "Run" Function ---
def run_pipeline_from_graph(initial_state: dict, graph_layout: dict, target_node_id=None):
    # Ensure H2O is running (Do not shutdown between runs)
    try:
        if h2o.connection() is None:
            h2o.init(nthreads=-1)#, max_mem_size="8g")
    except:
        h2o.init(nthreads=-1)#, max_mem_size="8g")

    app_graph = build_graph(graph_layout['nodes'], graph_layout['edges'])
    if not app_graph:
        raise ValueError("Invalid Graph: No Data Acquisition node found.")

    # Create directories
    results_dir = initial_state['results_dir']
    os.makedirs(os.path.join(results_dir, "charts"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "downloaded_data"), exist_ok=True)
    
    final_state = {}
    
    try:
        if 'node_configs' not in initial_state:
            initial_state['node_configs'] = {}
            
        # Stream execution step-by-step
        for s in app_graph.stream(initial_state):
            step_name = list(s.keys())[0]
            print(f"--- Finished Step: {step_name} ---")
            final_state = s.get(step_name)
            
            # --- CRITICAL: Save Intermediate Data for Download ---
            # Logic: Check for the most processed dataframe available and save it as 'active_data.csv'
            active_df = None
            if 'featured_df' in final_state: active_df = final_state['featured_df']
            elif 'cleaned_df' in final_state: active_df = final_state['cleaned_df']
            elif 'raw_df' in final_state: active_df = final_state['raw_df']
            
            if active_df is not None:
                final_state['data_shape'] = str(active_df.shape)
                active_path = os.path.join(results_dir, "active_data.csv")
                active_df.to_csv(active_path, index=False)
                # Update the HTML preview for the frontend
                final_state['data_preview_html'] = active_df.head(50).to_html(classes='table', border=0, index=False)

            # --- Stop Condition (Partial Run) ---
            if target_node_id and step_name == target_node_id:
                final_state['final_message'] = f"Run stopped after step {target_node_id}."
                return final_state

            # --- Pause Condition (Search/Generate) ---
            if final_state.get('search_results') or final_state.get('suggest_generate'):
                return final_state
        
        # --- Final Message Construction ---
        # If we have the full report text, use it. Otherwise fallback to simple message.
        if 'report_content' in final_state:
             final_state['final_message'] = final_state['report_content']
        else:
             final_state['final_message'] = f"✅ Pipeline complete!\nModel: {final_state.get('best_model_id', 'N/A')}\nPath: {final_state.get('final_model_path', 'N/A')}"
             
        return final_state

    except Exception as e:
        print(f"PIPELINE FAILED: {e}")
        # We do NOT shutdown H2O here, so the user can retry without restarting the server
        raise e