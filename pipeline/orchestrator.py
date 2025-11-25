import os
import h2o
from typing import TypedDict, Dict, Any, List
from langgraph.graph import StateGraph, END
import pandas as pd
from app import mongo 

from pipeline.agents.agent_1_data import DataAgent
from pipeline.agents.agent_2_analysis import AnalysisAgent
from pipeline.agents.agent_3_preprocess import PreprocessAgent
from pipeline.agents.agent_4_viz import VizAgent
from pipeline.agents.agent_5_feature import FeatureAgent
from pipeline.agents.agent_6_staging import StagingAgent
from pipeline.agents.agent_7_automl import AutoMLAgent
from pipeline.agents.agent_8_export import ExportAgent

class OptunaAgentPlaceholder:
    def run(self, state): return state

class AgentState(TypedDict):
    project_id: str; acquisition_mode: str; acquisition_input: Any; problem_description: str; results_dir: str; final_message: str
    raw_df: pd.DataFrame; analysis: Dict[str, Any]; cleaned_df: pd.DataFrame; chart_images: List[Any]; featured_df: pd.DataFrame
    X_train: pd.DataFrame; X_test: pd.DataFrame; y_train: pd.Series; y_test: pd.Series
    best_model: Any; best_model_id: str; final_model_path: str; leaderboard: Any
    search_results: List[Dict[str, Any]]; suggest_generate: bool

AGENT_NODE_MAP = {
    "agent_1_data": DataAgent, "agent_2_analysis": AnalysisAgent, "agent_3_preprocess": PreprocessAgent,
    "agent_4_viz": VizAgent, "agent_5_feature": FeatureAgent, "agent_6_staging": StagingAgent,
    "agent_7_automl": AutoMLAgent, "agent_7_optuna": OptunaAgentPlaceholder, "agent_8_export": ExportAgent,
}

def create_agent_node(agent_class):
    def agent_node(state: AgentState):
        print(f"--- EXECUTING {agent_class.__name__} ---")
        return agent_class().run(state)
    return agent_node

def build_graph(nodes_from_gui, edges_from_gui):
    workflow = StateGraph(AgentState)
    entry_point = ""
    added_node_ids = set()

    for node in nodes_from_gui:
        if node['type'] in AGENT_NODE_MAP:
            workflow.add_node(node['id'], create_agent_node(AGENT_NODE_MAP[node['type']]))
            added_node_ids.add(node['id'])
            if node['type'] == "agent_1_data": entry_point = node['id']

    if not entry_point: return None
    workflow.set_entry_point(entry_point)

    for edge in edges_from_gui:
        if edge['source'] in added_node_ids and edge['target'] in added_node_ids:
            if edge['source'] == entry_point:
                def decide(state):
                    if state.get("search_results") or state.get("suggest_generate"): return "pause"
                    if "raw_df" in state and not state['raw_df'].empty: return "continue"
                    return "error"
                workflow.add_conditional_edges(entry_point, decide, {"pause": END, "continue": edge['target'], "error": END})
            else:
                workflow.add_edge(edge['source'], edge['target'])

    sources = {e['source'] for e in edges_from_gui}
    for n in added_node_ids:
        if n not in sources and n != entry_point: workflow.add_edge(n, END)

    return workflow.compile()

def run_pipeline_from_graph(initial_state, graph_layout):
    try:
        if h2o.connection() is None: h2o.init(nthreads=-1, max_mem_size="8g")
    except: h2o.init(nthreads=-1, max_mem_size="8g")

    app_graph = build_graph(graph_layout['nodes'], graph_layout['edges'])
    if not app_graph: raise ValueError("Invalid Graph")

    results_dir = initial_state['results_dir']
    os.makedirs(os.path.join(results_dir, "charts"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "downloaded_data"), exist_ok=True)
    
    final_state = {}
    try:
        for s in app_graph.stream(initial_state):
            step = list(s.keys())[0]
            print(f"--- Finished {step} ---")
            final_state = s.get(step)
            if final_state.get('search_results') or final_state.get('suggest_generate'): return final_state
        
        final_state['final_message'] = f"✅ Pipeline complete!\nModel: {final_state.get('best_model_id', 'N/A')}\nPath: {final_state.get('final_model_path', 'N/A')}"
        return final_state
    except Exception as e:
        print(f"PIPELINE FAILED: {e}")
        raise e