import os
from typing import TypedDict, Dict, Any, List
from langgraph.graph import StateGraph, END
import pandas as pd
from app import mongo

# Agents
from pipeline.agents.agent_nlp_1_data import NLPDataAgent
from pipeline.agents.agent_nlp_2_task import NLPTaskAgent
from pipeline.agents.agent_nlp_3_preprocess import NLPPreprocessAgent
from pipeline.agents.agent_nlp_4_model import NLPModelAgent
from pipeline.agents.agent_nlp_5_export import NLPExportAgent

class NLPState(TypedDict):
    project_id: str
    acquisition_mode: str
    acquisition_input: Any
    problem_description: str
    results_dir: str
    final_message: str
    node_configs: Dict[str, Any]
    raw_df: pd.DataFrame
    text_column: str
    label_column: str
    task_type: str
    processed_df: pd.DataFrame
    vectorizer: Any
    model: Any
    is_finished: bool
    summary_result: str
    analysis_result: str
    dl_app: str
    dl_data: str

# Helper to run agent and trigger callback
def create_node(agent_class, node_id, callback=None):
    def node_fn(state):
        if callback: callback(f"START:{node_id}")
        try:
            result = agent_class().run(state)
            if callback: callback(f"FINISH:{node_id}")
            return result
        except Exception as e:
            if callback: callback(f"ERROR:{node_id}:{str(e)}")
            raise e
    return node_fn

def run_nlp_pipeline(initial_state: dict, graph_layout: dict, target_node_id=None, status_callback=None):
    workflow = StateGraph(NLPState)
    
    # Nodes (Mapped to IDs used in HTML)
    workflow.add_node("agent_nlp_1_data", create_node(NLPDataAgent, "agent_nlp_1_data", status_callback))
    workflow.add_node("agent_nlp_2_task", create_node(NLPTaskAgent, "agent_nlp_2_task", status_callback))
    workflow.add_node("agent_nlp_3_preprocess", create_node(NLPPreprocessAgent, "agent_nlp_3_preprocess", status_callback))
    workflow.add_node("agent_nlp_4_model", create_node(NLPModelAgent, "agent_nlp_4_model", status_callback))
    workflow.add_node("agent_nlp_5_export", create_node(NLPExportAgent, "agent_nlp_5_export", status_callback))

    # Conditional Logic
    def check_task_stop(state):
        return "end" if state.get('is_finished') else "continue"

    def check_prep_stop(state):
        return "end" if state.get('is_finished') else "continue"

    # Edges
    workflow.set_entry_point("agent_nlp_1_data")
    workflow.add_edge("agent_nlp_1_data", "agent_nlp_2_task")
    
    # Branch 1: Stop if task was Summary/Translation
    workflow.add_conditional_edges("agent_nlp_2_task", check_task_stop, {"end": END, "continue": "agent_nlp_3_preprocess"})
    
    # Branch 2: Stop if task was Preprocessing Only
    workflow.add_conditional_edges("agent_nlp_3_preprocess", check_prep_stop, {"end": END, "continue": "agent_nlp_4_model"})
    
    workflow.add_edge("agent_nlp_4_model", "agent_nlp_5_export")
    workflow.add_edge("agent_nlp_5_export", END)

    app = workflow.compile()
    
    # Run
    if status_callback: status_callback("PIPELINE_START")
    final_state = {}
    
    for s in app.stream(initial_state):
        step_name = list(s.keys())[0]
        final_state = s[step_name]
        # Allow running up to a specific node for debugging
        if target_node_id and step_name == target_node_id:
            return final_state

    return final_state