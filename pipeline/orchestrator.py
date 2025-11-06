# pipeline/orchestrator.py

import os
import h2o

from agents.agent_1_data import DataAgent
from agents.agent_2_analysis import AnalysisAgent
from agents.agent_3_preprocess import PreprocessAgent
from agents.agent_4_viz import VizAgent
from agents.agent_5_feature import FeatureAgent
from agents.agent_6_staging import StagingAgent
from agents.agent_7_automl import AutoMLAgent
from agents.agent_8_export import ExportAgent

def run_full_pipeline(state: dict):
    """
    Orchestrates the entire multi-agent AutoML pipeline.
    
    The 'state' dict is passed from agent to agent, gathering all data.
    """
    
    callback = state['callback']
    
    # Create results directory
    results_dir = state['results_dir']
    os.makedirs(os.path.join(results_dir, "charts"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "models"), exist_ok=True)
    
    agent1 = DataAgent()
    agent2 = AnalysisAgent()
    agent3 = PreprocessAgent()
    agent4 = VizAgent()
    agent5 = FeatureAgent()
    agent6 = StagingAgent()
    agent7 = AutoMLAgent()
    agent8 = ExportAgent()
    
    try:
        # Step 1: Data Acquisition
        callback("Step 1: Data Acquisition Agent is working...", 10)
        state = agent1.run(state)
        
        # Step 2: Problem Analysis
        callback("Step 2: Problem Analysis Agent is working...", 20)
        state = agent2.run(state)
        
        # Step 3: Preprocessing
        callback("Step 3: Preprocessing Agent is cleaning data...", 35)
        state = agent3.run(state)
        
        # Step 4: Visualization
        callback("Step 4: Visualization Agent is generating charts...", 50)
        state = agent4.run(state)
        
        # Step 5: Feature Engineering
        callback("Step 5: Feature Engineering Agent is creating features...", 60)
        state = agent5.run(state)
        
        # Step 6: Data Staging
        callback("Step 6: Data Staging Agent is splitting data...", 75)
        state = agent6.run(state)
        
        # Step 7: AutoML Training
        callback("Step 7: AutoML Agent (H2O.ai) is training models...", 90)
        state = agent7.run(state)
        
        # Step 8: Export
        callback("Step 8: Export Agent is saving the final model...", 100)
        state = agent8.run(state)

        # final message 
        final_message = f"✅ Pipeline complete!\nBest Model: {state['best_model_id']}\nSaved to: {state['final_model_path']}"
        state['final_message'] = final_message
        
        return state 

    except Exception as e:
        print(f"PIPELINE FAILED: {e}")
        import traceback
        traceback.print_exc()
        raise e
    finally:
        # Shutdown H2O cluster
        if h2o.connection() is not None:
            h2o.shutdown(prompt=False)
            print("\nShut down H2O cluster.")