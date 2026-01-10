import h2o
from h2o.automl import H2OAutoML
import pandas as pd

class AutoMLAgent: 
    def __init__(self):
        if h2o.connection() is None: h2o.init(nthreads=-1, max_mem_size="8g")

    def run(self, state):
        print("-> Agent 7: Running AutoML...")
        X_train, y_train = state['X_train'], state['y_train']
        target = state['analysis']['target_variable']
        
        # --- Read Configuration ---
        config = state.get('node_configs', {}).get('agent_7_automl', {})
        mode = config.get('mode', 'default')
        
        max_runtime = 60 if len(X_train) < 1000 else 120
        if mode == 'custom':
            print("   Custom Config: Extensive Training selected.")
            max_runtime = 300 # 5 minutes
        
        train_df = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True).rename(target)], axis=1)
        h2o_train = h2o.H2OFrame(train_df)
        x = h2o_train.columns; y = target; x.remove(y)

        if state['analysis']['problem_type'] == 'Classification':
            h2o_train[y] = h2o_train[y].asfactor()

        aml = H2OAutoML(max_runtime_secs=max_runtime, seed=1)
        aml.train(x=x, y=y, training_frame=h2o_train)

        best = aml.leader
        state['best_model'] = best
        state['best_model_id'] = best.model_id if best else "None"
        
        # Save raw leaderboard for Export, and HTML for GUI
        lb = aml.leaderboard
        state['leaderboard'] = lb.as_data_frame()
        state['leaderboard_html'] = lb.as_data_frame().head(10).to_html(classes='table', border=0, index=False)
        
        return state