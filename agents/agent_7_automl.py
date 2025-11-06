# agents/agent_7_automl.py

import h2o
from h2o.automl import H2OAutoML
import pandas as pd

class AutoMLAgent: 
    
    def __init__(self):
        try:
            if h2o.connection() is None:
                h2o.init(nthreads=-1, max_mem_size="8g")
                print("   H2O cluster initialized.")
            else:
                print("   H2O cluster already running.")
        except Exception as e:
            print(f"H2O init failed: {e}. Raising error.")
            raise e
    
    def run(self, state):
        print("-> Agent 7: Running H2O.ai AutoML...")
        
        X_train = state['X_train']
        y_train = state['y_train']
        target = state['analysis']['target_variable']

        train_df = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True).rename(target)], axis=1)
        h2o_train = h2o.H2OFrame(train_df)

        x = h2o_train.columns
        y = target
        x.remove(y)

        if state['analysis']['problem_type'] == 'Classification':
            h2o_train[y] = h2o_train[y].asfactor()

        # AutoML run (shorter for small data like Iris)
        max_runtime = 120
        if len(X_train) < 500:
            print("   Small dataset detected, running a shorter AutoML (60s).")
            max_runtime = 60
            
        aml = H2OAutoML(max_runtime_secs=max_runtime, seed=1)
        aml.train(x=x, y=y, training_frame=h2o_train)

        lb = aml.leaderboard
        print("\n   H2O AutoML Leaderboard:")
        print(lb.head(rows=lb.nrows))

        best_model = aml.leader
        best_model_id = best_model.model_id
        
        state['best_model'] = best_model
        state['best_model_id'] = best_model_id 
        return state