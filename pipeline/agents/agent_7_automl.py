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
        
        train_df = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True).rename(target)], axis=1)
        h2o_train = h2o.H2OFrame(train_df)
        
        if state['analysis']['problem_type'] == 'Classification':
            h2o_train[target] = h2o_train[target].asfactor()

        max_rt = 60 if len(X_train) < 1000 else 120
        aml = H2OAutoML(max_runtime_secs=max_rt, seed=1)
        aml.train(y=target, training_frame=h2o_train)

        best = aml.leader
        state['best_model'] = best
        state['best_model_id'] = best.model_id if best else "None"
        state['leaderboard'] = aml.leaderboard.as_data_frame() 
        return state