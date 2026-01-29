import h2o
from h2o.automl import H2OAutoML
import pandas as pd

class AutoMLAgent:
    def run(self, state):
        print("-> Agent 7: Running AutoML...")
        
        df = state.get('X_train')
        y = state.get('y_train')
        if df is None or y is None: raise ValueError("Training data missing.")
        
        # 1. Safe H2O Initialization
        try: 
            # Try to connect, if fails, start new with lower memory reqs if needed
            if h2o.connection() is None: h2o.init(nthreads=-1, max_mem_size="2G")
        except: 
            h2o.init(nthreads=-1)

        # 2. Prepare Data
        full_train = pd.concat([df, y], axis=1)
        hf_train = h2o.H2OFrame(full_train)
        target_col = y.name
        x_cols = [c for c in hf_train.columns if c != target_col]
        
        # 3. Handle Classification vs Regression
        problem_type = state['analysis'].get('problem_type', 'classification').lower()
        if 'classification' in problem_type:
            hf_train[target_col] = hf_train[target_col].asfactor()
            n_levels = hf_train[target_col].nlevels()[0]
            if n_levels < 2:
                raise ValueError("Training Error: Target variable has only 1 class. Need 2+ for classification.")

        # 4. Check User Config for Duration
        config = state.get('node_configs', {}).get('agent_7_automl', {})
        # Default to 60s to prevent browser timeout (Freeze)
        runtime = 60 if config.get('mode') == 'default' else 300 
        
        print(f"   Starting H2O AutoML (Limit: {runtime}s)...")
        aml = H2OAutoML(
            max_runtime_secs=runtime,  # <--- PREVENTS LONG WAITS
            seed=42,
            project_name=f"AutoML_{state.get('project_id', 'def')}",
            verbosity="info"
        )
        
        aml.train(x=x_cols, y=target_col, training_frame=hf_train)
        
        # 5. Save Results
        lb = aml.leaderboard
        state['best_model'] = aml.leader
        state['best_model_id'] = aml.leader.model_id
        state['leaderboard'] = lb.as_data_frame()
        # Safe HTML conversion
        lb_df = lb.as_data_frame()
        state['leaderboard_html'] = lb_df.head(10).to_html(classes='table', border=0, index=False)
        
        print(f"   AutoML Complete. Best Model: {aml.leader.model_id}")
        return state