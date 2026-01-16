import h2o
from h2o.automl import H2OAutoML
import pandas as pd

class AutoMLAgent:
    def run(self, state):
        print("-> Agent 7: Running AutoML...")
        
        df = state.get('X_train')
        y = state.get('y_train')
        
        # Validation
        if df is None or y is None: raise ValueError("Training data missing.")
        
        print(f"   Training on {len(df)} rows.")
        full_train = pd.concat([df, y], axis=1)
        
        # Init H2O
        try: 
            if h2o.connection() is None: h2o.init(nthreads=-1)
        except: 
            h2o.init(nthreads=-1)

        hf_train = h2o.H2OFrame(full_train)
        target_col = y.name
        x_cols = [c for c in hf_train.columns if c != target_col]
        
        # Check Problem Type
        problem_type = state['analysis'].get('problem_type', 'classification').lower()
        if 'classification' in problem_type:
            hf_train[target_col] = hf_train[target_col].asfactor()
            
            # --- CRITICAL FIX: H2O Level Check ---
            # Ask H2O explicitly how many classes it sees
            n_levels = hf_train[target_col].nlevels()[0]
            print(f"   H2O detected {n_levels} unique classes in target '{target_col}'.")
            
            if n_levels < 2:
                # Get the unique values to show user
                vals = hf_train[target_col].unique().as_data_frame().values.flatten()
                raise ValueError(
                    f"TRAINING STOPPED: The training set has only 1 class ({vals}). "
                    "AutoML needs at least 2 classes to learn. "
                    "Solution: Check if Agent 1 extracted the whole file, or if 'Staging' split accidentally removed a class."
                )

        # Configure AutoML
        aml = H2OAutoML(
            max_runtime_secs=300,
            seed=42,
            project_name=f"AutoML_{state.get('project_id', 'def')}",
            verbosity="info"
        )
        
        aml.train(x=x_cols, y=target_col, training_frame=hf_train)
        
        # Save Results
        lb = aml.leaderboard
        state['best_model'] = aml.leader
        state['best_model_id'] = aml.leader.model_id
        state['leaderboard'] = lb.as_data_frame()
        state['leaderboard_html'] = lb.as_data_frame().head(10).to_html(classes='table', border=0, index=False)
        
        return state