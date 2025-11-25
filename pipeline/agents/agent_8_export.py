import h2o
import os
import joblib

class ExportAgent: 
    def run(self, state):
        print("-> Agent 8: Exporting model...")
        model = state['best_model']
        models_dir = os.path.join(state['results_dir'], "models")
        
        try:
            model_path = model.download_mojo(path=models_dir, get_genmodel_jar=False)
        except:
            try: model_path = h2o.save_model(model=model, path=models_dir, force=True)
            except: model_path = "Error saving model"

        report_path = os.path.join(state['results_dir'], "final_report.txt")
        with open(report_path, "w") as f:
            f.write(f"Problem: {state['problem_description']}\n")
            f.write(f"Best Model: {state['best_model_id']}\n")
            f.write(f"Saved: {model_path}\n\n")
            if 'leaderboard' in state and state['leaderboard'] is not None:
                 f.write(state['leaderboard'].to_string())
            else:
                 f.write("Leaderboard unavailable.")

        state['final_model_path'] = model_path
        return state