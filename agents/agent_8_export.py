# agents/agent_8_export.py

import h2o
import os

class ExportAgent: 
    
    def run(self, state):
        print("-> Agent 8: Exporting the final trained model...")
        model = state['best_model']
        models_dir = os.path.join(state['results_dir'], "models")
        
        # H2O models are best saved as MOJO (Model Object, Optimized)
        # or as a binary file
        try:
            # MOJO is generally more flexible for production
            model_path = model.download_mojo(path=models_dir, get_genmodel_jar=False)
            print(f"   Model saved as MOJO: {model_path}")
        except Exception as e:
            print(f"   MOJO save failed: {e}. Trying H2O Binary.")
            model_path = h2o.save_model(model=model, path=models_dir, force=True)
            print(f"   Model saved as H2O Binary: {model_path}")

        state['final_model_path'] = model_path
        return state