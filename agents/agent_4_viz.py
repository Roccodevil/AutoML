# agents/agent_4_viz.py

import matplotlib 
matplotlib.use('Agg')

import seaborn as sns
import matplotlib.pyplot as plt 
import os
import io
from PIL import Image

class VizAgent: 
    
    def run(self, state):
        print("-> Agent 4: Generating data visualizations...")
        df = state['cleaned_df']
        charts_dir = os.path.join(state['results_dir'], "charts")
        chart_images = [] 

        try:
            # 1. Correlation Heatmap (for numeric features)
            numeric_df = df.select_dtypes(include=['float64', 'int64'])
            if not numeric_df.empty and len(numeric_df.columns) > 1:
                # Limit features for readability if too many
                cols_to_plot = numeric_df.columns
                if len(cols_to_plot) > 20: 
                    print("   Too many numeric features for heatmap, plotting top 20 correlations with target.")
                    target = state['analysis']['target_variable']
                    if target in numeric_df.columns:
                        top_corr_cols = numeric_df.corr()[target].abs().nlargest(21).index # Target + top 20
                        numeric_df = numeric_df[top_corr_cols]
                    else: # If target isn't numeric, just take first 20
                         numeric_df = numeric_df.iloc[:, :20]

                fig, ax = plt.subplots(figsize=(12, 8))
                sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
                ax.set_title("Correlation Heatmap")
                
                path = os.path.join(charts_dir, "correlation_heatmap.png")
                fig.savefig(path, bbox_inches='tight')
                
                img_buf = io.BytesIO()
                fig.savefig(img_buf, format='png', bbox_inches='tight')
                img_buf.seek(0)
                chart_images.append(Image.open(img_buf))
                plt.close(fig)

            # 2. Target Distribution
            target = state['analysis']['target_variable']
            if target in df:
                fig, ax = plt.subplots(figsize=(10, 6))
                if state['analysis']['problem_type'] == 'Classification':
                    # Check cardinality before countplot
                    if df[target].nunique() < 50: 
                        sns.countplot(data=df, x=target, ax=ax, palette="viridis")
                    else: # Use histplot for high cardinality categorical
                         sns.histplot(data=df, x=target, ax=ax)
                    ax.set_title(f"Distribution of Target Variable ({target})")
                else: # Regression
                    sns.histplot(data=df, x=target, kde=True, ax=ax)
                    ax.set_title(f"Distribution of Target Variable ({target})")
                
                path = os.path.join(charts_dir, "target_distribution.png")
                fig.savefig(path, bbox_inches='tight')
                
                img_buf = io.BytesIO()
                fig.savefig(img_buf, format='png', bbox_inches='tight')
                img_buf.seek(0)
                chart_images.append(Image.open(img_buf))
                plt.close(fig)

            print(f"   Charts saved to '{charts_dir}'")
            state['chart_images'] = chart_images
        
        except Exception as e:
            print(f"   Visualization failed: {e}")
            import traceback
            traceback.print_exc() # Print full error for debugging
            state['chart_images'] = []
            
        return state