import pandas as pd
import seaborn as sns
import matplotlib
# Force non-interactive backend for server environments
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import os
import uuid
import numpy as np
from core.llm_services import llm_powerful_api
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

class VizAgent:
    def __init__(self):
        # The AI Visualization Expert Prompt
        self.strategy_prompt = ChatPromptTemplate.from_template(
            """You are a Lead Data Visualization Expert.
            Analyze the dataset stats and User Requests to prescribe the best 3-6 charts.

            DATA PROFILE:
            - Rows: {rows}, Columns: {cols}
            - Numerical Cols: {num_cols}
            - Categorical Cols: {cat_cols}
            - Correlations (Top 5): {corr_info}
            - User Command: "{user_request}" (PRIORITIZE THIS)

            CAPABILITIES:
            - 'hist': Histogram (Distribution)
            - 'box': Boxplot (Outliers/Comparison)
            - 'scatter': Scatter Plot (Relationship between 2 numbers)
            - 'bar': Bar Chart (Categorical Counts or Aggregates)
            - 'heatmap': Correlation Matrix
            - 'pair': Pair Plot (Multivariate overview)
            - 'violin': Violin Plot (Distribution + Box)

            OUTPUT JSON RULES:
            - Return a list of objects in a "charts" key.
            - Each object must have: "type", "x", "y" (optional), "hue" (optional), "title".
            - If User Command is empty, suggest the most insightful charts automatically.
            
            EXAMPLE OUTPUT:
            {{
                "charts": [
                    {{"type": "hist", "x": "Age", "hue": "Survived", "title": "Age Distribution by Survival"}},
                    {{"type": "heatmap", "title": "Correlation Matrix"}}
                ]
            }}
            """
        )
        self.llm_chain = self.strategy_prompt | llm_powerful_api | JsonOutputParser()

    def run(self, state):
        print("-> Agent 4: Intelligent Visualization...")
        
        # 1. Get Data Safely
        df = state.get('cleaned_df')
        if df is None:
            df = state.get('raw_df')
            
        if df is None: 
            raise ValueError("No data found for visualization.")
        
        # 2. Analyze Column Types
        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        
        # Calculate Correlation only if we have enough numeric cols
        corr_info = "N/A"
        if len(num_cols) > 1:
            try:
                corr_matrix = df[num_cols].corr().abs()
                # Extract top correlations
                pairs = (corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                         .stack().sort_values(ascending=False).head(5))
                corr_info = str(pairs.to_dict())
            except: pass

        # 3. Get User Config
        config = state.get('node_configs', {}).get('agent_4_viz', {})
        user_request = config.get('user_request', '')

        # 4. Determine Strategy (LLM or Fallback)
        plan = {"charts": []}
        
        # FALLBACK: If no numeric columns exist, force Categorical plots
        if not num_cols and cat_cols:
            print("   [Agent 4] Warning: No numeric columns found. Creating categorical count plots.")
            # Plot top 3 categorical columns
            for col in cat_cols[:3]:
                plan['charts'].append({
                    "type": "bar", 
                    "x": col, 
                    "title": f"Distribution of {col}"
                })
        else:
            # Standard LLM Planning
            stats = {
                "rows": len(df), "cols": len(df.columns),
                "num_cols": num_cols[:10], "cat_cols": cat_cols[:10],
                "corr_info": corr_info,
                "user_request": user_request
            }
            
            print(f"   Analyzing with User Request: '{user_request}'")
            try:
                plan = self.llm_chain.invoke(stats)
                print(f"   LLM Planned {len(plan.get('charts', []))} charts.")
            except Exception as e:
                print(f"   LLM Error: {e}. Reverting to defaults.")
                # Basic default if LLM fails
                if num_cols:
                    plan['charts'].append({"type": "hist", "x": num_cols[0], "title": "Distribution"})
                if cat_cols:
                    plan['charts'].append({"type": "bar", "x": cat_cols[0], "title": "Counts"})

        # 5. Prepare Directory
        save_dir = os.path.join(state['results_dir'], "charts")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        else:
            # Clean old charts
            for f in os.listdir(save_dir):
                if f.endswith('.png'): os.remove(os.path.join(save_dir, f))
            
        # 6. Generate Charts
        sns.set_theme(style="darkgrid")
        plt.rcParams.update({'figure.max_open_warning': 0})
        
        img_paths = []
        charts_list = plan.get('charts', [])
        if not isinstance(charts_list, list): charts_list = []

        for i, chart in enumerate(charts_list):
            try:
                plt.figure(figsize=(10, 6))
                c_type = chart.get('type')
                x = chart.get('x')
                y = chart.get('y')
                hue = chart.get('hue')
                title = chart.get('title', f"Chart {i+1}")

                # Validation: Ensure columns actually exist
                if x and x not in df.columns: x = None
                if y and y not in df.columns: y = None
                if hue and hue not in df.columns: hue = None

                # Plot Logic
                if c_type == 'hist' and x:
                    sns.histplot(data=df, x=x, hue=hue, kde=True, multiple="stack")
                
                elif c_type == 'box' and x:
                    if y: sns.boxplot(data=df, x=x, y=y, hue=hue) # Bivariate
                    else: sns.boxplot(data=df, x=x, hue=hue)      # Univariate
                
                elif c_type == 'bar' and x:
                    # If 'y' is missing, countplot (counts). If 'y' exists, barplot (agg mean).
                    if y: sns.barplot(data=df, x=x, y=y, hue=hue)
                    else: sns.countplot(data=df, x=x, hue=hue)
                
                elif c_type == 'scatter' and x and y:
                    sns.scatterplot(data=df, x=x, y=y, hue=hue)
                
                elif c_type == 'violin' and x:
                    sns.violinplot(data=df, x=x, y=y, hue=hue)
                
                elif c_type == 'heatmap': 
                    # Only correlate numeric columns
                    num_df = df.select_dtypes(include=['number'])
                    if not num_df.empty:
                        sns.heatmap(num_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
                
                elif c_type == 'pair':
                    # Pairplot handles its own figure management
                    plt.close()
                    vars_to_plot = [c for c in num_cols[:5]] # Limit to 5 vars for speed
                    g = sns.pairplot(df, hue=hue, vars=vars_to_plot)
                    fname = f"chart_{uuid.uuid4().hex[:8]}.png"
                    path = os.path.join(save_dir, fname)
                    g.savefig(path)
                    img_paths.append(f"/results/charts/{fname}")
                    continue # Skip generic save

                # Finalize Generic Plot
                plt.title(title)
                plt.tight_layout()
                
                fname = f"chart_{uuid.uuid4().hex[:8]}.png"
                path = os.path.join(save_dir, fname)
                plt.savefig(path)
                plt.close()
                
                img_paths.append(f"/results/charts/{fname}")
                print(f"      Generated: {title}")

            except Exception as e:
                print(f"      Failed to plot chart {i}: {e}")
                plt.close()

        state['chart_images'] = img_paths
        
        # We DO NOT zip here. Zipping happens on demand in app/routes.py.
        return state