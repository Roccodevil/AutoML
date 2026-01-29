import matplotlib
matplotlib.use('Agg') # Force headless mode
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import uuid
import numpy as np
import difflib
import zipfile
import json
import re
from core.llm_services import llm_powerful_api
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class VizAgent:
    def __init__(self):
        # We use StrOutputParser now to get raw text and manually clean it (More robust than JsonOutputParser)
        self.strategy_prompt = ChatPromptTemplate.from_template(
            """You are a Lead Data Visualization Architect.
            
            DATA PROFILE:
            - Rows: {rows}, Columns: {cols}
            - Numerical Cols: {num_cols}
            - Categorical Cols: {cat_cols}
            
            USER COMMAND: "{user_request}"
            (CRITICAL: You MUST follow this command if provided. If the user asks for a specific chart, generate ONLY that.)
            
            PREFERRED TYPE: "{pref_type}"
            
            TASK:
            Return a JSON object with a key "charts" containing a list of chart definitions.
            
            FORMAT:
            {{
                "charts": [
                    {{
                        "type": "scatterplot",
                        "x": "ColumnName",
                        "y": "ColumnName",
                        "hue": "OptionalColumn", 
                        "title": "Chart Title"
                    }}
                ]
            }}
            
            VALID TYPES: 
            scatterplot, lineplot, histplot, kdeplot, barplot, countplot, boxplot, violinplot, heatmap, pairplot, jointplot, lmplot
            
            Return ONLY VALID JSON. No markdown formatting.
            """
        )
        self.llm_chain = self.strategy_prompt | llm_powerful_api | StrOutputParser()

    def _fuzzy_match_col(self, col_name, all_columns):
        if not col_name: return None
        if col_name in all_columns: return col_name
        # Find closest match
        matches = difflib.get_close_matches(col_name, all_columns, n=1, cutoff=0.4)
        return matches[0] if matches else None

    def _clean_and_parse_json(self, raw_text):
        """Robustly extracts JSON from LLM response, handling Markdown blocks."""
        try:
            # Remove Markdown fences
            text = raw_text.replace("```json", "").replace("```", "").strip()
            return json.loads(text)
        except:
            # Try finding the first { and last }
            try:
                start = text.find("{")
                end = text.rfind("}") + 1
                if start != -1 and end != -1:
                    return json.loads(text[start:end])
            except:
                pass
        return None

    def run(self, state):
        print("-> Agent 4: Visualization (Seaborn Ultra)...")
        
        # 1. Get Data
        df = state.get('current_data')
        if df is None: df = state.get('featured_df')
        if df is None: df = state.get('cleaned_df')
        if df is None: raise ValueError("No data found to visualize.")
        
        # 2. Setup
        save_dir = os.path.join(state['results_dir'], "charts")
        os.makedirs(save_dir, exist_ok=True)
        # Clean old charts
        for f in os.listdir(save_dir):
            if f.endswith('.png'): os.remove(os.path.join(save_dir, f))

        config = state.get('node_configs', {}).get('agent_4_viz', {})
        user_request = config.get('user_request', '')
        
        # 3. Downsample for Speed
        plot_df = df.sample(n=2000, random_state=42) if len(df) > 2000 else df.copy()
        
        # 4. Plan (LLM vs Heuristic)
        num_cols = plot_df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = plot_df.select_dtypes(exclude=np.number).columns.tolist()
        
        plan = None
        
        # A. Try LLM Planning
        try:
            print(f"   [Viz] Processing User Request: '{user_request}'")
            raw_response = self.llm_chain.invoke({
                "rows": len(df), "cols": len(df.columns),
                "num_cols": num_cols[:30], "cat_cols": cat_cols[:30],
                "user_request": user_request,
                "pref_type": config.get('pref_type', 'Auto')
            })
            
            plan = self._clean_and_parse_json(raw_response)
            if plan: print(f"   [Viz] LLM Plan: {len(plan.get('charts', []))} charts generated.")
            
        except Exception as e:
            print(f"   [Viz] LLM Planning Failed: {e}")

        # B. Fallback Heuristics (If LLM fails or returns empty)
        if not plan or not plan.get('charts'):
            print("   [Viz] Using Fallback Logic.")
            charts = []
            
            # Simple Keyword Matching from User Request
            req_lower = user_request.lower()
            
            if 'corr' in req_lower or 'heatmap' in req_lower:
                charts.append({"type": "heatmap", "title": "Correlation Heatmap"})
            elif 'pair' in req_lower:
                charts.append({"type": "pairplot", "hue": cat_cols[0] if cat_cols else None, "title": "Pair Plot"})
            elif 'scatter' in req_lower and len(num_cols) >= 2:
                charts.append({"type": "scatterplot", "x": num_cols[0], "y": num_cols[1], "title": f"Scatter: {num_cols[0]} vs {num_cols[1]}"})
            else:
                # Default Default
                for c in num_cols[:3]:
                    charts.append({"type": "histplot", "x": c, "title": f"Distribution of {c}"})
                if cat_cols:
                    charts.append({"type": "barplot", "x": cat_cols[0], "y": num_cols[0] if num_cols else None, "title": f"{cat_cols[0]} Counts"})
            
            plan = {"charts": charts}

        # 5. Render Engine
        img_paths = []
        sns.set_theme(style="whitegrid", context="notebook")
        
        for chart in plan.get('charts', []):
            try:
                plt.close('all')
                
                # Figure-Level vs Axes-Level Handling
                kind = chart.get('type', 'histplot').lower()
                
                # Determine columns
                x = self._fuzzy_match_col(chart.get('x'), plot_df.columns)
                y = self._fuzzy_match_col(chart.get('y'), plot_df.columns)
                hue = self._fuzzy_match_col(chart.get('hue'), plot_df.columns)
                
                # Logic Switch
                if kind == 'heatmap':
                    plt.figure(figsize=(10, 8))
                    corr = plot_df.select_dtypes(include=np.number).iloc[:, :15].corr()
                    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
                    plt.title("Correlation Matrix")
                    
                elif kind == 'pairplot':
                    # Pairplot creates its own figure
                    g = sns.pairplot(plot_df, hue=hue, vars=num_cols[:5])
                    
                elif kind == 'jointplot' and x and y:
                    g = sns.jointplot(data=plot_df, x=x, y=y, hue=hue, kind='reg')
                    
                elif kind == 'lmplot' and x and y:
                    g = sns.lmplot(data=plot_df, x=x, y=y, hue=hue)
                    
                elif hasattr(sns, kind):
                    # Standard Plots (Axes Level)
                    plt.figure(figsize=(10, 6))
                    k = {"data": plot_df}
                    if x: k['x'] = x
                    if y: k['y'] = y
                    if hue and plot_df[hue].nunique() < 10: k['hue'] = hue # Limit hue to low cardinality
                    
                    getattr(sns, kind)(**k)
                    
                    plt.title(chart.get('title', f"{kind} of {x}"))
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                else:
                    print(f"   [Viz] Unknown chart type: {kind}")
                    continue

                # Save
                fname = f"chart_{uuid.uuid4().hex[:6]}.png"
                save_path = os.path.join(save_dir, fname)
                
                # If we used a Figure-level plot (g), allow standard save to catch current figure
                try:
                    plt.savefig(save_path, bbox_inches='tight')
                except:
                    # Fallback for complex Seaborn grids
                    plt.gcf().savefig(save_path)
                    
                img_paths.append(f"/results/charts/{fname}")
                print(f"      Generated: {kind}")
                
            except Exception as e:
                print(f"      [Viz Error] Failed to render {chart.get('type')}: {e}")

        state['chart_images'] = img_paths
        
        # 6. Bundle for Download (Standalone Mode)
        if img_paths:
            zip_path = os.path.join(state['results_dir'], "charts_bundle.zip")
            with zipfile.ZipFile(zip_path, 'w') as zf:
                for img_url in img_paths:
                    fname = os.path.basename(img_url)
                    zf.write(os.path.join(save_dir, fname), fname)
            state['dl_charts'] = "/api/download/charts.zip"

        state['final_message'] = f"✅ Visualization Complete.\nGenerated {len(img_paths)} charts based on request: '{user_request}'"
        return state