import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt
import os
import io
from PIL import Image
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from core.llm_services import llm_fast_api

class VizAgent: 
    def __init__(self):
        self.parser = JsonOutputParser()
        self.prompt = ChatPromptTemplate.from_template(
            """You are a data visualization expert. For a {problem_type} problem with target '{target}',
            suggest {num_charts} useful charts to plot from these columns: {columns}.
            
            User Request: {user_request}
            
            Respond ONLY with a JSON list of plot objects.
            Each object must have "chart_type" (e.g., 'hist', 'box', 'scatter', 'count', 'heatmap'),
            "x" (string, column name, or null),
            "y" (string, column name, or null),
            "title" (string).
            {format_instructions}"""
        )
        self.chain = self.prompt | llm_fast_api | self.parser

    def run(self, state):
        print("-> Agent 4: Dynamically generating visualizations...")
        df = state['cleaned_df']
        analysis = state['analysis']
        target = analysis['target_variable']
        charts_dir = os.path.join(state['results_dir'], "charts")
        chart_images = []

        # --- Read Configuration ---
        config = state.get('node_configs', {}).get('agent_4_viz', {})
        mode = config.get('mode', 'default')
        user_request = config.get('user_request', 'None') if mode == 'custom' else 'None'
        num_charts = 6 if mode == 'custom' else 3
        
        try:
            chart_suggestions = self.chain.invoke({
                "problem_type": analysis['problem_type'],
                "target": target,
                "columns": df.columns.tolist()[:50],
                "num_charts": num_charts,
                "user_request": user_request,
                "format_instructions": self.parser.get_format_instructions()
            })
            
            print(f"   LLM suggested {len(chart_suggestions)} charts.")

            for chart in chart_suggestions:
                try:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    chart_type = chart['chart_type']
                    x, y = chart.get('x'), chart.get('y')
                    
                    # Validating columns exist
                    if x and x not in df.columns: x = None
                    if y and y not in df.columns: y = None

                    if chart_type == 'hist' and x: sns.histplot(data=df, x=x, kde=True, ax=ax)
                    elif chart_type == 'count' and x: sns.countplot(data=df, x=x, ax=ax)
                    elif chart_type == 'box' and x and y: sns.boxplot(data=df, x=x, y=y, ax=ax)
                    elif chart_type == 'scatter' and x and y: sns.scatterplot(data=df, x=x, y=y, ax=ax)
                    elif chart_type == 'heatmap':
                        num_cols = df.select_dtypes(include=['number']).columns
                        sns.heatmap(df[num_cols].corr(), annot=True, fmt=".1f", ax=ax)

                    ax.set_title(chart['title'])
                    
                    filename = f"{chart['title'].lower().replace(' ', '_')[:20]}.png"
                    path = os.path.join(charts_dir, filename)
                    fig.savefig(path, bbox_inches='tight')
                    
                    img_buf = io.BytesIO()
                    fig.savefig(img_buf, format='png', bbox_inches='tight')
                    img_buf.seek(0)
                    chart_images.append(Image.open(img_buf))
                    plt.close(fig)
                except: plt.close(fig)

            state['chart_images'] = chart_images
        except Exception as e:
            print(f"   Visualization agent failed: {e}")
            state['chart_images'] = []
            
        return state