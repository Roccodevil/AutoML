import matplotlib
matplotlib.use('Agg') # FIX: Use non-interactive backend
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
            suggest 3 useful charts to plot from these columns: {columns}.
            
            Respond ONLY with a JSON list of plot objects.
            Each object must have "chart_type" (e.g., 'hist', 'box', 'scatter', 'count', 'heatmap'),
            "x" (string, column name, or null),
            "y" (string, column name, or null),
            "title" (string).
            
            Example:
            [
                {{"chart_type": "hist", "x": "{target}", "y": null, "title": "Distribution of {target}"}},
                {{"chart_type": "scatter", "x": "feature_1", "y": "{target}", "title": "Feature 1 vs {target}"}},
                {{"chart_type": "heatmap", "x": null, "y": null, "title": "Correlation Heatmap"}}
            ]
            
            {format_instructions}
            """
        )
        self.chain = self.prompt | llm_fast_api | self.parser

    def run(self, state):
        print("-> Agent 4: Dynamically generating visualizations...")
        df = state['cleaned_df']
        analysis = state['analysis']
        target = analysis['target_variable']
        charts_dir = os.path.join(state['results_dir'], "charts")
        chart_images = []

        try:
            # Get numeric cols for heatmap/scatter
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            if target in numeric_cols: numeric_cols.remove(target)
            
            # --- Get chart suggestions from LLM ---
            chart_suggestions = self.chain.invoke({
                "problem_type": analysis['problem_type'],
                "target": target,
                "columns": df.columns.tolist(),
                "format_instructions": self.parser.get_format_instructions()
            })
            
            print(f"   LLM suggested {len(chart_suggestions)} charts.")

            for chart in chart_suggestions:
                try:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    chart_type = chart['chart_type']
                    x = chart.get('x')
                    y = chart.get('y')
                    
                    if chart_type == 'hist':
                        sns.histplot(data=df, x=x, kde=True, ax=ax)
                    elif chart_type == 'count':
                        sns.countplot(data=df, x=x, ax=ax, palette="viridis")
                    elif chart_type == 'box':
                        sns.boxplot(data=df, x=x, y=y, ax=ax)
                    elif chart_type == 'scatter':
                        sns.scatterplot(data=df, x=x, y=y, ax=ax)
                    elif chart_type == 'heatmap':
                        cols_to_plot = numeric_cols[:15] + [target] # Limit heatmap
                        sns.heatmap(df[cols_to_plot].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
                    else:
                        print(f"   Skipping unknown chart type: {chart_type}")
                        plt.close(fig)
                        continue
                        
                    ax.set_title(chart['title'])
                    
                    # Save to file
                    filename = f"{chart['title'].lower().replace(' ', '_')[:20]}.png"
                    path = os.path.join(charts_dir, filename)
                    fig.savefig(path, bbox_inches='tight')
                    
                    # Save to memory for GUI
                    img_buf = io.BytesIO()
                    fig.savefig(img_buf, format='png', bbox_inches='tight')
                    img_buf.seek(0)
                    chart_images.append(Image.open(img_buf))
                    plt.close(fig)
                
                except Exception as e:
                    print(f"   Failed to generate chart '{chart['title']}': {e}")
                    plt.close(fig)

            print(f"   Charts saved to '{charts_dir}'")
            state['chart_images'] = chart_images
        
        except Exception as e:
            print(f"   Visualization agent failed: {e}")
            state['chart_images'] = []
            
        return state