# agents/agent_2_analysis.py

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from core.llm_services import llm_fast_api  # <-- This is now the Groq fast model
import pandas as pd

class AnalysisAgent: 
    
    def __init__(self):
        self.parser = JsonOutputParser()
        self.prompt = ChatPromptTemplate.from_template(
            """You are an expert data scientist. Analyze the user's problem statement and dataset columns.
            
            Respond ONLY with a valid JSON object.
            
            Problem Statement: "{problem}"
            Columns: {columns}

            Respond with a JSON object with the following keys:
            1. "problem_type": (string) "Classification" or "Regression".
            2. "target_variable": (string) The most likely target column from the list.
            3. "reasoning": (string) A brief explanation for your choices.
            
            {format_instructions}
            """
        )
        self.chain = self.prompt | llm_fast_api | self.parser
        
    def run(self, state):
        print(f"-> Agent 2: Analyzing problem statement with LLM: '{state['problem_description']}'")
        
        try:
            analysis = self.chain.invoke({
                "problem": state['problem_description'],
                "columns": state['raw_df'].columns.tolist(),
                "format_instructions": self.parser.get_format_instructions()
            })
            
            print(f"   LLM Analysis complete. Task: {analysis['problem_type']}, Target: {analysis['target_variable']}")
            state['analysis'] = analysis
            return state

        except Exception as e:
            print(f"   LLM analysis failed: {e}. Attempting heuristic fallback.")
            # Simple Heuristic Fallback
            df = state['raw_df']
            target = df.columns[-1]
            if pd.api.types.is_numeric_dtype(df[target]) and df[target].nunique() > 30:
                ptype = "Regression"
            else:
                ptype = "Classification"
            
            state['analysis'] = {
                "problem_type": ptype,
                "target_variable": target,
                "reasoning": "LLM failed, used heuristic fallback (last column)."
            }
            print(f"   Heuristic fallback complete. Task: {ptype}, Target: {target}")
            return state