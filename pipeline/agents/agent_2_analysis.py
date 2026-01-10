from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from core.llm_services import llm_fast_api
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
            3. "reasoning_short": (string) A one-sentence explanation for your choices.
            4. "reasoning_detailed": (string) A multi-paragraph, clear explanation of why the problem is of this type and why the target variable was chosen. This will be shown to the user.
            5. "recommended_model": (string) A suitable and strong algorithm (e.g., "XGBoost", "H2O AutoML", "RandomForest").
            
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
            df = state['raw_df']
            target = df.columns[-1]
            ptype = "Classification"
            if pd.api.types.is_numeric_dtype(df[target]) and df[target].nunique() > 30:
                ptype = "Regression"
            
            state['analysis'] = {
                "problem_type": ptype,
                "target_variable": target,
                "reasoning_short": "LLM failed, used heuristic fallback.",
                "reasoning_detailed": f"The AI analysis failed, so a heuristic fallback was used. The problem was determined to be '{ptype}' and the target was assumed to be the last column, '{target}'.",
                "recommended_model": "H2O AutoML"
            }
            return state