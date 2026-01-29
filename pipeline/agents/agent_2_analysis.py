from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from core.llm_services import llm_fast_api
import pandas as pd

class AnalysisAgent: 
    def __init__(self):
        self.parser = JsonOutputParser()
        self.prompt = ChatPromptTemplate.from_template(
            """You are an expert Lead Data Scientist.
            
            PROBLEM CONTEXT:
            User Goal: "{problem}"
            
            DATASET PROFILE:
            Columns: {columns}
            Data Preview (First 3 rows):
            {head}
            
            TASK:
            1. Determine if this is a "Classification" or "Regression" problem.
            2. Identify the Target Variable (Label).
            3. Recommend a model algorithm.
            
            OUTPUT JSON RULES:
            - "problem_type": Must be "Classification" or "Regression".
            - "target_variable": Must be the exact column name from the list provided.
            - "reasoning_short": One sentence explanation.
            - "reasoning_detailed": Detailed explanation for the user.
            - "recommended_model": e.g., "XGBoost", "RandomForest", "LinearRegression".
            
            {format_instructions}
            """
        )
        self.chain = self.prompt | llm_fast_api | self.parser
        
    def run(self, state):
        print(f"-> Agent 2: Analyzing Data Structure...")
        
        # 1. DEPENDENCY CHECK (Previous Agent Validation)
        df = state.get('current_data')
        if df is None or df.empty:
            error_msg = "❌ Agent 2 Error: No data found. Please run Agent 1 (Data Acquisition) first."
            print(f"   {error_msg}")
            state['final_message'] = error_msg
            # We raise an error to stop the pipeline execution flow in the orchestrator
            raise ValueError(error_msg)
        
        # Heuristic Backup (Last column is usually target)
        heuristic_target = df.columns[-1]
        
        try:
            # 2. PREPARE CONTEXT
            # We convert head to string so LLM can see values (numbers vs strings)
            data_head = df.head(3).to_string(index=False)
            cols = df.columns.tolist()
            user_problem = state.get('problem_description', 'Predict the target variable')
            
            # 3. INVOKE LLM
            analysis = self.chain.invoke({
                "problem": user_problem,
                "columns": cols,
                "head": data_head,
                "format_instructions": self.parser.get_format_instructions()
            })
            
            # 4. VALIDATION (Anti-Hallucination)
            # Ensure the selected target actually exists in the dataframe
            if analysis['target_variable'] not in df.columns:
                print(f"   [Agent 2 Warning] Hallucinated target '{analysis['target_variable']}'. Reverting to heuristic.")
                analysis['target_variable'] = heuristic_target

            # Update State
            state['analysis'] = analysis
            print(f"   Decision: {analysis['problem_type']} -> Target: {analysis['target_variable']}")
            
            # 5. STANDALONE OUTPUT MESSAGE
            # This ensures the UI shows useful info if the user runs *only* this step.
            state['final_message'] = (
                f"✅ Analysis Complete.\n"
                f"• Problem Type: {analysis['problem_type']}\n"
                f"• Target Variable: {analysis['target_variable']}\n"
                f"• Recommended Model: {analysis.get('recommended_model', 'AutoML')}\n"
                f"• Reasoning: {analysis.get('reasoning_short', 'N/A')}"
            )

        except Exception as e:
            print(f"   [Agent 2 Error] LLM Analysis failed ({e}). Using Heuristic Fallback.")
            
            # Fallback Logic
            ptype = "Classification"
            # If target is numeric and has many unique values -> Regression
            if pd.api.types.is_numeric_dtype(df[heuristic_target]) and df[heuristic_target].nunique() > 20:
                ptype = "Regression"
                
            fallback_analysis = {
                "problem_type": ptype,
                "target_variable": heuristic_target,
                "reasoning_short": "Automatic fallback (LLM unavailable).",
                "reasoning_detailed": "The AI analysis service was unavailable, so the last column was selected as the target based on data type heuristics.",
                "recommended_model": "H2O AutoML"
            }
            state['analysis'] = fallback_analysis
            
            state['final_message'] = (
                f"⚠️ Analysis Fallback Used.\n"
                f"• Problem Type: {ptype}\n"
                f"• Target Variable: {heuristic_target}\n"
                f"• Note: LLM was unavailable, defaulted to last column."
            )
            
        return state