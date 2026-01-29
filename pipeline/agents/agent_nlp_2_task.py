from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from core.llm_services import llm_powerful_api

class NLPTaskAgent:
    def __init__(self):
        self.chain = (
            ChatPromptTemplate.from_template(
                """User Request: "{req}".
                
                Determine the NLP task.
                - If the user explicitly asks for "Summary" or "Translation" -> Return "summary" or "translation".
                - If the user asks to "Train", "Classify", "Analyze Sentiment", or is vague -> Return "classification" (This triggers the full training pipeline).
                - If the user asks to "Clean" -> Return "preprocessing".
                
                Return JSON: {{"task": "task_name"}}"""
            ) | llm_powerful_api | JsonOutputParser()
        )

    def run(self, state):
        print("-> NLP Agent 2: Routing...")
        df = state['raw_df']
        text_col = state['text_column']
        req = state['problem_description']
        
        try:
            res = self.chain.invoke({"req": req})
            task = res.get('task', 'classification') # Default to Full Pipeline
        except: 
            print("   [Agent 2] LLM Router failed. Defaulting to Classification (Full Pipeline).")
            task = 'classification'
        
        state['task_type'] = task
        print(f"   Identified Task: {task}")

        # STOP CONDITION: Only stop for Zero-Shot tasks (Summary/Translation)
        if task in ['summary', 'translation', 'keywords']:
            print(f"   Executing Zero-Shot Task ({task})...")
            # Run simple LLM inference on first few rows
            sample_texts = df[text_col].head(3).tolist()
            results = []
            
            prompt_map = {
                'summary': "Summarize this text:",
                'translation': "Translate this text to English:",
                'keywords': "Extract 5 keywords:"
            }
            
            for text in sample_texts:
                try:
                    p = f"{prompt_map.get(task, 'Analyze:')}\n\n{text[:1500]}"
                    res = llm_powerful_api.invoke(p).content
                    results.append(res)
                except: results.append("Error processing text.")
            
            state['summary_result'] = "\n\n---\n\n".join(results)
            state['final_message'] = f"Task '{task}' Completed successfully via LLM."
            state['is_finished'] = True # <--- STOPS HERE
            
        else:
            # CONTINUE CONDITION: Classification, Preprocessing, Sentiment
            print("   Continuing to Full Pipeline (Cleaning -> Modeling -> Export)...")
            state['is_finished'] = False # <--- CONTINUES TO AGENT 3
            
        return state