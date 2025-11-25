from sklearn.model_selection import train_test_split
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from core.llm_services import llm_fast_api
import pandas as pd

class StagingAgent: 
    
    def __init__(self):
        self.chain = (
            ChatPromptTemplate.from_template(
                """You are a data science expert. What is the best train-test split strategy for the following problem?
                Problem Type: {problem_type}
                Target Variable: {target}
                Target Unique Values: {nunique}
                Data shape: {shape}
                
                Respond with ONLY the name of the strategy.
                Options: 'Stratified', 'Standard'
                """
            )
            | llm_fast_api
            | StrOutputParser()
        )
        
    def run(self, state):
        print("-> Agent 6: Staging data and determining split strategy...")
        df = state['featured_df']
        target = state['analysis']['target_variable']
        problem_type = state['analysis']['problem_type']
        
        if target not in df.columns:
            print(f"   Target '{target}' not in featured_df. Using last column.")
            target = df.columns[-1]
            state['analysis']['target_variable'] = target
        
        X = df.drop(columns=[target])
        y = df[target]

        try:
            strategy_name = self.chain.invoke({
                "problem_type": problem_type,
                "target": target,
                "nunique": y.nunique(),
                "shape": df.shape,
            }).strip().replace("'", "").replace('"', '')
            
            print(f"   LLM suggested split strategy: {strategy_name}")

            if "Stratified" in strategy_name and problem_type == "Classification":
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            else:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        except Exception as e:
            print(f"   LLM split suggestion failed ({e}). Defaulting to standard train_test_split.")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print(f"   Data split complete. Train: {X_train.shape}, Test: {X_test.shape}")
        
        state['X_train'] = X_train
        state['X_test'] = X_test
        state['y_train'] = y_train
        state['y_test'] = y_test
        
        return state