import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

class NLPModelAgent:
    def run(self, state):
        print("-> NLP Agent 4: Modeling...")
        df = state['processed_df']
        
        # Heuristic Label Finding (Shortest column with <20 unique values)
        label_col = None
        for c in df.columns:
            if c != 'processed_text' and c != state['text_column']:
                if df[c].nunique() < 20:
                    label_col = c
                    break
        
        if not label_col:
            raise ValueError("No label column found for classification. Please upload labeled data.")

        print(f"   Training on Label: {label_col}")
        
        X = df['processed_text']
        y = df[label_col]
        
        # Simple Pipeline
        model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=3000)),
            ('clf', LogisticRegression())
        ])
        
        model.fit(X, y)
        state['model'] = model
        state['final_message'] = f"Model Trained on '{label_col}'. Ready for Export."
        
        return state