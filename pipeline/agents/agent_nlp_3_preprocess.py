import pandas as pd
import re
import nltk
from nltk.stem import WordNetLemmatizer

class NLPPreprocessAgent:
    def run(self, state):
        print("-> NLP Agent 3: Cleaning...")
        
        # --- CRASH PROTECTION: Safe NLTK Loading ---
        try:
            nltk.data.find('corpora/stopwords')
            nltk.data.find('corpora/wordnet')
        except LookupError:
            print("   [Agent 3] Downloading NLTK data (stopwords/wordnet)...")
            try:
                nltk.download('stopwords', quiet=True)
                nltk.download('wordnet', quiet=True)
            except Exception as e:
                print(f"   [Agent 3 Warning] NLTK Download Failed: {e}. Proceeding without advanced stopwords.")

        from nltk.corpus import stopwords
        try:
            stop_words = set(stopwords.words('english'))
        except: 
            stop_words = set() # Fallback empty set

        df = state['raw_df'].copy()
        col = state['text_column']
        
        # Get Config
        config = state.get('node_configs', {}).get('agent_nlp_3_preprocess', {})
        do_lower = config.get('lowercase', True)
        do_remove_html = config.get('remove_html', True)
        do_stopwords = config.get('remove_stopwords', True)
        do_lemmatize = config.get('lemmatize', True)
        
        lemmatizer = WordNetLemmatizer()

        def clean(text):
            if not isinstance(text, str): return ""
            
            # 1. Lowercase
            if do_lower: text = text.lower()
            
            # 2. HTML & Punctuation
            if do_remove_html: text = re.sub(r'<.*?>', '', text)
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            
            words = text.split()
            
            # 3. Stopwords
            if do_stopwords and stop_words:
                words = [w for w in words if w not in stop_words]
            
            # 4. Lemmatization
            if do_lemmatize:
                try: words = [lemmatizer.lemmatize(w) for w in words]
                except: pass # Safety skip
                
            return " ".join(words)

        print(f"   Cleaning column: {col}")
        df['processed_text'] = df[col].apply(clean)
        state['processed_df'] = df
        
        # STOP LOGIC: Only stop if user explicitly said "Just clean it"
        if state['task_type'] == 'preprocessing':
            state['final_message'] = "Preprocessing Done."
            try: state['data_preview_html'] = df.head(5).to_html(classes='table')
            except: pass
            state['is_finished'] = True
        else:
            state['is_finished'] = False # Continue to Agent 4 (Model)
            
        return state