# randomforest_model.py
import joblib
def load_model():
    # Load and return the trained RandomForestClassifier model
    model = joblib.load('models/best_sarcasm_model.joblib') 
    return model
def load_vectorizer():
    # Load and return the trained RandomForestClassifier model
    vectorizer = joblib.load('models/vectorizer_v5.joblib') 
    return vectorizer
def load_encoder():
    # Load and return the trained RandomForestClassifier model
    encoder = joblib.load('models/encoder_test_v5.joblib') 
    return encoder
