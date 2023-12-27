# randomforest_model.py
import joblib
def load_model():
    # Load and return the trained RandomForestClassifier model
    model = joblib.load('models/sarcasm_model_v4.joblib') 
    return model
def load_vectorizer():
    # Load and return the trained RandomForestClassifier model
    vectorizer = joblib.load('models/vectorizer.joblib') 
    return vectorizer
def load_encoder():
    # Load and return the trained RandomForestClassifier model
    encoder = joblib.load('models/encoder.joblib') 
    return encoder
