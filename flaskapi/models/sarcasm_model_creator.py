# randomforest_model.py
import joblib
def load_model():
    # Load and return the trained RandomForestClassifier model
    model = joblib.load('models/sarcasm_model_v4.joblib') 
    return model
