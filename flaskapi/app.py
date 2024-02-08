from flask import Flask, render_template, request, jsonify
from models.sarcasm_model_creator import load_model, load_vectorizer, load_encoder
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__, template_folder='./templates')

# Load the model during initialization
sarcasm_model = load_model()
encoder = load_encoder()
vectorizer = load_vectorizer()

@app.route('/')
def home():
    return render_template('app.html')  # Adjust the path based on your folder structure

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        tweet = data.get('tweet', '').strip('"')
        dialect = data.get('dialect', '')
        
        # Vectorize the text data using the pre-fitted vectorizer
        text_vectorized = vectorizer.transform([tweet])
  
        # One-hot encode the dialect using the pre-fitted encoder
        dialect_encoded = encoder.transform([[dialect]])
        
        # Combine vectorized text and encoded dialect
        X_final = hstack([text_vectorized, dialect_encoded])
    
        # Predict using the pre-trained model
        prediction = sarcasm_model.predict(X_final)
        
        # Print the prediction in the terminal
        print("Prediction (sarcasm value):", prediction[0])

        return jsonify({'prediction': int(prediction[0])})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)