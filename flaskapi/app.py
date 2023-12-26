from flask import Flask, render_template, request, jsonify
from models.sarcasm_model_creator import load_model
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack
import joblib
from sklearn.preprocessing import OneHotEncoder
app = Flask(__name__, template_folder='./templates')# Load the model outside the route function

@app.route('/')
def home():
    return render_template('app.html')  # Adjust the path based on your folder structure

# Load the vectorizer and model during initialization
vectorizer = TfidfVectorizer()  # You should fit this vectorizer during model training
sarcasm_model = load_model()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the request
        data = request.json

        # Preprocess the input data
        tweet = data.get('tweet', '').strip('"')
        dialect = data.get('dialect', '')  # Assuming 'dialect' is a key in the JSON payload

        # Create a DataFrame with the input data
        input_data = pd.DataFrame({'tweet': [tweet], 'dialect': [dialect]})

        # Vectorize the text data using the pre-fitted vectorizer
        text_vectorized = vectorizer.transform(input_data['tweet'])

        # One-hot encode the 'dialect' column
        dialect_encoded = encoder.transform(input_data[['dialect']])
        
        # Combine vectorized text and one-hot encoded features
        X_final = hstack([text_vectorized, dialect_encoded])

        # Make predictions
        prediction = sarcasm_model.predict(X_final)

        # Print the prediction in the terminal
        print("Prediction (sarcasm value):", prediction)

        # Return the prediction as JSON
        return jsonify({'prediction': int(prediction[0])})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)