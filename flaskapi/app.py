from flask import Flask, render_template, request, jsonify
from models.sarcasm_model_creator import load_model
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack
import joblib

app = Flask(__name__, template_folder='./templates')
sarcasm_model = load_model()  # Load the model outside the route function

@app.route('/')
def home():
    return render_template('app.html')  # Adjust the path based on your folder structure

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the request
        data = request.json

        # Preprocess the input data
        tweet = data['tweet'].strip('"')

        # Create a DataFrame with the input data
        input_data = pd.DataFrame({'tweet': [tweet]})

        # One-hot encode categorical columns ('sentiment' and 'dialect')
        encoder = OneHotEncoder(sparse_output=True)
        categorical_encoded = encoder.fit_transform(input_data[['sentiment', 'dialect']])
        X_encoded = pd.concat([pd.DataFrame(categorical_encoded.toarray()), input_data[['tweet']]], axis=1)

        # Vectorize the text data
        vectorizer = TfidfVectorizer()
        text_vectorized = vectorizer.transform(X_encoded['tweet'])

        # Combine vectorized text and one-hot encoded features
        X_final = hstack([text_vectorized, X_encoded.drop('tweet', axis=1)])

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