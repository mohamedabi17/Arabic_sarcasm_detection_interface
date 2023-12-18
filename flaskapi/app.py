from flask import Flask, render_template, request, jsonify
from models.sarcasm_model_creator import load_model
import numpy as np
from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
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

        # Preprocess the input data (similar to what you did during training)
        # For example, if your input data has a 'tweet' field:
        tweet = data['tweet']
        # Perform any necessary preprocessing, such as stripping quotes
        tweet = tweet.strip('"')

        # Create a DataFrame with the input data (you might need to adapt this based on your actual input)
        input_data = pd.DataFrame({'tweet': [tweet]})

        # One-hot encode categorical columns ('sentiment' and 'dialect')
        encoder = OneHotEncoder(sparse_output=False)
        categorical_encoded = encoder.fit_transform(input_data[['sentiment', 'dialect']])
        categorical_encoded_df = pd.DataFrame(categorical_encoded, columns=encoder.get_feature_names_out(['sentiment', 'dialect']))
        X_encoded = pd.concat([input_data[['tweet']], categorical_encoded_df], axis=1)

        # Vectorize the text data
        vectorizer = TfidfVectorizer()
        text_vectorized = vectorizer.transform(X_encoded['tweet'])

        # Combine vectorized text and one-hot encoded features
        X_final = pd.concat([pd.DataFrame(text_vectorized.toarray()), X_encoded.drop('tweet', axis=1)], axis=1)

        # Make predictions
        prediction = sarcasm_model.predict(X_final)
        print("Prediction:", prediction)
        # Return the prediction as JSON
        return jsonify({'prediction': int(prediction[0])})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)