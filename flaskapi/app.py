from flask import Flask, render_template, request, jsonify
from models.sarcasm_model_creator import load_model,load_vectorizer,load_encoder
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_classif

app = Flask(__name__, template_folder='./templates')

# Load the model during initialization
sarcasm_model = load_model()
encoder =load_encoder()
vectorizer=load_vectorizer()


# df_test = pd.read_csv('./models/testing_data.csv')

@app.route('/')
def home():
    return render_template('app.html')  # Adjust the path based on your folder structure

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
        text_vectorized = vectorizer.fit_transform(input_data['tweet'])
        # One-hot encode the 'dialect' column using the pre-fitted encoder # Use sparse_output instead of sparse
        categorical_encoded = encoder.fit_transform(input_data)
        
        # Combine vectorized text and one-hot encoded features
        X_final = hstack([text_vectorized, categorical_encoded])
        print(X_final) 

        print("Vectorizer shape:", vectorizer.get_feature_names_out().shape)
        print("Encoder shape:", encoder.get_feature_names_out().shape)
        print("X_final shape:", X_final.shape)
        # Make predictions
        y_test = [True]
        prediction = sarcasm_model.fit(X_final,y_test)
        prediction = sarcasm_model.predict(X_final)

        # Print the prediction in the terminal
        print("Prediction (sarcasm value):", int(prediction[0]))

        # Return the prediction as JSON
        return jsonify({'prediction': int(prediction[0])})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)