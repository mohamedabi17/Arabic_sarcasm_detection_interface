# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# Load the training data
df = pd.read_csv('training_data.csv')

# Load the testing data
df_test = pd.read_csv('testing_data.csv')

# Clean the 'tweet' column
df['tweet'] = df['tweet'].str.strip('"')

# Extract labels and texts
labels, texts = df["sarcasm"], df["tweet"]
labels_test, texts_test = df_test["sarcasm"], df_test["tweet"]

# One-hot encode categorical columns ('sentiment' and 'dialect')
encoder = OneHotEncoder(sparse_output=False)
categorical_encoded = encoder.fit_transform(df[['sentiment', 'dialect']])
categorical_encoded_df = pd.DataFrame(categorical_encoded, columns=encoder.get_feature_names_out(['sentiment', 'dialect']))
X_encoded = pd.concat([df[['tweet']], categorical_encoded_df], axis=1)

# Vectorize text data using TF-IDF
vectorizer = TfidfVectorizer()
text_vectorized = vectorizer.fit_transform(X_encoded['tweet'])

# Combine text features and one-hot encoded features
X_final = pd.concat([pd.DataFrame(text_vectorized.toarray()), X_encoded.drop('tweet', axis=1)], axis=1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_final, labels, test_size=0.5, random_state=42)


X_train.columns = X_train.columns.astype(str)
X_test.columns = X_test.columns.astype(str)


# Initialize and train the RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Display the results
print(f"Model Accuracy: {accuracy:.2%}")
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)

# Save the trained model to a joblib file
joblib.dump(model, 'sarcasm_model.joblib')
