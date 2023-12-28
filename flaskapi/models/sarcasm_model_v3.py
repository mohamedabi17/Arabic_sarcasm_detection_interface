# import numpy as np
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import cross_val_score
# from sklearn.metrics import classification_report
# from scipy.sparse import hstack
# import joblib

# # Load data
# df = pd.read_csv('training_data.csv')
# df_test = pd.read_csv('testing_data.csv')

# # Preprocess data
# df['tweet'] = df['tweet'].str.strip('"').dropna()

# # Extract labels and texts
# labels, texts = df["sarcasm"], df["tweet"]

# # One-hot encode categorical columns ('sentiment' and 'dialect')
# encoder = OneHotEncoder(sparse=True)  # Set sparse=True
# categorical_encoded = encoder.fit_transform(df[['sentiment', 'dialect']])
# X_encoded = hstack([categorical_encoded, TfidfVectorizer().fit_transform(texts)])

# # Train the model
# randomforest_classifier = RandomForestClassifier()
# randomforest_classifier.fit(X_encoded, labels)

# # Evaluate the model
# y_pred = randomforest_classifier.predict(X_encoded)
# classification_rep = classification_report(labels, y_pred)
# print("Classification Report:\n", classification_rep)

# # Save the trained model to a joblib file
# joblib.dump(randomforest_classifier, 'sarcasm_model_v3.joblib')
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from scipy.sparse import hstack
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# Load data
df = pd.read_csv('training_data.csv')
df_test = pd.read_csv('testing_data.csv')

# Preprocess data
df['tweet'] = df['tweet'].str.strip('"').dropna()

# Extract labels and texts
labels, texts = df["sarcasm"], df[["tweet", "dialect"]]


# One-hot encode categorical columns ('sentiment' and 'dialect')
encoder = OneHotEncoder(sparse_output=True)  # Use sparse_output instead of sparse
categorical_encoded = encoder.fit_transform(df[['dialect']])
joblib.dump(encoder, 'encoder.joblib')  # Save the encoder


vectorizer = TfidfVectorizer()
X_text = vectorizer.fit_transform(texts)
joblib.dump(vectorizer, 'vectorizer.joblib')
X_encoded = hstack([categorical_encoded, vectorizer.fit_transform(df["tweet"])])
print("X_final shape:", X_encoded.shape)
# Train the model
randomforest_classifier = RandomForestClassifier()
randomforest_classifier.fit(X_encoded, labels)

# Evaluate the model
y_pred = randomforest_classifier.predict(X_encoded)
classification_rep = classification_report(labels, y_pred)
print("Classification Report:\n", classification_rep)

# Evaluate the model
accuracy = accuracy_score(labels, y_pred)
conf_matrix = confusion_matrix(labels, y_pred)
class_report = classification_report(labels, y_pred)

# Display the results
print(f"Model Accuracy: {accuracy:.2%}")
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)


# Save the trained model to a joblib file
joblib.dump(randomforest_classifier, 'sarcasm_model_v4.joblib')
