import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from scipy.sparse import hstack
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv('training_data.csv')
df_test = pd.read_csv('testing_data.csv')

# Preprocess data
df['tweet'] = df['tweet'].str.strip('"').dropna()
df_test['tweet'] = df_test['tweet'].str.strip('"').dropna()

# Extract labels and texts for training set
labels, texts = df["sarcasm"], df[["tweet", "dialect"]]

# Extract labels and texts for test set
labels2, texts2 = df_test["sarcasm"], df_test[["tweet", "dialect"]]


# %%

# One-hot encode categorical columns ('dialect') for training set
encoder = OneHotEncoder(sparse_output=True)
categorical_encoded = encoder.fit_transform(df[['dialect']])
joblib.dump(encoder, 'encoder.joblib')  # Save the encoder


# %%

# TF-IDF Vectorization for training set
vectorizer = TfidfVectorizer()
X_text = vectorizer.fit_transform(df["tweet"])
joblib.dump(vectorizer, 'vectorizer.joblib')
X_encoded = hstack([categorical_encoded, vectorizer.transform(df["tweet"])])


# %%
# One-hot encode categorical columns ('dialect') for test set using the same encoder as training set
categorical_encoded_test = encoder.transform(df_test[['dialect']])
joblib.dump(encoder, 'encoder_test.joblib')  # Save the encoder



# %%
# TF-IDF Vectorization for test set using the same vectorizer as training set
X_encoded_test = hstack([categorical_encoded_test, vectorizer.transform(df_test["tweet"])])


# %%


# Define the Random Forest classifier with class weights
randomforest_classifier = RandomForestClassifier(class_weight='balanced', random_state=42)


# %%

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
}

grid_search = GridSearchCV(randomforest_classifier, param_grid, cv=5, scoring='f1', n_jobs=-1)
grid_search.fit(X_encoded, labels)



# %%
# Get the best model from the grid search
best_randomforest_classifier = grid_search.best_estimator_



# %%

# Evaluate the best model on the test set
y_pred_test = best_randomforest_classifier.predict(X_encoded_test)
classification_rep_test = classification_report(labels2, y_pred_test)

# Evaluate the best model on the training set
y_pred_train = best_randomforest_classifier.predict(X_encoded)
classification_rep_train = classification_report(labels, y_pred_train)

# Display the results for the training set
print("Best Model Classification Report (Training Set):\n", classification_rep_train)

# Display the results for the test set
print("Best Model Classification Report (Test Set):\n", classification_rep_test)

# Save the best model to a joblib file
joblib.dump(best_randomforest_classifier, 'best_sarcasm_model.joblib')

# %%
accuracy = accuracy_score(labels2, y_pred_test)
print(accuracy)
accuracy2 = accuracy_score(labels, y_pred_train)
print(accuracy2)

# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%




# %%



