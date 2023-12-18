# %%
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %%
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB, ComplementNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# %%
df = pd.read_csv('/kaggle/input/arabic-sarcasm-detection/Dataset/ArSarcasm-v2-main/ArSarcasm-v2-main/ArSarcasm-v2/training_data.csv')
print(df)
df_test = pd.read_csv('/kaggle/input/arabic-sarcasm-detection/Dataset/ArSarcasm-v2-main/ArSarcasm-v2-main/ArSarcasm-v2/testing_data.csv')
print(df_test)

# %%
print(df.describe())

# %%
df.head(5)

# %%
df.info()

# %%
print(df.columns)

# %%
df['tweet'] = df['tweet'].str.strip('"')
print(df['tweet'])

# %%
labels, texts = df["sarcasm"], df["tweet"]
print(labels)
print(texts)

# %%
labels_test, texts_test = df_test["sarcasm"], df_test["tweet"]
print(labels_test)
print(texts_test)

# %%
# One-hot encode categorical columns ('sentiment' and 'dialect')
encoder = OneHotEncoder(sparse_output=False)  # Use `sparse_output` instead of `sparse`
categorical_encoded = encoder.fit_transform(df[['sentiment', 'dialect']])
categorical_encoded_df = pd.DataFrame(categorical_encoded, columns=encoder.get_feature_names_out(['sentiment', 'dialect']))
X_encoded = pd.concat([df[['tweet']], categorical_encoded_df], axis=1)

# %%
vectorizer = TfidfVectorizer()
text_vectorized = vectorizer.fit_transform(X_encoded['tweet'])


# %%
X_final = pd.concat([pd.DataFrame(text_vectorized.toarray()), X_encoded.drop('tweet', axis=1)], axis=1)

# %% [markdown]
# 

# %%
X_train, X_test, y_train, y_test = train_test_split(X_final, labels, test_size=0.2, random_state=42)

# %%
from sklearn.ensemble import RandomForestClassifier
X_train.columns = X_train.columns.astype(str)
X_test.columns = X_test.columns.astype(str)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# %%

# Prédictions sur l'ensemble de test
y_pred = model.predict(X_test)


# %%

# Évaluation du modèle
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

#Affichage des résultats
print(f"Précision du modèle: {accuracy:.2%}")
print("Matrice de confusion:\n", conf_matrix)


