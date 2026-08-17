import pandas as pd
import numpy as np

df = pd.read_csv('IMDB Dataset.csv')  
print(df.shape)
print(df.head())

import re
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'<.*?>', ' ', text)          # remove HTML tags like <br />
    text = re.sub(r'[^a-zA-Z]', ' ', text)       # keep only letters
    text = text.lower()                          # lowercase everything
    words = text.split()
    words = [w for w in words if w not in stop_words]  # remove stopwords
    return ' '.join(words)

df['cleaned_review'] = df['review'].apply(clean_text)

print(df[['review', 'cleaned_review']].head())

from sklearn.model_selection import train_test_split

# Convert sentiment labels to numbers: positive=1, negative=0
df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# Split data: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    df['cleaned_review'], 
    df['label'], 
    test_size=0.2, 
    random_state=42
)

print("Train size:", X_train.shape[0])
print("Test size:", X_test.shape[0])

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

tfidf = TfidfVectorizer(max_features=5000)            #tdidf vectorization -- converts words into numeric features 
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

model = LogisticRegression(max_iter=1000)           # for spam we usess naive bayes and now we uses LR 
model.fit(X_train_tfidf, y_train)

print("Model trained successfully!")

from sklearn.metrics import accuracy_score, classification_report

# Predict on test data
y_pred = model.predict(X_test_tfidf)

# Check accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['negative', 'positive'])) 

def predict_sentiment(review):
    cleaned = clean_text(review)
    vector = tfidf.transform([cleaned])
    prediction = model.predict(vector)[0]
    return "positive" if prediction == 1 else "negative"

# Try it out
test_review = "This movie was absolutely terrible, I wasted two hours of my life"
print(predict_sentiment(test_review))

test_review2 = "One of the best films I've ever seen, brilliant acting!"
print(predict_sentiment(test_review2))

import pickle

# Save model and vectorizer
with open('sentiment_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

print("Model and vectorizer saved!")