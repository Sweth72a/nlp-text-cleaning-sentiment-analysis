import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# 1. LOAD DATA
# ============================================================

FILE = "data/processed/sentiment140_cleaned.csv"

print("=" * 70)
print("LOADING PROCESSED DATASET")
print("=" * 70)

df = pd.read_csv(FILE)

print("\nOriginal shape:")
print(df.shape)


# ============================================================
# 2. REMOVE MISSING TEXT
# ============================================================

print("\nRemoving missing final_text values...")

df = df.dropna(subset=["final_text"])

# Remove completely empty text
df = df[df["final_text"].str.strip() != ""]

print("Shape after removing missing/empty text:")
print(df.shape)


# ============================================================
# 3. SELECT FEATURES AND TARGET
# ============================================================

X = df["final_text"]
y = df["sentiment"]

print("\nFeatures:")
print("X =", X.shape)

print("\nTarget:")
print("y =", y.shape)


# ============================================================
# 4. TRAIN-TEST SPLIT
# ============================================================

print("\n" + "=" * 70)
print("TRAIN-TEST SPLIT")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))


# ============================================================
# 5. TF-IDF
# ============================================================

print("\n" + "=" * 70)
print("TF-IDF VECTORIZATION")
print("=" * 70)

tfidf = TfidfVectorizer(
    max_features=100000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

X_train_tfidf = tfidf.fit_transform(X_train)

X_test_tfidf = tfidf.transform(X_test)

print("\nTF-IDF completed!")

print("Training matrix shape:", X_train_tfidf.shape)
print("Testing matrix shape :", X_test_tfidf.shape)


# ============================================================
# 6. LOGISTIC REGRESSION
# ============================================================

print("\n" + "=" * 70)
print("TRAINING LOGISTIC REGRESSION")
print("=" * 70)

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(X_train_tfidf, y_train)

print("\nModel training completed!")


# ============================================================
# 7. PREDICTION
# ============================================================

print("\nMaking predictions...")

y_pred = model.predict(X_test_tfidf)


# ============================================================
# 8. EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("MODEL EVALUATION")
print("=" * 70)

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:")
print(f"{accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Negative", "Positive"]
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# ============================================================
# 9. TEST CUSTOM TWEETS
# ============================================================

print("\n" + "=" * 70)
print("CUSTOM TWEET TEST")
print("=" * 70)

sample_tweets = [
    "I love this product",
    "This is amazing",
    "I hate this",
    "Worst experience ever",
    "The movie was okay"
]

sample_tfidf = tfidf.transform(sample_tweets)

predictions = model.predict(sample_tfidf)

for tweet, prediction in zip(sample_tweets, predictions):

    sentiment = "Positive" if prediction == 1 else "Negative"

    print("\nTweet:", tweet)
    print("Prediction:", sentiment)


# ============================================================
# 10. SAVE MODEL
# ============================================================

import os
import joblib

os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/sentiment_model.pkl")
joblib.dump(tfidf, "models/tfidf_vectorizer.pkl")

print("\n" + "=" * 70)
print("MODELS SAVED")
print("=" * 70)

print("\nmodels/sentiment_model.pkl")
print("models/tfidf_vectorizer.pkl")

print("\nTraining pipeline completed successfully!")