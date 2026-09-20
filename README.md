# NLP Text Cleaning & Sentiment Analysis

An end-to-end Natural Language Processing (NLP) project that demonstrates text cleaning using **Regular Expressions (Regex)**, traditional NLP preprocessing, **TF-IDF feature extraction**, and **Logistic Regression** for sentiment classification.

---

## 📌 Project Overview

Text data collected from social media often contains noise such as:

- URLs
- User mentions
- Hashtags
- Numbers
- HTML tags
- Special characters
- Repeated characters
- Extra whitespace
- Contractions
- Stopwords

This project builds a complete NLP pipeline to clean and preprocess tweet text and then classify the tweets as:

- **Negative**
- **Positive**

The project uses the **Sentiment140 dataset**, which contains approximately **1.6 million tweets**.

---

## 🎯 Objectives

The main objectives of this project are:

1. Understand real-world text data.
2. Clean noisy tweet data using Regex.
3. Apply NLP preprocessing techniques.
4. Preserve important sentiment words such as `not`, `no`, and `never`.
5. Convert text into numerical features using TF-IDF.
6. Train a Logistic Regression sentiment classifier.
7. Evaluate the machine-learning model.
8. Save the trained model and TF-IDF vectorizer.
9. Use the trained model to predict sentiment for new text.

---

# 🔄 NLP Pipeline

```text
                 Sentiment140 Dataset
                         │
                         ↓
                   Raw Tweet Text
                         │
                         ↓
                  Regex Text Cleaning
                         │
        ┌────────────────┼────────────────┐
        ↓                ↓                ↓
      URLs            Mentions         HTML
      Hashtags        Numbers          Special chars
      Repeated chars  Extra spaces     etc.
                         │
                         ↓
                NLP Preprocessing
                         │
                ┌────────┴────────┐
                ↓                 ↓
            Tokenization     Stopword Removal
                                  │
                                  ↓
                            Lemmatization
                                  │
                                  ↓
                         Processed Text
                                  │
                                  ↓
                              TF-IDF
                                  │
                                  ↓
                       Logistic Regression
                                  │
                                  ↓
                       Sentiment Prediction
                         /              \
                        ↓                ↓
                   Negative          Positive
