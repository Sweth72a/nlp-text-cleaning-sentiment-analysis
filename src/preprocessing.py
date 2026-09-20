import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# Download NLTK resources
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")


# English stopwords
stop_words = set(stopwords.words("english"))

# Keep important negative words for sentiment analysis
important_words = {
    "not",
    "no",
    "never",
    "neither",
    "nor"
}

stop_words = stop_words - important_words


# Lemmatizer
lemmatizer = WordNetLemmatizer()


def tokenize_text(text):
    return word_tokenize(text)


def remove_stopwords(tokens):
    return [
        word
        for word in tokens
        if word not in stop_words
    ]


def lemmatize_tokens(tokens):
    return [
        lemmatizer.lemmatize(word)
        for word in tokens
    ]


def preprocess_text(text):

    # Tokenization
    tokens = tokenize_text(text)

    # Stopword removal
    tokens = remove_stopwords(tokens)

    # Lemmatization
    tokens = lemmatize_tokens(tokens)

    # Convert tokens back to text
    return " ".join(tokens)