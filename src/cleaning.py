import re
import contractions


def clean_text(text):
    """
    Clean raw tweet text using Regex and basic NLP cleaning.
    """

    text = str(text)

    # 1. Lowercase
    text = text.lower()

    # 2. Expand contractions
    # don't -> do not
    # can't -> cannot
    text = contractions.fix(text)

    # 3. Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # 4. Remove @mentions
    text = re.sub(r"@\w+", "", text)

    # 5. Remove hashtag symbol but keep the word
    text = re.sub(r"#", "", text)

    # 6. Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # 7. Remove numbers
    text = re.sub(r"\d+", "", text)

    # 8. Normalize repeated characters
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)

    # 9. Remove special characters and punctuation
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # 10. Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text