import pandas as pd

from src.cleaning import clean_text
from src.preprocessing import preprocess_text


# ============================================================
# FILE PATHS
# ============================================================

INPUT_FILE = "data/raw/training.1600000.processed.noemoticon.csv"

OUTPUT_FILE = "data/processed/sentiment140_cleaned.csv"


# ============================================================
# SENTIMENT140 COLUMN NAMES
# ============================================================

columns = [
    "sentiment",
    "id",
    "date",
    "query",
    "user",
    "text"
]


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading Sentiment140 dataset...")

df = pd.read_csv(
    INPUT_FILE,
    encoding="latin-1",
    names=columns
)

print("Dataset loaded successfully!")
print("Shape:", df.shape)


# ============================================================
# DISPLAY DATASET
# ============================================================

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 tweets:")
print(df[["sentiment", "text"]].head())


# ============================================================
# SENTIMENT CONVERSION
# ============================================================

# Sentiment140:
# 0 = Negative
# 4 = Positive

df["sentiment"] = df["sentiment"].map({
    0: 0,
    4: 1
})


# ============================================================
# REMOVE MISSING TEXT
# ============================================================

df = df.dropna(subset=["text"])


# ============================================================
# CLEAN TEXT
# ============================================================

print("\nCleaning text using Regex...")

df["clean_text"] = df["text"].apply(clean_text)

print("Text cleaning completed!")


# ============================================================
# NLP PREPROCESSING
# ============================================================

print("\nApplying NLP preprocessing...")

df["final_text"] = df["clean_text"].apply(
    preprocess_text
)

print("NLP preprocessing completed!")


# ============================================================
# SHOW BEFORE / AFTER
# ============================================================

print("\n" + "=" * 100)
print("ORIGINAL vs CLEANED vs FINAL")
print("=" * 100)

for i in range(5):

    print("\nORIGINAL:")
    print(df.iloc[i]["text"])

    print("\nCLEANED:")
    print(df.iloc[i]["clean_text"])

    print("\nFINAL:")
    print(df.iloc[i]["final_text"])

    print("-" * 100)


# ============================================================
# FINAL DATASET
# ============================================================

final_df = df[
    [
        "sentiment",
        "text",
        "clean_text",
        "final_text"
    ]
]


# ============================================================
# SAVE
# ============================================================

final_df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n" + "=" * 100)
print("PROCESS COMPLETED SUCCESSFULLY!")
print("=" * 100)

print("\nOutput file:")
print(OUTPUT_FILE)

print("\nFinal shape:")
print(final_df.shape)

print("\nFinal columns:")
print(final_df.columns.tolist())