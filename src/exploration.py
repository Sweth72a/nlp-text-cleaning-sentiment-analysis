import pandas as pd

FILE = "data/processed/sentiment140_cleaned.csv"

print("=" * 70)
print("LOADING PROCESSED DATASET")
print("=" * 70)

df = pd.read_csv(FILE)

print("\nDataset loaded successfully!")

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("\nSentiment distribution:")
print(df["sentiment"].value_counts())

print("\nSentiment percentage:")
print(df["sentiment"].value_counts(normalize=True) * 100)

print("\nText length statistics:")
print(df["final_text"].str.len().describe())

print("\n" + "=" * 70)
print("EXPLORATION COMPLETED")
print("=" * 70)