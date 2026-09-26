
import pandas as pd
import numpy as np


RAW_PATH = "winemag-data-130k-v2.csv"   # Input Kaggle info
CLEAN_PATH = "wine_clean.csv"           # Output clean file
MIN_DESCRIPTION_WORDS = 8               # tasting notes shorter than this get dropped


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0) # first column is just numbers

    print(f"Loading in: {len(df):,} rows, {df.shape[0]} columns") #start row/col
    return df


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame: #removing all duplicate taste descriptions that are the EXACT same
    before = len(df)  # remember the row count so we can report how many we removed
    df = df.drop_duplicates(subset=["description"])

    print(f"Dropped {before - len(df):,} duplicate rows")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = ["description", "variety", "points"]
    before = len(df)

    df = df.dropna(subset=required_columns) #remove all rows that are missing values that are really needed
    print(f"Dropped {before - len(df):,} rows missing required fields {required_columns}") #check

    # fill other columns that are null but have the values i need 
    df["region_1"] = df["region_1"].fillna("Unknown")
    df["country"] = df["country"].fillna("Unknown")
    df["designation"] = df["designation"].fillna("")
    df["price"] = df["price"].fillna(df["price"].median()) #placeholder for price .. change later?

    return df


def standardize_text_fields(df: pd.DataFrame) -> pd.DataFrame:
    for column in ["variety", "country", "region_1", "winery"]: #clean up all white spaces, and duplicate names
        df[column] = df[column].astype(str).str.strip()

    # Name inconsistencies if i find any:
    variety_fixes = {
        "GewÃ¼rztraminer": "Gewürztraminer",
    }
    df["variety"] = df["variety"].replace(variety_fixes)
    
    return df


def filter_short_descriptions(df: pd.DataFrame) -> pd.DataFrame:
    # check word count under my min
    word_counts = df["description"].str.split().str.len()

    before = len(df)

    df = df[word_counts >= MIN_DESCRIPTION_WORDS]

    print(f"Dropped {before - len(df):,} rows with tasting notes under {MIN_DESCRIPTION_WORDS} words")
    return df


def add_derived_fields(df: pd.DataFrame) -> pd.DataFrame:
    # separating my wines into budget tiers
    df["price_tier"] = pd.cut(
        df["price"],
        bins=[0, 15, 30, 60, 100, np.inf],  # np.inf means "no upper limit"
        labels=["budget", "everyday", "a little extra", "once in a while", "splurge"],
    )
    df["price_tier"] = df["price_tier"].astype(str)
    return df


def main():
    df = load_data(RAW_PATH)
    df = drop_duplicates(df)
    df = handle_missing_values(df)
    df = standardize_text_fields(df)
    df = filter_short_descriptions(df)
    df = add_derived_fields(df)

   
    df = df.reset_index(drop=True) 

   
    df.to_csv(CLEAN_PATH, index=False) #new and also clean file :b

    print(f"\nSaved {len(df):,} cleaned rows to {CLEAN_PATH}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nSample:\n{df[['title', 'variety', 'description']].head(4)}")

if __name__ == "__main__":
    main()