import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

FEATURES_PATH = "wine_features.csv"
EMBEDDINGS_PATH = "wine_embeddings.npy"  # .npy is numpy's format for saving arrays to disk

# small + fast embedding model
MODEL_NAME = "all-MiniLM-L6-v2"

TOP_N = 3


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    #print(f"Loaded {len(df):,} wines")
    return df


def save_embeddings(embeddings: np.ndarray, path: str):
    np.save(path, embeddings)
    #print(f"Saved embeddings to {path}")


def build_embeddings(df: pd.DataFrame) -> np.ndarray:
    model = SentenceTransformer(MODEL_NAME)
  
    descriptions = df["description"].tolist() #pandas series to python strings

    embeddings = model.encode(descriptions, batch_size=64)

    #print(f"Embeddings dimensions: {embeddings.shape}")
   
    return embeddings


def load_embeddings(path: str) -> np.ndarray:
    return np.load(path)


def nni(embeddings: np.ndarray) -> NearestNeighbors:
    index = NearestNeighbors(n_neighbors=TOP_N + 1, metric="cosine")
    index.fit(embeddings)
    #print("Built nearest-neighbor search index")
    return index


def find_similar_wines(df: pd.DataFrame, index: NearestNeighbors, embeddings: np.ndarray, wine_title: str, top_n: int = TOP_N,) -> pd.DataFrame:
    matches = df.index[df["title"] == wine_title] #find wine within df where true 

    if len(matches) == 0:
        raise ValueError(f"No wine found with title: {wine_title}") # catch in case user inputs a wine i dont have

    row_position = matches[0] #call wine position

    #find the similars
    query_vector = embeddings[row_position].reshape(1, -1) 
    distances, neighbor_positions = index.kneighbors(query_vector, n_neighbors=TOP_N + 1)

    #pull out the actual nums
    distances = distances[0]
    neighbor_positions = neighbor_positions[0]
    results = []
    for distance, position in zip(distances, neighbor_positions):
        if position == row_position: #no self matches
            continue
        results.append((position, distance))

    results = results[:TOP_N]
    result_rows = []
    for position, distance in results:
        row = df.iloc[position]
        similarity_percent = (1 - distance) * 100
        result_rows.append({
            "title": row["title"],
            "variety": row["variety"],
            "similarity": round(similarity_percent, 1),
            "description": row["description"],
        })

    return pd.DataFrame(result_rows)


def embeddpath(df:pd.DataFrame) -> np.ndarray:
    if os.path.exists(EMBEDDINGS_PATH):
        return load_embeddings(EMBEDDINGS_PATH)
    embeddings = build_embeddings(df)
    save_embeddings(embeddings, EMBEDDINGS_PATH)
    return embeddings
