"""
STEP 3: Build the similarity engine (embeddings + nearest-neighbor search).

WHAT THIS SCRIPT DOES:
This is the heart of the "Spotify for wine" idea. Given any wine, we
want to find other wines that taste similar to it. To do that, we need
a way to turn each wine's tasting note (just text) into something we
can do MATH on — because "similarity" is fundamentally a math problem,
not a text problem.

THE APPROACH (embeddings):
An "embedding" is a list of numbers (a vector) that represents the
MEANING of a piece of text. Similar meanings end up as similar numbers.
For example, "bright cherry and red fruit flavors" and "notes of ripe
raspberry and cranberry" would produce vectors that are close together,
even though they don't share many exact words — because a good
embedding model understands they're describing similar things.

We use a pre-trained model (built and trained by other people, which we
just reuse) to generate these embeddings. This is much more powerful
than counting shared words, because it captures MEANING, not just
vocabulary overlap.

Once every wine has a vector, "find similar wines" becomes: "find the
vectors that are closest to this one" — a well-solved math problem
called NEAREST NEIGHBOR SEARCH.

BEFORE RUNNING THIS:
1. Run clean_wine_data.py, then engineer_features.py first — this
   script needs wine_features.csv to exist.
2. Install two new libraries:
   pip install sentence-transformers scikit-learn
   (sentence-transformers is the embedding model; scikit-learn gives us
   the nearest-neighbor search)

NOTE ON RUNTIME:
Generating embeddings for ~120,000 wines takes a few minutes on a
normal laptop (no fancy hardware needed). This is a one-time cost —
we'll save the results to disk so future runs are instant.

HOW TO RUN THIS SCRIPT:
python build_similarity_engine.py
"""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

FEATURES_PATH = "wine_features.csv"
EMBEDDINGS_PATH = "wine_embeddings.npy"  # .npy is numpy's format for saving arrays to disk

# This is a small, fast, well-regarded embedding model — a good default
# for a portfolio project. There are bigger/more accurate models, but
# this one runs fine on a normal laptop without a GPU.
MODEL_NAME = "all-MiniLM-L6-v2"

# How many similar wines to return when we demo the search below.
DEMO_TOP_N = 5


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} wines")
    return df


def build_embeddings(df: pd.DataFrame) -> np.ndarray:
    """
    Converts every wine's tasting note into a vector using the
    pre-trained embedding model. Returns a 2D numpy array where each
    ROW is one wine's vector.
    """
    print(f"Loading embedding model ({MODEL_NAME})...")
    model = SentenceTransformer(MODEL_NAME)

    print("Generating embeddings — this takes a few minutes, grab a coffee...")
    # .tolist() converts the pandas column into a plain Python list,
    # which is what the model's encode() function expects.
    descriptions = df["description"].tolist()

    # show_progress_bar=True gives you visual feedback so you're not
    # staring at a frozen terminal wondering if it crashed.
    embeddings = model.encode(descriptions, show_progress_bar=True, batch_size=64)

    print(f"Done. Embeddings shape: {embeddings.shape}")
    # "shape" for a 2D array is (number_of_rows, number_of_columns) —
    # here that's (number_of_wines, vector_length). Each wine's vector
    # will have the same length (384 numbers, for this particular model).
    return embeddings


def save_embeddings(embeddings: np.ndarray, path: str):
    """
    Saves the embeddings to disk so we don't have to regenerate them
    (a multi-minute process) every single time we work on the project.
    """
    np.save(path, embeddings)
    print(f"Saved embeddings to {path}")


def load_embeddings(path: str) -> np.ndarray:
    return np.load(path)


def build_nearest_neighbors_index(embeddings: np.ndarray) -> NearestNeighbors:
    """
    Builds a searchable index over all the embeddings, so that later we
    can efficiently ask "which wines are closest to THIS one?" without
    manually comparing against all 120,000 others by hand every time.

    metric="cosine" means we measure similarity by the ANGLE between two
    vectors rather than their raw distance — this is the standard choice
    for text embeddings, because it cares about DIRECTION (meaning)
    rather than magnitude (which can vary for reasons unrelated to
    meaning, like sentence length).
    """
    # n_neighbors here is just a default; we can ask for a different
    # number of neighbors at search time, so this value doesn't lock
    # us in to anything.
    index = NearestNeighbors(n_neighbors=DEMO_TOP_N + 1, metric="cosine")
    index.fit(embeddings)
    print("Built nearest-neighbor search index")
    return index


def find_similar_wines(
    df: pd.DataFrame,
    index: NearestNeighbors,
    embeddings: np.ndarray,
    wine_title: str,
    top_n: int = DEMO_TOP_N,
) -> pd.DataFrame:
    """
    Given the exact title of a wine already in our dataset, returns the
    top_n most similar wines (excluding the wine itself).
    """
    # Find the row number(s) matching this title. df.index[...] gives us
    # the positions where the condition (df["title"] == wine_title) is True.
    matches = df.index[df["title"] == wine_title]

    if len(matches) == 0:
        raise ValueError(f"No wine found with title: {wine_title}")

    row_position = matches[0]  # if there are duplicates, just use the first

    # We look up this wine's own embedding vector, then ask the index
    # for its nearest neighbors. [row_position] wraps it in a list
    # because the search expects a batch of query vectors, even if
    # we're only searching for one.
    query_vector = embeddings[row_position].reshape(1, -1)

    # kneighbors returns two things: the distances, and the row
    # positions of the neighbors found. We ask for top_n + 1 because
    # the wine's own vector will always be its own closest match
    # (distance 0), and we want to exclude that.
    distances, neighbor_positions = index.kneighbors(query_vector, n_neighbors=top_n + 1)

    # distances and neighbor_positions come back as 2D arrays (one row
    # per query — we only had one query, so we take row [0]).
    distances = distances[0]
    neighbor_positions = neighbor_positions[0]

    # Drop the first result if it's the wine itself.
    results = []
    for distance, position in zip(distances, neighbor_positions):
        if position == row_position:
            continue
        results.append((position, distance))

    results = results[:top_n]  # just in case we still have one extra

    # Build a small DataFrame of the results to return, including a
    # similarity score. Cosine DISTANCE is 0 (identical) to 2 (opposite),
    # so we convert it to a more intuitive 0-100% similarity score.
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


def main():
    df = load_data(FEATURES_PATH)

    # Generate embeddings once, then reuse them on future runs instead
    # of waiting several minutes every time.
    import os
    if os.path.exists(EMBEDDINGS_PATH):
        print(f"Found existing embeddings at {EMBEDDINGS_PATH}, loading instead of regenerating.")
        embeddings = load_embeddings(EMBEDDINGS_PATH)
    else:
        embeddings = build_embeddings(df)
        save_embeddings(embeddings, EMBEDDINGS_PATH)

    index = build_nearest_neighbors_index(embeddings)

    # --- Demo: show similar wines for one example wine ---
    demo_title = df["title"].iloc[0]  # just use the first wine in the dataset as an example
    print(f"\nFinding wines similar to: {demo_title}")
    similar = find_similar_wines(df, index, embeddings, demo_title)
    print(similar[["title", "variety", "similarity"]].to_string(index=False))


if __name__ == "__main__":
    main()
