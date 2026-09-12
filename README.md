# Wine Recommender

A data science project that recommends wines based on wine attributes such as description, variety, and points awarded by tasters. Search for a wine you like, and get back others with a similar taste profile — tannin, acidity, body, and sweetness — along with a similarity score.

## Motivation

I work as a server and wanted to build something that could help recommend 
wines to customers based on their preferences and in turn, give them a better scope of what they might enjoy next

## Status

🚧 Work in progress

- [x] Load and clean the dataset
- [x] Exploratory data analysis
- [X] Build recommendation logic
- [X] Evaluate results
- [In progess] Build a simple interface + Make it easier to read

## Data

Dataset: Kaggle Wine Reviews: https://www.kaggle.com/datasets/zynicide/wine-reviews

## Setup
  1. Clone the repo:
    git clone https://github.com/emcavazo/Wine-Down-App.git
    cd Wine-Down-App

  3. Install dependencies:
   pip install -r requirements.txt

  4. Make sure `wine_features.csv` is present in the project root.
   (First run will generate `wine_embeddings.npy` automatically — this will
   take a few minutes)

### Running with Docker 
  docker build -t wine-api .
  docker run -p 5000:5000 wine-api


## Usage
  1. Open http://localhost:5000 in your browser
  2. Start typing a wine title in the search box -> matching suggestions
   appear as you type.
  3. Click a suggestion to see that wine's details 
  4. The app returns a list of similar wines, ranked by similarity, along with each one's description and detailed flavor     profile.

##API endpoints
  GET /search?q=<text>	Returns up to 10 wine titles matching the query
  GET /recommend?wine=<title>	Returns wines most similar to the given title, ranked by tasting-note similarity

## Tech Stack
  Python — pandas, NumPy, scikit-learn, sentence-transformers
  Flask — REST API
  Docker — containerized deployment
  Vanilla JS — frontend
