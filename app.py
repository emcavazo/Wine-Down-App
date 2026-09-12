from flask import Flask, jsonify, request, render_template
import os

# import everything directly from your existing script
# (rename "winedown" to whatever your actual filename is, minus .py)
from build_similarity_engine import (
    load_data,
    embeddpath,
    nni,
    find_similar_wines,
    FEATURES_PATH,
)

app = Flask(__name__)

df = load_data(FEATURES_PATH)
embeddings = embeddpath(df)
index = nni(embeddings)


@app.route("/recommend")
def recommend():
    # pull the wine title from the query string
    wine_title = request.args.get("wine")

    if not wine_title:
        return jsonify({"error": "pass a wine title, e.g. ?wine=Some Wine Title"}), 400

    try:
        results_df = find_similar_wines(df, index, embeddings, wine_title)
    except ValueError as e: #no title in my data
        return jsonify({"error": str(e)}), 404

    results_list = results_df.to_dict(orient="records")

    return jsonify({
        "query": wine_title,
        "results": results_list
    })

@app.route("/search")
def search_titles():
    query = request.args.get("q", "")

    if not query:
        return jsonify({"results": []})

    matches = df[df["title"].str.contains(query, case=False, na=False)]
    titles = matches["title"].head(10).tolist()

    return jsonify({"results": titles})

@app.route("/")
def home():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)