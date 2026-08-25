"""
STEP 2: Engineer wine features from the tasting note text.

WHAT THIS SCRIPT DOES:
Our cleaned dataset has a text description for each wine (the tasting
note), but no structured info like "how tannic is this wine?" or "is
this wine light-bodied or full-bodied?". Those structured attributes
are what we'll need later for both the similarity engine and the dish-
pairing logic, so this script's job is to read each tasting note and
tag the wine with those attributes.

THE APPROACH (keyword scanning):
We build small lists of words that tend to signal each attribute, e.g.
"tannic", "grippy", and "structured" all suggest high tannin. Then, for
each wine, we count how many of those words show up in its tasting note
and use that to assign a level (low / medium / high). This is a simple
technique called "lexicon-based" or "keyword-based" tagging. It's not
perfect — it can miss subtlety a human taster would catch — but it's a
solid, explainable starting point, and it's the same idea used in a lot
of real sentiment-analysis tools.

FALLBACK (variety defaults):
Keyword scanning alone leaves a lot of wines "unknown," because
reviewers often imply structure without using our exact keywords. To
fill those gaps, we fall back to well-established typical profiles for
each grape variety (e.g. Cabernet Sauvignon is reliably high-tannin).
So the priority order for every wine is: (1) what the text actually
says, and only if that finds nothing, (2) what's typical for the
grape.

BEFORE RUNNING THIS:
Run clean_wine_data.py first — this script expects wine_clean.csv to exist.

HOW TO RUN THIS SCRIPT:
python engineer_features.py
"""

import pandas as pd
import re  # Python's library for pattern matching in text ("regular expressions")

CLEAN_PATH = "wine_clean.csv"
FEATURES_PATH = "wine_features.csv"


# --- Keyword lexicons ---
# Each of these dictionaries maps a "level" (like "high") to a list of
# words that suggest that level. We keep these as module-level constants
# (defined outside any function) since they're fixed reference data, not
# something that changes while the script runs.

TANNIN_KEYWORDS = {
    "high": ["tannic", "tannin", "grippy", "firm tannins", "structured", "chewy", "astringent"],
    "medium": ["soft tannins", "smooth tannins", "supple", "polished"],
    "low": ["silky", "light tannins", "delicate", "soft and easy"],
}

ACIDITY_KEYWORDS = {
    "high": ["crisp", "zesty", "tart", "bright acidity", "racy", "vibrant acidity", "sharp"],
    "medium": ["balanced acidity", "fresh"],
    "low": ["soft acidity", "low acid", "flabby", "round"],
}

BODY_KEYWORDS = {
    "full": ["full-bodied", "full bodied", "rich", "dense", "weighty", "heavy", "powerful"],
    "medium": ["medium-bodied", "medium bodied", "mid-weight"],
    "light": ["light-bodied", "light bodied", "delicate", "thin", "airy"],
}

SWEETNESS_KEYWORDS = {
    "sweet": ["sweet", "honeyed", "dessert", "sugary", "off-dry"],
    "dry": ["dry", "bone-dry", "crisp and dry"],
}

# FALLBACK: variety-based typical profiles.
# Keyword scanning misses a lot of wines because reviewers often imply
# structure ("built to age," "coats the palate") without ever using our
# exact keywords. But grape variety is basically never missing from the
# data, and winemaking has well-established defaults for how each
# variety typically tastes. So: if the text gives us no signal, we fall
# back to what's typical for that grape instead of leaving it unknown.
#
# Each entry is (tannin, acidity, body, sweetness). This covers the
# highest-volume varieties in the dataset — check your data with
# df["variety"].value_counts() to see if it's worth extending further.
VARIETY_PROFILES = {
    "Cabernet Sauvignon": ("high", "medium", "full", "dry"),
    "Pinot Noir": ("low", "high", "light", "dry"),
    "Merlot": ("medium", "medium", "medium", "dry"),
    "Chardonnay": ("low", "medium", "medium", "dry"),
    "Sauvignon Blanc": ("low", "high", "light", "dry"),
    "Riesling": ("low", "high", "light", "sweet"),
    "Syrah": ("high", "medium", "full", "dry"),
    "Shiraz": ("high", "medium", "full", "dry"),
    "Zinfandel": ("medium", "medium", "full", "dry"),
    "Malbec": ("high", "medium", "full", "dry"),
    "Sangiovese": ("high", "high", "medium", "dry"),
    "Tempranillo": ("high", "medium", "medium", "dry"),
    "Nebbiolo": ("high", "high", "full", "dry"),
    "Grenache": ("medium", "medium", "medium", "dry"),
    "Pinot Grigio": ("low", "high", "light", "dry"),
    "Pinot Gris": ("low", "high", "light", "dry"),
    "Gewürztraminer": ("low", "medium", "medium", "sweet"),
    "Chenin Blanc": ("low", "high", "light", "sweet"),
    "Viognier": ("low", "medium", "medium", "dry"),
    "Cabernet Franc": ("medium", "medium", "medium", "dry"),
    "Petite Sirah": ("high", "medium", "full", "dry"),
    "Rosé": ("low", "high", "light", "dry"),
    "Port": ("high", "low", "full", "sweet"),
}


# Flavor categories work a bit differently — a wine can have several at
# once (e.g. both "fruity" AND "oaky"), so we don't assign a single
# level. Instead, later in the script, we'll tag a wine with ALL the
# categories whose keywords appear in its description.
FLAVOR_KEYWORDS = {
    "fruity": ["cherry", "berry", "citrus", "apple", "peach", "plum", "raspberry", "tropical fruit"],
    "earthy": ["earthy", "mineral", "forest floor", "mushroom", "leather", "tobacco"],
    "oaky": ["oak", "vanilla", "toast", "smoky", "cedar", "butterscotch"],
    "spicy": ["pepper", "clove", "cinnamon", "spice", "spicy"],
    "floral": ["floral", "blossom", "violet", "rose petal", "jasmine"],
    "herbal": ["herbal", "grassy", "green pepper", "sage", "eucalyptus"],
}


def count_keyword_hits(text: str, keywords: list) -> int:
    """
    Counts how many times any keyword in the given list appears in the
    text. We use this as our basic "signal strength" measurement for
    each attribute.
    """
    # re.escape() makes sure special regex characters in our keywords
    # (unlikely here, but good practice) are treated as literal text.
    # We join all keywords into a single pattern with "|" (meaning "OR"),
    # so one search checks for all of them at once — much faster than
    # looping through each keyword individually.
    pattern = "|".join(re.escape(word) for word in keywords)
    matches = re.findall(pattern, text)
    return len(matches)


def score_level(text: str, level_keywords: dict) -> str:
    """
    Given a tasting note and a dictionary of {level: [keywords]}, this
    figures out which level has the most keyword matches and returns
    that level's name. If nothing matches at all, we return "unknown"
    rather than guessing.
    """
    text = text.lower()  # lowercase everything so "Tannic" and "tannic" both match

    # This builds a dictionary of {level: hit_count} using a "dict
    # comprehension" — a compact way to build a dictionary in one line.
    # It's equivalent to writing a for-loop that fills in a dict.
    scores = {level: count_keyword_hits(text, words) for level, words in level_keywords.items()}

    best_level = max(scores, key=scores.get)  # the level with the highest score

    if scores[best_level] == 0:
        return "unknown"
    return best_level


def get_variety_default(variety: str, attribute_index: int) -> str:
    """
    Looks up the typical value for one attribute of a given grape
    variety. attribute_index picks which position in the profile tuple
    we want: 0=tannin, 1=acidity, 2=body, 3=sweetness.

    Returns "unknown" if we don't have a profile for this variety —
    keyword scanning gave us nothing AND we have no domain-knowledge
    fallback, so "unknown" is the honest answer.
    """
    profile = VARIETY_PROFILES.get(variety)  # .get() returns None instead of erroring if not found
    if profile is None:
        return "unknown"
    return profile[attribute_index]


def tag_flavors(text: str) -> str:
    """
    Unlike the level-based attributes above, flavors aren't mutually
    exclusive — a wine can be both fruity and oaky. So instead of
    picking one winner, we collect every category that had at least
    one keyword match.
    """
    text = text.lower()
    matched_flavors = []

    for flavor, words in FLAVOR_KEYWORDS.items():
        if count_keyword_hits(text, words) > 0:
            matched_flavors.append(flavor)

    if not matched_flavors:
        return "unknown"

    # We store multiple flavors as a single comma-separated string,
    # since a plain CSV column can't hold a Python list directly.
    return ", ".join(matched_flavors)


def add_wine_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies all the tagging functions above to every row in the
    DataFrame, adding one new column per attribute.
    """
    # .apply() runs a function once for every value in a column. Here,
    # for each row's description, we call score_level() with the
    # tannin lexicon, and store the result in a brand new column.
    df["tannin_level"] = df["description"].apply(lambda text: score_level(text, TANNIN_KEYWORDS))
    df["acidity_level"] = df["description"].apply(lambda text: score_level(text, ACIDITY_KEYWORDS))
    df["body_level"] = df["description"].apply(lambda text: score_level(text, BODY_KEYWORDS))
    df["sweetness_level"] = df["description"].apply(lambda text: score_level(text, SWEETNESS_KEYWORDS))
    df["flavor_tags"] = df["description"].apply(tag_flavors)

    # Now fill in the gaps using the variety-based fallback. df.apply()
    # with axis=1 runs a function once per ROW instead of once per
    # column (axis=0, the default) — we need that here because the
    # fallback logic needs to look at two columns at once (the current
    # level AND the variety).
    #
    # Each lambda below says: "if we already found a real value from
    # the text, keep it — otherwise, look up the variety's typical
    # value instead."
    df["tannin_level"] = df.apply(
        lambda row: row["tannin_level"] if row["tannin_level"] != "unknown"
        else get_variety_default(row["variety"], 0),
        axis=1,
    )
    df["acidity_level"] = df.apply(
        lambda row: row["acidity_level"] if row["acidity_level"] != "unknown"
        else get_variety_default(row["variety"], 1),
        axis=1,
    )
    df["body_level"] = df.apply(
        lambda row: row["body_level"] if row["body_level"] != "unknown"
        else get_variety_default(row["variety"], 2),
        axis=1,
    )
    df["sweetness_level"] = df.apply(
        lambda row: row["sweetness_level"] if row["sweetness_level"] != "unknown"
        else get_variety_default(row["variety"], 3),
        axis=1,
    )

    return df


def report_coverage(df: pd.DataFrame):
    """
    Prints out what percentage of wines got a real (non-"unknown") tag
    for each attribute. This tells us how well our keyword lexicons are
    working — low coverage means we should add more keywords.
    """
    print("\nFeature coverage (percent of wines with a detected value):")
    for column in ["tannin_level", "acidity_level", "body_level", "sweetness_level", "flavor_tags"]:
        known = (df[column] != "unknown").sum()
        percent = known / len(df) * 100
        print(f"  {column}: {percent:.1f}%")


def main():
    df = pd.read_csv(CLEAN_PATH)
    print(f"Loaded {len(df):,} cleaned wines")

    df = add_wine_features(df)
    report_coverage(df)

    df.to_csv(FEATURES_PATH, index=False)
    print(f"\nSaved {len(df):,} wines with features to {FEATURES_PATH}")

    print("\nSample:")
    sample_columns = ["title", "tannin_level", "acidity_level", "body_level", "flavor_tags"]
    print(df[sample_columns].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
