import pandas as pd
import re  

CLEAN_PATH = "wine_clean.csv"
FEATURES_PATH = "wine_features.csv"


# Create a general overview of words used that could indicate attributes

TANNIN_KEYWORDS = {
    "high": ["tannic","abrasive","phenolic", "scratchy", "choppy", "grippy", "firm tannins","structured", "chewy", "astringent","tight","grabby","clench","raspy","tartaric","fur","furry","youthfully","staunch"],
    "medium": ["soft tannins", "smooth tannins", "supple","polished"],
    "low": ["silky", "light tannins","delicate", "soft","soft and easy","rounded"],
}

ACIDITY_KEYWORDS = {
    "high": ["crisp", "gooseberry","gooseberries","grassy","grassiness","heraceousness","feline","taut","nettle","asparagus","slender","spry","honed","pristined","linear","lanolin","brisk", "zesty", "tart", "bright acidity", "racy", "vibrant acidity", "sharp","bristling"],
    "medium": ["balanced acidity", "fresh"],
    "low": ["soft acidity", "low acid", "flabby", "round"],
}

BODY_KEYWORDS = {
    "full": ["full-bodied","rich","full bodied", "rich", "dense", "weighty", "heavy", "powerful","inky","malolactic","browned","opaque","saturated","soupy","lusty","saturated"],
    "medium": ["medium-bodied", "medium bodied", "mid-weight","supple","good body"],
    "light": ["light-bodied", "light bodied", "delicate", "thin", "airy"],
}

SWEETNESS_KEYWORDS = {
    "sweet": ["sweet", "honeyed", "dessert", "sugary", "off-dry","lychee","peachy","honeysuckle","marmalade","residual","toffee","jelly","apricots","raisins","peaches"],
    "dry": ["dry", "bone-dry", "crisp and dry"],
}

# create usual structures as a fallback to missed keywords
VARIETY_PROFILES = {
    "Cabernet Sauvignon": ("high", "medium", "full", "dry"),
    "Pinot Noir": ("low", "high", "light", "dry"),
    "Merlot": ("medium", "medium", "medium", "dry"),
    "Chardonnay": ("low", "medium", "medium", "dry"),
    "Sauvignon Blanc": ("low", "high", "light", "dry"),
    "Syrah": ("high", "medium", "full", "dry"),
    "Shiraz": ("high", "medium", "full", "dry"),
    "Zinfandel": ("medium", "medium", "full", "dry"),
    "Malbec": ("high", "medium", "full", "dry"),
    "Sangiovese": ("high", "high", "medium", "dry"),
    "Tempranillo": ("high", "high", "medium", "dry"),
    "Nebbiolo": ("high", "high", "full", "dry"),
    "Grenache": ("medium", "medium", "full", "dry"),
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
# extras
STYLE_FALLBACKS = {
    ("Champagne", ("medium", "high", "light", "dry")),
    ("Sparkling", ("medium", "high", "light", "dry")),
    ("Prosecco", ("low", "high", "light", "sweet")),
    ("Portuguese Red", ("medium", "medium", "medium", "dry")),
    ("Bordeaux-style red", ("high", "medium", "full", "dry")),
    ("Rhône-style red", ("medium", "medium", "full", "dry")),
    ("Red blend", ("medium", "medium", "full", "dry")),
    ("Bordeaux-style white", ("low", "high", "medium", "dry")),
    ("White blend", ("low", "medium", "medium", "dry")),
}

# create flavor  categories 
FLAVOR_KEYWORDS = {
    "fruity": ["cherry", "berry", "citrus", "apple", "peach", "plum", "raspberry", "tropical fruit", "pineapple"],
    "earthy": ["cocoa", "coffee", "earthy", "mineral", "forest floor", "mushroom", "leather", "tobacco"],
    "oaky": ["oak", "vanilla", "toast", "smoke", "smoky", "cedar", "butterscotch"],
    "spicy": ["pepper", "clove", "cinnamon", "spice", "spicy"],
    "floral": ["floral", "flower", "blossom", "violet", "rose petal", "jasmine","broom","rose","iris"],
    "herbal": ["herbal", "grassy", "green pepper", "sage", "eucalyptus"],
    "sulfurous": ["brimstone", "sulfur", "sulphur","struck match", "gunpowder"],
    "mineral": ["mineral", "flinty", "flint", "wet stone", "slate"]
}

#see how many times words are actually caught
def count_keyword_hits(text: str, keywords: list) -> int:
    count = 0
    for word in keywords:
        count = count + text.count(word)
    return count

#create a scoreboard that keeps track of how many times the wine is called a specific keyword and its associated label
def score_level(text: str, level_keywords: dict) -> str:
    text = text.lower()  # lowercase it all
    scores = {} # empty scoreboard
    for level in level_keywords: # go through each level name and interate through it's associated keywords
        words = level_keywords[level]
        hits = count_keyword_hits(text,words)
        scores[level] = hits

    best_level = None
    best_score = -1

    for level in scores:
        if scores[level] > best_score:
            best_score = scores[level]
            best_level = level

    if best_score == 0:
        return "unknown"

    return best_level

#return typical values of a grape for unknowns
def get_variety_default(variety: str, attribute_index: int) -> str:
    
    profile = VARIETY_PROFILES.get(variety)  
    if profile is None:
        return "unknown"
    return profile[attribute_index]

#function to tag all the flavors
def tag_flavors(text: str) -> str:
    text = text.lower()

    matched_flavors = []

    for flavor, words in FLAVOR_KEYWORDS.items():
        if count_keyword_hits(text, words) > 0:
            matched_flavors.append(flavor)

    if len(matched_flavors) == 0:
        return "unknown"
    
    # combine list into csv
    return ", ".join(matched_flavors)


def add_wine_features(df: pd.DataFrame) -> pd.DataFrame:
  
    tannin_list = []
    acidity_list = []
    body_list = []
    sweetness_list = []
    flavor_list = []

    for index, row in df.iterrows():
        text = row["description"]
        variety = row["variety"]

        tannin = score_level(text, TANNIN_KEYWORDS)
        if tannin == "unknown":
            tannin = get_variety_default(variety, 0)

        acidity = score_level(text, ACIDITY_KEYWORDS)
        if acidity == "unknown":
            acidity = get_variety_default(variety, 1)

        body = score_level(text, BODY_KEYWORDS)
        if body == "unknown":
            body = get_variety_default(variety, 2)

        sweetness = score_level(text, SWEETNESS_KEYWORDS)
        if sweetness == "unknown":
            sweetness = get_variety_default(variety, 3)

        flavors = tag_flavors(text)

        tannin_list.append(tannin)
        acidity_list.append(acidity)
        body_list.append(body)
        sweetness_list.append(sweetness)
        flavor_list.append(flavors)

    df["tannin_level"] = tannin_list
    df["acidity_level"] = acidity_list
    df["body_level"] = body_list
    df["sweetness_level"] = sweetness_list
    df["flavor_tags"] = flavor_list

    return df

#check check check
def report_coverage(df: pd.DataFrame):
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
    sample_columns = ["title", "tannin_level", "acidity_level", "body_level","sweetness_level","flavor_tags"]
    print(df[sample_columns].head(10).to_string(index=False))


if __name__ == "__main__":
    main()


