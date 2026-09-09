import pandas as pd
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS  

#IGNORE_LIST = {"wine","flavors","aromas","palate","finish","acidity","tannins","drink","notes","blend","texture","bodied","offers","cabernet","shows","character","like","sauvignon","structure"}

WORD_PATTERN = re.compile(r"\b[a-z]+\b")
SMOOTHING = 1 
df = pd.read_csv("wine_clean.csv")

def build_global_exclusion_set(df, columns):
    exclusion_words = set()
    for col in columns:
        if col in df.columns:
            for value in df[col].dropna().unique():
                for word in str(value).lower().split():
                    exclusion_words.add(word)
    return exclusion_words

label_columns = ["country","variety", "region_1", "region_2","province","title" "winery", "designation"]
global_exclusions = build_global_exclusion_set(df, label_columns)
global_exclusions.update({"cab", "later","chardonnays","chards", "cabs","ports","zinny","crozes","rieslings","zins","zinfandels", "sauv","cabernets", "malbecs", "syrahs","doles","marasca","barolos","importers","spirits","cooking","culinary","percentages"})

def wordCount(df, col1="description",col2="variety",global_exclude = None):
    if global_exclude is None:
        global_exclude = set()
    
    word_counts = {}

    for index, row in df.iterrows():
        description = row[col1]
        variety = row[col2]
        
        if pd.isna(description):
            continue
   
        words = WORD_PATTERN.findall(description.lower())

        for w in words:
            if w in ENGLISH_STOP_WORDS or len(w) <= 2 or w in global_exclude:
                continue
            word_counts[w] = word_counts.get(w,0) + 1

    return word_counts




def runComparison(df,highV,lowV, global_exclude, label =""):
    high_group = df[df["variety"].isin(highV)]
    low_group = df[df["variety"].isin(lowV)]

    high_counts = wordCount(high_group,global_exclude = global_exclusions)
    low_counts = wordCount(low_group,global_exclude = global_exclusions)

    high_total = sum(high_counts.values())
    low_total = sum(low_counts.values())


    minCount = 20

    signal_words = []
   

    for word, count in high_counts.items():
        high_freq = (count + 1) / (high_total+ 1)
        low_freq = (low_counts.get(word, 0) + 1) / (low_total + 1)
        ratio = high_freq / low_freq
        signal_words.append((word, ratio, count))

    signal_words = [item for item in signal_words if item[2] >= minCount]
    signal_words.sort(key=lambda x: x[1], reverse=True)

    print(f"\n--- Top words for: {label} ---")
    for word, ratio, count in signal_words[:30]:
        print(f"{word}: {ratio:.2f}x (n={count})")
        

#print(df["variety"].value_counts().head(50))



#TANNINS high acid vs low acid filter
runComparison(
    df,
    highV=["Cabernet Sauvignon", "Syrah", "Malbec", "Nebbiolo","Petit Sirah","Petit Syrah", "Tempranillo"],
    lowV=["Pinot Noir", "Gamay", "Riesling", "Grenache", "Sauvignon Blanc"],
    global_exclude=global_exclusions,
    label="Tannins"
)

#ACIDITY high acid vs low acid filter
runComparison(
    df,
    highV=["Riesling", "Sauvignon Blanc", "Barbera", "Chenin Blanc","Barbera","Chenin Blanc"],
    lowV=["Viognier", "Gewürztraminer", "Marsanne", "Grenache"],
    global_exclude=global_exclusions,
    label="Acidity"
)

#BODY full body vs light body
runComparison(
    df,
    highV=["Cabernet Sauvignon", "Syrah", "Malbec", "Zinfandel","Tempranillo","Grenache","Chardonnay"],
    lowV=["Pinot Noir", "Gamay", "Pinot Grigio", "Riesling","Grüner Veltliner"],
    global_exclude=global_exclusions,
    label="Body"
)

#SWEETNESS sweet vs dry
runComparison(
    df,
    highV=["Muscat","Orange Muscat","Muscat Black","Gewürztraminer","Port"],
    lowV=["Sauvignon Blanc", "Cabernet Sauvignon", "Merlot","Sangiovese","Nebbiolo"],
    global_exclude=global_exclusions,
    label="Sweetness"
)
