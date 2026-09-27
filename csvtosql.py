import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv


DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set")

engine = create_engine(DATABASE_URL) 

df = pd.read_csv("wine_features.csv")

df = df.astype(object).where(df.notna(), None)

#adding an index :b
df = df.reset_index(drop=False).rename(columns={"index": "id"})

df.to_sql(
    "wines",
    engine,
    if_exists = "replace", ##if already exists drop and replace
    index = False,
    chunksize = 1000,
    method = "multi"
)

with engine.connect() as conn:
    count = conn.execute(text("SELECT COUNT(*) FROM wines")).scalar()

print(f"Load {count} rows (expected {len(df)})")
