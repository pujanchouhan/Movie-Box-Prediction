"""
save_model.py — Train the best model and save artifacts for the web app.
Saves: model, scaler, feature metadata, and label encoder mappings.
"""

import os, json, warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)
os.makedirs("model_artifacts", exist_ok=True)

# ── Load & preprocess ──
df = pd.read_csv("data/movies.csv")
for col in ["budget", "runtime", "IMDB_rating", "number_of_votes"]:
    if df[col].isnull().any():
        df[col].fillna(df[col].median(), inplace=True)
df.drop_duplicates(subset=["movie_id"], keep="first", inplace=True)
df.drop(columns=["movie_id", "title"], inplace=True, errors="ignore")

# ── Feature engineering ──
df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
df["release_month"] = df["release_date"].dt.month
df["release_year"] = df["release_date"].dt.year

season_map = {12: "Winter", 1: "Winter", 2: "Winter",
              3: "Spring", 4: "Spring", 5: "Spring",
              6: "Summer", 7: "Summer", 8: "Summer",
              9: "Fall", 10: "Fall", 11: "Fall"}
df["release_season"] = df["release_month"].map(season_map)

TOP_CAST = [
    "Leonardo DiCaprio", "Meryl Streep", "Tom Hanks",
    "Scarlett Johansson", "Robert Downey Jr.", "Cate Blanchett",
    "Denzel Washington", "Margot Robbie", "Brad Pitt",
    "Viola Davis", "Dwayne Johnson", "Jennifer Lawrence",
    "Christian Bale", "Emma Stone", "Morgan Freeman",
    "Natalie Portman", "Will Smith", "Saoirse Ronan",
    "Ryan Gosling", "Lupita Nyong'o",
]
TOP_DIRECTORS = [
    "Steven Spielberg", "Christopher Nolan", "Martin Scorsese",
    "James Cameron", "Ridley Scott", "Denis Villeneuve",
    "Quentin Tarantino", "Greta Gerwig", "Jordan Peele", "David Fincher",
]

df["cast_popularity"] = df["cast"].apply(
    lambda c: sum(1 for n in str(c).split(",") if n.strip() in TOP_CAST))
df["director_is_top"] = df["director"].apply(lambda d: int(d in TOP_DIRECTORS))

broad = {
    "Action": "Action/Adventure", "Adventure": "Action/Adventure",
    "Animation": "Family/Animation", "Family": "Family/Animation",
    "Comedy": "Comedy/Romance", "Romance": "Comedy/Romance",
    "Drama": "Drama/Biography", "Biography": "Drama/Biography",
    "Horror": "Horror/Thriller", "Thriller": "Horror/Thriller",
    "Mystery": "Horror/Thriller", "Crime": "Crime/War",
    "War": "Crime/War", "Sci-Fi": "Sci-Fi/Fantasy",
    "Fantasy": "Sci-Fi/Fantasy", "Documentary": "Other",
    "Western": "Other", "Musical": "Other",
}
df["genre_group"] = df["genre"].map(broad).fillna("Other")

df["budget_x_rating"] = df["budget"] * df["IMDB_rating"]
df["budget_per_minute"] = df["budget"] / df["runtime"].replace(0, np.nan)
df["votes_per_rating"] = df["number_of_votes"] / df["IMDB_rating"].replace(0, np.nan)
df["budget_per_minute"].fillna(df["budget_per_minute"].median(), inplace=True)
df["votes_per_rating"].fillna(df["votes_per_rating"].median(), inplace=True)
df["log_budget"] = np.log1p(df["budget"])
df["is_english"] = (df["language"] == "English").astype(int)
df["is_summer_release"] = df["release_month"].isin([5, 6, 7]).astype(int)
df["is_holiday_release"] = df["release_month"].isin([11, 12]).astype(int)

df.drop(columns=["cast", "release_date"], inplace=True)

# ── Encode categoricals ──
cat_cols = ["genre", "director", "production_company",
            "language", "country", "release_season", "genre_group"]
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# ── Train/test split ──
TARGET = "box_office_revenue"
log_rev = np.log1p(df[TARGET])
X = df.drop(columns=[TARGET])
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)
feature_names = list(X.columns)

# ── Train best model (tuned HistGradientBoosting) ──
model = HistGradientBoostingRegressor(
    learning_rate=0.05, max_depth=4, max_iter=200,
    min_samples_leaf=20, random_state=SEED,
)
model.fit(X_train_sc, y_train)

y_pred = np.maximum(model.predict(X_test_sc), 0)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"  Model R² = {r2:.4f}, MAE = ${mae:,.0f}")

# ── Compute feature medians (for filling defaults in web app) ──
medians = X.median().to_dict()

# ── Save label encoder mappings as plain dicts ──
le_mappings = {}
for col, le in label_encoders.items():
    le_mappings[col] = {cls: int(idx) for idx, cls in enumerate(le.classes_)}

# ── Save everything ──
joblib.dump(model, "model_artifacts/model.pkl")
joblib.dump(scaler, "model_artifacts/scaler.pkl")

metadata = {
    "feature_names": feature_names,
    "medians": {k: float(v) for k, v in medians.items()},
    "label_encoders": le_mappings,
    "genre_group_map": broad,
    "r2": round(r2, 4),
    "mae": round(mae, 0),
    "top_cast": TOP_CAST,
    "top_directors": TOP_DIRECTORS,
}
with open("model_artifacts/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("  ✅ Saved model_artifacts/model.pkl")
print("  ✅ Saved model_artifacts/scaler.pkl")
print("  ✅ Saved model_artifacts/metadata.json")
