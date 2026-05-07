"""
Movie Box Office Prediction using Regression & Feature Engineering
===================================================================
A complete ML pipeline: load → clean → engineer features → train → evaluate → visualize.

Usage:
    python generate_dataset.py        # create data/movies.csv first
    python movie_box_office_prediction.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
)

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")
os.makedirs("outputs", exist_ok=True)
SEED = 42

# ──────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ──────────────────────────────────────────────────────────────────────
DATA_PATH = "data/movies.csv"
print("=" * 70)
print("  STEP 1 · Loading Dataset")
print("=" * 70)
df = pd.read_csv(DATA_PATH)
print(f"  Shape: {df.shape}")
print(f"  Columns: {list(df.columns)}\n")
print(df.head())

# ──────────────────────────────────────────────────────────────────────
# 2. DATA PREPROCESSING
# ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  STEP 2 · Data Preprocessing")
print("=" * 70)

# 2a. Missing values
print("\n--- Missing Values (before) ---")
print(df.isnull().sum()[df.isnull().sum() > 0])

num_cols = ["budget", "runtime", "IMDB_rating", "number_of_votes"]
for col in num_cols:
    if df[col].isnull().any():
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        print(f"  Filled '{col}' with median = {median_val}")

print("\n--- Missing Values (after) ---")
print(df.isnull().sum().sum(), "total missing values remaining")

# 2b. Duplicates
dup_count = df.duplicated(subset=["movie_id"]).sum()
print(f"\n  Duplicate movie_id rows: {dup_count}")
df.drop_duplicates(subset=["movie_id"], keep="first", inplace=True)
print(f"  Shape after dedup: {df.shape}")

# 2c. Drop helper columns
df.drop(columns=["movie_id", "title"], inplace=True, errors="ignore")

# ──────────────────────────────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  STEP 3 · Feature Engineering")
print("=" * 70)

# 3a. Release date → month, season, year
df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
df["release_month"] = df["release_date"].dt.month
df["release_year"] = df["release_date"].dt.year

season_map = {12: "Winter", 1: "Winter", 2: "Winter",
              3: "Spring", 4: "Spring", 5: "Spring",
              6: "Summer", 7: "Summer", 8: "Summer",
              9: "Fall", 10: "Fall", 11: "Fall"}
df["release_season"] = df["release_month"].map(season_map)
print("  ✓ Extracted release_month, release_year, release_season")

# 3b. Cast popularity score (number of "top-20" actors in the cast)
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
    "Quentin Tarantino", "Greta Gerwig", "Jordan Peele",
    "David Fincher",
]

def cast_popularity(cast_str):
    names = [n.strip() for n in str(cast_str).split(",")]
    return sum(1 for n in names if n in TOP_CAST)

df["cast_popularity"] = df["cast"].apply(cast_popularity)
df["director_is_top"] = df["director"].apply(lambda d: int(d in TOP_DIRECTORS))
print("  ✓ Created cast_popularity, director_is_top")

# 3c. Genre grouping
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
print("  ✓ Grouped genres into broader categories")

# 3d. Interaction features
df["budget_x_rating"] = df["budget"] * df["IMDB_rating"]
df["budget_per_minute"] = df["budget"] / df["runtime"].replace(0, np.nan)
df["votes_per_rating"] = df["number_of_votes"] / df["IMDB_rating"].replace(0, np.nan)
df["budget_per_minute"].fillna(df["budget_per_minute"].median(), inplace=True)
df["votes_per_rating"].fillna(df["votes_per_rating"].median(), inplace=True)
print("  ✓ Created interaction features")

# 3e. Log-transform budget and revenue (right-skewed)
df["log_budget"] = np.log1p(df["budget"])
df["log_revenue"] = np.log1p(df["box_office_revenue"])
print("  ✓ Applied log transformations")

# Drop raw text / date columns before modelling
df.drop(columns=["cast", "release_date"], inplace=True)

# ──────────────────────────────────────────────────────────────────────
# 4. ENCODING CATEGORICAL VARIABLES
# ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  STEP 4 · Encoding Categorical Variables")
print("=" * 70)

cat_cols = ["genre", "director", "production_company",
            "language", "country", "release_season", "genre_group"]
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le
    print(f"  Label-encoded '{col}' ({len(le.classes_)} classes)")

# ──────────────────────────────────────────────────────────────────────
# 5. EXPLORATORY DATA ANALYSIS (EDA) – saved to outputs/
# ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  STEP 5 · Exploratory Data Analysis")
print("=" * 70)

# 5a. Distribution of target
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(df["box_office_revenue"], bins=50, color="#6366f1", edgecolor="white")
axes[0].set_title("Box Office Revenue Distribution", fontsize=13)
axes[0].set_xlabel("Revenue ($)")
axes[1].hist(df["log_revenue"], bins=50, color="#14b8a6", edgecolor="white")
axes[1].set_title("Log-Transformed Revenue Distribution", fontsize=13)
axes[1].set_xlabel("log(1 + Revenue)")
plt.tight_layout()
plt.savefig("outputs/01_revenue_distribution.png", dpi=150)
plt.close()
print("  Saved → outputs/01_revenue_distribution.png")

# 5b. Correlation heatmap (top features)
numeric_df = df.select_dtypes(include=[np.number])
corr = numeric_df.corr()
top_corr = corr["box_office_revenue"].abs().sort_values(ascending=False).head(12).index
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr.loc[top_corr, top_corr], annot=True, fmt=".2f",
            cmap="coolwarm", center=0, ax=ax, linewidths=0.5)
ax.set_title("Correlation Heatmap – Top 12 Features", fontsize=14)
plt.tight_layout()
plt.savefig("outputs/02_correlation_heatmap.png", dpi=150)
plt.close()
print("  Saved → outputs/02_correlation_heatmap.png")

# 5c. Budget vs Revenue scatter
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(df["budget"], df["box_office_revenue"],
           alpha=0.35, s=12, c="#8b5cf6")
ax.set_xlabel("Budget ($)")
ax.set_ylabel("Box Office Revenue ($)")
ax.set_title("Budget vs. Box Office Revenue", fontsize=14)
plt.tight_layout()
plt.savefig("outputs/03_budget_vs_revenue.png", dpi=150)
plt.close()
print("  Saved → outputs/03_budget_vs_revenue.png")

# 5d. Box plot – revenue by season
fig, ax = plt.subplots(figsize=(8, 5))
season_order = sorted(df["release_season"].unique())
sns.boxplot(x="release_season", y="log_revenue", data=df,
            palette="Set2", ax=ax)
ax.set_title("Log Revenue by Release Season (encoded)", fontsize=14)
plt.tight_layout()
plt.savefig("outputs/04_revenue_by_season.png", dpi=150)
plt.close()
print("  Saved → outputs/04_revenue_by_season.png")

# ──────────────────────────────────────────────────────────────────────
# 6. PREPARE FEATURES & TRAIN/TEST SPLIT
# ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  STEP 6 · Train / Test Split")
print("=" * 70)

TARGET = "box_office_revenue"
DROP = ["box_office_revenue", "log_revenue"]  # keep log_budget as feature
X = df.drop(columns=DROP)
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED)

# Scale features
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

print(f"  X_train: {X_train.shape}  |  X_test: {X_test.shape}")
print(f"  Features used ({X.shape[1]}): {list(X.columns)}")

# ──────────────────────────────────────────────────────────────────────
# 7. MODEL BUILDING & EVALUATION
# ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  STEP 7 · Model Training & Evaluation")
print("=" * 70)

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression":  Ridge(alpha=1.0),
    "Lasso Regression":  Lasso(alpha=1000, max_iter=10000),
    "Decision Tree":     DecisionTreeRegressor(max_depth=12, random_state=SEED),
    "Random Forest":     RandomForestRegressor(
                             n_estimators=200, max_depth=15,
                             random_state=SEED, n_jobs=-1),
}

results = []
predictions = {}

for name, model in models.items():
    model.fit(X_train_sc, y_train)
    y_pred = model.predict(X_test_sc)
    y_pred = np.maximum(y_pred, 0)  # revenue can't be negative

    mae  = mean_absolute_error(y_test, y_pred)
    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_test, y_pred)

    results.append({"Model": name, "MAE": mae, "MSE": mse,
                     "RMSE": rmse, "R²": r2})
    predictions[name] = y_pred

    print(f"\n  {name}")
    print(f"    MAE  = {mae:,.0f}")
    print(f"    RMSE = {rmse:,.0f}")
    print(f"    R²   = {r2:.4f}")

results_df = pd.DataFrame(results).sort_values("R²", ascending=False)
results_df.to_csv("outputs/model_results.csv", index=False)
print("\n  Saved → outputs/model_results.csv")

# ──────────────────────────────────────────────────────────────────────
# 8. RESULTS VISUALISATION
# ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  STEP 8 · Visualising Results")
print("=" * 70)

# 8a. Model comparison bar chart
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

colors = ["#6366f1", "#ec4899", "#f59e0b", "#14b8a6", "#3b82f6"]
axes[0].barh(results_df["Model"], results_df["R²"], color=colors)
axes[0].set_xlabel("R² Score")
axes[0].set_title("Model Comparison – R² Score", fontsize=14)
axes[0].set_xlim(0, 1)
for i, v in enumerate(results_df["R²"]):
    axes[0].text(v + 0.01, i, f"{v:.4f}", va="center", fontsize=10)

axes[1].barh(results_df["Model"], results_df["RMSE"], color=colors)
axes[1].set_xlabel("RMSE ($)")
axes[1].set_title("Model Comparison – RMSE", fontsize=14)
for i, v in enumerate(results_df["RMSE"]):
    axes[1].text(v + v*0.01, i, f"{v:,.0f}", va="center", fontsize=10)

plt.tight_layout()
plt.savefig("outputs/05_model_comparison.png", dpi=150)
plt.close()
print("  Saved → outputs/05_model_comparison.png")

# 8b. Actual vs Predicted for best model
best_name = results_df.iloc[0]["Model"]
best_pred = predictions[best_name]

fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(y_test, best_pred, alpha=0.4, s=15, c="#8b5cf6")
lims = [0, max(y_test.max(), best_pred.max()) * 1.05]
ax.plot(lims, lims, "--", color="#ef4444", lw=2, label="Perfect prediction")
ax.set_xlabel("Actual Revenue ($)", fontsize=12)
ax.set_ylabel("Predicted Revenue ($)", fontsize=12)
ax.set_title(f"Actual vs Predicted – {best_name}", fontsize=14)
ax.legend()
plt.tight_layout()
plt.savefig("outputs/06_actual_vs_predicted.png", dpi=150)
plt.close()
print("  Saved → outputs/06_actual_vs_predicted.png")

# 8c. Feature importance (Random Forest)
rf_model = models["Random Forest"]
importances = rf_model.feature_importances_
feat_imp = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importances
}).sort_values("Importance", ascending=True).tail(15)

fig, ax = plt.subplots(figsize=(9, 7))
ax.barh(feat_imp["Feature"], feat_imp["Importance"], color="#14b8a6")
ax.set_title("Random Forest – Top 15 Feature Importances", fontsize=14)
ax.set_xlabel("Importance")
plt.tight_layout()
plt.savefig("outputs/07_feature_importance.png", dpi=150)
plt.close()
print("  Saved → outputs/07_feature_importance.png")

# 8d. Residual plot for best model
residuals = y_test.values - best_pred
fig, ax = plt.subplots(figsize=(9, 5))
ax.scatter(best_pred, residuals, alpha=0.35, s=12, c="#ec4899")
ax.axhline(0, color="#374151", lw=1.5, ls="--")
ax.set_xlabel("Predicted Revenue ($)")
ax.set_ylabel("Residual ($)")
ax.set_title(f"Residual Plot – {best_name}", fontsize=14)
plt.tight_layout()
plt.savefig("outputs/08_residual_plot.png", dpi=150)
plt.close()
print("  Saved → outputs/08_residual_plot.png")

# ──────────────────────────────────────────────────────────────────────
# 9. SUMMARY
# ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  SUMMARY")
print("=" * 70)
print(results_df.to_string(index=False))
print(f"\n  🏆 Best model: {best_name} (R² = {results_df.iloc[0]['R²']:.4f})")
print(f"\n  All charts saved in outputs/ directory.")
print("=" * 70)
