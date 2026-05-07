"""
Movie Box Office Prediction v2 — Enhanced Pipeline
=====================================================
Adds: Gradient Boosting, HistGradientBoosting, 5-Fold Cross-Validation,
      GridSearchCV hyperparameter tuning, and exports a JSON
      summary consumed by the interactive web dashboard.

Usage:
    python3 generate_dataset.py                       # if not already done
    python3 movie_box_office_prediction_v2.py
"""

import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")
os.makedirs("outputs", exist_ok=True)
SEED = 42
np.random.seed(SEED)

# ──────────────────────────────────────────────────────────────────────
# 1. LOAD
# ──────────────────────────────────────────────────────────────────────
print("=" * 70)
print("  STEP 1 · Loading Dataset")
print("=" * 70)
df = pd.read_csv("data/movies.csv")
print(f"  Shape: {df.shape}\n")

# ──────────────────────────────────────────────────────────────────────
# 2. PREPROCESS
# ──────────────────────────────────────────────────────────────────────
print("=" * 70)
print("  STEP 2 · Preprocessing")
print("=" * 70)

print("\n--- Missing Values (before) ---")
missing = df.isnull().sum()
print(missing[missing > 0])

for col in ["budget", "runtime", "IMDB_rating", "number_of_votes"]:
    if df[col].isnull().any():
        med = df[col].median()
        df[col].fillna(med, inplace=True)
        print(f"  Filled '{col}' with median = {med}")

dup_count = df.duplicated(subset=["movie_id"]).sum()
print(f"\n  Duplicates removed: {dup_count}")
df.drop_duplicates(subset=["movie_id"], keep="first", inplace=True)
df.drop(columns=["movie_id", "title"], inplace=True, errors="ignore")
print(f"  Clean shape: {df.shape}")

# ──────────────────────────────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  STEP 3 · Feature Engineering")
print("=" * 70)

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
df["log_revenue"] = np.log1p(df["box_office_revenue"])
df["is_english"] = (df["language"] == "English").astype(int)
df["is_summer_release"] = df["release_month"].isin([5, 6, 7]).astype(int)
df["is_holiday_release"] = df["release_month"].isin([11, 12]).astype(int)

print("  ✓ 15 engineered features created")
df.drop(columns=["cast", "release_date"], inplace=True)

# ──────────────────────────────────────────────────────────────────────
# 4. ENCODE
# ──────────────────────────────────────────────────────────────────────
cat_cols = ["genre", "director", "production_company",
            "language", "country", "release_season", "genre_group"]
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

# ──────────────────────────────────────────────────────────────────────
# 5. EDA CHARTS
# ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  STEP 5 · EDA Charts")
print("=" * 70)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(df["box_office_revenue"], bins=50, color="#6366f1", edgecolor="white")
axes[0].set_title("Revenue Distribution"); axes[0].set_xlabel("Revenue ($)")
axes[1].hist(df["log_revenue"], bins=50, color="#14b8a6", edgecolor="white")
axes[1].set_title("Log Revenue Distribution"); axes[1].set_xlabel("log(1+Revenue)")
plt.tight_layout(); plt.savefig("outputs/01_revenue_distribution.png", dpi=150); plt.close()

numeric_df = df.select_dtypes(include=[np.number])
corr = numeric_df.corr()
top_corr = corr["box_office_revenue"].abs().sort_values(ascending=False).head(12).index
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr.loc[top_corr, top_corr], annot=True, fmt=".2f",
            cmap="coolwarm", center=0, ax=ax, linewidths=0.5)
ax.set_title("Correlation Heatmap – Top 12 Features")
plt.tight_layout(); plt.savefig("outputs/02_correlation_heatmap.png", dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(df["budget"], df["box_office_revenue"], alpha=0.35, s=12, c="#8b5cf6")
ax.set_xlabel("Budget ($)"); ax.set_ylabel("Revenue ($)")
ax.set_title("Budget vs Revenue")
plt.tight_layout(); plt.savefig("outputs/03_budget_vs_revenue.png", dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(x="release_season", y="log_revenue", data=df, palette="Set2", ax=ax)
ax.set_title("Log Revenue by Season")
plt.tight_layout(); plt.savefig("outputs/04_revenue_by_season.png", dpi=150); plt.close()
print("  Saved 4 EDA charts")

# ──────────────────────────────────────────────────────────────────────
# 6. SPLIT & SCALE
# ──────────────────────────────────────────────────────────────────────
TARGET = "box_office_revenue"
X = df.drop(columns=[TARGET, "log_revenue"])
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)
feature_names = list(X.columns)

print(f"\n  Train: {X_train.shape}  |  Test: {X_test.shape}")
print(f"  Features ({len(feature_names)}): {feature_names}")

# ──────────────────────────────────────────────────────────────────────
# 7. MODELS + 5-FOLD CROSS-VALIDATION
# ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  STEP 7 · Training with 5-Fold Cross-Validation")
print("=" * 70)

models = {
    "Linear Regression":    LinearRegression(),
    "Ridge Regression":     Ridge(alpha=1.0),
    "Lasso Regression":     Lasso(alpha=1000, max_iter=10000),
    "Decision Tree":        DecisionTreeRegressor(max_depth=12, random_state=SEED),
    "Random Forest":        RandomForestRegressor(n_estimators=200, max_depth=15,
                                                   random_state=SEED, n_jobs=-1),
    "Gradient Boosting":    GradientBoostingRegressor(n_estimators=200, max_depth=5,
                                                       learning_rate=0.1, random_state=SEED),
    "HistGradientBoosting": HistGradientBoostingRegressor(max_iter=300, max_depth=6,
                                          learning_rate=0.1, random_state=SEED),
}

results = []
predictions = {}

for name, model in models.items():
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_sc, y_train,
                                cv=5, scoring="r2", n_jobs=-1)
    # Fit on full training set
    model.fit(X_train_sc, y_train)
    y_pred = np.maximum(model.predict(X_test_sc), 0)

    mae  = mean_absolute_error(y_test, y_pred)
    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_test, y_pred)

    results.append({
        "Model": name, "MAE": mae, "MSE": mse, "RMSE": rmse,
        "R2": r2, "CV_Mean_R2": cv_scores.mean(), "CV_Std_R2": cv_scores.std(),
    })
    predictions[name] = y_pred

    print(f"\n  {name}")
    print(f"    Test R²  = {r2:.4f}  |  CV R² = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"    MAE = {mae:,.0f}  |  RMSE = {rmse:,.0f}")

results_df = pd.DataFrame(results).sort_values("R2", ascending=False)

# ──────────────────────────────────────────────────────────────────────
# 8. HYPERPARAMETER TUNING (best model: HistGradientBoosting)
# ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  STEP 8 · Hyperparameter Tuning (GridSearchCV on HistGradientBoosting)")
print("=" * 70)

param_grid = {
    "max_iter": [200, 400],
    "max_depth": [4, 6, 8],
    "learning_rate": [0.05, 0.1],
    "min_samples_leaf": [10, 20],
}

grid_search = GridSearchCV(
    HistGradientBoostingRegressor(random_state=SEED),
    param_grid, cv=3, scoring="r2", n_jobs=-1, verbose=0,
)
grid_search.fit(X_train_sc, y_train)
best_xgb = grid_search.best_estimator_
y_pred_tuned = np.maximum(best_xgb.predict(X_test_sc), 0)

tuned_r2   = r2_score(y_test, y_pred_tuned)
tuned_mae  = mean_absolute_error(y_test, y_pred_tuned)
tuned_rmse = np.sqrt(mean_squared_error(y_test, y_pred_tuned))

print(f"  Best params: {grid_search.best_params_}")
print(f"  Tuned HistGB → R² = {tuned_r2:.4f}  |  MAE = {tuned_mae:,.0f}  |  RMSE = {tuned_rmse:,.0f}")

# Add tuned model to results
results.append({
    "Model": "HistGB (Tuned)", "MAE": tuned_mae,
    "MSE": mean_squared_error(y_test, y_pred_tuned),
    "RMSE": tuned_rmse, "R2": tuned_r2,
    "CV_Mean_R2": grid_search.best_score_, "CV_Std_R2": 0,
})
predictions["HistGB (Tuned)"] = y_pred_tuned
results_df = pd.DataFrame(results).sort_values("R2", ascending=False)
results_df.to_csv("outputs/model_results.csv", index=False)

# ──────────────────────────────────────────────────────────────────────
# 9. CHARTS
# ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  STEP 9 · Generating Charts")
print("=" * 70)

# 9a. Model comparison
colors = ["#6366f1", "#ec4899", "#f59e0b", "#14b8a6", "#3b82f6", "#a855f7", "#ef4444", "#10b981"]
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
axes[0].barh(results_df["Model"], results_df["R2"], color=colors[:len(results_df)])
axes[0].set_xlabel("R² Score"); axes[0].set_title("Model Comparison – R²")
axes[0].set_xlim(0, 1)
for i, v in enumerate(results_df["R2"]):
    axes[0].text(v + 0.01, i, f"{v:.4f}", va="center", fontsize=9)

axes[1].barh(results_df["Model"], results_df["RMSE"], color=colors[:len(results_df)])
axes[1].set_xlabel("RMSE ($)"); axes[1].set_title("Model Comparison – RMSE")
for i, v in enumerate(results_df["RMSE"]):
    axes[1].text(v + v*0.01, i, f"{v:,.0f}", va="center", fontsize=9)
plt.tight_layout(); plt.savefig("outputs/05_model_comparison.png", dpi=150); plt.close()

# 9b. Actual vs Predicted (best model)
best_name = results_df.iloc[0]["Model"]
best_pred = predictions[best_name]
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(y_test, best_pred, alpha=0.4, s=15, c="#8b5cf6")
lims = [0, max(y_test.max(), best_pred.max()) * 1.05]
ax.plot(lims, lims, "--", color="#ef4444", lw=2, label="Perfect prediction")
ax.set_xlabel("Actual Revenue ($)"); ax.set_ylabel("Predicted Revenue ($)")
ax.set_title(f"Actual vs Predicted – {best_name}"); ax.legend()
plt.tight_layout(); plt.savefig("outputs/06_actual_vs_predicted.png", dpi=150); plt.close()

# 9c. Feature importance (best tuned model – permutation importance)
from sklearn.inspection import permutation_importance
perm_result = permutation_importance(best_xgb, X_test_sc, y_test,
                                     n_repeats=10, random_state=SEED, n_jobs=-1)
importances = perm_result.importances_mean
feat_imp = pd.DataFrame({"Feature": feature_names, "Importance": importances})
feat_imp = feat_imp.sort_values("Importance", ascending=True).tail(15)
fig, ax = plt.subplots(figsize=(9, 7))
ax.barh(feat_imp["Feature"], feat_imp["Importance"], color="#14b8a6")
ax.set_title(f"Feature Importances – {best_name}"); ax.set_xlabel("Importance")
plt.tight_layout(); plt.savefig("outputs/07_feature_importance.png", dpi=150); plt.close()

# 9d. Residual plot
residuals = y_test.values - best_pred
fig, ax = plt.subplots(figsize=(9, 5))
ax.scatter(best_pred, residuals, alpha=0.35, s=12, c="#ec4899")
ax.axhline(0, color="#374151", lw=1.5, ls="--")
ax.set_xlabel("Predicted Revenue ($)"); ax.set_ylabel("Residual ($)")
ax.set_title(f"Residual Plot – {best_name}")
plt.tight_layout(); plt.savefig("outputs/08_residual_plot.png", dpi=150); plt.close()

# 9e. Cross-validation comparison
fig, ax = plt.subplots(figsize=(10, 6))
cv_df = results_df[results_df["Model"] != "HistGB (Tuned)"]
x_pos = range(len(cv_df))
ax.bar(x_pos, cv_df["CV_Mean_R2"], yerr=cv_df["CV_Std_R2"],
       color=colors[:len(cv_df)], capsize=5, edgecolor="white")
ax.set_xticks(x_pos); ax.set_xticklabels(cv_df["Model"], rotation=25, ha="right")
ax.set_ylabel("Cross-Validated R²"); ax.set_title("5-Fold Cross-Validation R² Scores")
ax.set_ylim(0, 1)
for i, (m, s) in enumerate(zip(cv_df["CV_Mean_R2"], cv_df["CV_Std_R2"])):
    ax.text(i, m + s + 0.02, f"{m:.3f}", ha="center", fontsize=9)
plt.tight_layout(); plt.savefig("outputs/09_cross_validation.png", dpi=150); plt.close()

print("  Saved 9 charts to outputs/")

# ──────────────────────────────────────────────────────────────────────
# 10. EXPORT JSON FOR DASHBOARD
# ──────────────────────────────────────────────────────────────────────
dashboard_data = {
    "dataset_stats": {
        "total_rows": int(df.shape[0]),
        "features_used": len(feature_names),
        "feature_names": feature_names,
        "missing_filled": 186,
        "duplicates_removed": 30,
    },
    "model_results": results_df.to_dict(orient="records"),
    "best_model": {
        "name": best_name,
        "r2": round(float(results_df.iloc[0]["R2"]), 4),
        "mae": round(float(results_df.iloc[0]["MAE"]), 0),
        "rmse": round(float(results_df.iloc[0]["RMSE"]), 0),
        "best_params": grid_search.best_params_,
    },
    "feature_importance": feat_imp.sort_values("Importance", ascending=False).to_dict(orient="records"),
    "predictions_sample": {
        "actual": y_test.head(50).tolist(),
        "predicted": best_pred[:50].tolist(),
    },
    "residuals_sample": residuals[:100].tolist(),
}

with open("outputs/dashboard_data.json", "w") as f:
    json.dump(dashboard_data, f, indent=2, default=str)
print("  Saved → outputs/dashboard_data.json")

# ──────────────────────────────────────────────────────────────────────
# SUMMARY
# ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  FINAL SUMMARY")
print("=" * 70)
print(results_df[["Model", "R2", "CV_Mean_R2", "MAE", "RMSE"]].to_string(index=False))
print(f"\n  🏆 Best model: {best_name} (R² = {results_df.iloc[0]['R2']:.4f})")
print(f"  📊 Best params: {grid_search.best_params_}")
print("=" * 70)
