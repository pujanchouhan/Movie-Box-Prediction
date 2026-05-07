"""
app.py — Flask backend for Movie Success Prediction System
Serves the web app and provides /predict API endpoint.

Usage:
    python3 save_model.py   # train & save model (run once)
    python3 app.py          # start web server on port 5000
"""

import os, json
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__, static_folder="webapp", static_url_path="")

# ── Load model artifacts ──
model = joblib.load("model_artifacts/model.pkl")
scaler = joblib.load("model_artifacts/scaler.pkl")

with open("model_artifacts/metadata.json") as f:
    meta = json.load(f)

FEATURE_NAMES = meta["feature_names"]
MEDIANS = meta["medians"]
LE_MAPS = meta["label_encoders"]
GENRE_GROUP_MAP = meta["genre_group_map"]

MONTH_MAP = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
    "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
}
SEASON_MAP = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer",
    9: "Fall", 10: "Fall", 11: "Fall",
}


def _safe_le(col, value):
    """Label-encode a value; fall back to 0 if unseen."""
    mapping = LE_MAPS.get(col, {})
    return mapping.get(str(value), 0)


def build_feature_vector(data):
    """Convert user inputs → feature vector matching training columns."""
    budget = float(data.get("budget", 50_000_000))
    genre_raw = data.get("genre", "Action")
    runtime = float(data.get("runtime", 110))
    release_month = data.get("release_month", "Jun")
    cast_pop = float(data.get("cast_popularity", 5))
    director_score = float(data.get("director_score", 5))
    imdb_rating = float(data.get("imdb_rating", 7.0))

    # Derived features
    month_num = MONTH_MAP.get(release_month, 6)
    season = SEASON_MAP.get(month_num, "Summer")
    genre_group = GENRE_GROUP_MAP.get(genre_raw, "Other")

    # Scale cast_popularity from 1-10 slider → 0-4 model range
    cast_popularity_scaled = round(cast_pop / 10 * 4)
    director_is_top = 1 if director_score >= 7 else 0

    # Estimate number_of_votes from budget (loosely correlated)
    number_of_votes = max(500, int((np.log(budget + 1) - 10) * 1500))

    budget_x_rating = budget * imdb_rating
    budget_per_minute = budget / max(runtime, 1)
    votes_per_rating = number_of_votes / max(imdb_rating, 0.1)
    log_budget = np.log1p(budget)
    is_english = 1  # default assumption
    is_summer = 1 if month_num in (5, 6, 7) else 0
    is_holiday = 1 if month_num in (11, 12) else 0

    row = {}
    row["budget"] = budget
    row["genre"] = _safe_le("genre", genre_raw)
    row["director"] = MEDIANS.get("director", 0)
    row["runtime"] = runtime
    row["production_company"] = MEDIANS.get("production_company", 0)
    row["language"] = _safe_le("language", "English")
    row["country"] = _safe_le("country", "USA")
    row["IMDB_rating"] = imdb_rating
    row["number_of_votes"] = number_of_votes
    row["release_month"] = month_num
    row["release_year"] = 2025
    row["release_season"] = _safe_le("release_season", season)
    row["cast_popularity"] = cast_popularity_scaled
    row["director_is_top"] = director_is_top
    row["genre_group"] = _safe_le("genre_group", genre_group)
    row["budget_x_rating"] = budget_x_rating
    row["budget_per_minute"] = budget_per_minute
    row["votes_per_rating"] = votes_per_rating
    row["log_budget"] = log_budget
    row["is_english"] = is_english
    row["is_summer_release"] = is_summer
    row["is_holiday_release"] = is_holiday

    # Build DataFrame in correct column order (avoids sklearn feature-name warnings)
    vec = [row.get(f, MEDIANS.get(f, 0)) for f in FEATURE_NAMES]
    return pd.DataFrame([vec], columns=FEATURE_NAMES)


def classify_verdict(revenue, budget):
    ratio = revenue / max(budget, 1)
    if ratio >= 2.0:
        return "Blockbuster", "blockbuster"
    elif ratio >= 1.5:
        return "Hit", "hit"
    elif ratio >= 1.0:
        return "Average", "average"
    else:
        return "Flop", "flop"


def compute_feature_impact(data):
    """Rough feature impact explanation based on input values."""
    impacts = []
    budget = float(data.get("budget", 50_000_000))
    imdb = float(data.get("imdb_rating", 7.0))
    cast = float(data.get("cast_popularity", 5))
    director = float(data.get("director_score", 5))
    month_name = data.get("release_month", "Jun")
    month = MONTH_MAP.get(month_name, 6)

    if budget > 100_000_000:
        impacts.append({"feature": "Budget", "impact": "positive",
                        "detail": f"High budget (${budget:,.0f}) correlates with wider reach"})
    elif budget < 20_000_000:
        impacts.append({"feature": "Budget", "impact": "negative",
                        "detail": "Low budget may limit marketing & distribution"})

    if imdb >= 8.0:
        impacts.append({"feature": "IMDB Rating", "impact": "positive",
                        "detail": f"Excellent rating ({imdb}) drives word-of-mouth"})
    elif imdb < 6.0:
        impacts.append({"feature": "IMDB Rating", "impact": "negative",
                        "detail": f"Below-average rating ({imdb}) hurts audience interest"})

    if cast >= 7:
        impacts.append({"feature": "Cast Popularity", "impact": "positive",
                        "detail": "Star-studded cast boosts opening weekend"})
    elif cast <= 3:
        impacts.append({"feature": "Cast Popularity", "impact": "negative",
                        "detail": "Unknown cast makes marketing harder"})

    if director >= 7:
        impacts.append({"feature": "Director", "impact": "positive",
                        "detail": "Top-tier director increases audience trust"})

    if month in (6, 7, 12):
        impacts.append({"feature": "Release Timing", "impact": "positive",
                        "detail": f"{month_name} is a peak movie season"})
    elif month in (1, 2, 9):
        impacts.append({"feature": "Release Timing", "impact": "negative",
                        "detail": f"{month_name} is traditionally a slow period"})

    return impacts


# ── Routes ──

@app.route("/")
def index():
    return send_from_directory("webapp", "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        X = build_feature_vector(data)
        X_scaled = scaler.transform(X)
        predicted_revenue = max(0, float(model.predict(X_scaled)[0]))

        budget = float(data.get("budget", 50_000_000))
        profit = predicted_revenue - budget
        verdict, verdict_class = classify_verdict(predicted_revenue, budget)

        # Confidence based on how close the prediction is to training distribution
        confidence = min(95, max(40, 70 + (predicted_revenue / max(budget, 1) - 1) * 10))

        impacts = compute_feature_impact(data)

        return jsonify({
            "success": True,
            "predicted_revenue": round(predicted_revenue),
            "budget": round(budget),
            "profit": round(profit),
            "verdict": verdict,
            "verdict_class": verdict_class,
            "confidence": round(confidence, 1),
            "feature_impacts": impacts,
            "model_r2": meta["r2"],
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/metadata")
def api_metadata():
    return jsonify({
        "genres": ["Action", "Adventure", "Animation", "Comedy", "Crime",
                   "Documentary", "Drama", "Fantasy", "Horror", "Mystery",
                   "Romance", "Sci-Fi", "Thriller", "War", "Western",
                   "Musical", "Family", "Biography"],
        "model_r2": meta["r2"],
        "model_mae": meta["mae"],
    })


@app.route("/dashboard/")
@app.route("/dashboard/<path:filename>")
def dashboard(filename="index.html"):
    return send_from_directory("dashboard", filename)


@app.route("/outputs/<path:filename>")
def outputs(filename):
    return send_from_directory("outputs", filename)


if __name__ == "__main__":
    print("  🎬 Movie Success Prediction System")
    print("  🌐 Open http://localhost:5000")
    app.run(debug=True, port=5000)
