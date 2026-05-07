# 🎬 Movie Box Office Prediction using Regression & Feature Engineering

A machine learning project that predicts movie box office revenue using multiple
regression models, advanced feature engineering, cross-validation, and hyperparameter tuning.
Includes an interactive web dashboard for visualizing results.

## 📁 Project Structure

```
Movie Box Prediction ML/
├── generate_dataset.py                 # Synthetic dataset generator (3,000 movies)
├── movie_box_office_prediction.py      # Basic ML pipeline (v1)
├── movie_box_office_prediction_v2.py   # Enhanced pipeline (v2) with CV & tuning
├── requirements.txt                    # Python dependencies
├── data/
│   └── movies.csv                      # Generated dataset
├── outputs/
│   ├── 01–09 *.png                     # 9 publication-quality charts
│   ├── model_results.csv               # All evaluation metrics
│   └── dashboard_data.json             # JSON export for web dashboard
├── dashboard/
│   ├── index.html                      # Interactive web dashboard
│   ├── style.css                       # Premium dark theme
│   └── app.js                          # Chart.js visualizations
└── README.md
```

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate the synthetic dataset
python3 generate_dataset.py

# 3. Run the enhanced ML pipeline (v2)
python3 movie_box_office_prediction_v2.py

# 4. Launch the interactive dashboard
python3 -m http.server 8080
# Then open: http://localhost:8080/dashboard/
```

## 📊 Dataset Features

| Feature             | Type        | Description                          |
|---------------------|-------------|--------------------------------------|
| budget              | Numerical   | Production budget ($)                |
| genre               | Categorical | Primary genre                        |
| cast                | Text        | Top 2–4 actors                       |
| director            | Categorical | Director name                        |
| runtime             | Numerical   | Duration in minutes                  |
| release_date        | Date        | Theatrical release date              |
| production_company  | Categorical | Production studio                    |
| language            | Categorical | Original language                    |
| country             | Categorical | Country of origin                    |
| IMDB_rating         | Numerical   | Average IMDB score (1–10)            |
| number_of_votes     | Numerical   | Total IMDB votes                     |
| **box_office_revenue** | **Target** | **Revenue to predict ($)**        |

## 🔧 Pipeline Steps

### 1. Data Preprocessing
- Imputed 186 missing values (median strategy)
- Removed 30 duplicate rows
- Cleaned and validated data types

### 2. Feature Engineering (22 total features)
- **Temporal**: `release_month`, `release_year`, `release_season`
- **Popularity**: `cast_popularity`, `director_is_top`
- **Genre grouping**: 18 genres → 8 broad categories
- **Interactions**: `budget × rating`, `budget / minute`, `votes / rating`
- **Binary flags**: `is_english`, `is_summer_release`, `is_holiday_release`
- **Log transforms**: `log(1 + budget)`

### 3. Models Trained (v2)

| Model                  | Test R² | CV R² (mean) | MAE ($M) | RMSE ($M) |
|------------------------|---------|--------------|----------|-----------|
| **HistGB (Tuned)**     | **0.676** | **0.707**  | **51.0** | **143.9** |
| Gradient Boosting      | 0.672   | 0.676        | 53.6     | 144.7     |
| HistGradientBoosting   | 0.661   | 0.684        | 53.7     | 147.2     |
| Random Forest          | 0.648   | 0.701        | 54.0     | 149.9     |
| Ridge Regression       | 0.589   | 0.636        | 59.6     | 162.1     |
| Lasso Regression       | 0.589   | 0.636        | 59.7     | 162.2     |
| Linear Regression      | 0.589   | 0.636        | 59.8     | 162.2     |
| Decision Tree          | 0.515   | 0.441        | 73.8     | 176.0     |

### 4. Hyperparameter Tuning
- **GridSearchCV** on HistGradientBoosting with 3-fold CV
- Best parameters: `learning_rate=0.05`, `max_depth=4`, `max_iter=200`, `min_samples_leaf=20`

🏆 **Best Model: HistGB (Tuned) — R² = 0.676, MAE = $51M**

## 🖥️ Interactive Dashboard

The project includes a premium dark-themed web dashboard built with Chart.js:
- Animated KPI cards
- Model performance comparison charts
- 5-fold cross-validation visualization
- Actual vs Predicted scatter plot
- Residual distribution histogram
- Feature importance ranking
- Detailed results table
- Tuned hyperparameter display

## 🛠️ Tools & Technologies

- **Python 3.9+** — ML pipeline
- **Pandas & NumPy** — data manipulation
- **Scikit-learn** — models, CV, GridSearchCV, metrics
- **Matplotlib & Seaborn** — static charts
- **Chart.js** — interactive dashboard charts
- **HTML/CSS/JS** — web dashboard

## 📈 Key Insights

- **budget × IMDB_rating** is the strongest predictor (interaction feature)
- **Language** (English vs non-English) significantly impacts global revenue
- **Gradient Boosting** methods outperform linear models by ~15% R²
- **Summer & holiday** releases earn ~25% more on average
- **5-fold cross-validation** confirms model robustness (CV R² ≈ 0.71)
- **Hyperparameter tuning** improved MAE by ~$3M over the default model
