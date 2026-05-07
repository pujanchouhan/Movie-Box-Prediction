"""
generate_dataset.py
====================
Generates a realistic synthetic movie dataset (3 000 movies) for the
Box Office Revenue Prediction project.

The revenue formula is loosely modelled on observed real-world
correlations so that the downstream regression models have genuine
signal to learn from.

Run:
    python generate_dataset.py

Output:
    data/movies.csv
"""

import os
import datetime
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
NUM_MOVIES = 3000
SEED = 42
OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "movies.csv")

np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Lookup tables
# ---------------------------------------------------------------------------
GENRES = [
    "Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary",
    "Drama", "Fantasy", "Horror", "Mystery", "Romance", "Sci-Fi",
    "Thriller", "War", "Western", "Musical", "Family", "Biography",
]

DIRECTORS = [
    "Steven Spielberg", "Christopher Nolan", "Martin Scorsese",
    "James Cameron", "Ridley Scott", "Denis Villeneuve",
    "Quentin Tarantino", "Greta Gerwig", "Jordan Peele",
    "David Fincher", "Wes Anderson", "Chloe Zhao",
    "Peter Jackson", "Ava DuVernay", "Bong Joon-ho",
    "Ryan Coogler", "Damien Chazelle", "Patty Jenkins",
    "Michael Bay", "Zack Snyder", "Tim Burton", "Guy Ritchie",
    "Kathryn Bigelow", "Barry Jenkins", "Taika Waititi",
    "Sam Mendes", "Guillermo del Toro", "Sofia Coppola",
    "John Smith", "Jane Doe", "Alex Rivera", "Chris Park",
    "Morgan Blake", "Taylor Green", "Casey Brown",
    "Drew White", "Jamie Lee", "Pat Quinn",
]

CAST_POOL = [
    "Leonardo DiCaprio", "Meryl Streep", "Tom Hanks",
    "Scarlett Johansson", "Robert Downey Jr.", "Cate Blanchett",
    "Denzel Washington", "Margot Robbie", "Brad Pitt",
    "Viola Davis", "Dwayne Johnson", "Jennifer Lawrence",
    "Christian Bale", "Emma Stone", "Morgan Freeman",
    "Natalie Portman", "Will Smith", "Saoirse Ronan",
    "Ryan Gosling", "Lupita Nyong'o", "Timothée Chalamet",
    "Zendaya", "Florence Pugh", "Pedro Pascal",
    "Chris Evans", "Ana de Armas", "Joaquin Phoenix",
    "Gal Gadot", "Idris Elba", "Sandra Bullock",
    "Samuel L. Jackson", "Amy Adams", "Jake Gyllenhaal",
    "Charlize Theron", "Benedict Cumberbatch", "Awkwafina",
    "Tom Holland", "Brie Larson", "Oscar Isaac",
    "Anya Taylor-Joy", "John Doe A", "Jane Doe B",
    "Alex Actor", "Sam Star", "Pat Player",
    "Jordan Talent", "Riley Role", "Morgan Screen",
    "Taylor Thespian", "Casey Celeb",
]

PRODUCTION_COMPANIES = [
    "Warner Bros.", "Walt Disney Pictures", "Universal Pictures",
    "20th Century Studios", "Paramount Pictures", "Columbia Pictures",
    "Lionsgate", "A24", "New Line Cinema", "Legendary Entertainment",
    "Metro-Goldwyn-Mayer", "DreamWorks", "Amblin Entertainment",
    "Blumhouse Productions", "Focus Features", "Searchlight Pictures",
    "NEON", "Annapurna Pictures", "Regency Enterprises", "Village Roadshow",
]

LANGUAGES = ["English", "Spanish", "French", "Hindi", "Korean", "Japanese",
             "German", "Chinese", "Italian", "Portuguese"]

COUNTRIES = ["USA", "UK", "France", "India", "South Korea", "Japan",
             "Germany", "Canada", "Australia", "Brazil", "China", "Italy"]

# ---------------------------------------------------------------------------
# Star-power scores (higher → bigger draw)
# ---------------------------------------------------------------------------
_director_power = {}
for i, d in enumerate(DIRECTORS):
    if i < 15:
        _director_power[d] = np.random.uniform(0.7, 1.0)
    elif i < 25:
        _director_power[d] = np.random.uniform(0.4, 0.7)
    else:
        _director_power[d] = np.random.uniform(0.1, 0.4)

_cast_power = {}
for i, c in enumerate(CAST_POOL):
    if i < 20:
        _cast_power[c] = np.random.uniform(0.6, 1.0)
    elif i < 35:
        _cast_power[c] = np.random.uniform(0.3, 0.6)
    else:
        _cast_power[c] = np.random.uniform(0.05, 0.3)


def _random_date(start_year=2000, end_year=2025):
    start = datetime.date(start_year, 1, 1)
    end = datetime.date(end_year, 12, 31)
    delta = (end - start).days
    return start + datetime.timedelta(days=np.random.randint(0, delta))


def _pick_cast(n=3):
    chosen = np.random.choice(CAST_POOL, size=n, replace=False)
    return ", ".join(chosen)


# ---------------------------------------------------------------------------
# Generate rows
# ---------------------------------------------------------------------------
rows = []
for movie_id in range(1, NUM_MOVIES + 1):
    # --- categorical ---
    genre = np.random.choice(GENRES)
    director = np.random.choice(DIRECTORS)
    cast = _pick_cast(n=np.random.choice([2, 3, 4]))
    production_company = np.random.choice(PRODUCTION_COMPANIES)
    language = np.random.choice(LANGUAGES, p=[
        0.50, 0.08, 0.07, 0.07, 0.05, 0.05,
        0.04, 0.05, 0.05, 0.04,
    ])
    country = np.random.choice(COUNTRIES)

    # --- numeric ---
    budget = int(np.random.lognormal(mean=17.0, sigma=1.2))
    budget = max(50_000, min(budget, 400_000_000))  # clamp
    runtime = int(np.clip(np.random.normal(110, 25), 60, 240))
    release_date = _random_date()

    # IMDB rating: slightly correlated with director power
    dp = _director_power[director]
    imdb_base = np.random.normal(6.5 + dp * 1.5, 1.0)
    imdb_rating = round(float(np.clip(imdb_base, 1.0, 10.0)), 1)

    # Votes: correlated with budget (bigger movies get more attention)
    vote_base = (np.log(budget) - 10) * 1500 + np.random.normal(0, 3000)
    number_of_votes = int(max(50, vote_base))

    # --- target: box_office_revenue ---
    # Cast power average
    cast_names = [c.strip() for c in cast.split(",")]
    cast_score = np.mean([_cast_power.get(c, 0.2) for c in cast_names])

    # Seasonal multiplier (summer / holiday releases do better)
    month = release_date.month
    if month in (6, 7, 12):
        season_mult = 1.25
    elif month in (5, 11):
        season_mult = 1.10
    else:
        season_mult = 1.0

    # Genre multiplier
    genre_mult = {
        "Action": 1.3, "Adventure": 1.25, "Animation": 1.2,
        "Sci-Fi": 1.2, "Fantasy": 1.15, "Comedy": 1.05,
        "Family": 1.15, "Thriller": 1.05, "Horror": 0.9,
        "Drama": 0.95, "Romance": 0.85, "Documentary": 0.5,
        "Crime": 1.0, "War": 0.9, "Western": 0.75,
        "Musical": 0.95, "Mystery": 0.95, "Biography": 0.85,
    }.get(genre, 1.0)

    # Language multiplier (English films have wider reach)
    lang_mult = 1.0 if language == "English" else np.random.uniform(0.3, 0.7)

    # Revenue formula
    revenue = (
        budget * np.random.uniform(0.8, 3.5)
        * (0.5 + dp)
        * (0.5 + cast_score)
        * season_mult
        * genre_mult
        * lang_mult
        * (0.5 + imdb_rating / 10.0)
    )
    # Add noise
    noise = np.random.normal(1.0, 0.15)
    revenue = max(0, int(revenue * noise))

    rows.append({
        "movie_id": movie_id,
        "title": f"Movie_{movie_id}",
        "budget": budget,
        "genre": genre,
        "cast": cast,
        "director": director,
        "runtime": runtime,
        "release_date": str(release_date),
        "production_company": production_company,
        "language": language,
        "country": country,
        "IMDB_rating": imdb_rating,
        "number_of_votes": number_of_votes,
        "box_office_revenue": revenue,
    })

# ---------------------------------------------------------------------------
# Inject realistic imperfections (missing values, duplicates)
# ---------------------------------------------------------------------------
df = pd.DataFrame(rows)

# ~3 % missing budgets
mask = np.random.rand(len(df)) < 0.03
df.loc[mask, "budget"] = np.nan

# ~2 % missing runtimes
mask = np.random.rand(len(df)) < 0.02
df.loc[mask, "runtime"] = np.nan

# ~1 % missing IMDB ratings
mask = np.random.rand(len(df)) < 0.01
df.loc[mask, "IMDB_rating"] = np.nan

# ~1 % missing number_of_votes
mask = np.random.rand(len(df)) < 0.01
df.loc[mask, "number_of_votes"] = np.nan

# Inject ~30 near-duplicate rows (same movie_id, slightly tweaked revenue)
dup_idx = np.random.choice(df.index, size=30, replace=False)
dups = df.loc[dup_idx].copy()
dups["box_office_revenue"] = (dups["box_office_revenue"] * 1.001).astype(int)
df = pd.concat([df, dups], ignore_index=True)
df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)
print(f"✅ Dataset saved to {OUTPUT_FILE}  ({len(df)} rows, {df.shape[1]} columns)")
print(df.head())
