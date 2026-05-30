from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import joblib
import re
import requests
import traceback
import pandas as pd
import os
from dotenv import load_dotenv

# look for the .env file and load the keys inside it
load_dotenv() 

# pull the keys from the secure environment
TMDB_BEARER_TOKEN = os.getenv("TMDB_BEARER_TOKEN")
OMDB_API_KEY = os.getenv("OMDB_API_KEY")
MDBLIST_API_KEY = os.getenv("MDBLIST_API_KEY")

# Path to your training CSV — expects columns: 'review' and 'sentiment'
# If your columns differ, update the names below in the fallback section.
TRAINING_CSV_PATH = "notebooks/IMDB Dataset.csv"
REVIEW_COLUMN     = "review"     # column name for review text
SENTIMENT_COLUMN  = "sentiment"  # column name for label (not used for prediction — model predicts it)

# Minimum live reviews before we attempt the dataset fallback
LIVE_REVIEW_THRESHOLD = 20

headers = {
    "accept": "application/json",
    "Authorization": f"Bearer {TMDB_BEARER_TOKEN}"
}

app = FastAPI(title="Multi-Criteria Entertainment Recommender")

# ==========================================
# 2. ML MODEL LOADING
# ==========================================
print("⏳ Loading machine learning models...")
try:
    model = joblib.load('models/logistic_regression_baseline.pkl')
    tfidf = joblib.load('models/tfidf_vectorizer.pkl')
    print("🚀 Models loaded successfully!")
except Exception as e:
    print(f"⚠️ Warning: Could not load ML models. Ensure the 'models' folder exists. Error: {e}")
    model = None
    tfidf = None

# ==========================================
# 3. TRAINING DATASET LOADING (for fallback)
# ==========================================
training_df = None
if os.path.exists(TRAINING_CSV_PATH):
    try:
        training_df = pd.read_csv(TRAINING_CSV_PATH)
        print(f"📚 Training dataset loaded: {len(training_df):,} rows from '{TRAINING_CSV_PATH}'")
    except Exception as e:
        print(f"⚠️ Could not load training CSV: {e}")
else:
    print(f"ℹ️  No training CSV found at '{TRAINING_CSV_PATH}' — dataset fallback disabled.")


def clean_text(text: str) -> str:
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.lower().strip()


def predict_reviews(reviews: list[str]) -> tuple[int, int]:
    """Run the ML model over a list of review strings.
    Returns (pos_count, neg_count).
    """
    if not reviews or model is None or tfidf is None:
        return 0, 0
    predictions = [
        str(model.predict(tfidf.transform([clean_text(r)]))[0]).lower()
        for r in reviews
    ]
    pos = sum(1 for p in predictions if p == "positive")
    neg = len(predictions) - pos
    return pos, neg


def fetch_dataset_reviews(title: str, want: int = 100) -> list[str]:
    """Search the training dataframe for rows that mention the title.
    Returns up to `want` review strings, or [] if the dataset isn't loaded.
    This gives the model more signal when live TMDB reviews are sparse.
    """
    if training_df is None or REVIEW_COLUMN not in training_df.columns:
        return []

    pattern = re.escape(title.lower())
    mask = training_df[REVIEW_COLUMN].str.lower().str.contains(pattern, na=False)
    matched = training_df.loc[mask, REVIEW_COLUMN].dropna().tolist()

    if matched:
        print(f"📚 Dataset fallback: found {len(matched)} reviews mentioning '{title}' — capping at {want}")
        return matched[:want]

    print(f"📚 Dataset fallback: no exact title match for '{title}' — sampling random reviews for baseline")
    # No title match — draw a random balanced sample so the model still gets
    # a representative spread rather than returning nothing.
    sample_size = min(want, len(training_df))
    return training_df[REVIEW_COLUMN].dropna().sample(sample_size, random_state=42).tolist()


# Serve the frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def serve_frontend():
    return FileResponse("static/index.html")

# ==========================================
# 4. CORE MULTI-FORMAT API ENDPOINT
# ==========================================
@app.get("/api/analyze/{media_type}/{name}")
def analyze_media(media_type: str, name: str):
    """
    Dynamically routes logic based on media_type ('movie' or 'tv').
    """
    try:
        if media_type not in ['movie', 'tv']:
            return {"error": "Invalid media type. Use 'movie' or 'tv'."}

        print(f"\n--- Multi-Criteria Analysis [{media_type.upper()}]: {name} ---")

        # ── Step A: Search TMDB ──
        search_url = (
            f"https://api.themoviedb.org/3/search/{media_type}"
            f"?query={requests.utils.quote(name)}&language=en-US&page=1"
        )
        search_data = requests.get(search_url, headers=headers, timeout=10).json()

        if not search_data.get('results'):
            return {"error": f"Could not find any {media_type} matching '{name}'."}

        media_id = search_data['results'][0]['id']

        # ── Step B: Full Metadata from TMDB ──
        details_url = f"https://api.themoviedb.org/3/{media_type}/{media_id}?language=en-US"
        details = requests.get(details_url, headers=headers, timeout=10).json()

        imdb_id = details.get('imdb_id')
        if not imdb_id and media_type == 'tv':
            ext_url = f"https://api.themoviedb.org/3/tv/{media_id}/external_ids"
            ext_data = requests.get(ext_url, headers=headers, timeout=10).json()
            imdb_id = ext_data.get('imdb_id')

        if not imdb_id:
            return {"error": "This title does not have a linked IMDb ID available."}

        genres = [g['name'] for g in details.get('genres', [])]

        if media_type == 'movie':
            title        = details.get("title")
            release_date = details.get("release_date", "Unknown")
            runtime_val  = f"{details.get('runtime', 0)} minutes"
            stat_label_1 = "Budget"
            stat_value_1 = f"${details.get('budget', 0):,.2f}" if details.get('budget') else "N/A"
            stat_label_2 = "Revenue"
            stat_value_2 = f"${details.get('revenue', 0):,.2f}" if details.get('revenue') else "N/A"
        else:
            title        = details.get("name")
            release_date = details.get("first_air_date", "Unknown")
            runtime_val  = f"{details.get('number_of_seasons', 0)} Seasons"
            stat_label_1 = "Total Seasons"
            stat_value_1 = str(details.get('number_of_seasons', 0))
            stat_label_2 = "Total Episodes"
            stat_value_2 = str(details.get('number_of_episodes', 0))

        # ── Step C: Primary Metrics via OMDb ──
        print("Fetching primary metrics from OMDb...")
        omdb_url  = f"https://www.omdbapi.com/?i={imdb_id}&apikey={OMDB_API_KEY}"
        omdb_data = requests.get(omdb_url, timeout=10).json()

        imdb_rating = omdb_data.get("imdbRating", "N/A")
        imdb_votes  = omdb_data.get("imdbVotes", "0")
        metascore   = omdb_data.get("Metascore", "N/A")

        rt_critic = "N/A"
        for r in omdb_data.get("Ratings", []):
            if r['Source'] == 'Rotten Tomatoes':
                rt_critic = r['Value']
                break

        # ── Step D: MDBList rescue for missing scores ──
        if rt_critic == "N/A" or metascore == "N/A":
            print("OMDb missing critical scores. Deploying MDBList rescue scraper...")
            try:
                mdblist_url  = f"https://mdblist.com/api/?apikey={MDBLIST_API_KEY}&i={imdb_id}"
                mdblist_data = requests.get(mdblist_url, timeout=10).json()

                for rating in mdblist_data.get("ratings", []):
                    source = rating.get("source", "").lower()
                    score  = rating.get("score") or rating.get("value")
                    if score:
                        if "tomatoes" in source and rt_critic == "N/A":
                            rt_critic = f"{score}%"
                            print(f"✅ Rescued Rotten Tomatoes: {rt_critic}")
                        elif "metacritic" in source and metascore == "N/A":
                            metascore = str(score)
                            print(f"✅ Rescued Metascore: {metascore}")

                if mdblist_data.get("tomatoes")   and rt_critic == "N/A":
                    rt_critic = f"{mdblist_data['tomatoes']}%"
                if mdblist_data.get("metacritic") and metascore == "N/A":
                    metascore = str(mdblist_data['metacritic'])

            except Exception as e:
                print(f"⚠️ MDBList fallback failed: {e}")

        # ── Step E: Fetch live TMDB reviews (up to 10 pages) ──
        print("Fetching live TMDB reviews for AI Sentiment Analysis...")
        live_reviews: list[str] = []
        for page in range(1, 11):
            reviews_url = (
                f"https://api.themoviedb.org/3/{media_type}/{media_id}"
                f"/reviews?language=en-US&page={page}"
            )
            res = requests.get(reviews_url, headers=headers, timeout=10)
            if res.status_code == 200:
                data = res.json()
                live_reviews.extend([r['content'] for r in data.get('results', [])])
                if page >= data.get('total_pages', 1):
                    break
            else:
                break

        print(f"  Live reviews fetched: {len(live_reviews)}")

        # ── Step F: Dataset fallback if live reviews are sparse ──
        reviews_source = "live"
        all_reviews    = live_reviews

        if len(live_reviews) < LIVE_REVIEW_THRESHOLD:
            print(f"  Live count ({len(live_reviews)}) below threshold ({LIVE_REVIEW_THRESHOLD}). "
                  f"Augmenting from training dataset…")
            dataset_reviews = fetch_dataset_reviews(title, want=100)
            if dataset_reviews:
                # Combine: live first (higher relevance), then dataset
                all_reviews    = live_reviews + dataset_reviews
                reviews_source = "live + dataset"
                print(f"  Combined review pool: {len(all_reviews)} ({len(live_reviews)} live + {len(dataset_reviews)} dataset)")

        # ── Step G: Run ML predictions ──
        total_reviews = len(all_reviews)
        pos_count, neg_count = predict_reviews(all_reviews)
        ai_score = (pos_count / total_reviews * 100) if total_reviews > 0 else 0.0

        print(f"  Sentiment result: {pos_count} positive / {neg_count} negative "
              f"({ai_score:.1f}%) — source: {reviews_source}")

        # ── Step H: Weighted Super Score ──
        try:
            imdb_norm = float(imdb_rating) * 10 if imdb_rating != "N/A" else 70.0
            meta_norm = float(metascore)          if metascore  != "N/A" else 70.0
            rt_norm   = float(rt_critic.replace('%', '')) if rt_critic != "N/A" else 70.0
            nlp_norm  = ai_score if total_reviews > 0 else imdb_norm

            # 30% IMDb · 20% Metacritic · 20% Rotten Tomatoes · 30% AI/NLP
            super_score = (imdb_norm * 0.3) + (meta_norm * 0.2) + (rt_norm * 0.2) + (nlp_norm * 0.3)
        except Exception:
            super_score = ai_score if total_reviews > 0 else 70.0

        # ── Step I: Return payload ──
        return {
            "title":        title,
            "release_date": release_date,
            "runtime":      runtime_val,
            "genres":       ", ".join(genres) if genres else "N/A",
            "stat_label_1": stat_label_1,
            "stat_value_1": stat_value_1,
            "stat_label_2": stat_label_2,
            "stat_value_2": stat_value_2,
            "summary":      details.get("overview", "No summary available."),
            "poster_url": (
                f"https://image.tmdb.org/t/p/w500{details['poster_path']}"
                if details.get('poster_path')
                else "https://via.placeholder.com/500x750?text=No+Poster"
            ),

            # Platform ratings
            "imdb_rating":           imdb_rating,
            "imdb_votes":            imdb_votes,
            "metascore":             metascore,
            "rotten_tomatoes_critic": rt_critic,

            # AI sentiment breakdown
            "total_reviews_analyzed":  total_reviews,
            "live_reviews_count":      len(live_reviews),
            "positive_reviews":        pos_count,
            "negative_reviews":        neg_count,
            "ai_sentiment_percentage": round(ai_score, 1),
            "reviews_source":          reviews_source,

            # Final verdict
            "sentiment_score": round(super_score, 1),
            "verdict": (
                "HIGHLY RECOMMENDED" if super_score >= 75
                else "MIXED FEEDBACK" if super_score >= 50
                else "SKIP IT"
            ),
        }

    except Exception as e:
        traceback.print_exc()
        return {"error": f"Internal Server Error: {str(e)}"}