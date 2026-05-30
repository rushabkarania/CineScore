

</p>

# CineScore: AI-Powered Entertainment Recommendation Engine

This repository houses the entire codebase for **CineScore**, a highly optimized, multi-platform entertainment recommendation engine. CineScore completely discards traditional, static rating aggregations by fusing real-time web scraping, dynamic API failover routing, and an on-the-fly Natural Language Processing (NLP) inference pipeline.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Online-brightgreen?style=for-the-badge&logo=render)](https://cinescore-alcg.onrender.com/)

> 🔗 **Live Web Application:** [https://cinescore-alcg.onrender.com/](https://cinescore-alcg.onrender.com/)

This repository houses the entire codebase for **CineScore**, a highly optimized, multi-platform entertainment recommendation engine...

---

## Background & The Problem

We’ve all experienced the "Friday Night Scroll"—staring blankly at a streaming interface, trying to decide what to watch. You see a movie with a 6.5 on IMDb. Is that a *good* 6.5 because it's a cult-classic action flick, or a *bad* 6.5 because the plot unravels completely in the third act?

Traditional platforms suffer from the **"Black Box Rating Problem."** They present a single, flat, unyielding number and expect you to trust it. But audience reviews are messy, complex, and deeply nuanced. They are filled with sarcasm, highly specific critiques, and divided consensus that a flat number completely obliterates.

### The CineScore Solution

CineScore solves this by acting as an automated data scientist in your browser. When you search for a movie or TV show, the system doesn't just pull a static score; it actively scrapes live, unstructured text reviews written by real audiences. It then passes those reviews through a custom Machine Learning pipeline to evaluate the actual emotional sentiment of the viewers.

By combining this live AI sentiment analysis with critical scores harvested across multiple global movie databases, CineScore computes an authoritative, mathematically optimized **System Super Score**. No more guessing—just pure, data-backed movie recommendations.

---

## Under the Hood: How We Built It

CineScore is engineered as a clean, fully decoupled web architecture featuring a high-performance **FastAPI** backend serving a highly customized, responsive native frontend.

### 1. The Multi-Platform "Waterfall" Rescue Routing

Data sparsity is a massive issue when pulling live metadata, especially for newly released indie titles or streaming-exclusive TV shows where Metacritic or Rotten Tomatoes scores are notoriously incomplete.

To achieve maximum data density, CineScore implements a **Conditional Fallback Routing Architecture**:

* **Primary Query:** The backend queries **The Movie Database (TMDB)** using strict string matching to grab foundational metadata, artwork assets, genre IDs, and unique identifier maps (`imdb_id`).
* **Secondary Pull:** The backend routes the `imdb_id` directly to the **Open Movie Database (OMDb)** to harvest primary audience and critical rating matrixes.
* **The Rescue Scraper:** If OMDb returns a critical score as `"N/A"`, the backend automatically catches the omission and dispatches an asynchronous API call to **MDBList** as a failover rescue layer, patching the missing metrics before any data is sent to the client.

### 2. Live NLP Sentiment Inference Engine

The "AI Sentiment" score is built entirely from scratch and calculated live on your server.

* **Preprocessing:** Raw text reviews are scraped directly from the TMDB review endpoint. The backend sanitizes the incoming text strings using custom Regular Expressions (Regex), stripping out raw HTML tags, escape characters, and markdown artifacts left behind by users.
* **Vectorization:** Text data is inherently unreadable by algorithms. The system loads a pre-trained **Term Frequency-Inverse Document Frequency (TF-IDF)** vectorizer. This maps the unstructured reviews into an optimized, high-dimensional numerical matrix, weighting the semantic importance of distinct words while completely filtering out standard English stop words.
* **Classification:** The normalized vector matrix is injected into a trained **Logistic Regression** classifier (`.pkl`). The model runs binary inference on every single review, explicitly categorizing each review as either `Positive` (1) or `Negative` (0).
* **Aggregation:** The engine sums the total positive classifications against the sample size to yield a precise, real-time audience approval percentage ($S_{nlp}$).

### 3. The Super Score Mathematical Algorithm

To balance the inherent biases across different platforms, CineScore applies an algebraic weighting matrix. All external metrics are programmatically normalized to a base-100 scale before running through the final pipeline formula:

$$
\text{Super Score} = (0.30 \times S_{imdb}) + (0.20 \times S_{meta}) + (0.20 \times S_{rt}) + (0.30 \times S_{nlp})
$$

Where:

* $S_{imdb}$ = The raw IMDb rating scaled perfectly into a base-100 metric $(x \times 10)$.
* $S_{meta}$ = The Metacritic Critic Consensus score (0–100).
* $S_{rt}$ = The Rotten Tomatoes Tomatometer percentage (0–100).
* $S_{nlp}$ = CineScore's custom Live AI Sentiment Classification percentage (0–100).

> ⚠️ **Dynamic Weight Redistribution:** If an asset is highly obscure and a critical score remains completely unrecoverable across all fallback APIs, the system automatically triggers an error-handling wrapper. It zeroes out the missing platform and dynamically redistributes its mathematical weight to the AI Sentiment score, ensuring indie films are never mathematically penalized.

### 4. Math-Driven Dynamic Fluid UI

The user interface features a custom, hand-coded frontend built with vanilla web technologies. It utilizes an advanced, multi-layered SVG liquid wave parallax animation system.

Instead of static styling, the frontend reads the incoming JSON response payload from the FastAPI server and dynamically alters the DOM:

* **Green Fluid Theme (Super Score $\ge$ 75):** Visually triggers a high-frequency liquid wave using vibrant, energetic blending profiles to signal a definitive **"Highly Recommended"** status.
* **Amber Fluid Theme (Super Score 50–74):** Morphs the waves into a slow, sweeping amber profile to signal **"Mixed Feedback / Proceed with Caution"**.
* **Red Fluid Theme (Super Score $<$ 50):** Alters the visual states into a deep, slow-bleeding crimson profile to signal a definitive **"Skip It"** verdict.

---

## Tech Stack

* **Backend Framework:** Python 3.10+, FastAPI, Uvicorn
* **Machine Learning Pipeline:** Scikit-Learn, Pandas, NumPy, Joblib
* **Data Ingestion & Extraction:** HTTPX, Requests, Python-Dotenv
* **Frontend Architecture:** HTML5, CSS3 (Grid/Flexbox), JavaScript (Vanilla ES6+), SVG Animation Engines

---

## Supporting Data

The application operates as a real-time data transformer. Below is the strict data dictionary mapping out how properties are managed during a live application lifecycle:

### Data Dictionary Table

| Feature Name     | Technical Description                                           | Data Type         | Data Source / Engine Layer       | Used in Algorithm? |
| :--------------- | :-------------------------------------------------------------- | :---------------- | :------------------------------- | :----------------: |
| `title`        | The verified name of the movie or TV show asset                 | `str`           | TMDB API                         |         ❌         |
| `media_type`   | Schema flag distinguishing between `movie` or `tv` tracking | `str`           | Core Backend Router              |         ❌         |
| `budget`       | Financial production cost recorded in USD                       | `int64`         | TMDB API                         |         ❌         |
| `imdb_rating`  | Original user-voted rating metric spanning 0.0 to 10.0          | `float64`       | OMDb / MDBList Failover          |         ✅         |
| `rt_critic`    | Tomatometer percentage representing official critical score     | `str` / `int` | OMDb / MDBList Failover          |         ✅         |
| `metascore`    | Weighted critical consensus aggregate scoring metric (0-100)    | `int64`         | OMDb / MDBList Failover          |         ✅         |
| `raw_reviews`  | Array of raw, unstructured audience textual reviews             | `list[str]`     | TMDB Reviews Scraper             |         ❌         |
| `ai_sentiment` | Final computed probability percentage of positive text reviews  | `float64`       | Custom Logistic Regression Model |         ✅         |
| `super_score`  | Final weighted mathematical system recommendation score         | `float64`       | CineScore Core Math Engine       |         ✅         |

### Data Sources and Attributions

* **[The Movie Database (TMDB)](https://www.themoviedb.org/):** Sourced for all master metadata schemas, backdrop/poster images, and the textual audience reviews processed by the NLP step.
* **[The Open Movie Database (OMDb)](https://www.omdbapi.com/):** Sourced for structural critical score syncing and IMDb verification indices.
* **[MDBList](https://mdblist.com/):** Utilized as our robust data aggregation failover network for missing streaming parameters and show information.

---

## How to Run Locally

Follow these exact steps to build, secure, and deploy CineScore locally on your machine:

### 1. Clone & Set Up Environment

Clone this repository to your local system and open the project directory in your terminal.

### 2. Configure Dependencies

CineScore requires a clear, isolated Python environment. Run the following command to install the required production and ML libraries:

```bash
pip install -r requirements.txt**
```

### 3. Secure Your API Keys Local Environment

Never push raw API credentials directly into code blocks or upload them to a public GitHub repository.

1. Look at `.env.example` to see the structure.
2. Create a local file in the project root directory and name it exactly `.env`.
3. Open the `.env` file and securely paste your personal developer access keys:

**Plaintext**

```
TMDB_BEARER_TOKEN="your_jwt_bearer_token_here"
OMDB_API_KEY="your_omdb_key_here"
MDBLIST_API_KEY="your_mdblist_key_here"
```

### 4. Boot the Server

Initialize the asynchronous server using Uvicorn:

**Bash**

```
uvicorn app:app --reload
```

Once initialized, open your browser and point it directly to `http://localhost:8000` to interact with the system.

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information. Anyone is free to use, modify, and build upon this code, provided that credit is attributed
