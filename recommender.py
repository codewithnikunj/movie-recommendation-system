"""
recommender.py
Core AI/ML Recommendation Engine
----------------------------------
Methods:
  - User-Based Collaborative Filtering (Cosine Similarity)
  - Item-Based Collaborative Filtering (Cosine Similarity)
  - Content-Based Filtering (TF-IDF on Genre + Director)
  - SVD Matrix Factorization (Truncated SVD)
  - Hybrid (weighted blend of User-CF + Content-Based)
  - Model Evaluation (RMSE, MAE, Precision@K, Recall@K)
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error


class MovieRecommender:
    def __init__(self, matrix_path="user_movie_matrix.csv",
                 movies_path="movies.csv",
                 sim_path="user_sim_df.csv"):
        # Load data
        self.matrix = pd.read_csv(matrix_path, index_col=0)
        self.movies = pd.read_csv(movies_path)
        self.user_sim = pd.read_csv(sim_path, index_col=0)

        # Pre-compute item similarity matrix
        self._item_sim = None
        # Pre-compute content similarity matrix
        self._content_sim = None
        # SVD model components
        self._svd_pred = None

    # ------------------------------------------------------------------ #
    #  1. USER-BASED COLLABORATIVE FILTERING
    # ------------------------------------------------------------------ #
    def user_cf(self, user_id: str, n: int = 5, min_rating: float = 4.0) -> pd.DataFrame:
        """Return top-N movie recommendations using User-CF."""
        if user_id not in self.user_sim.columns:
            return pd.DataFrame()

        sim_scores = self.user_sim[user_id].drop(user_id).sort_values(ascending=False)
        seen = self._seen_movies(user_id)
        recommendations = {}

        for sim_user, sim_score in sim_scores.items():
            if sim_score <= 0:
                continue
            user_ratings = self.matrix.loc[sim_user]
            high_rated = user_ratings[user_ratings >= min_rating].index.tolist()
            for movie in high_rated:
                if movie not in seen:
                    if movie not in recommendations:
                        recommendations[movie] = {"score": 0.0, "voters": 0}
                    recommendations[movie]["score"] += sim_score * user_ratings[movie]
                    recommendations[movie]["voters"] += 1

        if not recommendations:
            return pd.DataFrame()

        rec_df = pd.DataFrame(recommendations).T.reset_index()
        rec_df.columns = ["MovieID", "score", "voters"]
        rec_df["score"] = rec_df["score"] / rec_df["voters"].clip(lower=1)
        rec_df = rec_df.sort_values("score", ascending=False).head(n)
        return self._enrich(rec_df, "User-CF")

    # ------------------------------------------------------------------ #
    #  2. ITEM-BASED COLLABORATIVE FILTERING
    # ------------------------------------------------------------------ #
    def item_cf(self, user_id: str, n: int = 5) -> pd.DataFrame:
        """Return top-N recommendations using Item-CF."""
        if self._item_sim is None:
            item_matrix = self.matrix.T
            sim = cosine_similarity(item_matrix)
            self._item_sim = pd.DataFrame(sim,
                                          index=item_matrix.index,
                                          columns=item_matrix.index)

        seen = self._seen_movies(user_id)
        user_ratings = self.matrix.loc[user_id]
        rated_movies = user_ratings[user_ratings > 0]

        scores = {}
        for movie in self.matrix.columns:
            if movie in seen:
                continue
            sim_sum = 0.0
            weighted_sum = 0.0
            for rated_movie, rating in rated_movies.items():
                s = self._item_sim.loc[movie, rated_movie]
                if s > 0:
                    weighted_sum += s * rating
                    sim_sum += s
            if sim_sum > 0:
                scores[movie] = weighted_sum / sim_sum

        if not scores:
            return pd.DataFrame()

        rec_df = pd.DataFrame(list(scores.items()), columns=["MovieID", "score"])
        rec_df = rec_df.sort_values("score", ascending=False).head(n)
        return self._enrich(rec_df, "Item-CF")

    # ------------------------------------------------------------------ #
    #  3. CONTENT-BASED FILTERING
    # ------------------------------------------------------------------ #
    def content_based(self, user_id: str, n: int = 5) -> pd.DataFrame:
        """Return top-N recommendations using TF-IDF on Genre + Director."""
        if self._content_sim is None:
            self.movies["content"] = (
                self.movies["Genre"].fillna("") + " " +
                self.movies["Director"].fillna("") + " " +
                self.movies["Year"].astype(str)
            )
            tfidf = TfidfVectorizer(stop_words="english")
            tfidf_matrix = tfidf.fit_transform(self.movies["content"])
            sim = cosine_similarity(tfidf_matrix)
            self._content_sim = pd.DataFrame(sim,
                                             index=self.movies["MovieID"],
                                             columns=self.movies["MovieID"])

        seen = self._seen_movies(user_id)
        user_ratings = self.matrix.loc[user_id]
        liked = user_ratings[user_ratings >= 4].index.tolist()

        scores = {}
        for movie in self.movies["MovieID"]:
            if movie in seen or movie not in self._content_sim.columns:
                continue
            sim_score = 0.0
            count = 0
            for liked_movie in liked:
                if liked_movie in self._content_sim.index:
                    sim_score += self._content_sim.loc[liked_movie, movie]
                    count += 1
            if count > 0:
                scores[movie] = sim_score / count

        if not scores:
            return pd.DataFrame()

        rec_df = pd.DataFrame(list(scores.items()), columns=["MovieID", "score"])
        rec_df = rec_df.sort_values("score", ascending=False).head(n)
        return self._enrich(rec_df, "Content-Based")

    # ------------------------------------------------------------------ #
    #  4. SVD MATRIX FACTORIZATION
    # ------------------------------------------------------------------ #
    def svd_recommend(self, user_id: str, n: int = 5, n_components: int = 20) -> pd.DataFrame:
        """Return top-N recommendations using Truncated SVD."""
        if self._svd_pred is None:
            svd = TruncatedSVD(n_components=n_components, random_state=42)
            matrix_values = self.matrix.values
            latent = svd.fit_transform(matrix_values)
            self._svd_pred = pd.DataFrame(
                np.dot(latent, svd.components_),
                index=self.matrix.index,
                columns=self.matrix.columns
            )

        if user_id not in self._svd_pred.index:
            return pd.DataFrame()

        seen = self._seen_movies(user_id)
        pred_scores = self._svd_pred.loc[user_id]
        unseen = pred_scores[~pred_scores.index.isin(seen)]
        top_n = unseen.sort_values(ascending=False).head(n)

        rec_df = top_n.reset_index()
        rec_df.columns = ["MovieID", "score"]
        return self._enrich(rec_df, "SVD")

    # ------------------------------------------------------------------ #
    #  5. HYBRID RECOMMENDATION
    # ------------------------------------------------------------------ #
    def hybrid(self, user_id: str, n: int = 5,
               cf_weight: float = 0.6, cb_weight: float = 0.4) -> pd.DataFrame:
        """Weighted hybrid of User-CF + Content-Based."""
        cf = self.user_cf(user_id, n=n * 2)
        cb = self.content_based(user_id, n=n * 2)

        if cf.empty and cb.empty:
            return pd.DataFrame()

        scores = {}

        if not cf.empty:
            cf_max = cf["score"].max() or 1
            for _, row in cf.iterrows():
                mid = row["MovieID"]
                scores[mid] = scores.get(mid, 0) + cf_weight * (row["score"] / cf_max)

        if not cb.empty:
            cb_max = cb["score"].max() or 1
            for _, row in cb.iterrows():
                mid = row["MovieID"]
                scores[mid] = scores.get(mid, 0) + cb_weight * (row["score"] / cb_max)

        rec_df = pd.DataFrame(list(scores.items()), columns=["MovieID", "score"])
        rec_df = rec_df.sort_values("score", ascending=False).head(n)
        return self._enrich(rec_df, "Hybrid")

    # ------------------------------------------------------------------ #
    #  6. MODEL EVALUATION
    # ------------------------------------------------------------------ #
    def evaluate(self, test_size: float = 0.2, random_state: int = 42) -> dict:
        """
        Evaluate User-CF using RMSE, MAE, Precision@5, Recall@5.
        Splits known ratings into train/test.
        """
        # Collect all known ratings
        records = []
        for user in self.matrix.index:
            for movie in self.matrix.columns:
                rating = self.matrix.loc[user, movie]
                if rating > 0:
                    records.append((user, movie, rating))

        if len(records) < 10:
            return {}

        df = pd.DataFrame(records, columns=["user", "movie", "rating"])
        train_df, test_df = train_test_split(df, test_size=test_size,
                                             random_state=random_state)

        # Build train matrix
        train_matrix = self.matrix.copy() * 0
        for _, row in train_df.iterrows():
            train_matrix.loc[row["user"], row["movie"]] = row["rating"]

        # Predict using mean of similar users (simple CF prediction)
        y_true, y_pred = [], []
        for _, row in test_df.iterrows():
            user, movie, true_rating = row["user"], row["movie"], row["rating"]
            if user not in self.user_sim.index or movie not in train_matrix.columns:
                continue
            sim_scores = self.user_sim[user].drop(user)
            sim_scores = sim_scores[sim_scores > 0].sort_values(ascending=False).head(10)
            num, den = 0.0, 0.0
            for sim_user, sim in sim_scores.items():
                r = train_matrix.loc[sim_user, movie]
                if r > 0:
                    num += sim * r
                    den += sim
            if den > 0:
                y_pred.append(num / den)
                y_true.append(true_rating)

        if not y_true:
            return {}

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)

        # Precision@5 and Recall@5
        threshold = 4.0
        k = 5
        precisions, recalls = [], []
        for user in test_df["user"].unique():
            relevant = set(test_df[(test_df["user"] == user) &
                                   (test_df["rating"] >= threshold)]["movie"])
            recs = self.user_cf(user, n=k)
            if recs.empty:
                continue
            recommended = set(recs["MovieID"])
            hits = len(recommended & relevant)
            precisions.append(hits / k)
            recalls.append(hits / len(relevant) if relevant else 0)

        return {
            "RMSE": float(round(rmse, 4)),
            "MAE": float(round(mae, 4)),
            "Precision@5": float(round(np.mean(precisions), 4)) if precisions else 0.0,
            "Recall@5": float(round(np.mean(recalls), 4)) if recalls else 0.0,
            "Test Samples": len(y_true),
        }

    # ------------------------------------------------------------------ #
    #  HELPERS
    # ------------------------------------------------------------------ #
    def _seen_movies(self, user_id: str) -> list:
        """Return list of movies already rated by user."""
        if user_id not in self.matrix.index:
            return []
        return self.matrix.loc[user_id][self.matrix.loc[user_id] > 0].index.tolist()

    def _enrich(self, rec_df: pd.DataFrame, method: str) -> pd.DataFrame:
        """Merge movie metadata into recommendation dataframe."""
        merged = rec_df.merge(self.movies, on="MovieID", how="left")
        merged["Method"] = method
        merged["Confidence"] = (merged["score"] / merged["score"].max() * 100).round(1)
        return merged

    def get_user_history(self, user_id: str) -> pd.DataFrame:
        """Return movies rated by the user with metadata."""
        if user_id not in self.matrix.index:
            return pd.DataFrame()
        rated = self.matrix.loc[user_id]
        rated = rated[rated > 0].reset_index()
        rated.columns = ["MovieID", "Rating"]
        return rated.merge(self.movies, on="MovieID", how="left").sort_values(
            "Rating", ascending=False)

    def get_top_movies(self, n: int = 10) -> pd.DataFrame:
        """Return globally most-rated movies."""
        avg_ratings = self.matrix.replace(0, np.nan).mean()
        count_ratings = (self.matrix > 0).sum()
        top_df = pd.DataFrame({
            "MovieID": avg_ratings.index,
            "AvgRating": avg_ratings.values.round(2),
            "NumRatings": count_ratings.values
        }).sort_values("AvgRating", ascending=False).head(n)
        return top_df.merge(self.movies, on="MovieID", how="left")

    def get_genre_stats(self) -> pd.DataFrame:
        """Return genre distribution from the movies dataset."""
        return self.movies["Genre"].value_counts().reset_index()

    def get_rating_distribution(self) -> pd.Series:
        """Return distribution of all non-zero ratings."""
        flat = self.matrix.values.flatten()
        return pd.Series(flat[flat > 0])

    def get_similar_users(self, user_id: str, n: int = 5) -> pd.DataFrame:
        """Return top-N most similar users."""
        if user_id not in self.user_sim.columns:
            return pd.DataFrame()
        sim = self.user_sim[user_id].drop(user_id).sort_values(ascending=False).head(n)
        return pd.DataFrame({"UserID": sim.index, "Similarity": sim.values.round(4)})


if __name__ == "__main__":
    rec = MovieRecommender()
    print("=== User-CF ===")
    print(rec.user_cf("U001", n=5)[["MovieID", "Title", "Genre", "score"]].to_string(index=False))
    print("\n=== Item-CF ===")
    print(rec.item_cf("U001", n=5)[["MovieID", "Title", "Genre", "score"]].to_string(index=False))
    print("\n=== Content-Based ===")
    print(rec.content_based("U001", n=5)[["MovieID", "Title", "Genre", "score"]].to_string(index=False))
    print("\n=== SVD ===")
    print(rec.svd_recommend("U001", n=5)[["MovieID", "Title", "Genre", "score"]].to_string(index=False))
    print("\n=== Hybrid ===")
    print(rec.hybrid("U001", n=5)[["MovieID", "Title", "Genre", "score"]].to_string(index=False))
    print("\n=== Evaluation ===")
    print(rec.evaluate())
