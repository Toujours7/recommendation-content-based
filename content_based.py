from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


REQUIRED_MOVIE_COLUMNS = {"movieId", "title", "genres"}
REQUIRED_RATING_COLUMNS = {"userId", "movieId", "rating"}


def load_movielens(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """Load MovieLens csv files from data_dir."""
    data_path = Path(data_dir)
    movies_path = data_path / "movies.csv"
    ratings_path = data_path / "ratings.csv"
    tags_path = data_path / "tags.csv"

    if not movies_path.exists() or not ratings_path.exists():
        raise FileNotFoundError(
            "movies.csv and ratings.csv are required. "
            f"Checked directory: {data_path.resolve()}"
        )

    movies = pd.read_csv(movies_path)
    ratings = pd.read_csv(ratings_path)
    tags = pd.read_csv(tags_path) if tags_path.exists() else None
    return movies, ratings, tags


class ContentBasedRecommender:
    """Content-based MovieLens recommender using TF-IDF over genres and tags."""

    def __init__(
        self,
        *,
        min_like_rating: float = 4.0,
        fallback_profile_size: int = 5,
        use_tags: bool = True,
        stop_words: str | None = "english",
    ) -> None:
        self.min_like_rating = min_like_rating
        self.fallback_profile_size = fallback_profile_size
        self.use_tags = use_tags
        self.stop_words = stop_words

        self.movies_: Optional[pd.DataFrame] = None
        self.ratings_: Optional[pd.DataFrame] = None
        self.movie_matrix_: Optional[csr_matrix] = None
        self.movie_index_: Optional[pd.Series] = None
        self.vectorizer_: Optional[TfidfVectorizer] = None

    def fit(
        self,
        movies: pd.DataFrame,
        ratings: pd.DataFrame,
        tags: Optional[pd.DataFrame] = None,
    ) -> "ContentBasedRecommender":
        """Fit TF-IDF movie features and keep user rating history."""
        self._validate_input(movies, ratings)

        movies_content = self._build_movie_content(movies, tags)
        vectorizer = TfidfVectorizer(stop_words=self.stop_words)
        movie_matrix = vectorizer.fit_transform(movies_content["content"])

        self.movies_ = movies_content.reset_index(drop=True)
        self.ratings_ = ratings.copy()
        self.movie_matrix_ = movie_matrix
        self.movie_index_ = pd.Series(self.movies_.index, index=self.movies_["movieId"])
        self.vectorizer_ = vectorizer
        return self

    def recommend(
        self,
        user_id: int,
        *,
        top_n: int = 10,
        include_watched: bool = False,
    ) -> pd.DataFrame:
        """Return Top-N content-based recommendations for one user."""
        scored = self.score_movies(user_id, include_watched=include_watched)
        return scored.head(top_n).reset_index(drop=True)

    def score_movies(
        self,
        user_id: int,
        *,
        include_watched: bool = False,
        normalize_scores: bool = True,
    ) -> pd.DataFrame:
        """Score movies for one user so the result can be used by a hybrid model."""
        self._check_is_fitted()

        assert self.movies_ is not None
        assert self.ratings_ is not None
        assert self.movie_matrix_ is not None

        user_ratings = self.ratings_[self.ratings_["userId"] == user_id]
        if user_ratings.empty:
            return pd.DataFrame(columns=["movieId", "title", "genres", "content_score"])

        profile = self._build_user_profile(user_ratings)
        if profile is None:
            return pd.DataFrame(columns=["movieId", "title", "genres", "content_score"])

        scores = cosine_similarity(profile, self.movie_matrix_).ravel()
        result = self.movies_[["movieId", "title", "genres"]].copy()
        result["raw_similarity"] = scores

        if not include_watched:
            watched_movie_ids = set(user_ratings["movieId"])
            result = result[~result["movieId"].isin(watched_movie_ids)]

        if normalize_scores:
            result["content_score"] = _minmax_scale(result["raw_similarity"].to_numpy())
        else:
            result["content_score"] = result["raw_similarity"]

        return (
            result.sort_values(["content_score", "raw_similarity", "title"], ascending=[False, False, True])
            .reset_index(drop=True)
        )

    def get_user_profile_movies(self, user_id: int) -> pd.DataFrame:
        """Return movies used to build the user's content profile."""
        self._check_is_fitted()

        assert self.movies_ is not None
        assert self.ratings_ is not None

        user_ratings = self.ratings_[self.ratings_["userId"] == user_id]
        liked = self._select_profile_ratings(user_ratings)
        if liked.empty:
            return pd.DataFrame(columns=["movieId", "title", "genres", "rating"])

        return (
            liked.merge(self.movies_[["movieId", "title", "genres"]], on="movieId", how="inner")
            [["movieId", "title", "genres", "rating"]]
            .sort_values("rating", ascending=False)
            .reset_index(drop=True)
        )

    def _build_movie_content(
        self,
        movies: pd.DataFrame,
        tags: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        movies_content = movies.copy()
        movies_content["genres_text"] = (
            movies_content["genres"]
            .fillna("")
            .str.replace("(no genres listed)", "", regex=False)
            .str.replace("|", " ", regex=False)
        )

        if self.use_tags and tags is not None and {"movieId", "tag"}.issubset(tags.columns):
            tag_text = (
                tags.dropna(subset=["tag"])
                .assign(tag=lambda df: df["tag"].astype(str).str.lower())
                .groupby("movieId")["tag"]
                .apply(lambda values: " ".join(values))
                .reset_index(name="tag_text")
            )
            movies_content = movies_content.merge(tag_text, on="movieId", how="left")
            movies_content["tag_text"] = movies_content["tag_text"].fillna("")
        else:
            movies_content["tag_text"] = ""

        movies_content["content"] = (
            movies_content["genres_text"].fillna("") + " " + movies_content["tag_text"].fillna("")
        ).str.strip()
        movies_content.loc[movies_content["content"] == "", "content"] = "unknown"
        return movies_content

    def _build_user_profile(self, user_ratings: pd.DataFrame) -> Optional[csr_matrix]:
        assert self.movie_matrix_ is not None
        assert self.movie_index_ is not None

        profile_ratings = self._select_profile_ratings(user_ratings)
        if profile_ratings.empty:
            return None

        indices: list[int] = []
        weights: list[float] = []
        for row in profile_ratings.itertuples(index=False):
            movie_id = getattr(row, "movieId")
            rating = float(getattr(row, "rating"))
            if movie_id in self.movie_index_:
                indices.append(int(self.movie_index_[movie_id]))
                weights.append(rating)

        if not indices:
            return None

        selected_matrix = self.movie_matrix_[indices]
        weight_array = np.asarray(weights, dtype=float)
        weighted_sum = selected_matrix.multiply(weight_array[:, None]).sum(axis=0)
        profile = csr_matrix(weighted_sum / weight_array.sum())
        return normalize(profile)

    def _select_profile_ratings(self, user_ratings: pd.DataFrame) -> pd.DataFrame:
        liked = user_ratings[user_ratings["rating"] >= self.min_like_rating]
        if not liked.empty:
            return liked.sort_values("rating", ascending=False)

        return user_ratings.sort_values("rating", ascending=False).head(self.fallback_profile_size)

    def _check_is_fitted(self) -> None:
        if self.movies_ is None or self.ratings_ is None or self.movie_matrix_ is None:
            raise RuntimeError("Call fit(movies, ratings, tags) before recommending.")

    @staticmethod
    def _validate_input(movies: pd.DataFrame, ratings: pd.DataFrame) -> None:
        missing_movies = REQUIRED_MOVIE_COLUMNS - set(movies.columns)
        missing_ratings = REQUIRED_RATING_COLUMNS - set(ratings.columns)
        if missing_movies:
            raise ValueError(f"movies is missing columns: {sorted(missing_movies)}")
        if missing_ratings:
            raise ValueError(f"ratings is missing columns: {sorted(missing_ratings)}")


def _minmax_scale(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values

    min_value = values.min()
    max_value = values.max()
    if np.isclose(min_value, max_value):
        return np.ones_like(values, dtype=float)

    return (values - min_value) / (max_value - min_value)
