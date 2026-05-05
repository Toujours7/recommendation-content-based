from __future__ import annotations

import argparse
from pathlib import Path

from content_based import ContentBasedRecommender, load_movielens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the MovieLens content-based recommender demo.")
    parser.add_argument("--data-dir", default="data/ml-latest-small", help="Directory containing movies.csv and ratings.csv.")
    parser.add_argument("--user-id", type=int, default=1, help="MovieLens userId to recommend for.")
    parser.add_argument("--top-n", type=int, default=10, help="Number of recommendations to print.")
    parser.add_argument("--no-tags", action="store_true", help="Use genres only, ignoring tags.csv.")
    parser.add_argument("--output", default="", help="Optional csv path for saving recommendations.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    movies, ratings, tags = load_movielens(args.data_dir)

    recommender = ContentBasedRecommender(use_tags=not args.no_tags)
    recommender.fit(movies, ratings, tags)

    profile_movies = recommender.get_user_profile_movies(args.user_id)
    recommendations = recommender.recommend(args.user_id, top_n=args.top_n)

    print(f"\nUser {args.user_id} profile movies:")
    if profile_movies.empty:
        print("No rating history found for this user.")
    else:
        print(profile_movies.head(10).to_string(index=False))

    print(f"\nTop {args.top_n} content-based recommendations:")
    if recommendations.empty:
        print("No recommendations could be generated.")
    else:
        printable = recommendations[["movieId", "title", "genres", "content_score"]].copy()
        printable["content_score"] = printable["content_score"].round(4)
        print(printable.to_string(index=False))

    if args.output and not recommendations.empty:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        recommendations.to_csv(output_path, index=False)
        print(f"\nSaved recommendations to {output_path}")


if __name__ == "__main__":
    main()
