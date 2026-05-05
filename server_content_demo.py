from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from content_based import ContentBasedRecommender, load_movielens


class ContentRecommendationService:
    def __init__(self, data_dir: str, use_tags: bool = True) -> None:
        movies, ratings, tags = load_movielens(data_dir)
        self.recommender = ContentBasedRecommender(use_tags=use_tags)
        self.recommender.fit(movies, ratings, tags)

    def recommend(self, user_id: int, top_n: int) -> list[dict]:
        result = self.recommender.recommend(user_id=user_id, top_n=top_n)
        if result.empty:
            return []

        result = result[["movieId", "title", "genres", "raw_similarity", "content_score"]].copy()
        result["movieId"] = result["movieId"].astype(int)
        result["raw_similarity"] = result["raw_similarity"].astype(float).round(6)
        result["content_score"] = result["content_score"].astype(float).round(6)
        return result.to_dict(orient="records")


def build_handler(service: ContentRecommendationService):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)

            if parsed.path == "/health":
                self._send_json({"status": "ok"})
                return

            if parsed.path != "/recommend/content":
                self._send_json({"error": "not found"}, status=404)
                return

            query = parse_qs(parsed.query)
            try:
                user_id = int(query.get("user_id", ["1"])[0])
                top_n = int(query.get("top_n", ["10"])[0])
            except ValueError:
                self._send_json({"error": "user_id and top_n must be integers"}, status=400)
                return

            if top_n <= 0 or top_n > 100:
                self._send_json({"error": "top_n must be between 1 and 100"}, status=400)
                return

            recommendations = service.recommend(user_id=user_id, top_n=top_n)
            self._send_json(
                {
                    "userId": user_id,
                    "topN": top_n,
                    "count": len(recommendations),
                    "recommendations": recommendations,
                }
            )

        def log_message(self, format: str, *args) -> None:
            return

        def _send_json(self, payload: dict, status: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a simple content recommendation HTTP service.")
    parser.add_argument("--data-dir", default="data/ml-latest-small", help="Directory containing MovieLens csv files.")
    parser.add_argument("--host", default="127.0.0.1", help="Server host.")
    parser.add_argument("--port", type=int, default=8000, help="Server port.")
    parser.add_argument("--no-tags", action="store_true", help="Use genres only, ignoring tags.csv.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    service = ContentRecommendationService(data_dir=args.data_dir, use_tags=not args.no_tags)
    server = ThreadingHTTPServer((args.host, args.port), build_handler(service))

    print(f"Content recommendation service running at http://{args.host}:{args.port}")
    print(f"Try: http://{args.host}:{args.port}/recommend/content?user_id=1&top_n=10")
    server.serve_forever()


if __name__ == "__main__":
    main()
