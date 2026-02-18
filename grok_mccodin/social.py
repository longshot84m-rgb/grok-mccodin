"""Social integrations — X/Twitter posting, Giphy search."""

from __future__ import annotations

import logging
from pathlib import Path

import requests
from rich.console import Console

from grok_mccodin.config import Config

logger = logging.getLogger(__name__)
console = Console()


# ---------------------------------------------------------------------------
# Giphy
# ---------------------------------------------------------------------------

def search_giphy(query: str, config: Config, limit: int = 5) -> list[dict[str, str]]:
    """Search Giphy and return a list of {title, url} dicts."""
    if not config.has_giphy_key:
        console.print("[yellow]Giphy API key not set. Add GIPHY_API_KEY to .env[/yellow]")
        return []

    url = "https://api.giphy.com/v1/gifs/search"
    params = {"api_key": config.giphy_api_key, "q": query, "limit": limit, "rating": "g"}
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json().get("data", [])
        return [
            {
                "title": g.get("title", ""),
                "url": g["images"]["original"]["url"],
            }
            for g in data
            if "images" in g
        ]
    except requests.RequestException as exc:
        logger.error("Giphy search failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# X / Twitter
# ---------------------------------------------------------------------------

def post_to_x(
    text: str,
    config: Config,
    *,
    media_path: str | None = None,
) -> str:
    """Post a tweet (with optional media) using Tweepy / X API v2.

    Returns the tweet URL on success or an error string.
    """
    if not config.has_x_credentials:
        return "[error] X credentials not configured. Add X_API_KEY, X_API_SECRET, X_ACCESS_TOKEN, X_ACCESS_SECRET to .env"

    try:
        import tweepy  # type: ignore[import-untyped]
    except ImportError:
        return "[error] tweepy not installed — run: pip install tweepy"

    try:
        # Authenticate
        auth = tweepy.OAuth1UserHandler(
            config.x_api_key,
            config.x_api_secret,
            config.x_access_token,
            config.x_access_secret,
        )
        api_v1 = tweepy.API(auth)
        client = tweepy.Client(
            consumer_key=config.x_api_key,
            consumer_secret=config.x_api_secret,
            access_token=config.x_access_token,
            access_token_secret=config.x_access_secret,
        )

        media_ids = []
        if media_path and Path(media_path).is_file():
            media = api_v1.media_upload(filename=media_path)
            media_ids = [media.media_id]
            logger.info("Uploaded media: %s", media.media_id)

        response = client.create_tweet(text=text, media_ids=media_ids or None)
        tweet_id = response.data["id"]
        # Derive tweet URL (username not available from response, but id is enough)
        return f"https://x.com/i/status/{tweet_id}"

    except Exception as exc:
        logger.error("X posting failed: %s", exc)
        return f"[error] {exc}"
