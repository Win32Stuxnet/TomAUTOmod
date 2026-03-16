from __future__ import annotations

import logging

import praw

from ..models import Sample

log = logging.getLogger(__name__)


def scrape(config: dict, subreddits: list[str] | None = None, limit: int | None = None, sort: str | None = None) -> list[Sample]:
    reddit_cfg = config.get("reddit", {})

    reddit = praw.Reddit(
        client_id=reddit_cfg["client_id"],
        client_secret=reddit_cfg["client_secret"],
        user_agent=reddit_cfg["user_agent"],
    )

    subs = subreddits or reddit_cfg.get("subreddits", [])
    max_comments = limit or reddit_cfg.get("limit", 1000)
    sort_mode = sort or reddit_cfg.get("sort", "controversial")

    if not subs:
        log.error("No subreddits specified. Use --subreddits or set in config.yaml")
        return []

    samples: list[Sample] = []

    for sub_name in subs:
        log.info("Scraping r/%s (%s, limit=%d)", sub_name, sort_mode, max_comments)
        subreddit = reddit.subreddit(sub_name)

        getter = getattr(subreddit, sort_mode, subreddit.controversial)
        submissions = getter(limit=None)

        count = 0
        for submission in submissions:
            if count >= max_comments:
                break

            submission.comments.replace_more(limit=0)
            for comment in submission.comments.list():
                if count >= max_comments:
                    break

                body = comment.body.strip()
                if not body or body in ("[deleted]", "[removed]"):
                    continue

                is_removed = body == "[removed]" or getattr(comment, "removed", False)

                samples.append(Sample(
                    text=body,
                    label="toxic" if is_removed else None,
                    source="reddit",
                    metadata={
                        "subreddit": sub_name,
                        "score": comment.score,
                        "author": str(comment.author) if comment.author else None,
                        "created_utc": comment.created_utc,
                        "submission_title": submission.title,
                        "is_removed": is_removed,
                    },
                ))
                count += 1

        log.info("Scraped %d comments from r/%s", count, sub_name)

    log.info("Total scraped: %d samples", len(samples))
    return samples
