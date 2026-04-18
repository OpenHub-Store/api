"""
Write fetcher output to Postgres (github-store-backend database).

Requires: pip install psycopg2-binary
Env var: DATABASE_URL (e.g. postgresql://githubstore:pass@89.167.115.83:5432/githubstore)

Usage from fetch_all_categories.py:
    from db_writer import save_to_postgres
    save_to_postgres(category, platform, repos)
"""

import os
import sys
from typing import List, Dict, Optional

try:
    import psycopg2
    import psycopg2.extras
except ImportError:
    psycopg2 = None

DATABASE_URL = os.environ.get("DATABASE_URL")


def _get_connection():
    if psycopg2 is None:
        print("  ⚠ psycopg2 not installed — skipping Postgres write", file=sys.stderr)
        return None
    if not DATABASE_URL:
        print("  ⚠ DATABASE_URL not set — skipping Postgres write", file=sys.stderr)
        return None
    return psycopg2.connect(DATABASE_URL)


def save_to_postgres(category: str, platform: str, repos: List[Dict]):
    """UPSERT repos into Postgres and update category/topic rankings."""
    conn = _get_connection()
    if conn is None:
        return

    try:
        with conn:
            with conn.cursor() as cur:
                # Upsert each repo
                for repo in repos:
                    _upsert_repo(cur, repo, platform)

                # Update category or topic rankings
                if category in ("trending", "new-releases", "most-popular"):
                    _update_categories(cur, category, platform, repos)
                else:
                    # Topic bucket (privacy, media, etc.) — strip "topics/" prefix
                    bucket = category.removeprefix("topics/")
                    _update_topic_bucket(cur, bucket, platform, repos)

        print(f"  ✓ Postgres: upserted {len(repos)} repos for {category}/{platform}")
    except Exception as e:
        print(f"  ✗ Postgres error: {e}", file=sys.stderr)
    finally:
        conn.close()


def _upsert_repo(cur, repo: Dict, platform: str):
    """UPSERT a single repo into the repos table."""
    owner = repo.get("owner", {})
    repo_id = repo["id"]

    # Determine platform installer flags
    platform_flags = {
        "has_installers_android": platform == "android",
        "has_installers_windows": platform == "windows",
        "has_installers_macos": platform == "macos",
        "has_installers_linux": platform == "linux",
    }

    cur.execute("""
        INSERT INTO repos (
            id, full_name, owner, name, owner_avatar_url, description,
            default_branch, html_url, stars, forks, language,
            topics, latest_release_date, latest_release_tag,
            has_installers_android, has_installers_windows,
            has_installers_macos, has_installers_linux,
            download_count, trending_score, popularity_score,
            created_at_gh, updated_at_gh, indexed_at
        ) VALUES (
            %(id)s, %(full_name)s, %(owner)s, %(name)s, %(avatar)s, %(description)s,
            %(default_branch)s, %(html_url)s, %(stars)s, %(forks)s, %(language)s,
            %(topics)s, %(release_date)s, NULL,
            %(android)s, %(windows)s, %(macos)s, %(linux)s,
            %(download_count)s, %(trending_score)s, %(popularity_score)s,
            %(created_at)s, %(updated_at)s, NOW()
        )
        ON CONFLICT (id) DO UPDATE SET
            full_name = EXCLUDED.full_name,
            owner = EXCLUDED.owner,
            name = EXCLUDED.name,
            owner_avatar_url = EXCLUDED.owner_avatar_url,
            description = EXCLUDED.description,
            default_branch = EXCLUDED.default_branch,
            html_url = EXCLUDED.html_url,
            stars = EXCLUDED.stars,
            forks = EXCLUDED.forks,
            language = EXCLUDED.language,
            topics = EXCLUDED.topics,
            latest_release_date = COALESCE(EXCLUDED.latest_release_date, repos.latest_release_date),
            trending_score = COALESCE(EXCLUDED.trending_score, repos.trending_score),
            popularity_score = COALESCE(EXCLUDED.popularity_score, repos.popularity_score),
            updated_at_gh = EXCLUDED.updated_at_gh,
            indexed_at = NOW(),
            has_installers_android = repos.has_installers_android OR EXCLUDED.has_installers_android,
            has_installers_windows = repos.has_installers_windows OR EXCLUDED.has_installers_windows,
            has_installers_macos = repos.has_installers_macos OR EXCLUDED.has_installers_macos,
            has_installers_linux = repos.has_installers_linux OR EXCLUDED.has_installers_linux,
            download_count = GREATEST(EXCLUDED.download_count, repos.download_count)
    """, {
        "id": repo_id,
        "full_name": repo.get("fullName"),
        "owner": owner.get("login"),
        "name": repo.get("name"),
        "avatar": owner.get("avatarUrl"),
        "description": repo.get("description"),
        "default_branch": repo.get("defaultBranch"),
        "html_url": repo.get("htmlUrl"),
        "stars": repo.get("stargazersCount", 0),
        "forks": repo.get("forksCount", 0),
        "language": repo.get("language"),
        "topics": repo.get("topics", []),
        "release_date": repo.get("latestReleaseDate"),
        "download_count": repo.get("downloadCount", 0),
        "android": platform_flags["has_installers_android"],
        "windows": platform_flags["has_installers_windows"],
        "macos": platform_flags["has_installers_macos"],
        "linux": platform_flags["has_installers_linux"],
        "trending_score": repo.get("trendingScore"),
        "popularity_score": repo.get("popularityScore"),
        "created_at": repo.get("createdAt"),
        "updated_at": repo.get("updatedAt"),
    })


def _update_categories(cur, category: str, platform: str, repos: List[Dict]):
    """Replace category rankings for this category+platform."""
    # Delete old rankings
    cur.execute(
        "DELETE FROM repo_categories WHERE category = %s AND platform = %s",
        (category, platform)
    )

    # Insert new rankings
    for rank, repo in enumerate(repos, start=1):
        cur.execute(
            "INSERT INTO repo_categories (repo_id, category, platform, rank) VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING",
            (repo["id"], category, platform, rank)
        )


def _update_topic_bucket(cur, bucket: str, platform: str, repos: List[Dict]):
    """Replace topic bucket rankings for this bucket+platform."""
    cur.execute(
        "DELETE FROM repo_topic_buckets WHERE bucket = %s AND platform = %s",
        (bucket, platform)
    )

    for rank, repo in enumerate(repos, start=1):
        cur.execute(
            "INSERT INTO repo_topic_buckets (repo_id, bucket, platform, rank) VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING",
            (repo["id"], bucket, platform, rank)
        )
