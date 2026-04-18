"""
Sync repos from Postgres to Meilisearch.

Usage:
    python3 meili_sync.py              # bulk sync all repos
    python3 meili_sync.py --configure  # configure index settings only

Env vars:
    DATABASE_URL   - Postgres connection string
    MEILI_URL      - Meilisearch URL (default: http://localhost:7700)
    MEILI_MASTER_KEY - Meilisearch API key
"""

import os
import sys
import json
import urllib.request
import urllib.error

try:
    import psycopg2
    import psycopg2.extras
except ImportError:
    print("ERROR: psycopg2 not installed", file=sys.stderr)
    sys.exit(1)

DATABASE_URL = os.environ.get("DATABASE_URL")
MEILI_URL = os.environ.get("MEILI_URL", "http://localhost:7700")
MEILI_KEY = os.environ.get("MEILI_MASTER_KEY", "")
INDEX_NAME = "repos"


def meili_request(method, path, body=None):
    """Make a request to Meilisearch."""
    url = f"{MEILI_URL}{path}"
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {MEILI_KEY}")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        print(f"Meilisearch error {e.code}: {e.read().decode()}", file=sys.stderr)
        raise


def configure_index():
    """Set up Meilisearch index settings for optimal search."""
    print("Configuring Meilisearch index settings...")

    # Create index if it doesn't exist
    meili_request("POST", "/indexes", {"uid": INDEX_NAME, "primaryKey": "id"})

    # Searchable attributes (order matters for relevance)
    meili_request("PUT", f"/indexes/{INDEX_NAME}/settings/searchable-attributes", [
        "full_name",
        "description",
        "owner",
        "name",
        "language",
    ])

    # Filterable attributes
    meili_request("PUT", f"/indexes/{INDEX_NAME}/settings/filterable-attributes", [
        "has_installers_android",
        "has_installers_windows",
        "has_installers_macos",
        "has_installers_linux",
    ])

    # Sortable attributes
    meili_request("PUT", f"/indexes/{INDEX_NAME}/settings/sortable-attributes", [
        "stars",
        "latest_release_date",
        "trending_score",
        "popularity_score",
    ])

    # Ranking rules
    meili_request("PUT", f"/indexes/{INDEX_NAME}/settings/ranking-rules", [
        "words",
        "typo",
        "proximity",
        "attribute",
        "sort",
        "exactness",
        "stars:desc",
    ])

    # Typo tolerance
    meili_request("PATCH", f"/indexes/{INDEX_NAME}/settings/typo-tolerance", {
        "enabled": True,
        "minWordSizeForTypos": {"oneTypo": 4, "twoTypos": 8},
    })

    print("  Index settings configured.")


def fetch_all_repos():
    """Fetch all repos from Postgres."""
    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT id, full_name, owner, name, owner_avatar_url,
                       description, default_branch, html_url, stars, forks,
                       language, topics, latest_release_date, latest_release_tag,
                       download_count,
                       has_installers_android, has_installers_windows,
                       has_installers_macos, has_installers_linux,
                       trending_score, popularity_score
                FROM repos
            """)
            return cur.fetchall()
    finally:
        conn.close()


def repo_to_meili_doc(row):
    """Convert a Postgres row to a Meilisearch document."""
    return {
        "id": row["id"],
        "full_name": row["full_name"],
        "owner": row["owner"],
        "name": row["name"],
        "owner_avatar_url": row["owner_avatar_url"],
        "description": row["description"],
        "default_branch": row["default_branch"],
        "html_url": row["html_url"],
        "stars": row["stars"],
        "forks": row["forks"],
        "language": row["language"],
        "topics": row["topics"] if row["topics"] else [],
        "latest_release_date": row["latest_release_date"].isoformat() if row["latest_release_date"] else None,
        "latest_release_tag": row["latest_release_tag"],
        "download_count": row["download_count"] or 0,
        "has_installers_android": row["has_installers_android"],
        "has_installers_windows": row["has_installers_windows"],
        "has_installers_macos": row["has_installers_macos"],
        "has_installers_linux": row["has_installers_linux"],
        "trending_score": row["trending_score"],
        "popularity_score": row["popularity_score"],
    }


def bulk_sync():
    """Sync all repos from Postgres to Meilisearch."""
    print("Fetching repos from Postgres...")
    repos = fetch_all_repos()
    print(f"  Found {len(repos)} repos")

    if not repos:
        print("  Nothing to sync.")
        return

    # Convert to Meilisearch documents
    docs = [repo_to_meili_doc(r) for r in repos]

    # Push in batches of 500
    batch_size = 500
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        result = meili_request("POST", f"/indexes/{INDEX_NAME}/documents", batch)
        print(f"  Pushed batch {i // batch_size + 1}: {len(batch)} docs (task: {result.get('taskUid')})")

    print(f"  Sync complete: {len(docs)} repos pushed to Meilisearch")


if __name__ == "__main__":
    if not DATABASE_URL:
        print("ERROR: DATABASE_URL not set", file=sys.stderr)
        sys.exit(1)

    configure_index()

    if "--configure" not in sys.argv:
        bulk_sync()
