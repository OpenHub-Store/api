"""
Backfill download_count for all repos in the database.

Paginates /releases?per_page=100 following the Link: next header so the sum
is accurate for repos with more than 100 releases. Rotates across the four
GH_TOKEN_* tokens (same ones run_fetcher.sh uses) so the backfill finishes
in ~20-30 min instead of ~2.5 hours on a single token.
"""

import os
import sys
import json
import re
import time
import itertools
import urllib.request
import urllib.error

try:
    import psycopg2
    import psycopg2.extras
except ImportError:
    print("ERROR: psycopg2 not installed")
    sys.exit(1)

DATABASE_URL = os.environ.get("DATABASE_URL")
MEILI_URL = os.environ.get("MEILI_URL")
MEILI_KEY = os.environ.get("MEILI_MASTER_KEY")

# Round-robin the same pool of tokens run_fetcher.sh already uses. Fall back
# to GITHUB_TOKEN (single) if the pool env vars aren't set.
GH_TOKENS = [
    t for t in (
        os.environ.get("GH_TOKEN_TRENDING"),
        os.environ.get("GH_TOKEN_NEW_RELEASES"),
        os.environ.get("GH_TOKEN_MOST_POPULAR"),
        os.environ.get("GH_TOKEN_TOPICS"),
    ) if t
]
if not GH_TOKENS:
    solo = os.environ.get("GITHUB_TOKEN")
    if solo:
        GH_TOKENS = [solo]

_token_cycle = itertools.cycle(GH_TOKENS) if GH_TOKENS else None

LINK_NEXT_RE = re.compile(r'<([^>]+)>;\s*rel="next"')


def parse_next_link(link_header):
    """Return the rel=next URL from a Link header, or None."""
    if not link_header:
        return None
    m = LINK_NEXT_RE.search(link_header)
    return m.group(1) if m else None


def github_request(url):
    """Single GET with rate-limit awareness. Returns (body, next_url) or (None, None)."""
    req = urllib.request.Request(url)
    req.add_header("Accept", "application/vnd.github+json")
    if _token_cycle:
        req.add_header("Authorization", f"token {next(_token_cycle)}")
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            remaining = resp.headers.get("X-RateLimit-Remaining", "?")
            if remaining != "?" and int(remaining) < 5:
                reset = int(resp.headers.get("X-RateLimit-Reset", "0"))
                wait = max(reset - int(time.time()), 1)
                print(f"  Token low ({remaining} left), sleeping {wait}s...")
                time.sleep(wait + 1)
            body = json.loads(resp.read())
            next_url = parse_next_link(resp.headers.get("Link"))
            return body, next_url
    except urllib.error.HTTPError as e:
        if e.code == 403:
            reset = int(e.headers.get("X-RateLimit-Reset", "0"))
            wait = max(reset - int(time.time()), 10)
            print(f"  Rate limited, sleeping {wait}s...")
            time.sleep(wait + 1)
            return github_request(url)
        return None, None
    except Exception:
        return None, None


def get_total_downloads(full_name):
    """Sum download_count across every asset of every release, following pagination."""
    url = f"https://api.github.com/repos/{full_name}/releases?per_page=100"
    total = 0
    pages = 0
    while url:
        releases, next_url = github_request(url)
        if not releases or not isinstance(releases, list):
            break
        pages += 1
        for release in releases:
            for asset in release.get("assets", []):
                total += asset.get("download_count", 0) or 0
        url = next_url
    return total, pages


def backfill():
    conn = psycopg2.connect(DATABASE_URL)
    meili_updates = []

    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT id, full_name FROM repos ORDER BY stars DESC")
            repos = cur.fetchall()

        print(f"Backfilling download counts for {len(repos)} repos "
              f"(tokens: {len(GH_TOKENS)})...")
        updated = 0
        total_pages = 0

        for repo in repos:
            downloads, pages = get_total_downloads(repo["full_name"])
            total_pages += pages

            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE repos SET download_count = %s WHERE id = %s",
                        (downloads, repo["id"])
                    )

            meili_updates.append({"id": repo["id"], "download_count": downloads})

            updated += 1
            if updated % 25 == 0:
                print(f"  {updated}/{len(repos)} repos updated "
                      f"({total_pages} pages fetched so far)...")
                if meili_updates:
                    meili_sync(meili_updates)
                    meili_updates = []

            time.sleep(0.3)

        if meili_updates:
            meili_sync(meili_updates)

        print(f"\nDone! Updated {updated} repos with download counts "
              f"({total_pages} total pages fetched).")

    finally:
        conn.close()


def meili_sync(docs):
    url = f"{MEILI_URL}/indexes/repos/documents"
    data = json.dumps(docs).encode()
    req = urllib.request.Request(url, data=data, method="PUT")
    req.add_header("Authorization", f"Bearer {MEILI_KEY}")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=30):
            pass
    except Exception as e:
        print(f"  Meili sync error: {e}")


if __name__ == "__main__":
    if not DATABASE_URL or not GH_TOKENS:
        print("ERROR: DATABASE_URL and at least one GH_TOKEN_* required")
        sys.exit(1)
    backfill()
