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
import socket
import threading
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed

# urllib's per-call timeout argument doesn't always cover DNS / TCP handshake
# edge cases. A process-wide default guarantees no call ever hangs forever.
socket.setdefaulttimeout(25)

# Hard cap on pages per repo so one pathological repo can't stall the backfill.
# 30 pages = 3000 releases — safely above any app-store-relevant megaproject.
MAX_PAGES_PER_REPO = 30

# Per-worker thread-local storage (Postgres connection + assigned token).
_local = threading.local()

# Shared state guarded by this lock (progress counters + Meili batch).
_lock = threading.Lock()
_updated = 0
_total_pages = 0
_meili_batch = []

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

LINK_NEXT_RE = re.compile(r'<([^>]+)>;\s*rel="next"')


def parse_next_link(link_header):
    """Return the rel=next URL from a Link header, or None."""
    if not link_header:
        return None
    m = LINK_NEXT_RE.search(link_header)
    return m.group(1) if m else None


def github_request(url, token):
    """Single GET with rate-limit awareness. Returns (body, next_url) or (None, None).

    Token is the per-worker token, not a rotation — so one worker hitting its
    token's limit sleeps that worker only, the other three keep going.
    """
    req = urllib.request.Request(url)
    req.add_header("Accept", "application/vnd.github+json")
    if token:
        req.add_header("Authorization", f"token {token}")
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            remaining = resp.headers.get("X-RateLimit-Remaining", "?")
            if remaining != "?" and int(remaining) < 5:
                reset = int(resp.headers.get("X-RateLimit-Reset", "0"))
                wait = max(reset - int(time.time()), 1)
                print(f"  [worker {threading.current_thread().name}] token low "
                      f"({remaining} left), sleeping {wait}s...", flush=True)
                time.sleep(wait + 1)
            body = json.loads(resp.read())
            next_url = parse_next_link(resp.headers.get("Link"))
            return body, next_url
    except urllib.error.HTTPError as e:
        if e.code == 403:
            reset = int(e.headers.get("X-RateLimit-Reset", "0"))
            wait = max(reset - int(time.time()), 10)
            print(f"  [worker {threading.current_thread().name}] 403 rate-limited, "
                  f"sleeping {wait}s...", flush=True)
            time.sleep(wait + 1)
            return github_request(url, token)
        return None, None
    except Exception:
        return None, None


def get_total_downloads(full_name, token):
    """Sum download_count across every asset of every release, following pagination."""
    url = f"https://api.github.com/repos/{full_name}/releases?per_page=100"
    total = 0
    pages = 0
    while url and pages < MAX_PAGES_PER_REPO:
        releases, next_url = github_request(url, token)
        if not releases or not isinstance(releases, list):
            break
        pages += 1
        for release in releases:
            for asset in release.get("assets", []):
                total += asset.get("download_count", 0) or 0
        url = next_url
    return total, pages


def _init_worker(worker_idx):
    """Give this thread its dedicated token + Postgres connection."""
    _local.token = GH_TOKENS[worker_idx % len(GH_TOKENS)]
    _local.conn = psycopg2.connect(DATABASE_URL, connect_timeout=10)
    with _local.conn.cursor() as cur:
        cur.execute("SET statement_timeout = '10s'")
        cur.execute("SET idle_in_transaction_session_timeout = '30s'")
    _local.conn.commit()


def _close_worker():
    if hasattr(_local, "conn"):
        _local.conn.close()


def _process_repo(repo, total_repos):
    global _updated, _total_pages, _meili_batch
    t0 = time.time()
    downloads, pages = get_total_downloads(repo["full_name"], _local.token)
    elapsed = time.time() - t0

    with _local.conn:
        with _local.conn.cursor() as cur:
            cur.execute(
                "UPDATE repos SET download_count = %s WHERE id = %s",
                (downloads, repo["id"])
            )

    # Shared progress + Meili batch under lock.
    flush = None
    with _lock:
        _updated += 1
        _total_pages += pages
        _meili_batch.append({"id": repo["id"], "download_count": downloads})
        current = _updated
        pages_so_far = _total_pages
        if current % 25 == 0:
            flush = _meili_batch
            _meili_batch = []

    if elapsed > 5.0:
        print(f"  [slow] {repo['full_name']}: {pages} pages, {elapsed:.1f}s, "
              f"{downloads:,} downloads", flush=True)

    if current % 25 == 0:
        print(f"  {current}/{total_repos} repos updated "
              f"({pages_so_far} pages fetched so far)...", flush=True)
        if flush:
            meili_sync(flush)

    # Gentle pacing per worker — with 4 workers this yields ~13 req/s aggregate,
    # well under GitHub's secondary abuse threshold.
    time.sleep(0.3)


def backfill():
    # Workers each own a token + PG connection. Fewer workers than tokens is
    # fine (leaves the extras unused); more workers than tokens would share.
    worker_count = len(GH_TOKENS)

    # Load the repo list on the main thread using a short-lived connection.
    bootstrap = psycopg2.connect(DATABASE_URL, connect_timeout=10)
    try:
        with bootstrap.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT id, full_name FROM repos ORDER BY stars DESC")
            repos = cur.fetchall()
    finally:
        bootstrap.close()

    print(f"Backfilling download counts for {len(repos)} repos "
          f"({worker_count} parallel workers, 1 token each)...", flush=True)

    # ThreadPoolExecutor with a deterministic initializer so each worker
    # gets exactly one token by its index in the pool.
    worker_idx_counter = {"i": 0}
    idx_lock = threading.Lock()

    def init():
        with idx_lock:
            i = worker_idx_counter["i"]
            worker_idx_counter["i"] += 1
        _init_worker(i)

    with ThreadPoolExecutor(max_workers=worker_count, initializer=init) as ex:
        futures = [ex.submit(_process_repo, r, len(repos)) for r in repos]
        for f in as_completed(futures):
            # Surface any exception so it doesn't silently swallow failures.
            try:
                f.result()
            except Exception as e:
                print(f"  [error] worker failed: {e}", flush=True)

    # Final Meili flush for whatever's left in the shared batch.
    with _lock:
        if _meili_batch:
            final = _meili_batch
            _meili_batch = []
        else:
            final = None
    if final:
        meili_sync(final)

    print(f"\nDone! Updated {_updated} repos with download counts "
          f"({_total_pages} total pages fetched).", flush=True)


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
