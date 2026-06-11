"""
Publish the live discovery feed to static JSON for the client's offline
waterfall.

The client falls back to raw.githubusercontent.com/OpenHub-Store/api when the
backend is unreachable. The category/topic endpoints already have cached
mirrors under cached-data/; this does the same for GET /v1/feed so a
backend-down user still gets a populated feed-first home instead of an empty
screen.

Stdlib only — this just curls one already-assembled, anonymous endpoint
(no GitHub API, no tokens, no rate limits). Output matches the
CachedRepoResponse envelope the client decodes for every other cached file:
  { category, platform, lastUpdated, totalCount, repositories: [...] }
so the existing fallback reader needs zero new code — it reads
feed/<platform>.json exactly like trending/<platform>.json.
"""

import json
import os
import sys
import urllib.request
import urllib.error
from datetime import datetime, timezone

BACKEND = os.environ.get("FEED_BACKEND_ORIGIN", "https://api.github-store.org")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "cached-data", "feed")

# "all" → no platform param (all-platform mix). The four named platforms map
# to the ?platform= filter. File name mirrors the key the client uses.
PLATFORMS = {
    "all": None,
    "android": "android",
    "windows": "windows",
    "macos": "macos",
    "linux": "linux",
}

# Offline depth: 3 pages × 50 = up to 150 items per platform — enough for a
# few screens of scroll before the user is likely back online. The live feed
# holds up to 500, so this is a head slice, not the whole thing.
PAGES = 3
LIMIT = 50
TIMEOUT_S = 20


def fetch_page(platform_param, page):
    url = f"{BACKEND}/v1/feed?page={page}&limit={LIMIT}"
    if platform_param:
        url += f"&platform={platform_param}"
    req = urllib.request.Request(url, headers={"User-Agent": "feed-cache-publisher/1.0"})
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
        if resp.status != 200:
            raise RuntimeError(f"{url} → HTTP {resp.status}")
        return json.loads(resp.read().decode("utf-8"))


def build_platform(name, platform_param):
    items = []
    rotation = None
    for page in range(1, PAGES + 1):
        body = fetch_page(platform_param, page)
        rotation = body.get("rotation")
        items.extend(body.get("items", []))
        if not body.get("hasMore"):
            break
    return {
        "category": "feed",
        "platform": name,
        "lastUpdated": datetime.now(timezone.utc).isoformat(),
        "rotation": rotation,
        "totalCount": len(items),
        "repositories": items,
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    failures = 0
    for name, param in PLATFORMS.items():
        try:
            payload = build_platform(name, param)
            path = os.path.join(OUT_DIR, f"{name}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            print(f"wrote {path}: {payload['totalCount']} repos (rotation={payload['rotation']})")
        except (urllib.error.URLError, RuntimeError, ValueError) as e:
            # One platform failing must not blank the others. Leave the
            # previous committed file in place (stale is better than empty)
            # and surface a non-zero exit so the workflow logs the miss.
            print(f"ERROR building feed/{name}.json: {e}", file=sys.stderr)
            failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
