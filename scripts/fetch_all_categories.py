"""
Fetch GitHub repository categories: trending, new-releases, most-popular.
Optimized for speed using asyncio + aiohttp with smart batching.

Typical runtime: 5-15 minutes (down from 60+ minutes).
"""

import os
import sys
import json
import asyncio
import aiohttp
import time
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field

# ─── Configuration ────────────────────────────────────────────────────────────

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    print("ERROR: GITHUB_TOKEN environment variable is not set.", file=sys.stderr)
    sys.exit(1)

HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json",
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
CACHE_DIR = os.path.join(REPO_ROOT, "cached-data")
CACHE_VALIDITY_HOURS = 23

# Concurrency knobs
MAX_CONCURRENT_REQUESTS = 25        # simultaneous HTTP requests
MAX_SEARCH_CONCURRENT = 5           # simultaneous search API calls (stricter limit)
RELEASE_CHECK_BATCH = 40            # repos to check releases for at once
REQUEST_TIMEOUT = 20                # seconds
MAX_RETRIES = 3
DESIRED_COUNT = 100                 # repos per category/platform

# ─── Platform definitions ─────────────────────────────────────────────────────

PLATFORMS = {
    "android": {
        "topics": ["android", "android-app", "kotlin-android"],
        "installer_extensions": [".apk", ".aab"],
        "score_keywords": {
            "high": ["android", "kotlin-android"],
            "medium": ["mobile", "kotlin", "jetpack-compose"],
            "low": ["java", "apk", "gradle"],
        },
        "languages": {"primary": ["kotlin", "java"], "secondary": ["dart", "c++"]},
        "frameworks": ["jetpack-compose", "android-jetpack"],
    },
    "windows": {
        "topics": ["windows", "electron", "desktop", "windows-app"],
        "installer_extensions": [".msi", ".exe", ".msix"],
        "score_keywords": {
            "high": ["windows", "windows-app", "wpf", "winui"],
            "medium": ["desktop", "electron", "dotnet"],
            "low": ["app", "gui", "win32"],
        },
        "languages": {"primary": ["c#", "c++", "rust"], "secondary": ["javascript", "typescript"]},
        "frameworks": ["wpf", "winui", "avalonia"],
    },
    "macos": {
        "topics": ["macos", "osx", "mac", "swiftui"],
        "installer_extensions": [".dmg", ".pkg", ".app.zip"],
        "score_keywords": {
            "high": ["macos", "swiftui", "appkit"],
            "medium": ["desktop", "swift", "cocoa"],
            "low": ["app", "mac"],
        },
        "languages": {"primary": ["swift", "objective-c"], "secondary": ["c++", "rust"]},
        "frameworks": ["swiftui", "combine"],
    },
    "linux": {
        "topics": ["linux", "gtk", "qt", "gnome", "kde"],
        "installer_extensions": [".appimage", ".deb", ".rpm"],
        "score_keywords": {
            "high": ["linux", "gtk", "qt", "gnome"],
            "medium": ["desktop", "gnome", "kde", "flatpak"],
            "low": ["app", "unix", "gui"],
        },
        "languages": {"primary": ["c++", "rust", "c"], "secondary": ["python", "go", "vala"]},
        "frameworks": ["gtk4", "qt6"],
    },
}

# ─── Data classes ──────────────────────────────────────────────────────────────


@dataclass
class ReleaseInfo:
    """Cached release information for a repo."""
    has_release: bool = False
    published_at: Optional[str] = None
    has_installers: Dict[str, bool] = field(default_factory=dict)  # platform -> bool


@dataclass
class RepoCandidate:
    id: int
    name: str
    full_name: str
    owner_login: str
    owner_avatar: str
    description: Optional[str]
    default_branch: str
    html_url: str
    stars: int
    forks: int
    language: Optional[str]
    topics: List[str]
    releases_url: str
    updated_at: str
    created_at: str
    score: int = 0
    has_installers: bool = False
    recent_stars_velocity: float = 0.0
    latest_release_date: Optional[str] = None

    def to_summary(self, category: str = "trending") -> Dict:
        base = {
            "id": self.id,
            "name": self.name,
            "fullName": self.full_name,
            "owner": {"login": self.owner_login, "avatarUrl": self.owner_avatar},
            "description": self.description,
            "defaultBranch": self.default_branch,
            "htmlUrl": self.html_url,
            "stargazersCount": self.stars,
            "forksCount": self.forks,
            "language": self.language,
            "topics": self.topics,
            "releasesUrl": self.releases_url,
            "updatedAt": self.updated_at,
            "createdAt": self.created_at,
        }
        if category == "trending":
            base["trendingScore"] = round(self.score + (self.recent_stars_velocity * 10), 2)
        elif category == "new-releases":
            base["latestReleaseDate"] = self.latest_release_date
            recency = self._release_age_days()
            base["releaseRecency"] = recency
            if recency == 0:
                base["releaseRecencyText"] = "Released today"
            elif recency == 1:
                base["releaseRecencyText"] = "Released yesterday"
            else:
                base["releaseRecencyText"] = f"Released {recency} days ago"
        elif category == "most-popular":
            base["popularityScore"] = self.stars + (self.forks * 2)
        return base

    def _release_age_days(self) -> int:
        if not self.latest_release_date:
            return 999
        try:
            s = self.latest_release_date.replace("Z", "")
            if "+" in s:
                s = s.split("+")[0]
            rd = datetime.fromisoformat(s)
            return max(0, (datetime.utcnow() - rd).days)
        except Exception:
            return 999


# ─── Async HTTP layer ──────────────────────────────────────────────────────────


class GitHubClient:
    """Async GitHub API client with rate-limit awareness and retry."""

    def __init__(self):
        self._sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        self._search_sem = asyncio.Semaphore(MAX_SEARCH_CONCURRENT)
        self._session: Optional[aiohttp.ClientSession] = None
        self._request_count = 0
        self._rate_remaining = 5000
        self._rate_reset: Optional[float] = None
        # Cross-repo release cache: full_name -> ReleaseInfo
        self.release_cache: Dict[str, ReleaseInfo] = {}

    async def __aenter__(self):
        timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
        self._session = aiohttp.ClientSession(headers=HEADERS, timeout=timeout)
        return self

    async def __aexit__(self, *exc):
        if self._session:
            await self._session.close()

    async def _wait_for_rate_limit(self):
        """Pause if we're close to hitting the rate limit."""
        if self._rate_remaining < 50 and self._rate_reset:
            wait = self._rate_reset - time.time() + 2
            if wait > 0:
                print(f"  ⏳ Rate limit low ({self._rate_remaining}), waiting {wait:.0f}s...")
                await asyncio.sleep(wait)

    def _update_rate_info(self, headers):
        remaining = headers.get("X-RateLimit-Remaining")
        reset = headers.get("X-RateLimit-Reset")
        if remaining is not None:
            self._rate_remaining = int(remaining)
        if reset is not None:
            self._rate_reset = float(reset)

    async def get(self, url: str, params: Optional[Dict] = None) -> Tuple[Optional[Dict], Optional[str]]:
        """GET with retry, rate-limit handling, and concurrency control."""
        async with self._sem:
            await self._wait_for_rate_limit()
            for attempt in range(MAX_RETRIES):
                try:
                    async with self._session.get(url, params=params) as resp:
                        self._request_count += 1
                        self._update_rate_info(resp.headers)

                        if resp.status == 200:
                            return await resp.json(), None

                        if resp.status in (403, 429):
                            retry_after = resp.headers.get("Retry-After")
                            wait = int(retry_after) if retry_after and retry_after.isdigit() else (2 ** (attempt + 1))
                            if attempt < MAX_RETRIES - 1:
                                await asyncio.sleep(wait)
                                continue
                            return None, "Rate limited"

                        if 500 <= resp.status < 600 and attempt < MAX_RETRIES - 1:
                            await asyncio.sleep(2 ** attempt)
                            continue

                        return None, f"HTTP {resp.status}"

                except asyncio.TimeoutError:
                    if attempt < MAX_RETRIES - 1:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    return None, "Timeout"
                except Exception as e:
                    if attempt < MAX_RETRIES - 1:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    return None, str(e)

            return None, "Max retries exceeded"

    async def search_repos(self, query: str, sort: str = "stars", order: str = "desc",
                           pages: int = 3) -> List[Dict]:
        """Search repositories, fetching multiple pages concurrently."""
        async with self._search_sem:
            # Fetch page 1 first to know total
            data, err = await self.get(
                "https://api.github.com/search/repositories",
                params={"q": query, "sort": sort, "order": order, "per_page": 100, "page": 1},
            )
            if not data:
                return []

            items = data.get("items", [])
            total = data.get("total_count", 0)
            actual_pages = min(pages, (total // 100) + 1, 10)  # GitHub caps at 1000 results

            if actual_pages > 1:
                # Fetch remaining pages concurrently
                tasks = []
                for page in range(2, actual_pages + 1):
                    tasks.append(
                        self.get(
                            "https://api.github.com/search/repositories",
                            params={"q": query, "sort": sort, "order": order, "per_page": 100, "page": page},
                        )
                    )
                results = await asyncio.gather(*tasks)
                for page_data, page_err in results:
                    if page_data:
                        items.extend(page_data.get("items", []))

            return items

    async def get_latest_stable_release(self, owner: str, repo: str) -> ReleaseInfo:
        """Get latest stable release with caching."""
        full_name = f"{owner}/{repo}"
        if full_name in self.release_cache:
            return self.release_cache[full_name]

        info = ReleaseInfo()

        # Try /releases/latest first (one API call)
        data, err = await self.get(f"https://api.github.com/repos/{full_name}/releases/latest")
        if data and not data.get("draft") and not data.get("prerelease"):
            info.has_release = True
            info.published_at = data.get("published_at")
            assets = data.get("assets", [])
            # Pre-check all platforms
            for platform, cfg in PLATFORMS.items():
                for asset in assets:
                    name = asset.get("name", "").lower()
                    if any(name.endswith(ext) for ext in cfg["installer_extensions"]):
                        info.has_installers[platform] = True
                        break
            self.release_cache[full_name] = info
            return info

        # Fallback: search recent releases
        data, err = await self.get(
            f"https://api.github.com/repos/{full_name}/releases",
            params={"per_page": 5},
        )
        if data and isinstance(data, list):
            for release in data:
                if not release.get("draft") and not release.get("prerelease"):
                    info.has_release = True
                    info.published_at = release.get("published_at")
                    assets = release.get("assets", [])
                    for platform, cfg in PLATFORMS.items():
                        for asset in assets:
                            name = asset.get("name", "").lower()
                            if any(name.endswith(ext) for ext in cfg["installer_extensions"]):
                                info.has_installers[platform] = True
                                break
                    break

        self.release_cache[full_name] = info
        return info


# ─── Scoring ───────────────────────────────────────────────────────────────────


def calculate_platform_score(repo: Dict, platform: str) -> int:
    score = 0
    topics = [t.lower() for t in repo.get("topics", [])]
    language = (repo.get("language") or "").lower()
    desc = (repo.get("description") or "").lower()
    config = PLATFORMS[platform]
    kw = config["score_keywords"]
    langs = config["languages"]

    for k in kw["high"]:
        if k in topics:
            score += 15
    for k in kw["medium"]:
        if k in topics:
            score += 8
    for k in kw["low"]:
        if k in topics:
            score += 3

    if language in langs["primary"]:
        score += 20
    elif language in langs["secondary"]:
        score += 10

    for k in kw["high"]:
        if k in desc:
            score += 5
    for k in kw["medium"]:
        if k in desc:
            score += 3

    cross = ["cross-platform", "multiplatform", "multi-platform"]
    if any(c in topics or c in desc for c in cross):
        score += 15

    for fw in config.get("frameworks", []):
        if fw in topics:
            score += 10
            break

    return score


def calculate_velocity(repo: Dict) -> float:
    try:
        created = datetime.fromisoformat(repo["created_at"].replace("Z", "+00:00"))
        updated = datetime.fromisoformat(repo["updated_at"].replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        age_days = max((now - created).days, 1)
        days_since = (now - updated).days
        spd = repo["stargazers_count"] / age_days
        if days_since <= 7:
            spd *= 2.0
        elif days_since <= 30:
            spd *= 1.5
        elif days_since <= 90:
            spd *= 1.2
        return spd
    except Exception:
        return 0.0


def make_candidate(repo: Dict, platform: str, velocity: float = 0.0, score_weight: float = 1.0) -> RepoCandidate:
    score = int(calculate_platform_score(repo, platform) * score_weight)
    return RepoCandidate(
        id=repo["id"],
        name=repo["name"],
        full_name=repo["full_name"],
        owner_login=repo["owner"]["login"],
        owner_avatar=repo["owner"]["avatar_url"],
        description=repo.get("description"),
        default_branch=repo.get("default_branch", "main"),
        html_url=repo["html_url"],
        stars=repo["stargazers_count"],
        forks=repo["forks_count"],
        language=repo.get("language"),
        topics=repo.get("topics", []),
        releases_url=repo["releases_url"],
        updated_at=repo["updated_at"],
        created_at=repo["created_at"],
        score=score,
        recent_stars_velocity=velocity,
    )


# ─── Installer verification (async, batched) ──────────────────────────────────


async def verify_installers(
    client: GitHubClient,
    candidates: List[RepoCandidate],
    platform: str,
    need_release_date: bool = False,
    max_age_days: Optional[int] = None,
) -> List[RepoCandidate]:
    """
    Check candidates for platform-specific installers concurrently.
    Uses the cross-repo release cache so repeated checks are free.
    """
    print(f"  Verifying installers for {len(candidates)} candidates...")
    results = []
    now = datetime.utcnow()

    async def check_one(candidate: RepoCandidate):
        info = await client.get_latest_stable_release(candidate.owner_login, candidate.name)
        if not info.has_release or not info.has_installers.get(platform, False):
            return None

        # Age filter
        if max_age_days and info.published_at:
            try:
                s = info.published_at.replace("Z", "")
                if "+" in s:
                    s = s.split("+")[0]
                rd = datetime.fromisoformat(s)
                if (now - rd).days > max_age_days:
                    return None
                # Reject future dates
                if rd > now + timedelta(hours=1):
                    return None
            except Exception:
                return None

        candidate.has_installers = True
        if need_release_date:
            candidate.latest_release_date = info.published_at
        return candidate

    # Process in batches to avoid overwhelming the event loop
    for i in range(0, len(candidates), RELEASE_CHECK_BATCH):
        batch = candidates[i : i + RELEASE_CHECK_BATCH]
        tasks = [check_one(c) for c in batch]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in batch_results:
            if isinstance(r, RepoCandidate):
                results.append(r)
        done = min(i + RELEASE_CHECK_BATCH, len(candidates))
        if done % 100 == 0 or done == len(candidates):
            print(f"    Progress: {done}/{len(candidates)} checked, {len(results)} verified")

    return results


# ─── Category fetchers ─────────────────────────────────────────────────────────


async def fetch_trending(client: GitHubClient, platform: str) -> List[Dict]:
    print(f"\n{'='*60}")
    print(f"TRENDING — {platform.upper()}")
    print(f"{'='*60}")

    topics = PLATFORMS[platform]["topics"]
    primary_lang = PLATFORMS[platform]["languages"]["primary"][0]
    seen: Set[str] = set()
    all_candidates: List[RepoCandidate] = []

    strategies = [
        {"days": 30, "min_stars": 100, "topics": topics, "pages": 3, "weight": 1.5},
        {"days": 90, "min_stars": 50, "topics": topics[:1], "pages": 3, "weight": 1.2},
        {"days": 180, "min_stars": 500, "topics": [], "pages": 2, "weight": 1.0},
        {"days": 365, "min_stars": 200, "topics": topics[:1], "pages": 3, "weight": 0.9},
    ]

    # Run all search strategies concurrently
    async def run_strategy(strat):
        past = (datetime.utcnow() - timedelta(days=strat["days"])).strftime("%Y-%m-%d")
        base = f'stars:>{strat["min_stars"]} archived:false pushed:>={past}'
        if strat["topics"]:
            tq = " OR ".join(f"topic:{t}" for t in strat["topics"])
            q = f"{base} ({tq})"
        else:
            q = f"{base} language:{primary_lang}"
        return await client.search_repos(q, sort="stars", order="desc", pages=strat["pages"]), strat["weight"]

    tasks = [run_strategy(s) for s in strategies]
    results = await asyncio.gather(*tasks)

    for items, weight in results:
        for repo in items:
            fn = repo["full_name"]
            if fn in seen:
                continue
            seen.add(fn)
            vel = calculate_velocity(repo)
            c = make_candidate(repo, platform, velocity=vel, score_weight=weight)
            if c.score >= 5:
                all_candidates.append(c)

    print(f"  Collected {len(all_candidates)} candidates from {len(seen)} unique repos")

    all_candidates.sort(key=lambda c: c.score + c.recent_stars_velocity * 10, reverse=True)
    top = all_candidates[: DESIRED_COUNT * 3]
    verified = await verify_installers(client, top, platform)
    verified.sort(key=lambda c: c.score + c.recent_stars_velocity * 10, reverse=True)
    final = verified[:DESIRED_COUNT]

    print(f"  ✓ {len(final)} trending repos")
    return [r.to_summary("trending") for r in final]


async def fetch_new_releases(client: GitHubClient, platform: str) -> List[Dict]:
    print(f"\n{'='*60}")
    print(f"NEW RELEASES — {platform.upper()}")
    print(f"{'='*60}")

    topics = PLATFORMS[platform]["topics"]
    primary_lang = PLATFORMS[platform]["languages"]["primary"][0]
    seen: Set[str] = set()
    all_candidates: List[RepoCandidate] = []

    # Two rounds of strategies, but all searches within a round run concurrently
    rounds = [
        [
            {"days": 7, "min_stars": 10, "topics": topics, "pages": 5, "sort": "updated"},
            {"days": 14, "min_stars": 5, "topics": topics, "pages": 5, "sort": "updated"},
            {"days": 21, "min_stars": 20, "topics": topics[:1], "pages": 4, "sort": "updated"},
            {"days": 21, "min_stars": 50, "topics": [], "pages": 3, "sort": "stars"},
        ],
        [
            {"days": 21, "min_stars": 1, "topics": topics, "pages": 6, "sort": "updated"},
            {"days": 14, "min_stars": 0, "topics": topics, "pages": 5, "sort": "updated"},
        ],
    ]

    verified_total: List[RepoCandidate] = []

    for round_idx, strategies in enumerate(rounds, 1):
        if len(verified_total) >= DESIRED_COUNT:
            break

        print(f"\n  Round {round_idx}...")
        round_candidates: List[RepoCandidate] = []

        async def run_strategy(strat):
            past = (datetime.utcnow() - timedelta(days=strat["days"])).strftime("%Y-%m-%d")
            base = f'stars:>{strat["min_stars"]} archived:false pushed:>={past}'
            if strat["topics"]:
                tq = " OR ".join(f"topic:{t}" for t in strat["topics"])
                q = f"{base} ({tq})"
            else:
                q = f"{base} language:{primary_lang}"
            return await client.search_repos(q, sort=strat.get("sort", "updated"), order="desc", pages=strat["pages"])

        tasks = [run_strategy(s) for s in strategies]
        results = await asyncio.gather(*tasks)

        for items in results:
            for repo in items:
                fn = repo["full_name"]
                if fn in seen:
                    continue
                seen.add(fn)
                c = make_candidate(repo, platform)
                round_candidates.append(c)

        print(f"    {len(round_candidates)} new candidates")

        if not round_candidates:
            continue

        round_candidates.sort(key=lambda c: c.updated_at, reverse=True)
        verified = await verify_installers(
            client, round_candidates, platform, need_release_date=True, max_age_days=21
        )
        verified_total.extend(verified)
        print(f"    Running total: {len(verified_total)} verified")

    verified_total.sort(key=lambda r: r.latest_release_date or "", reverse=True)
    final = verified_total[:DESIRED_COUNT]

    print(f"  ✓ {len(final)} repos with new releases (≤21 days)")
    return [r.to_summary("new-releases") for r in final]


async def fetch_most_popular(client: GitHubClient, platform: str) -> List[Dict]:
    print(f"\n{'='*60}")
    print(f"MOST POPULAR — {platform.upper()}")
    print(f"{'='*60}")

    topics = PLATFORMS[platform]["topics"]
    primary_lang = PLATFORMS[platform]["languages"]["primary"][0]
    seen: Set[str] = set()
    all_candidates: List[RepoCandidate] = []

    six_months = (datetime.utcnow() - timedelta(days=180)).strftime("%Y-%m-%d")
    one_year = (datetime.utcnow() - timedelta(days=365)).strftime("%Y-%m-%d")

    strategies = [
        {"min_stars": 5000, "topics": topics, "pages": 5, "before": six_months},
        {"min_stars": 2000, "topics": topics[:1], "pages": 5, "before": six_months},
        {"min_stars": 1000, "topics": [], "pages": 3, "before": one_year},
    ]

    async def run_strategy(strat):
        base = f'stars:>{strat["min_stars"]} archived:false pushed:>={one_year} created:<{strat["before"]}'
        if strat["topics"]:
            tq = " OR ".join(f"topic:{t}" for t in strat["topics"])
            q = f"{base} ({tq})"
        else:
            q = f"{base} language:{primary_lang}"
        return await client.search_repos(q, sort="stars", order="desc", pages=strat["pages"])

    tasks = [run_strategy(s) for s in strategies]
    results = await asyncio.gather(*tasks)

    for items in results:
        for repo in items:
            fn = repo["full_name"]
            if fn in seen:
                continue
            seen.add(fn)
            c = make_candidate(repo, platform)
            all_candidates.append(c)

    print(f"  Collected {len(all_candidates)} candidates")

    all_candidates.sort(key=lambda c: c.stars, reverse=True)
    top = all_candidates[: DESIRED_COUNT * 3]
    verified = await verify_installers(client, top, platform)
    verified.sort(key=lambda c: c.stars, reverse=True)
    final = verified[:DESIRED_COUNT]

    print(f"  ✓ {len(final)} most popular repos")
    return [r.to_summary("most-popular") for r in final]


# ─── Cache I/O ─────────────────────────────────────────────────────────────────


def load_cache(category: str, platform: str) -> Optional[Dict]:
    cache_file = os.path.join(CACHE_DIR, category, f"{platform}.json")
    if not os.path.exists(cache_file):
        return None
    try:
        with open(cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        count = data.get("totalCount", 0)
        min_threshold = 10 if category == "new-releases" else 30
        if count < min_threshold:
            print(f"  Cache for {category}/{platform}: only {count} repos — refetching")
            return None
        last = datetime.fromisoformat(data["lastUpdated"].replace("Z", "+00:00"))
        age_h = (datetime.now(timezone.utc) - last).total_seconds() / 3600
        if age_h < CACHE_VALIDITY_HOURS:
            print(f"  ✓ Cache hit: {category}/{platform} ({age_h:.1f}h old, {count} repos)")
            return data
    except Exception as e:
        print(f"  Cache error {category}/{platform}: {e}")
    return None


def save_data(category: str, platform: str, repos: List[Dict], timestamp: str):
    out = {
        "category": category,
        "platform": platform,
        "lastUpdated": timestamp,
        "totalCount": len(repos),
        "repositories": repos,
    }
    cat_dir = os.path.join(CACHE_DIR, category)
    os.makedirs(cat_dir, exist_ok=True)
    path = os.path.join(cat_dir, f"{platform}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Saved {len(repos)} repos → {path}")


# ─── Main orchestrator ─────────────────────────────────────────────────────────


async def process_category(client: GitHubClient, category: str, fetch_fn, timestamp: str):
    """Process all platforms for a category, running platforms concurrently."""
    print(f"\n{'#'*70}")
    print(f"# CATEGORY: {category.upper()}")
    print(f"{'#'*70}")

    async def process_platform(platform: str):
        cached = load_cache(category, platform)
        if cached:
            return
        repos = await fetch_fn(client, platform)
        save_data(category, platform, repos, timestamp)

    # Run all 4 platforms concurrently within each category
    await asyncio.gather(*[process_platform(p) for p in PLATFORMS])


async def main():
    timestamp = datetime.utcnow().isoformat() + "Z"
    start = time.time()

    async with GitHubClient() as client:
        # Check rate limit
        data, _ = await client.get("https://api.github.com/rate_limit")
        if data:
            remaining = data.get("resources", {}).get("core", {}).get("remaining", 0)
            limit = data.get("resources", {}).get("core", {}).get("limit", 0)
            print(f"GitHub API: {remaining}/{limit} requests remaining\n")
            if remaining < 500:
                print("WARNING: Low rate limit!", file=sys.stderr)

        categories = [
            ("trending", fetch_trending),
            ("new-releases", fetch_new_releases),
            ("most-popular", fetch_most_popular),
        ]

        # Process categories sequentially (each has concurrent platforms inside)
        # This prevents search API abuse while still being fast
        for cat_name, cat_fn in categories:
            await process_category(client, cat_name, cat_fn, timestamp)

        elapsed = time.time() - start
        print(f"\n{'='*70}")
        print(f"✓ DONE in {elapsed:.0f}s ({elapsed/60:.1f} min)")
        print(f"  Total API requests: {client._request_count}")
        print(f"  Release cache entries: {len(client.release_cache)}")
        print(f"{'='*70}")


if __name__ == "__main__":
    asyncio.run(main())