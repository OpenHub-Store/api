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
import collections
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field

try:
    from db_writer import save_to_postgres
except ImportError:
    save_to_postgres = None

# ─── Configuration ────────────────────────────────────────────────────────────

# Per-category tokens (3 classic PATs, one per category).
# Fallback: single GITHUB_TOKEN used for all categories.
CATEGORY_TOKENS = {
    "trending": os.environ.get("GH_TOKEN_TRENDING"),
    "new-releases": os.environ.get("GH_TOKEN_NEW_RELEASES"),
    "most-popular": os.environ.get("GH_TOKEN_MOST_POPULAR"),
    "topics": os.environ.get("GH_TOKEN_TOPICS"),
}
FALLBACK_TOKEN = os.environ.get("GITHUB_TOKEN")

# Ensure at least one token is available
_any_token = any(CATEGORY_TOKENS.values()) or FALLBACK_TOKEN
if not _any_token:
    print("ERROR: No GitHub tokens set. Set GH_TOKEN_TRENDING / GH_TOKEN_NEW_RELEASES / "
          "GH_TOKEN_MOST_POPULAR / GH_TOKEN_TOPICS, or GITHUB_TOKEN as fallback.", file=sys.stderr)
    sys.exit(1)


def get_token(category: str) -> str:
    """Get the token for a category, falling back to GITHUB_TOKEN."""
    token = CATEGORY_TOKENS.get(category) or FALLBACK_TOKEN
    if not token:
        print(f"ERROR: No token for category '{category}' and no GITHUB_TOKEN fallback.", file=sys.stderr)
        sys.exit(1)
    return token


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
RATE_LIMIT_FLOOR = 50               # stop verifying when remaining requests drop below this
SEARCH_RATE_LIMIT = 28              # max search API calls per window (GitHub allows 30, leave buffer)
SEARCH_RATE_WINDOW = 60             # sliding window in seconds
FORCE_REFRESH = os.environ.get("FORCE_REFRESH", "").lower() in ("true", "1", "yes")

# Topics / keywords that indicate NSFW or inappropriate content.
# Repos whose topics or description match any of these are excluded.
BLOCKED_TOPICS = {
    "nsfw", "porn", "pornography", "hentai", "e-hentai", "ehentai",
    "adult", "adult-content", "xxx", "erotic", "erotica", "sex",
    "nude", "nudes", "nudity", "lewd", "r18", "r-18",
    "rule34", "rule-34", "booru", "gelbooru", "danbooru",
    "nhentai", "hanime", "ecchi", "yaoi", "yuri", "doujin", "doujinshi",
    "onlyfans", "fansly", "chaturbate", "xvideos", "pornhub",
    "xhamster", "xnxx", "redtube", "cam-girl", "camgirl",
    "fetish", "bdsm", "harem", "waifu", "18+",
}


def _is_blocked(repo: Dict) -> bool:
    """Return True if the repo's topics or description contain blocked terms."""
    topics = {t.lower() for t in repo.get("topics", [])}
    if topics & BLOCKED_TOPICS:
        return True
    desc = (repo.get("description") or "").lower()
    return any(term in desc for term in BLOCKED_TOPICS)


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
        # Broad terms used in catch-all searches to find repos that have
        # platform installers but may not carry a platform topic.
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
        "installer_extensions": [".dmg", ".pkg"],
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

        if self.latest_release_date:
            base["latestReleaseDate"] = self.latest_release_date
            recency = self._release_age_days()
            base["releaseRecency"] = recency
            if recency == 0:
                base["releaseRecencyText"] = "Released today"
            elif recency == 1:
                base["releaseRecencyText"] = "Released yesterday"
            else:
                base["releaseRecencyText"] = f"Released {recency} days ago"

        if category == "trending":
            base["trendingScore"] = round(self.score + (self.recent_stars_velocity * 10), 2)
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

    def __init__(self, token: str):
        self._token = token
        self._headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
        }
        self._sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        self._search_sem = asyncio.Semaphore(MAX_SEARCH_CONCURRENT)
        self._session: Optional[aiohttp.ClientSession] = None
        self._request_count = 0
        self._rate_remaining = 5000
        self._rate_reset: Optional[float] = None
        # Search API rate limiter (sliding window)
        self._search_call_times: collections.deque = collections.deque()
        self._search_rate_lock = asyncio.Lock()
        # Cross-repo release cache: full_name -> ReleaseInfo
        self.release_cache: Dict[str, ReleaseInfo] = {}

    async def __aenter__(self):
        timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
        self._session = aiohttp.ClientSession(headers=self._headers, timeout=timeout)
        return self

    async def __aexit__(self, *exc):
        if self._session:
            await self._session.close()

    async def _wait_for_rate_limit(self):
        """Short pause for temporary rate-limit dips. Never waits more than 60s."""
        if self._rate_remaining < 10 and self._rate_reset:
            wait = self._rate_reset - time.time() + 2
            if wait > 60:
                # Don't block for long resets — let the budget system handle it
                print(f"  ⚠ Rate limit exhausted ({self._rate_remaining} remaining, "
                      f"reset in {wait:.0f}s) — skipping wait")
                return
            if wait > 0:
                print(f"  ⏳ Rate limit low ({self._rate_remaining}), short wait {wait:.0f}s...")
                await asyncio.sleep(wait)

    def _update_rate_info(self, headers, url: str):
        # Only track the core rate limit; ignore search API headers
        # (search has its own 30/min limit that would pollute _rate_remaining)
        resource = headers.get("X-RateLimit-Resource", "")
        if resource == "search" or "/search/" in url:
            return
        remaining = headers.get("X-RateLimit-Remaining")
        reset = headers.get("X-RateLimit-Reset")
        if remaining is not None:
            self._rate_remaining = int(remaining)
        if reset is not None:
            self._rate_reset = float(reset)

    async def _acquire_search_slot(self):
        """Ensure we stay within the search API rate limit (30 req/min)."""
        async with self._search_rate_lock:
            now = time.time()
            while self._search_call_times and now - self._search_call_times[0] >= SEARCH_RATE_WINDOW:
                self._search_call_times.popleft()
            if len(self._search_call_times) >= SEARCH_RATE_LIMIT:
                oldest = self._search_call_times[0]
                wait = SEARCH_RATE_WINDOW - (now - oldest) + 1
                if wait > 0:
                    print(f"    ⏳ Search rate limit: {len(self._search_call_times)}/{SEARCH_RATE_LIMIT} in last 60s, pausing {wait:.0f}s")
                    await asyncio.sleep(wait)
            self._search_call_times.append(time.time())

    async def get(self, url: str, params: Optional[Dict] = None) -> Tuple[Optional[Dict], Optional[str]]:
        """GET with retry, rate-limit handling, and concurrency control."""
        if "/search/" in url:
            await self._acquire_search_slot()
        async with self._sem:
            await self._wait_for_rate_limit()
            for attempt in range(MAX_RETRIES):
                try:
                    async with self._session.get(url, params=params) as resp:
                        self._request_count += 1
                        self._update_rate_info(resp.headers, url)

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
        query = f"fork:true {query}"
        async with self._search_sem:
            # Fetch page 1 first to know total
            data, err = await self.get(
                "https://api.github.com/search/repositories",
                params={"q": query, "sort": sort, "order": order, "per_page": 100, "page": 1},
            )
            if not data:
                print(f"    ⚠ Search query failed: {err}")
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

        def _check_assets(assets: List[Dict]):
            """Detect platform installers from release assets."""
            for platform, cfg in PLATFORMS.items():
                if info.has_installers.get(platform):
                    continue
                for asset in assets:
                    name = asset.get("name", "").lower()
                    if any(name.endswith(ext) for ext in cfg["installer_extensions"]):
                        info.has_installers[platform] = True
                        break

        # Try /releases/latest first (one API call)
        data, err = await self.get(f"https://api.github.com/repos/{full_name}/releases/latest")
        if data and not data.get("draft") and not data.get("prerelease"):
            info.has_release = True
            info.published_at = data.get("published_at")
            _check_assets(data.get("assets", []))
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
                    _check_assets(release.get("assets", []))
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
    budget: Optional[int] = None,
) -> List[RepoCandidate]:
    """
    Check candidates for platform-specific installers concurrently.
    Uses the cross-repo release cache so repeated checks are free.
    Stops when rate limit drops below RATE_LIMIT_FLOOR or when
    the per-platform budget is exhausted.
    """
    print(f"  Verifying installers for {len(candidates)} candidates...")
    results = []
    now = datetime.utcnow()
    start_requests = client._request_count

    # Effective floor: the higher of RATE_LIMIT_FLOOR or (remaining - budget)
    if budget is not None:
        budget_floor = max(RATE_LIMIT_FLOOR, client._rate_remaining - budget)
        print(f"  Per-platform budget: ~{budget} requests (floor at {budget_floor})")
    else:
        budget_floor = RATE_LIMIT_FLOOR

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
            print(f"    Progress: {done}/{len(candidates)} checked, {len(results)} verified "
                  f"(rate limit: {client._rate_remaining})")

        # Stop when rate limit is running low or budget exhausted
        if client._rate_remaining < budget_floor:
            used = client._request_count - start_requests
            reason = "budget exhausted" if budget is not None else "rate limit low"
            print(f"    ⚠ {reason.capitalize()} ({client._rate_remaining} remaining, "
                  f"used {used} this platform), "
                  f"stopping after {done}/{len(candidates)} checked, {len(results)} verified")
            break

    return results


# ─── Search helpers ────────────────────────────────────────────────────────────


def _build_query(
    base_filters: str,
    topics: Optional[List[str]] = None,
    language: Optional[str] = None,
    description_kw: Optional[str] = None,
) -> str:
    """Build a GitHub search query from components."""
    parts = [base_filters]
    if topics:
        tq = " OR ".join(f"topic:{t}" for t in topics)
        parts.append(f"({tq})")
    if language:
        parts.append(f"language:{language}")
    if description_kw:
        parts.append(f"{description_kw} in:name,description")
    return " ".join(parts)


async def _collect_candidates(
    client: GitHubClient,
    search_specs: List[Dict],
    platform: str,
    seen: Set[str],
    compute_velocity: bool = False,
    min_score: int = 0,
) -> List[RepoCandidate]:
    """
    Run a list of search specs concurrently and return de-duped candidates.
    Each spec: {query, sort, order, pages, weight (optional)}.
    """
    candidates: List[RepoCandidate] = []

    async def _run(spec):
        return (
            await client.search_repos(
                spec["query"],
                sort=spec.get("sort", "stars"),
                order=spec.get("order", "desc"),
                pages=spec.get("pages", 3),
            ),
            spec.get("weight", 1.0),
        )

    results = await asyncio.gather(*[_run(s) for s in search_specs])

    for items, weight in results:
        for repo in items:
            fn = repo["full_name"]
            if fn in seen:
                continue
            seen.add(fn)
            if _is_blocked(repo):
                continue
            vel = calculate_velocity(repo) if compute_velocity else 0.0
            c = make_candidate(repo, platform, velocity=vel, score_weight=weight)
            if c.score >= min_score:
                candidates.append(c)

    return candidates


# ─── Category fetchers ─────────────────────────────────────────────────────────


async def fetch_trending(client: GitHubClient, platform: str, budget: Optional[int] = None) -> List[Dict]:
    print(f"\n{'='*60}")
    print(f"TRENDING — {platform.upper()}")
    print(f"{'='*60}")

    topics = PLATFORMS[platform]["topics"]
    all_langs = PLATFORMS[platform]["languages"]["primary"] + PLATFORMS[platform]["languages"]["secondary"]
    seen: Set[str] = set()

    # All search specs in one pass — no artificial caps, no escalation rounds
    specs = []

    # Topic-based queries across multiple time windows
    for days, min_stars, weight in [
        (30, 100, 1.5),
        (90, 50, 1.2),
        (180, 200, 1.0),
        (365, 200, 0.9),
    ]:
        past = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
        base = f"stars:>{min_stars} archived:false pushed:>={past}"
        specs.append({
            "query": _build_query(base, topics=topics),
            "sort": "stars", "pages": 5, "weight": weight,
        })

    # Wider date ranges, lower star thresholds
    for days, min_stars in [(365, 50), (730, 200)]:
        past = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
        base = f"stars:>{min_stars} archived:false pushed:>={past}"
        specs.append({
            "query": _build_query(base, topics=topics),
            "sort": "stars", "pages": 5, "weight": 0.8,
        })

    # Language-only queries (catches repos that don't use platform topics)
    past = (datetime.utcnow() - timedelta(days=365)).strftime("%Y-%m-%d")
    for lang in all_langs[:2]:
        base = f"stars:>1000 archived:false pushed:>={past}"
        specs.append({
            "query": _build_query(base, language=lang),
            "sort": "stars", "pages": 5, "weight": 0.8,
        })

    # High-star catch-all (no topic/language filter)
    base = f"stars:>5000 archived:false pushed:>={past}"
    specs.append({"query": base, "sort": "stars", "pages": 5, "weight": 0.6})

    # Cross-platform frameworks
    for ct in ["electron", "flutter", "tauri", "cross-platform"]:
        base = f"stars:>100 archived:false pushed:>={past}"
        specs.append({
            "query": _build_query(base, topics=[ct]),
            "sort": "stars", "pages": 5, "weight": 0.7,
        })

    candidates = await _collect_candidates(
        client, specs, platform, seen, compute_velocity=True, min_score=0,
    )
    print(f"  {len(candidates)} candidates from {len(seen)} unique repos")

    # Sort by trending score and verify ALL
    candidates.sort(key=lambda c: c.score + c.recent_stars_velocity * 10, reverse=True)
    verified = await verify_installers(client, candidates, platform, need_release_date=True, budget=budget)
    print(f"  ✓ {len(verified)} trending repos verified")
    return [r.to_summary("trending") for r in verified]


async def fetch_new_releases(client: GitHubClient, platform: str, budget: Optional[int] = None) -> List[Dict]:
    """Fetch repos with new releases in the last 14 days."""
    print(f"\n{'='*60}")
    print(f"NEW RELEASES — {platform.upper()}")
    print(f"{'='*60}")

    MAX_RELEASE_AGE_DAYS = 14

    topics = PLATFORMS[platform]["topics"]
    all_langs = PLATFORMS[platform]["languages"]["primary"] + PLATFORMS[platform]["languages"]["secondary"]
    seen: Set[str] = set()

    # All search specs in one pass
    specs = []

    for days, min_stars, sort in [
        (7, 5, "updated"),
        (14, 10, "updated"),
        (21, 50, "stars"),
    ]:
        past = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
        base = f"stars:>{min_stars} archived:false pushed:>={past}"
        specs.append({
            "query": _build_query(base, topics=topics),
            "sort": sort, "pages": 5,
        })

    # Lower star threshold, broader reach
    for days in [14, 21]:
        past = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
        base = f"stars:>0 archived:false pushed:>={past}"
        specs.append({
            "query": _build_query(base, topics=topics),
            "sort": "updated", "pages": 5,
        })

    # Language queries
    past14 = (datetime.utcnow() - timedelta(days=14)).strftime("%Y-%m-%d")
    for lang in all_langs[:2]:
        base = f"stars:>50 archived:false pushed:>={past14}"
        specs.append({
            "query": _build_query(base, language=lang),
            "sort": "updated", "pages": 5,
        })

    # High-star catch-all
    base = f"stars:>1000 archived:false pushed:>={past14}"
    specs.append({"query": base, "sort": "updated", "pages": 5})

    # Cross-platform frameworks
    for ct in ["electron", "flutter", "tauri"]:
        base = f"stars:>10 archived:false pushed:>={past14}"
        specs.append({
            "query": _build_query(base, topics=[ct]),
            "sort": "updated", "pages": 5,
        })

    candidates = await _collect_candidates(
        client, specs, platform, seen, compute_velocity=False, min_score=0,
    )
    print(f"  {len(candidates)} candidates")

    # Sort by update recency and verify ALL
    candidates.sort(key=lambda c: c.updated_at, reverse=True)
    verified = await verify_installers(
        client, candidates, platform,
        need_release_date=True, max_age_days=MAX_RELEASE_AGE_DAYS, budget=budget,
    )

    verified.sort(key=lambda r: r.latest_release_date or "", reverse=True)
    print(f"  ✓ {len(verified)} repos with new releases ≤{MAX_RELEASE_AGE_DAYS} days")
    return [r.to_summary("new-releases") for r in verified]


async def fetch_most_popular(client: GitHubClient, platform: str, budget: Optional[int] = None) -> List[Dict]:
    """Fetch the most popular repos — minimum 5000 stars."""
    print(f"\n{'='*60}")
    print(f"MOST POPULAR — {platform.upper()}")
    print(f"{'='*60}")

    MIN_STARS = 5000

    topics = PLATFORMS[platform]["topics"]
    all_langs = PLATFORMS[platform]["languages"]["primary"] + PLATFORMS[platform]["languages"]["secondary"]
    seen: Set[str] = set()

    one_year = (datetime.utcnow() - timedelta(days=365)).strftime("%Y-%m-%d")
    two_years = (datetime.utcnow() - timedelta(days=730)).strftime("%Y-%m-%d")

    # All search specs in one pass
    specs = []

    # Topic-based (1-year and 2-year windows)
    for pushed_since in [one_year, two_years]:
        base = f"stars:>{MIN_STARS} archived:false pushed:>={pushed_since}"
        specs.append({
            "query": _build_query(base, topics=topics),
            "sort": "stars", "pages": 5,
        })

    # Language queries (all primary + first secondary)
    base = f"stars:>{MIN_STARS} archived:false pushed:>={one_year}"
    for lang in all_langs[:3]:
        specs.append({
            "query": _build_query(base, language=lang),
            "sort": "stars", "pages": 5,
        })

    # High-star catch-all — no topic/language filter
    for min_stars in [20000, MIN_STARS]:
        b = f"stars:>{min_stars} archived:false pushed:>={one_year}"
        specs.append({"query": b, "sort": "stars", "pages": 5})

    # Cross-platform frameworks
    for ct in ["electron", "flutter", "tauri", "cross-platform"]:
        specs.append({
            "query": _build_query(base, topics=[ct]),
            "sort": "stars", "pages": 5,
        })

    candidates = await _collect_candidates(
        client, specs, platform, seen, compute_velocity=False, min_score=0,
    )
    print(f"  {len(candidates)} candidates from {len(seen)} unique repos")

    # Sort by stars and verify ALL
    candidates.sort(key=lambda c: c.stars, reverse=True)
    verified = await verify_installers(client, candidates, platform, need_release_date=True, budget=budget)

    verified.sort(key=lambda c: c.stars, reverse=True)
    print(f"  ✓ {len(verified)} most popular repos")
    return [r.to_summary("most-popular") for r in verified]


# ─── Topic definitions ─────────────────────────────────────────────────────

# Maps topic category name → list of GitHub topics to search for.
# Mirrors TopicCategory enum in the app (feature/home/domain/model/TopicCategory.kt).
TOPIC_CATEGORIES = {
    "privacy": {
        "topics": [
            "privacy", "security", "encryption", "vpn", "firewall",
            "password-manager", "privacy-tools", "e2ee", "secure",
            "anonymity", "tor", "pgp", "2fa", "auth",
        ],
        "keywords": ["privacy", "security", "encryption", "vpn", "firewall"],
    },
    "media": {
        "topics": [
            "music-player", "video-player", "media", "podcast", "streaming",
            "audio", "video", "media-player", "music", "player",
            "mpv", "vlc", "recorder", "screen-recorder", "gallery",
        ],
        "keywords": ["music-player", "video-player", "media", "podcast", "audio"],
    },
    "productivity": {
        "topics": [
            "productivity", "file-manager", "notes", "launcher", "keyboard",
            "browser", "calendar", "todo", "note-taking", "editor",
            "organizer", "task-manager", "markdown", "writing",
        ],
        "keywords": ["productivity", "file-manager", "notes", "launcher", "browser"],
    },
    "networking": {
        "topics": [
            "proxy", "dns", "ad-blocker", "torrent", "downloader",
            "network", "ssh", "wireguard", "adblock", "download-manager",
            "firewall", "socks5", "http-proxy", "p2p", "ftp",
        ],
        "keywords": ["proxy", "dns", "ad-blocker", "torrent", "downloader", "network"],
    },
    "dev-tools": {
        "topics": [
            "terminal", "developer-tools", "git-client", "editor", "cli",
            "ide", "devtools", "code-editor", "terminal-emulator", "development",
            "adb", "debugger", "api-client", "shell", "sdk",
        ],
        "keywords": ["terminal", "developer-tools", "git-client", "code-editor", "cli"],
    },
}


async def search_topic_candidates(
    client: GitHubClient,
    topic_name: str,
    topic_config: Dict,
) -> List[RepoCandidate]:
    """Search GitHub for repos matching a topic category (platform-agnostic).

    Returns unverified candidates — call verify_installers() per platform.
    """
    print(f"\n{'='*60}")
    print(f"TOPIC SEARCH: {topic_name.upper()} (all platforms)")
    print(f"{'='*60}")

    topics = topic_config["topics"]
    keywords = topic_config["keywords"]
    seen: Set[str] = set()

    one_year = (datetime.utcnow() - timedelta(days=365)).strftime("%Y-%m-%d")
    two_years = (datetime.utcnow() - timedelta(days=730)).strftime("%Y-%m-%d")

    specs = []

    # Strategy: search by CATEGORY TOPICS ONLY — do NOT cross with platform topics.
    # Almost no repos tag themselves with both "terminal" AND "windows".
    # Platform filtering happens at the installer-verification step (checks for
    # .apk, .exe, .dmg, .deb etc. in release assets), which runs per-platform.

    # 1) Batch category topics into groups of 4, sorted by stars
    topic_batches = [topics[i:i+4] for i in range(0, min(len(topics), 12), 4)]
    for batch in topic_batches:
        base = f"stars:>50 archived:false pushed:>={one_year}"
        cat_or = " OR ".join(f"topic:{t}" for t in batch)
        specs.append({
            "query": f"{base} ({cat_or})",
            "sort": "stars", "pages": 3, "weight": 1.5,
        })

    # 2) Keywords in name/description
    for kw in keywords[:3]:
        base = f"stars:>100 archived:false pushed:>={one_year}"
        specs.append({
            "query": f"{base} {kw} in:name,description",
            "sort": "stars", "pages": 2, "weight": 1.2,
        })

    # 3) All category topics combined, lower star threshold, recently updated
    cat_all_or = " OR ".join(f"topic:{t}" for t in topics[:8])
    base = f"stars:>20 archived:false pushed:>={one_year}"
    specs.append({
        "query": f"{base} ({cat_all_or})",
        "sort": "updated", "pages": 3, "weight": 1.0,
    })

    # 4) Broader: high stars, wider time window (catches established projects)
    base = f"stars:>500 archived:false pushed:>={two_years}"
    specs.append({
        "query": f"{base} ({cat_all_or})",
        "sort": "stars", "pages": 2, "weight": 0.8,
    })

    print(f"  {len(specs)} search specs ({sum(s.get('pages', 3) for s in specs)} API calls)")

    # Use "android" as dummy platform for make_candidate scoring — score is
    # recalculated in verify_installers anyway, and we just need dedup here.
    candidates = await _collect_candidates(
        client, specs, "android", seen, compute_velocity=False, min_score=0,
    )
    print(f"  {len(candidates)} candidates from {len(seen)} unique repos")
    candidates.sort(key=lambda c: c.stars, reverse=True)
    return candidates


async def verify_topic_for_platform(
    client: GitHubClient,
    candidates: List[RepoCandidate],
    platform: str,
    topic_name: str,
    budget: Optional[int] = None,
) -> List[Dict]:
    """Verify which candidates have installers for a specific platform."""
    print(f"\n  --- {topic_name}/{platform}: verifying {len(candidates)} candidates ---")
    verified = await verify_installers(
        client, candidates, platform, need_release_date=True, budget=budget,
    )
    verified.sort(key=lambda c: (c.score, c.stars), reverse=True)
    print(f"  ✓ {len(verified)} {topic_name} repos verified for {platform}")
    return [r.to_summary("topic") for r in verified]


async def process_topics(client: GitHubClient, timestamp: str):
    """Process all 5 topic categories across all platforms.

    Optimization: search queries are platform-agnostic, so we search ONCE per
    topic and then verify installers per-platform. This saves ~75% of search
    API calls compared to searching per topic×platform.
    """
    print(f"\n{'#'*70}")
    print(f"# TOPICS (5 categories × {len(PLATFORMS)} platforms)")
    print(f"{'#'*70}")

    # ─── Rate limit check ─────────────────────────────────────────────────
    data, _ = await client.get("https://api.github.com/rate_limit")
    if data:
        core = data.get("resources", {}).get("core", {})
        search = data.get("resources", {}).get("search", {})
        print(f"  Rate limit — core: {core.get('remaining', '?')}/{core.get('limit', '?')}, "
              f"search: {search.get('remaining', '?')}/{search.get('limit', '?')}")
        remaining = core.get("remaining", 0)
        if remaining < 500:
            print(f"  ⚠ WARNING: Low core rate limit ({remaining}) — results may be incomplete")
        search_remaining = search.get("remaining", 0)
        if search_remaining < 5:
            print(f"  ⚠ WARNING: Search rate limit nearly exhausted ({search_remaining})")

    num_topics = len(TOPIC_CATEGORIES)
    num_platforms = len(PLATFORMS)
    # Budget slots: 1 search + N platform verifications per topic
    total_verify_slots = num_topics * num_platforms

    # Track results for final summary
    results_summary: Dict[str, Dict[str, int]] = {}

    for topic_idx, (topic_name, topic_config) in enumerate(TOPIC_CATEGORIES.items()):
        print(f"\n--- Topic {topic_idx + 1}/{num_topics}: {topic_name} ---")
        results_summary[topic_name] = {}

        # Check which platforms still need data
        platforms_needed = []
        for platform in PLATFORMS.keys():
            cached = load_cache(f"topics/{topic_name}", platform)
            if cached:
                # Count existing cached repos for summary
                existing = _load_existing_count(f"topics/{topic_name}", platform)
                results_summary[topic_name][platform] = existing
                print(f"  {platform}: cached ({existing} repos)")
            else:
                platforms_needed.append(platform)

        if not platforms_needed:
            print(f"  All platforms cached — skipping")
            continue

        # Search once for this topic (platform-agnostic)
        candidates = await search_topic_candidates(client, topic_name, topic_config)

        if not candidates:
            print(f"  ⚠ 0 candidates found — skipping all platforms")
            for platform in platforms_needed:
                existing = _load_existing_count(f"topics/{topic_name}", platform)
                results_summary[topic_name][platform] = existing
                print(f"    {platform}: existing cache has {existing} repos")
            continue

        # Verify installers per platform using the shared candidate list
        for platform_idx, platform in enumerate(platforms_needed):
            verify_slot = topic_idx * num_platforms + platform_idx
            slots_remaining = max(total_verify_slots - verify_slot, 1)
            budget = max((client._rate_remaining - RATE_LIMIT_FLOOR) // slots_remaining, 50)

            repos = await verify_topic_for_platform(
                client, candidates, platform, topic_name, budget,
            )

            if len(repos) == 0:
                existing = _load_existing_count(f"topics/{topic_name}", platform)
                results_summary[topic_name][platform] = existing
                print(f"  ⚠ 0 repos fetched — skipping save (existing cache: {existing} repos)")
                continue

            # Lower threshold for topics — even 5 repos is useful
            min_threshold = 5
            if len(repos) < min_threshold:
                existing = _load_existing_count(f"topics/{topic_name}", platform)
                if existing >= min_threshold:
                    results_summary[topic_name][platform] = existing
                    print(f"  ⚠ Only {len(repos)} repos fetched but cache has {existing} — keeping cached data")
                    continue

            save_data(f"topics/{topic_name}", platform, repos, timestamp)
            results_summary[topic_name][platform] = len(repos)

    # ─── Final summary ────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"TOPICS SUMMARY")
    print(f"{'='*70}")

    # Header
    platforms_list = list(PLATFORMS.keys())
    header = f"  {'topic':<16}" + "".join(f"{p:>10}" for p in platforms_list) + f"{'total':>10}"
    print(header)
    print(f"  {'-'*16}" + "".join(f"{'-'*10}" for _ in platforms_list) + f"{'-'*10}")

    grand_total = 0
    for topic_name in TOPIC_CATEGORIES:
        counts = results_summary.get(topic_name, {})
        row = f"  {topic_name:<16}"
        topic_total = 0
        for p in platforms_list:
            c = counts.get(p, 0)
            topic_total += c
            row += f"{c:>10}"
        row += f"{topic_total:>10}"
        grand_total += topic_total
        print(row)

    print(f"  {'-'*16}" + "".join(f"{'-'*10}" for _ in platforms_list) + f"{'-'*10}")
    print(f"  {'TOTAL':<16}" + "".join(
        f"{sum(results_summary.get(t, {}).get(p, 0) for t in TOPIC_CATEGORIES):>10}"
        for p in platforms_list
    ) + f"{grand_total:>10}")

    # Rate limit after
    data, _ = await client.get("https://api.github.com/rate_limit")
    if data:
        core = data.get("resources", {}).get("core", {})
        search = data.get("resources", {}).get("search", {})
        print(f"\n  Rate limit remaining — core: {core.get('remaining', '?')}/{core.get('limit', '?')}, "
              f"search: {search.get('remaining', '?')}/{search.get('limit', '?')}")


# ─── Cache I/O ─────────────────────────────────────────────────────────────────


def load_cache(category: str, platform: str) -> Optional[Dict]:
    if FORCE_REFRESH:
        print(f"  Force refresh: skipping cache for {category}/{platform}")
        return None
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


def _load_existing_count(category: str, platform: str) -> int:
    """Return the repo count from an existing cache file, or 0 if none."""
    cache_file = os.path.join(CACHE_DIR, category, f"{platform}.json")
    try:
        with open(cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("totalCount", 0)
    except Exception:
        return 0


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

    # Write to Postgres (if DATABASE_URL is set)
    if save_to_postgres is not None:
        save_to_postgres(category, platform, repos)


# ─── Main orchestrator ─────────────────────────────────────────────────────────


async def process_category(client: GitHubClient, category: str, fetch_fn, timestamp: str):
    """Process all platforms for a category, running platforms concurrently."""
    print(f"\n{'#'*70}")
    print(f"# CATEGORY: {category.upper()}")
    print(f"{'#'*70}")

    num_platforms = len(PLATFORMS)

    async def process_platform(platform: str, budget: int):
        """Fetch and save repos for one platform. Returns True if searches were run."""
        cached = load_cache(category, platform)
        if cached:
            return False
        repos = await fetch_fn(client, platform, budget)

        # Never save 0 repos — preserve whatever exists in cache
        if len(repos) == 0:
            existing = _load_existing_count(category, platform)
            print(f"  ⚠ 0 repos fetched — skipping save (existing cache: {existing} repos)")
            return True

        # Don't overwrite good cached data with poor results
        min_threshold = 10 if category == "new-releases" else 30
        if len(repos) < min_threshold:
            existing = _load_existing_count(category, platform)
            if existing >= min_threshold:
                print(f"  ⚠ Only {len(repos)} repos fetched but cache has {existing} — keeping cached data")
                return True

        save_data(category, platform, repos, timestamp)
        return True

    # Compute per-platform budget: divide remaining requests evenly
    remaining = client._rate_remaining
    per_platform = max(remaining // num_platforms, 100)  # at least 100 each
    print(f"  Rate limit: {remaining} remaining, ~{per_platform} per platform")

    # Process platforms SEQUENTIALLY to avoid rate-limit thrashing.
    # The release cache still benefits later platforms from earlier ones.
    # Search API pacing is handled by _acquire_search_slot() (sliding window rate limiter).
    platforms_left = list(PLATFORMS.keys())
    for i, p in enumerate(platforms_left):
        # Recalculate budget for remaining platforms so unused budget carries forward
        platforms_remaining = num_platforms - i
        budget = max((client._rate_remaining - RATE_LIMIT_FLOOR) // platforms_remaining, 100)

        await process_platform(p, budget)


async def main():
    timestamp = datetime.utcnow().isoformat() + "Z"
    start = time.time()
    total_requests = 0
    total_cache_entries = 0

    categories = [
        ("trending", fetch_trending),
        ("new-releases", fetch_new_releases),
        ("most-popular", fetch_most_popular),
    ]

    # Detect shared tokens so we can split the budget fairly across categories
    tokens = [get_token(name) for name, _ in categories]
    num_categories = len(categories)
    shared_token = len(set(tokens)) < num_categories

    if shared_token:
        print(f"\n⚠ Some categories share the same token — budget will be split evenly")
        print(f"  TIP: Set GH_TOKEN_TRENDING, GH_TOKEN_NEW_RELEASES, GH_TOKEN_MOST_POPULAR "
              f"to 3 separate PATs for 3× the rate limit")

    for cat_idx, (cat_name, cat_fn) in enumerate(categories):
        token = tokens[cat_idx]

        async with GitHubClient(token) as client:
            # Check actual rate limit for this token
            data, _ = await client.get("https://api.github.com/rate_limit")
            if data:
                remaining = data.get("resources", {}).get("core", {}).get("remaining", 0)
                limit = data.get("resources", {}).get("core", {}).get("limit", 0)

                if shared_token:
                    # Token is shared — only use this category's fair share
                    categories_left = num_categories - cat_idx
                    category_budget = (remaining - RATE_LIMIT_FLOOR) // categories_left
                    # Cap _rate_remaining so process_category divides only our share
                    client._rate_remaining = category_budget + RATE_LIMIT_FLOOR
                    print(f"\n[{cat_name}] API budget: {remaining}/{limit} remaining, "
                          f"category share: ~{category_budget} (1/{categories_left} of shared token)")
                else:
                    print(f"\n[{cat_name}] API budget: {remaining}/{limit} requests remaining (dedicated token)")

                if remaining < 500:
                    print(f"WARNING: Low rate limit for {cat_name}!", file=sys.stderr)

            await process_category(client, cat_name, cat_fn, timestamp)

            total_requests += client._request_count
            total_cache_entries += len(client.release_cache)
            print(f"  [{cat_name}] Used {client._request_count} API requests, "
                  f"{len(client.release_cache)} release cache entries, "
                  f"{client._rate_remaining} requests remaining")

    # ─── Phase 2: Topics ─────────────────────────────────────────────────────
    # Use GH_TOKEN_TOPICS (dedicated) plus scavenge leftover budget from
    # the 3 category tokens.

    topics_token = CATEGORY_TOKENS.get("topics") or FALLBACK_TOKEN
    if topics_token:
        print(f"\n{'#'*70}")
        print(f"# PHASE 2: TOPICS")
        print(f"{'#'*70}")

        # Collect all usable tokens: dedicated topics token + leftover from category tokens
        topic_tokens = [topics_token]
        for cat_name in ["trending", "new-releases", "most-popular"]:
            cat_token = CATEGORY_TOKENS.get(cat_name)
            if cat_token and cat_token != topics_token:
                topic_tokens.append(cat_token)

        # Deduplicate while preserving order
        seen_tokens: Set[str] = set()
        unique_tokens = []
        for t in topic_tokens:
            if t not in seen_tokens:
                seen_tokens.add(t)
                unique_tokens.append(t)

        print(f"  Using {len(unique_tokens)} token(s) for topics")

        # Check remaining budget on each token
        usable_clients = []
        for i, tok in enumerate(unique_tokens):
            async with GitHubClient(tok) as probe:
                data, _ = await probe.get("https://api.github.com/rate_limit")
                if data:
                    remaining = data.get("resources", {}).get("core", {}).get("remaining", 0)
                    limit = data.get("resources", {}).get("core", {}).get("limit", 0)
                    label = "dedicated" if i == 0 else f"leftover-{i}"
                    print(f"  Token {label}: {remaining}/{limit} remaining")
                    if remaining > RATE_LIMIT_FLOOR + 100:
                        usable_clients.append((tok, remaining))

        if usable_clients:
            # Use the token with the most remaining budget first
            usable_clients.sort(key=lambda x: x[1], reverse=True)
            best_token, best_remaining = usable_clients[0]
            print(f"  Selected token with {best_remaining} remaining requests")

            async with GitHubClient(best_token) as client:
                client._rate_remaining = best_remaining
                await process_topics(client, timestamp)
                total_requests += client._request_count
                total_cache_entries += len(client.release_cache)
                print(f"  [topics] Used {client._request_count} API requests, "
                      f"{len(client.release_cache)} release cache entries, "
                      f"{client._rate_remaining} requests remaining")

                # If first token ran low, try remaining tokens
                if client._rate_remaining <= RATE_LIMIT_FLOOR + 50 and len(usable_clients) > 1:
                    for tok, rem in usable_clients[1:]:
                        print(f"\n  Switching to next token ({rem} remaining)...")
                        async with GitHubClient(tok) as fallback_client:
                            fallback_client._rate_remaining = rem
                            # Share release cache for efficiency
                            fallback_client.release_cache = client.release_cache
                            await process_topics(fallback_client, timestamp)
                            total_requests += fallback_client._request_count
                            total_cache_entries += len(fallback_client.release_cache)
                            print(f"  [topics-fallback] Used {fallback_client._request_count} API requests, "
                                  f"{fallback_client._rate_remaining} remaining")
                            if fallback_client._rate_remaining > RATE_LIMIT_FLOOR + 50:
                                break  # Still has budget, done
        else:
            print("  ⚠ No tokens with enough budget for topics — skipping")
    else:
        print("\n⚠ No token available for topics — set GH_TOKEN_TOPICS or GITHUB_TOKEN")

    elapsed = time.time() - start
    print(f"\n{'='*70}")
    print(f"✓ DONE in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Total API requests across all tokens: {total_requests}")
    print(f"  Total release cache entries: {total_cache_entries}")
    print(f"{'='*70}")


if __name__ == "__main__":
    asyncio.run(main())