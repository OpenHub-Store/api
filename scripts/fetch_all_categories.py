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
        # Broad terms used in catch-all searches to find repos that have
        # platform installers but may not carry a platform topic.
        "broad_terms": ["app", "mobile", "application"],
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
        "broad_terms": ["app", "desktop", "tool", "application", "gui"],
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
        "broad_terms": ["app", "desktop", "tool", "application"],
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
        "broad_terms": ["app", "desktop", "tool", "cli", "application", "gui"],
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
    target_count: Optional[int] = None,
) -> List[RepoCandidate]:
    """
    Check candidates for platform-specific installers concurrently.
    Uses the cross-repo release cache so repeated checks are free.

    If target_count is set, stops early once that many are verified
    (with a small overshoot buffer to avoid cutting off mid-batch).
    """
    print(f"  Verifying installers for {len(candidates)} candidates" +
          (f" (target: {target_count})" if target_count else "") + "...")
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

        # Early termination: stop once we have well above the target
        # Use 1.5x buffer so we don't under-collect
        if target_count and len(results) >= int(target_count * 1.5):
            print(f"    ✓ Early stop: {len(results)} verified (target {target_count}) after {done}/{len(candidates)} checked")
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
            vel = calculate_velocity(repo) if compute_velocity else 0.0
            c = make_candidate(repo, platform, velocity=vel, score_weight=weight)
            if c.score >= min_score:
                candidates.append(c)

    return candidates


# ─── Category fetchers ─────────────────────────────────────────────────────────


async def fetch_trending(client: GitHubClient, platform: str) -> List[Dict]:
    print(f"\n{'='*60}")
    print(f"TRENDING — {platform.upper()}")
    print(f"{'='*60}")

    topics = PLATFORMS[platform]["topics"]
    all_langs = PLATFORMS[platform]["languages"]["primary"] + PLATFORMS[platform]["languages"]["secondary"]
    seen: Set[str] = set()
    verified_all: List[RepoCandidate] = []

    # ── Round 1: topic-based + language-based + high-star catch-all ────────────
    specs_r1 = []
    for days, min_stars, weight in [
        (30, 100, 1.5),
        (90, 50, 1.2),
        (180, 200, 1.0),
        (365, 200, 0.9),
    ]:
        past = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
        base = f"stars:>{min_stars} archived:false pushed:>={past}"
        # Topic-based query (all topics combined)
        specs_r1.append({
            "query": _build_query(base, topics=topics),
            "sort": "stars", "pages": 3, "weight": weight,
        })

    # Language-only queries (catches repos like ollama that lack platform topics)
    for lang in all_langs[:2]:
        past = (datetime.utcnow() - timedelta(days=365)).strftime("%Y-%m-%d")
        base = f"stars:>500 archived:false pushed:>={past}"
        specs_r1.append({
            "query": _build_query(base, language=lang),
            "sort": "stars", "pages": 3, "weight": 0.8,
        })

    # High-star catch-all (no topic/language filter — finds mega-popular repos)
    for min_stars in [10000, 5000]:
        past = (datetime.utcnow() - timedelta(days=365)).strftime("%Y-%m-%d")
        base = f"stars:>{min_stars} archived:false pushed:>={past}"
        specs_r1.append({
            "query": base,
            "sort": "stars", "pages": 3, "weight": 0.6,
        })

    r1_candidates = await _collect_candidates(
        client, specs_r1, platform, seen, compute_velocity=True, min_score=0,
    )
    print(f"  Round 1: {len(r1_candidates)} candidates from {len(seen)} unique repos")

    # Sort by trending score; take top pool for verification
    r1_candidates.sort(key=lambda c: c.score + c.recent_stars_velocity * 10, reverse=True)
    pool_size = min(len(r1_candidates), DESIRED_COUNT * 5)
    pool = r1_candidates[:pool_size]
    verified_r1 = await verify_installers(
        client, pool, platform, need_release_date=True,
        target_count=DESIRED_COUNT,
    )
    verified_all.extend(verified_r1)
    print(f"  Round 1 verified: {len(verified_r1)}")

    # ── Round 2 (escalation): only if we still need more ─────────────────────
    if len(verified_all) < DESIRED_COUNT:
        print(f"\n  ⚠ Only {len(verified_all)} verified — running escalation round...")
        specs_r2 = []

        # Wider date ranges, lower star thresholds
        for days, min_stars in [(365, 50), (730, 200)]:
            past = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
            base = f"stars:>{min_stars} archived:false pushed:>={past}"
            specs_r2.append({
                "query": _build_query(base, topics=topics),
                "sort": "stars", "pages": 5, "weight": 0.8,
            })
            for lang in all_langs[:2]:
                specs_r2.append({
                    "query": _build_query(base, language=lang),
                    "sort": "stars", "pages": 3, "weight": 0.7,
                })

        # Cross-platform frameworks that often ship multi-platform binaries
        cross_topics = ["electron", "flutter", "tauri", "react-native", "cross-platform"]
        for ct in cross_topics:
            past = (datetime.utcnow() - timedelta(days=365)).strftime("%Y-%m-%d")
            base = f"stars:>100 archived:false pushed:>={past}"
            specs_r2.append({
                "query": _build_query(base, topics=[ct]),
                "sort": "stars", "pages": 3, "weight": 0.7,
            })

        r2_candidates = await _collect_candidates(
            client, specs_r2, platform, seen, compute_velocity=True, min_score=0,
        )
        print(f"  Round 2: {len(r2_candidates)} new candidates")

        if r2_candidates:
            r2_candidates.sort(key=lambda c: c.score + c.recent_stars_velocity * 10, reverse=True)
            remaining_needed = DESIRED_COUNT - len(verified_all)
            pool_size = min(len(r2_candidates), remaining_needed * 5)
            pool2 = r2_candidates[:pool_size]
            verified_r2 = await verify_installers(
                client, pool2, platform, need_release_date=True,
                target_count=remaining_needed,
            )
            verified_all.extend(verified_r2)
            print(f"  Round 2 verified: {len(verified_r2)}")

    # ── Round 3 (last resort): even broader ──────────────────────────────────
    if len(verified_all) < DESIRED_COUNT:
        print(f"\n  ⚠ Still only {len(verified_all)} — running last-resort round...")
        specs_r3 = []
        for days, min_stars in [(365, 20), (730, 100)]:
            past = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
            base = f"stars:>{min_stars} archived:false pushed:>={past}"
            for t in topics:
                specs_r3.append({
                    "query": _build_query(base, topics=[t]),
                    "sort": "updated", "pages": 5, "weight": 0.6,
                })
            for lang in all_langs:
                specs_r3.append({
                    "query": _build_query(base, language=lang),
                    "sort": "updated", "pages": 3, "weight": 0.5,
                })

        r3_candidates = await _collect_candidates(
            client, specs_r3, platform, seen, compute_velocity=True, min_score=0,
        )
        print(f"  Round 3: {len(r3_candidates)} new candidates")

        if r3_candidates:
            r3_candidates.sort(key=lambda c: c.score + c.recent_stars_velocity * 10, reverse=True)
            remaining_needed = DESIRED_COUNT - len(verified_all)
            pool_size = min(len(r3_candidates), remaining_needed * 5)
            verified_r3 = await verify_installers(
                client, r3_candidates[:pool_size], platform, need_release_date=True,
                target_count=remaining_needed,
            )
            verified_all.extend(verified_r3)
            print(f"  Round 3 verified: {len(verified_r3)}")

    # ── De-duplicate verified (same repo could appear via different rounds) ───
    seen_ids: Set[int] = set()
    deduped: List[RepoCandidate] = []
    for c in verified_all:
        if c.id not in seen_ids:
            seen_ids.add(c.id)
            deduped.append(c)

    deduped.sort(key=lambda c: c.score + c.recent_stars_velocity * 10, reverse=True)

    # Return ALL verified repos — do NOT cap at DESIRED_COUNT
    print(f"  ✓ {len(deduped)} trending repos (target: ≥{DESIRED_COUNT})")
    if len(deduped) < DESIRED_COUNT:
        print(f"  ⚠ WARNING: Only found {len(deduped)}, below target {DESIRED_COUNT}")
    return [r.to_summary("trending") for r in deduped]


async def fetch_new_releases(client: GitHubClient, platform: str) -> List[Dict]:
    """Fetch repos with new releases in the last 14 days."""
    print(f"\n{'='*60}")
    print(f"NEW RELEASES — {platform.upper()}")
    print(f"{'='*60}")

    MAX_RELEASE_AGE_DAYS = 14  # ← Changed from 21 to 14

    topics = PLATFORMS[platform]["topics"]
    all_langs = PLATFORMS[platform]["languages"]["primary"] + PLATFORMS[platform]["languages"]["secondary"]
    seen: Set[str] = set()
    verified_all: List[RepoCandidate] = []

    # ── Round 1: topic + language + high-star searches ────────────────────────
    specs_r1 = []

    for days, min_stars, sort in [
        (7, 5, "updated"),
        (14, 5, "updated"),
        (14, 10, "stars"),
        (21, 20, "updated"),   # slightly wider window to catch edge cases
        (21, 50, "stars"),
    ]:
        past = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
        base = f"stars:>{min_stars} archived:false pushed:>={past}"
        specs_r1.append({
            "query": _build_query(base, topics=topics),
            "sort": sort, "pages": 5,
        })

    # Language queries
    for lang in all_langs[:3]:
        for days, min_stars in [(14, 20), (21, 100)]:
            past = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
            base = f"stars:>{min_stars} archived:false pushed:>={past}"
            specs_r1.append({
                "query": _build_query(base, language=lang),
                "sort": "updated", "pages": 3,
            })

    # High-star catch-all (recently updated)
    for min_stars in [5000, 1000]:
        past = (datetime.utcnow() - timedelta(days=14)).strftime("%Y-%m-%d")
        base = f"stars:>{min_stars} archived:false pushed:>={past}"
        specs_r1.append({
            "query": base,
            "sort": "updated", "pages": 3,
        })

    r1_candidates = await _collect_candidates(
        client, specs_r1, platform, seen, compute_velocity=False, min_score=0,
    )
    print(f"  Round 1: {len(r1_candidates)} candidates")

    # Sort by update recency; cap pool
    r1_candidates.sort(key=lambda c: c.updated_at, reverse=True)
    pool_size = min(len(r1_candidates), DESIRED_COUNT * 5)
    verified_r1 = await verify_installers(
        client, r1_candidates[:pool_size], platform,
        need_release_date=True, max_age_days=MAX_RELEASE_AGE_DAYS,
        target_count=DESIRED_COUNT,
    )
    verified_all.extend(verified_r1)
    print(f"  Round 1 verified: {len(verified_r1)}")

    # ── Round 2: escalation with broader queries ──────────────────────────────
    if len(verified_all) < DESIRED_COUNT:
        print(f"\n  ⚠ Only {len(verified_all)} verified — running escalation round...")
        specs_r2 = []

        # Very low star threshold
        for days in [14, 21]:
            past = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
            base = f"stars:>0 archived:false pushed:>={past}"
            specs_r2.append({
                "query": _build_query(base, topics=topics),
                "sort": "updated", "pages": 8,
            })

        # Cross-platform framework repos
        cross_topics = ["electron", "flutter", "tauri", "react-native"]
        for ct in cross_topics:
            past = (datetime.utcnow() - timedelta(days=14)).strftime("%Y-%m-%d")
            base = f"stars:>10 archived:false pushed:>={past}"
            specs_r2.append({
                "query": _build_query(base, topics=[ct]),
                "sort": "updated", "pages": 3,
            })

        r2_candidates = await _collect_candidates(
            client, specs_r2, platform, seen, compute_velocity=False, min_score=0,
        )
        print(f"  Round 2: {len(r2_candidates)} new candidates")

        if r2_candidates:
            r2_candidates.sort(key=lambda c: c.updated_at, reverse=True)
            remaining_needed = DESIRED_COUNT - len(verified_all)
            pool_size = min(len(r2_candidates), remaining_needed * 5)
            verified_r2 = await verify_installers(
                client, r2_candidates[:pool_size], platform,
                need_release_date=True, max_age_days=MAX_RELEASE_AGE_DAYS,
                target_count=remaining_needed,
            )
            verified_all.extend(verified_r2)
            print(f"  Round 2 verified: {len(verified_r2)}")

    # ── Round 3: last resort ─────────────────────────────────────────────────
    if len(verified_all) < DESIRED_COUNT:
        print(f"\n  ⚠ Still only {len(verified_all)} — running last-resort round...")
        specs_r3 = []
        for lang in all_langs:
            past = (datetime.utcnow() - timedelta(days=21)).strftime("%Y-%m-%d")
            base = f"stars:>0 archived:false pushed:>={past}"
            specs_r3.append({
                "query": _build_query(base, language=lang),
                "sort": "updated", "pages": 5,
            })

        r3_candidates = await _collect_candidates(
            client, specs_r3, platform, seen, compute_velocity=False, min_score=0,
        )
        print(f"  Round 3: {len(r3_candidates)} new candidates")
        if r3_candidates:
            r3_candidates.sort(key=lambda c: c.updated_at, reverse=True)
            remaining_needed = DESIRED_COUNT - len(verified_all)
            pool_size = min(len(r3_candidates), remaining_needed * 5)
            verified_r3 = await verify_installers(
                client, r3_candidates[:pool_size], platform,
                need_release_date=True, max_age_days=MAX_RELEASE_AGE_DAYS,
                target_count=remaining_needed,
            )
            verified_all.extend(verified_r3)
            print(f"  Round 3 verified: {len(verified_r3)}")

    # De-duplicate
    seen_ids: Set[int] = set()
    deduped: List[RepoCandidate] = []
    for c in verified_all:
        if c.id not in seen_ids:
            seen_ids.add(c.id)
            deduped.append(c)

    deduped.sort(key=lambda r: r.latest_release_date or "", reverse=True)

    # Return ALL — do NOT cap
    print(f"  ✓ {len(deduped)} repos with new releases ≤{MAX_RELEASE_AGE_DAYS} days (target: ≥{DESIRED_COUNT})")
    if len(deduped) < DESIRED_COUNT:
        print(f"  ⚠ WARNING: Only found {len(deduped)}, below target {DESIRED_COUNT}")
    return [r.to_summary("new-releases") for r in deduped]


async def fetch_most_popular(client: GitHubClient, platform: str) -> List[Dict]:
    """
    Fetch the most popular repos — minimum 5000 stars.

    Uses an aggressive multi-strategy approach:
      1. Topic-filtered queries (narrow but precise)
      2. Language-only queries (catches repos like ollama)
      3. High-star catch-all (no topic/language filter at all)
      4. Cross-platform framework queries
      5. Escalation rounds if needed
    """
    print(f"\n{'='*60}")
    print(f"MOST POPULAR — {platform.upper()}")
    print(f"{'='*60}")

    MIN_STARS = 5000  # ← Changed from 1000 to 5000

    topics = PLATFORMS[platform]["topics"]
    all_langs = PLATFORMS[platform]["languages"]["primary"] + PLATFORMS[platform]["languages"]["secondary"]
    seen: Set[str] = set()
    verified_all: List[RepoCandidate] = []

    one_year = (datetime.utcnow() - timedelta(days=365)).strftime("%Y-%m-%d")
    two_years = (datetime.utcnow() - timedelta(days=730)).strftime("%Y-%m-%d")

    # ── Round 1: comprehensive search battery ─────────────────────────────────
    specs_r1 = []

    # Topic-based at various star thresholds
    for min_stars in [MIN_STARS, 10000]:
        base = f"stars:>{min_stars} archived:false pushed:>={one_year}"
        specs_r1.append({
            "query": _build_query(base, topics=topics),
            "sort": "stars", "pages": 5,
        })

    # Language-only queries — key to finding repos like ollama
    for lang in all_langs[:3]:
        for min_stars in [MIN_STARS, 10000]:
            base = f"stars:>{min_stars} archived:false pushed:>={one_year}"
            specs_r1.append({
                "query": _build_query(base, language=lang),
                "sort": "stars", "pages": 3,
            })

    # High-star catch-all: no topic or language filter
    # These find mega-popular repos that ship multi-platform installers.
    for min_stars in [30000, 15000, MIN_STARS]:
        base = f"stars:>{min_stars} archived:false pushed:>={one_year}"
        specs_r1.append({
            "query": base,
            "sort": "stars", "pages": 5,
        })

    # Cross-platform framework repos
    cross_topics = ["electron", "flutter", "tauri", "react-native", "cross-platform"]
    for ct in cross_topics:
        base = f"stars:>{MIN_STARS} archived:false pushed:>={one_year}"
        specs_r1.append({
            "query": _build_query(base, topics=[ct]),
            "sort": "stars", "pages": 3,
        })

    r1_candidates = await _collect_candidates(
        client, specs_r1, platform, seen, compute_velocity=False, min_score=0,
    )
    print(f"  Round 1: {len(r1_candidates)} candidates from {len(seen)} unique repos")

    # Sort by stars; cap pool to avoid rate-limit exhaustion
    r1_candidates.sort(key=lambda c: c.stars, reverse=True)
    pool_size = min(len(r1_candidates), DESIRED_COUNT * 5)
    verified_r1 = await verify_installers(
        client, r1_candidates[:pool_size], platform, need_release_date=True,
        target_count=DESIRED_COUNT,
    )
    verified_all.extend(verified_r1)
    print(f"  Round 1 verified: {len(verified_r1)}")

    # ── Round 2: escalation ──────────────────────────────────────────────────
    if len(verified_all) < DESIRED_COUNT:
        print(f"\n  ⚠ Only {len(verified_all)} verified — running escalation round...")
        specs_r2 = []

        # Wider date range + remaining unchecked from round 1
        base = f"stars:>{MIN_STARS} archived:false pushed:>={two_years}"
        specs_r2.append({
            "query": _build_query(base, topics=topics),
            "sort": "stars", "pages": 5,
        })
        for lang in all_langs:
            specs_r2.append({
                "query": _build_query(base, language=lang),
                "sort": "stars", "pages": 5,
            })
        specs_r2.append({
            "query": base,
            "sort": "stars", "pages": 5,
        })

        r2_candidates = await _collect_candidates(
            client, specs_r2, platform, seen, compute_velocity=False, min_score=0,
        )
        print(f"  Round 2: {len(r2_candidates)} new candidates")

        if r2_candidates:
            r2_candidates.sort(key=lambda c: c.stars, reverse=True)
            remaining_needed = DESIRED_COUNT - len(verified_all)
            pool_size = min(len(r2_candidates), remaining_needed * 5)
            verified_r2 = await verify_installers(
                client, r2_candidates[:pool_size], platform, need_release_date=True,
                target_count=remaining_needed,
            )
            verified_all.extend(verified_r2)
            print(f"  Round 2 verified: {len(verified_r2)}")

        # Also verify remaining unchecked from round 1 if we had capped
        if len(verified_all) < DESIRED_COUNT and len(r1_candidates) > DESIRED_COUNT * 5:
            leftover = r1_candidates[DESIRED_COUNT * 5:]
            print(f"  Checking {len(leftover)} leftover candidates from round 1...")
            remaining_needed = DESIRED_COUNT - len(verified_all)
            pool_size = min(len(leftover), remaining_needed * 5)
            verified_leftover = await verify_installers(
                client, leftover[:pool_size], platform, need_release_date=True,
                target_count=remaining_needed,
            )
            verified_all.extend(verified_leftover)
            print(f"  Leftover verified: {len(verified_leftover)}")

    # De-duplicate
    seen_ids: Set[int] = set()
    deduped: List[RepoCandidate] = []
    for c in verified_all:
        if c.id not in seen_ids:
            seen_ids.add(c.id)
            deduped.append(c)

    deduped.sort(key=lambda c: c.stars, reverse=True)

    # Return ALL — do NOT cap
    print(f"  ✓ {len(deduped)} most popular repos (target: ≥{DESIRED_COUNT})")
    if len(deduped) < DESIRED_COUNT:
        print(f"  ⚠ WARNING: Only found {len(deduped)}, below target {DESIRED_COUNT}")
    return [r.to_summary("most-popular") for r in deduped]


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