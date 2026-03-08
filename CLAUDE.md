# CLAUDE.md

## Project Overview

Automated GitHub repository discovery tool. Fetches and categorizes open-source repos by **platform** (android, windows, macos, linux) and **category** (trending, new-releases, most-popular), then commits the results as cached JSON. Runs daily via GitHub Actions.

## Architecture

```
scripts/
  fetch_all_categories.py   — Main fetcher (async Python, aiohttp)
  validate_releases.py      — Release date validator (sync, requests)
  requirements.txt          — Python deps: requests, aiohttp
cached-data/
  trending/{platform}.json
  new-releases/{platform}.json
  most-popular/{platform}.json
.github/workflows/
  fetch_all_categories_workflow.yml — Daily cron (2am UTC) + manual dispatch
```

## How It Works

1. `fetch_all_categories.py` reads 3 per-category tokens (`GH_TOKEN_TRENDING`, `GH_TOKEN_NEW_RELEASES`, `GH_TOKEN_MOST_POPULAR`), falling back to `GITHUB_TOKEN`
2. Each category gets its own `GitHubClient` with a dedicated token
3. If tokens are shared (same underlying user), the budget is split evenly across categories
4. For each category × platform (12 combos), checks if cached JSON is fresh (<23h)
5. If stale, queries GitHub Search API with platform-specific topics/languages/keywords
6. Filters repos that have **real releases with platform installers** via two methods:
   - **Extension matching**: Direct installer files (`.apk`, `.exe`, `.dmg`, `.deb`, etc.)
   - **Keyword matching**: Generic archives (`.zip`, `.tar.gz`) with platform keywords in the filename (e.g. `myapp-macos-arm64.zip`, `myapp-win-x64.tar.gz`)
7. Repos with NSFW/inappropriate topics or descriptions are excluded via `BLOCKED_TOPICS`
8. Verifies ALL candidates — no artificial caps. Stops gracefully when per-platform budget is exhausted or rate limit drops below `RATE_LIMIT_FLOOR` (50)
9. Never saves 0-repo results; never overwrites good cached data with poor results
10. Waits 65s between platforms for search API rate limit (30 req/min) to reset
11. Saves results to `cached-data/{category}/{platform}.json`
12. GitHub Actions commits and pushes changes

### Token Strategy
- 3 GitHub Classic PATs (scope: `public_repo`), each from a **separate GitHub account**
- GitHub rate limits are per-user (not per-token), so 3 accounts = 3 independent 5,000 req/hr pools = 15,000 total
- Stored as GitHub Actions repository secrets: `GH_TOKEN_TRENDING`, `GH_TOKEN_NEW_RELEASES`, `GH_TOKEN_MOST_POPULAR`
- Backward compatible: falls back to single `GITHUB_TOKEN` if per-category tokens aren't set
- If shared tokens detected, budget is automatically split evenly across categories

### Rate Limit Management

**Two independent rate limits at play:**

| Limit | Pool | Scope |
|---|---|---|
| Core API | 5,000/hr per user | Release checks, rate_limit endpoint |
| Search API | 30/min per user | `search/repositories` queries |

**Core API budget system:**
- `main()` detects shared tokens and caps each category to its fair share
- `process_category()` divides the category's budget evenly across 4 platforms
- Budget recalculates after each platform — unused budget carries forward
- `verify_installers()` stops when per-platform budget is exhausted (not just global floor)

**Search API throttling:**
- 65-second pause between platforms within a category to let the 30 req/min limit reset
- Only pauses if the previous platform actually ran searches (cached platforms skip it)
- `_update_rate_info()` ignores search API headers to prevent core rate tracking pollution

**Safety caps:**
- `_wait_for_rate_limit()` never sleeps more than 60s (prevents workflow timeout)
- Minimum budget of 100 requests per platform regardless of remaining
- Workflow timeout: 45 minutes

### Categories
- **trending**: High star velocity + recent activity. Sorted by trending score (platform score + velocity × 10)
- **new-releases**: Repos with stable releases in last 14 days. Sorted by release date
- **most-popular**: Repos with 5,000+ stars. Sorted by star count

### Platform Detection

Each platform has: topics, installer file extensions, scoring keywords (high/medium/low), primary/secondary languages, and frameworks. See `PLATFORMS` dict.

**Installer detection** uses two layers:
1. **Extension matching** — dedicated installer files:
   - Android: `.apk`, `.aab`
   - Windows: `.msi`, `.exe`, `.msix`
   - macOS: `.dmg`, `.pkg`, `.app.zip`
   - Linux: `.appimage`, `.deb`, `.rpm`
2. **Keyword matching** — generic archives (`.zip`, `.tar.gz`, `.tar.xz`, `.tar.bz2`, `.7z`) with platform keywords in the filename:
   - Android: `android`
   - Windows: `win64`, `win32`, `windows`, `-win-`, etc.
   - macOS: `macos`, `darwin`, `osx`, `-mac-`, etc.
   - Linux: `linux`, `-linux-`, etc.

### Content Filtering

`BLOCKED_TOPICS` set (~40 terms) excludes repos with NSFW/inappropriate content. Checked against both repo topics (set intersection) and description (substring match) during candidate collection, before any API calls are wasted on verification.

### Cache Protection

- Cache files are valid for 23 hours (`CACHE_VALIDITY_HOURS`)
- Stale caches with fewer than the minimum threshold repos are refetched (30 for trending/most-popular, 10 for new-releases)
- **Never saves 0 repos** — if a fetch returns 0, existing cache is preserved
- **Never overwrites good data with poor results** — if fetch returns fewer than threshold but cache has more, cache is kept
- `FORCE_REFRESH` env var bypasses cache loading entirely

### Fork Inclusion

All search queries include `fork:true` to discover forked repositories with platform installers.

## Key Constants

| Constant | Value | Notes |
|---|---|---|
| `RATE_LIMIT_FLOOR` | 50 | Global minimum — stop verifying below this |
| `CACHE_VALIDITY_HOURS` | 23 | Cache TTL |
| `MAX_CONCURRENT_REQUESTS` | 25 | HTTP concurrency (core API) |
| `MAX_SEARCH_CONCURRENT` | 5 | Search API concurrency |
| `RELEASE_CHECK_BATCH` | 40 | Repos verified per batch |
| `REQUEST_TIMEOUT` | 20 | Per-request timeout (seconds) |
| `MAX_RETRIES` | 3 | Per-request retry limit |
| `MIN_STARS` (most-popular) | 5000 | Minimum stars |
| `MAX_RELEASE_AGE_DAYS` (new-releases) | 14 | Max release age |

## Commands

```bash
# Run with 3 dedicated tokens (recommended — each from a different GitHub account)
GH_TOKEN_TRENDING=ghp_a GH_TOKEN_NEW_RELEASES=ghp_b GH_TOKEN_MOST_POPULAR=ghp_c python scripts/fetch_all_categories.py

# Run with single fallback token
GITHUB_TOKEN=ghp_xxx python scripts/fetch_all_categories.py

# Force refresh (ignore all caches)
FORCE_REFRESH=true GITHUB_TOKEN=ghp_xxx python scripts/fetch_all_categories.py

# Validate release dates
GITHUB_TOKEN=ghp_xxx python scripts/validate_releases.py [platform]

# Install deps
pip install -r scripts/requirements.txt
```

## Output JSON Format

Each `{platform}.json`:
```json
{
  "category": "trending",
  "platform": "android",
  "lastUpdated": "2026-03-08T...",
  "totalCount": 150,
  "repositories": [{ "id", "name", "fullName", "owner", "description", "stargazersCount", "forksCount", "language", "topics", "latestReleaseDate", ... }]
}
```

## Workflow Details

**Jobs:**
1. `check-rate-limit` — Checks all 3 tokens, reports dedicated vs fallback, gates on >1000 remaining
2. `fetch-and-update` — Runs the script, validates JSON, commits and pushes (retries push up to 3 times with rebase)
3. `notify-on-failure` — Auto-creates a GitHub issue labeled `automation, category-fetch, bug`

**Workflow inputs:**
- `force_refresh` (boolean) — Skip all caches when triggered manually

**Timeout:** 45 minutes

## Development Notes

- Python 3.11, no type-checking or linting configured
- No tests beyond `validate_releases.py`
- Each category creates its own `GitHubClient` — release cache is per-client, shared across platforms within the same category
- Platforms are processed sequentially within each category (with 65s search rate limit pause between)
- Typical runtime: 15-25 minutes with 3 dedicated tokens
- The `_check_assets()` helper inside `get_latest_stable_release()` detects installers for ALL platforms in one pass, so cross-platform repos benefit all platforms from a single release check
