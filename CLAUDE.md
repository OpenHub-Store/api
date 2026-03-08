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
2. Each category gets its own `GitHubClient` with a dedicated token (5,000 req/hr each = 15,000 total)
3. For each category × platform (12 combos), checks if cached JSON is fresh (<23h)
4. If stale, queries GitHub Search API with platform-specific topics/languages/keywords
5. Filters repos that have **real releases with platform installers** (e.g. `.apk` for android, `.exe`/`.msi` for windows)
6. Verifies ALL candidates — no artificial caps. Stops gracefully when rate limit drops below `RATE_LIMIT_FLOOR` (50)
7. Saves results to `cached-data/{category}/{platform}.json`
8. GitHub Actions commits and pushes changes

### Token Strategy
- 3 GitHub Classic PATs (scope: `public_repo`), one per category
- Stored as GitHub Actions secrets: `GH_TOKEN_TRENDING`, `GH_TOKEN_NEW_RELEASES`, `GH_TOKEN_MOST_POPULAR`
- Backward compatible: falls back to single `GITHUB_TOKEN` if per-category tokens aren't set

### Categories
- **trending**: High star velocity + recent activity. Sorted by trending score (platform score + velocity x 10)
- **new-releases**: Repos with stable releases in last 14 days. Sorted by release date
- **most-popular**: Repos with 5000+ stars. Sorted by star count

### Platform Detection
Each platform has defined: topics, installer file extensions, scoring keywords (high/medium/low), primary/secondary languages, and frameworks. See `PLATFORMS` dict in fetch script.

## Key Constants

| Constant | Value | Notes |
|---|---|---|
| `RATE_LIMIT_FLOOR` | 50 | Stop verifying when rate limit drops below this |
| `CACHE_VALIDITY_HOURS` | 23 | Cache TTL |
| `MAX_CONCURRENT_REQUESTS` | 25 | HTTP concurrency limit |
| `MAX_SEARCH_CONCURRENT` | 5 | Search API concurrency (stricter) |
| `MAX_RETRIES` | 3 | Per-request retry limit |
| `MIN_STARS` (most-popular) | 5000 | Minimum stars for most-popular |
| `MAX_RELEASE_AGE_DAYS` (new-releases) | 14 | Max release age |

## Commands

```bash
# Run with 3 dedicated tokens (recommended)
GH_TOKEN_TRENDING=ghp_a GH_TOKEN_NEW_RELEASES=ghp_b GH_TOKEN_MOST_POPULAR=ghp_c python scripts/fetch_all_categories.py

# Run with single fallback token
GITHUB_TOKEN=ghp_xxx python scripts/fetch_all_categories.py

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

## Development Notes

- Python 3.11, no type-checking or linting configured
- No tests beyond `validate_releases.py`
- Each category creates its own `GitHubClient` — release cache is per-client, shared across platforms within the same category
- Platforms are processed sequentially within each category to avoid rate-limit thrashing
- The workflow retries `git push` up to 3 times with rebase on conflict
- On failure, the workflow auto-creates a GitHub issue labeled `automation, category-fetch, bug`
