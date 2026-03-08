# GitHub Store API

Automated pipeline that discovers open-source desktop and mobile applications on GitHub and categorizes them by platform and popularity. Runs daily via GitHub Actions and outputs structured JSON consumed by the GitHub Store frontend.

## What It Does

The script searches GitHub for repositories that ship **real platform installers** (`.apk`, `.exe`, `.dmg`, `.deb`, etc.) in their releases, then categorizes and ranks them:

| Category | Description | Sorting |
|---|---|---|
| **Trending** | Repos with high star velocity and recent activity | Trending score (platform relevance + star velocity) |
| **New Releases** | Repos with a stable release in the last 14 days | Release date (newest first) |
| **Most Popular** | Repos with 5,000+ stars | Star count |

Each category is fetched across 4 platforms:

| Platform | Installer types | Primary languages |
|---|---|---|
| **Android** | `.apk`, `.aab` | Kotlin, Java |
| **Windows** | `.exe`, `.msi`, `.msix` | C#, C++, Rust |
| **macOS** | `.dmg`, `.pkg`, `.app.zip` | Swift, Objective-C |
| **Linux** | `.AppImage`, `.deb`, `.rpm` | C++, Rust, C |

## Requirements

- Python 3.11+
- GitHub Personal Access Tokens (Classic) with `public_repo` scope

### Install dependencies

```bash
pip install -r scripts/requirements.txt
```

## Authentication

The script supports two modes:

### 3-token mode (recommended)

One dedicated token per category. Each gets its own 5,000 requests/hour rate limit (15,000 total).

```bash
export GH_TOKEN_TRENDING=ghp_...
export GH_TOKEN_NEW_RELEASES=ghp_...
export GH_TOKEN_MOST_POPULAR=ghp_...
```

**Creating tokens:** Go to [github.com/settings/tokens](https://github.com/settings/tokens) (Classic) and create 3 tokens with the `public_repo` scope.

### Single-token mode (fallback)

All categories share one token (5,000 requests/hour).

```bash
export GITHUB_TOKEN=ghp_...
```

If per-category tokens are set, they take priority. `GITHUB_TOKEN` is used as a fallback for any category missing a dedicated token.

## Usage

```bash
# Run with 3 dedicated tokens
GH_TOKEN_TRENDING=ghp_a GH_TOKEN_NEW_RELEASES=ghp_b GH_TOKEN_MOST_POPULAR=ghp_c \
  python scripts/fetch_all_categories.py

# Run with single token
GITHUB_TOKEN=ghp_x python scripts/fetch_all_categories.py

# Force refresh (ignore cache)
FORCE_REFRESH=true GITHUB_TOKEN=ghp_x python scripts/fetch_all_categories.py

# Validate release dates against GitHub API
GITHUB_TOKEN=ghp_x python scripts/validate_releases.py [platform]
```

## Output

Results are saved to `cached-data/{category}/{platform}.json`:

```
cached-data/
  trending/
    android.json, windows.json, macos.json, linux.json
  new-releases/
    android.json, windows.json, macos.json, linux.json
  most-popular/
    android.json, windows.json, macos.json, linux.json
```

Each file contains:

```json
{
  "category": "trending",
  "platform": "android",
  "lastUpdated": "2026-03-08T02:00:00Z",
  "totalCount": 130,
  "repositories": [
    {
      "id": 123456,
      "name": "example-app",
      "fullName": "owner/example-app",
      "owner": { "login": "owner", "avatarUrl": "https://..." },
      "description": "An example app",
      "defaultBranch": "main",
      "htmlUrl": "https://github.com/owner/example-app",
      "stargazersCount": 5000,
      "forksCount": 300,
      "language": "Kotlin",
      "topics": ["android", "kotlin"],
      "releasesUrl": "https://api.github.com/repos/owner/example-app/releases{/id}",
      "updatedAt": "2026-03-07T...",
      "createdAt": "2024-01-01T...",
      "latestReleaseDate": "2026-03-05T...",
      "releaseRecency": 3,
      "releaseRecencyText": "Released 3 days ago",
      "trendingScore": 42.5
    }
  ]
}
```

Cache validity is 23 hours. Cached results are reused unless expired, below a minimum repo threshold, or `FORCE_REFRESH=true` is set.

## GitHub Actions

The workflow runs daily at **02:00 UTC** and can be triggered manually.

### Setup

Add these **repository secrets** (Settings > Secrets and variables > Actions):

| Secret | Required | Description |
|---|---|---|
| `GH_TOKEN_TRENDING` | Yes | Classic PAT for trending category |
| `GH_TOKEN_NEW_RELEASES` | Yes | Classic PAT for new-releases category |
| `GH_TOKEN_MOST_POPULAR` | Yes | Classic PAT for most-popular category |
| `GITHUB_TOKEN` | Auto | Provided by GitHub Actions (used as fallback) |

### Manual trigger

Go to **Actions > Fetch All Repository Categories > Run workflow**. Check **Force refresh** to ignore all caches.

### Failure handling

If the workflow fails, it automatically creates a GitHub issue labeled `automation`, `category-fetch`, `bug`.

## How It Works

1. For each category, creates a `GitHubClient` with its dedicated token
2. Checks if cached JSON is fresh (< 23 hours old)
3. Queries GitHub Search API with platform-specific topics, languages, and keywords
4. Collects all candidates across multiple search strategies (topic-based, language-based, cross-platform frameworks, high-star catch-all)
5. Verifies each candidate has a **stable release** with **platform-specific installer assets**
6. Ranks and saves results
7. Stops gracefully if rate limit drops below 50 remaining requests

### Rate limit strategy

- Each category uses its own token with an independent 5,000 req/hr budget
- Search API rate limits (30/min) are tracked separately from core API limits
- Release verification results are cached across platforms within the same category
- Platforms are processed sequentially to avoid rate-limit thrashing

## Project Structure

```
scripts/
  fetch_all_categories.py   -- Main fetcher (async Python, aiohttp)
  validate_releases.py      -- Release date validation tool
  requirements.txt          -- Python dependencies
cached-data/                -- Output JSON files (committed by CI)
.github/workflows/
  fetch_all_categories_workflow.yml  -- Daily cron + manual dispatch
```

## License

Apache 2.0
