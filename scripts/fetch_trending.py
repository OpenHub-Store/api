import os
import sys
import json
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from functools import lru_cache
import hashlib

# Validate GITHUB_TOKEN early
GITHUB_TOKEN = os.environ.get('GITHUB_TOKEN')
if not GITHUB_TOKEN:
    print("ERROR: GITHUB_TOKEN environment variable is not set or is empty.", file=sys.stderr)
    print("Please set GITHUB_TOKEN before running this script.", file=sys.stderr)
    sys.exit(1)

HEADERS = {
    'Authorization': f'token {GITHUB_TOKEN}',
    'Accept': 'application/vnd.github.v3+json'
}

# Platform configurations with enhanced scoring
PLATFORMS = {
    'android': {
        'topics': ['android', 'android-app', 'kotlin-android'],
        'installer_extensions': ['.apk', '.aab'],
        'score_keywords': {
            'high': ['android', 'kotlin-android'],
            'medium': ['mobile', 'kotlin', 'jetpack-compose'],
            'low': ['java', 'apk', 'gradle']
        },
        'languages': {
            'primary': ['kotlin', 'java'],
            'secondary': ['dart', 'c++']
        }
    },
    'windows': {
        'topics': ['windows', 'electron', 'desktop', 'windows-app'],
        'installer_extensions': ['.msi', '.exe', '.msix'],
        'score_keywords': {
            'high': ['windows', 'windows-app', 'wpf', 'winui'],
            'medium': ['desktop', 'electron', 'dotnet'],
            'low': ['app', 'gui', 'win32']
        },
        'languages': {
            'primary': ['c#', 'c++', 'rust'],
            'secondary': ['javascript', 'typescript']
        }
    },
    'macos': {
        'topics': ['macos', 'osx', 'mac', 'swiftui'],
        'installer_extensions': ['.dmg', '.pkg', '.app.zip'],
        'score_keywords': {
            'high': ['macos', 'swiftui', 'appkit'],
            'medium': ['desktop', 'swift', 'cocoa'],
            'low': ['app', 'mac']
        },
        'languages': {
            'primary': ['swift', 'objective-c'],
            'secondary': ['c++', 'rust']
        }
    },
    'linux': {
        'topics': ['linux', 'gtk', 'qt'],
        'installer_extensions': ['.appimage', '.deb', '.rpm', '.flatpak', '.snap'],
        'score_keywords': {
            'high': ['linux', 'gtk', 'qt'],
            'medium': ['desktop', 'gnome', 'kde'],
            'low': ['app', 'unix']
        },
        'languages': {
            'primary': ['c++', 'rust', 'c'],
            'secondary': ['python', 'go']
        }
    }
}

# Configuration
MAX_RETRIES = 3
INITIAL_BACKOFF = 2
MAX_WORKERS = 5  # Parallel installer checks

# Use absolute path from script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)  # Go up one level from scripts/
CACHE_DIR = os.path.join(REPO_ROOT, 'cached-data', 'trending')

CACHE_VALIDITY_HOURS = 23  # Slightly less than 24h to ensure fresh data

@dataclass
class RepoCandidate:
    """Structured repository candidate"""
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
    
    # Trending metrics
    recent_stars_velocity: float = 0.0  # Stars per day
    commit_frequency: float = 0.0  # Commits per week
    issue_activity: float = 0.0  # Recent issues/PRs
    
    def to_summary(self) -> Dict:
        """Convert to output format"""
        return {
            'id': self.id,
            'name': self.name,
            'fullName': self.full_name,
            'owner': {
                'login': self.owner_login,
                'avatarUrl': self.owner_avatar
            },
            'description': self.description,
            'defaultBranch': self.default_branch,
            'htmlUrl': self.html_url,
            'stargazersCount': self.stars,
            'forksCount': self.forks,
            'language': self.language,
            'topics': self.topics,
            'releasesUrl': self.releases_url,
            'updatedAt': self.updated_at,
            'trendingScore': round(self.score + (self.recent_stars_velocity * 10), 2)
        }

def exponential_backoff_sleep(attempt: int, retry_after: Optional[int] = None) -> None:
    """Sleep with exponential backoff, respecting Retry-After if provided"""
    if retry_after:
        sleep_time = retry_after
        print(f"Rate limited - sleeping for {sleep_time}s (from Retry-After header)")
    else:
        sleep_time = min(INITIAL_BACKOFF * (2 ** attempt), 60)
        print(f"Backoff attempt {attempt + 1} - sleeping for {sleep_time}s")
    time.sleep(sleep_time)

def make_request_with_retry(url: str, params: Optional[Dict] = None, timeout: int = 30) -> Tuple[Optional[requests.Response], Optional[str]]:
    """Make HTTP request with retry logic for rate limits and server errors"""
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, headers=HEADERS, params=params, timeout=timeout)

            if response.status_code == 200:
                return response, None

            if response.status_code in [403, 429]:
                retry_after = response.headers.get('Retry-After')
                retry_after_int = int(retry_after) if retry_after and retry_after.isdigit() else None

                try:
                    error_data = response.json()
                    is_rate_limit = 'rate limit' in error_data.get('message', '').lower()
                except:
                    is_rate_limit = response.status_code == 429

                if is_rate_limit:
                    if attempt < MAX_RETRIES - 1:
                        exponential_backoff_sleep(attempt, retry_after_int)
                        continue
                    else:
                        return None, f"Rate limit exceeded after {MAX_RETRIES} retries"
                else:
                    return None, f"Access forbidden (403)"

            if 500 <= response.status_code < 600:
                if attempt < MAX_RETRIES - 1:
                    exponential_backoff_sleep(attempt)
                    continue
                else:
                    return None, f"Server error {response.status_code}"

            return None, f"Request failed with status {response.status_code}"

        except requests.Timeout:
            if attempt < MAX_RETRIES - 1:
                exponential_backoff_sleep(attempt)
                continue
            else:
                return None, "Timeout after retries"

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                exponential_backoff_sleep(attempt)
                continue
            else:
                return None, str(e)

    return None, "Max retries exceeded"

def calculate_platform_score(repo: Dict, platform: str) -> int:
    """
    Enhanced scoring algorithm with weighted factors:
    - Topic relevance (highest weight)
    - Language fit (high weight)
    - Description keywords (medium weight)
    - Cross-platform bonus
    """
    score = 0
    topics = [t.lower() for t in repo.get('topics', [])]
    language = (repo.get('language') or '').lower()
    desc = (repo.get('description') or '').lower()
    
    config = PLATFORMS[platform]
    keywords = config['score_keywords']
    languages = config['languages']

    # Topic scoring (0-40 points)
    for keyword in keywords['high']:
        if keyword in topics:
            score += 15
    for keyword in keywords['medium']:
        if keyword in topics:
            score += 8
    for keyword in keywords['low']:
        if keyword in topics:
            score += 3

    # Language scoring (0-20 points)
    if language in languages['primary']:
        score += 20
    elif language in languages['secondary']:
        score += 10

    # Description scoring (0-15 points)
    for keyword in keywords['high']:
        if keyword in desc:
            score += 5
    for keyword in keywords['medium']:
        if keyword in desc:
            score += 3

    # Cross-platform bonus (0-15 points)
    cross_platform_keywords = ['cross-platform', 'multiplatform', 'multi-platform']
    for kw in cross_platform_keywords:
        if kw in topics or kw in desc:
            score += 15
            break

    # Popular framework bonus (0-10 points)
    popular_frameworks = {
        'android': ['jetpack-compose', 'android-jetpack'],
        'windows': ['wpf', 'winui', 'avalonia'],
        'macos': ['swiftui', 'combine'],
        'linux': ['gtk4', 'qt6']
    }
    for framework in popular_frameworks.get(platform, []):
        if framework in topics:
            score += 10
            break

    return score

def calculate_trending_metrics(repo: Dict) -> Tuple[float, float]:
    """
    Calculate trending velocity metrics
    Returns: (stars_per_day, age_in_days)
    """
    try:
        created = datetime.fromisoformat(repo['created_at'].replace('Z', '+00:00'))
        updated = datetime.fromisoformat(repo['updated_at'].replace('Z', '+00:00'))
        now = datetime.now(created.tzinfo)
        
        age_days = max((now - created).days, 1)
        days_since_update = (now - updated).days
        
        # Stars velocity with recency weight
        stars = repo['stargazers_count']
        stars_per_day = stars / age_days
        
        # Boost recently updated repos
        recency_multiplier = 1.0
        if days_since_update <= 7:
            recency_multiplier = 2.0
        elif days_since_update <= 30:
            recency_multiplier = 1.5
        elif days_since_update <= 90:
            recency_multiplier = 1.2
        
        adjusted_velocity = stars_per_day * recency_multiplier
        
        return adjusted_velocity, age_days
        
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return 0.0, 365

def check_repo_has_installers(owner: str, repo_name: str, platform: str) -> bool:
    """Check if repository has relevant installer files with caching"""
    cache_key = f"{owner}/{repo_name}/{platform}"
    
    url = f'https://api.github.com/repos/{owner}/{repo_name}/releases'
    response, error = make_request_with_retry(url, params={'per_page': 5}, timeout=10)

    if response is None:
        return False

    try:
        releases = response.json()
        
        # Check latest 3 releases (including prereleases for better coverage)
        for release in releases[:3]:
            if release.get('draft'):
                continue
                
            assets = release.get('assets', [])
            if not assets:
                continue

            extensions = PLATFORMS[platform]['installer_extensions']
            for asset in assets:
                asset_name = asset['name'].lower()
                if any(asset_name.endswith(ext) for ext in extensions):
                    return True

        return False

    except Exception as e:
        print(f"Error checking installers for {owner}/{repo_name}: {e}")
        return False

def check_installers_batch(candidates: List[RepoCandidate], platform: str) -> List[RepoCandidate]:
    """Check installers in parallel for better performance"""
    print(f"Checking {len(candidates)} candidates for installers (parallel)...")
    
    results = []
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_repo = {
            executor.submit(
                check_repo_has_installers,
                candidate.owner_login,
                candidate.name,
                platform
            ): candidate
            for candidate in candidates
        }
        
        for future in as_completed(future_to_repo):
            candidate = future_to_repo[future]
            try:
                has_installers = future.result()
                candidate.has_installers = has_installers
                
                if has_installers:
                    results.append(candidate)
                    print(f"✓ {candidate.full_name} (score: {candidate.score})")
                
            except Exception as e:
                print(f"Error checking {candidate.full_name}: {e}")
    
    return results

def fetch_trending_repos(platform: str, desired_count: int = 100) -> List[Dict]:
    """Fetch trending repositories with optimized search strategy"""
    print(f"\n{'='*60}")
    print(f"Fetching trending repos for {platform.upper()}")
    print(f"{'='*60}")

    url = 'https://api.github.com/search/repositories'
    topics = PLATFORMS[platform]['topics']
    
    all_candidates: List[RepoCandidate] = []
    seen: Set[str] = set()
    
    # Multi-strategy search for comprehensive coverage
    search_strategies = [
        # Strategy 1: Recent highly-starred with topics (most relevant)
        {
            'days': 30,
            'min_stars': 100,
            'topics': topics,
            'max_pages': 5,  # Increased from 3
            'weight': 1.5
        },
        # Strategy 2: Medium timeframe, broader (catch rising stars)
        {
            'days': 90,
            'min_stars': 50,
            'topics': topics[:1],  # Primary topic only
            'max_pages': 5,  # Increased from 3
            'weight': 1.2
        },
        # Strategy 3: Established projects (quality baseline)
        {
            'days': 180,
            'min_stars': 500,
            'topics': [],
            'max_pages': 3,  # Increased from 2
            'weight': 1.0
        },
        # Strategy 4: NEW - Longer timeframe for more coverage
        {
            'days': 365,
            'min_stars': 200,
            'topics': topics[:1] if topics else [],
            'max_pages': 3,
            'weight': 0.9
        }
    ]
    
    for strategy_idx, strategy in enumerate(search_strategies):
        print(f"\n--- Strategy {strategy_idx + 1} ---")
        print(f"Days: {strategy['days']}, Min stars: {strategy['min_stars']}, Topics: {strategy['topics'] or 'none'}")
        
        past_date = (datetime.utcnow() - timedelta(days=strategy['days'])).strftime('%Y-%m-%d')
        
        # Build query
        base_query = f"stars:>{strategy['min_stars']} archived:false pushed:>={past_date}"
        if strategy['topics']:
            topic_query = " OR ".join([f"topic:{t}" for t in strategy['topics']])
            query = f"{base_query} ({topic_query})"
        else:
            # Strategy 3: Use primary language for the platform instead of complex OR
            primary_lang = PLATFORMS[platform]['languages']['primary'][0]
            query = f"{base_query} language:{primary_lang}"
        
        page = 1
        while page <= strategy['max_pages']:
            print(f"Fetching page {page}...")
            
            params = {
                'q': query,
                'sort': 'stars',
                'order': 'desc',
                'per_page': 100,
                'page': page
            }
            
            response, error = make_request_with_retry(url, params=params, timeout=30)
            
            if response is None:
                print(f"Failed to fetch page {page}: {error}")
                break
            
            try:
                data = response.json()
                items = data.get('items', [])
                
                if not items:
                    break
                
                # Process repositories
                for repo in items:
                    full_name = repo['full_name']
                    if full_name in seen:
                        continue
                    
                    seen.add(full_name)
                    
                    # Calculate scores
                    base_score = calculate_platform_score(repo, platform)
                    velocity, age = calculate_trending_metrics(repo)
                    
                    # Apply strategy weight
                    weighted_score = int(base_score * strategy['weight'])
                    
                    # Filter: minimum score threshold (lowered to get more results)
                    if weighted_score < 5:  # Changed from 10
                        continue
                    
                    candidate = RepoCandidate(
                        id=repo['id'],
                        name=repo['name'],
                        full_name=full_name,
                        owner_login=repo['owner']['login'],
                        owner_avatar=repo['owner']['avatar_url'],
                        description=repo.get('description'),
                        default_branch=repo.get('default_branch', 'main'),
                        html_url=repo['html_url'],
                        stars=repo['stargazers_count'],
                        forks=repo['forks_count'],
                        language=repo.get('language'),
                        topics=repo.get('topics', []),
                        releases_url=repo['releases_url'],
                        updated_at=repo['updated_at'],
                        created_at=repo['created_at'],
                        score=weighted_score,
                        recent_stars_velocity=velocity
                    )
                    
                    all_candidates.append(candidate)
                
                print(f"Collected {len(items)} repos from page {page} (total candidates: {len(all_candidates)})")
                
            except Exception as e:
                print(f"Error processing page {page}: {e}", file=sys.stderr)
                break
            
            page += 1
            time.sleep(0.5)  # Rate limiting courtesy
    
    # Sort candidates by combined score (base score + velocity)
    all_candidates.sort(
        key=lambda c: c.score + (c.recent_stars_velocity * 10),
        reverse=True
    )
    
    # Take top N candidates for installer check (more than desired to account for filtering)
    top_candidates = all_candidates[:min(len(all_candidates), desired_count * 4)]  # Changed from 3x to 4x
    
    print(f"\nTop {len(top_candidates)} candidates selected for installer verification")
    
    # Parallel installer check
    verified_repos = check_installers_batch(top_candidates, platform)
    
    # Sort verified repos and take desired count
    verified_repos.sort(
        key=lambda c: c.score + (c.recent_stars_velocity * 10),
        reverse=True
    )
    
    final_repos = verified_repos[:desired_count]
    
    print(f"\n{'='*60}")
    print(f"Final count: {len(final_repos)} repositories for {platform}")
    print(f"{'='*60}\n")
    
    return [repo.to_summary() for repo in final_repos]

def load_cache(platform: str) -> Optional[Dict]:
    """Load cached data if valid"""
    cache_file = f'{CACHE_DIR}/{platform}.json'
    
    if not os.path.exists(cache_file):
        return None
    
    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Reject empty or invalid cache
        repo_count = data.get('totalCount', 0)
        if repo_count < 30:  # Minimum threshold (30% of target)
            print(f"Cache for {platform} has insufficient data ({repo_count} repos), refetching...")
            return None
        
        # Check cache validity
        last_updated = datetime.fromisoformat(data['lastUpdated'].replace('Z', '+00:00'))
        age_hours = (datetime.now(last_updated.tzinfo) - last_updated).total_seconds() / 3600
        
        if age_hours < CACHE_VALIDITY_HOURS:
            print(f"Using cached data for {platform} (age: {age_hours:.1f}h, {repo_count} repos)")
            return data
        
    except Exception as e:
        print(f"Error loading cache for {platform}: {e}")
    
    return None

def main():
    """Main function to fetch and save trending repos for all platforms"""
    timestamp = datetime.utcnow().isoformat() + 'Z'
    
    # Check rate limit before starting
    response, _ = make_request_with_retry('https://api.github.com/rate_limit')
    if response:
        rate_data = response.json()
        core_remaining = rate_data.get('resources', {}).get('core', {}).get('remaining', 0)
        print(f"GitHub API rate limit: {core_remaining} requests remaining\n")
        
        if core_remaining < 100:
            print("WARNING: Low rate limit remaining. Consider running later.", file=sys.stderr)
    
    for platform in PLATFORMS.keys():
        print(f"\n{'#'*60}")
        print(f"# Processing {platform.upper()}")
        print(f"{'#'*60}")
        
        # Check cache first
        cached_data = load_cache(platform)
        if cached_data:
            print(f"Skipping {platform} - using cache")
            continue
        
        repos = fetch_trending_repos(platform, desired_count=100)
        
        output = {
            'platform': platform,
            'lastUpdated': timestamp,
            'totalCount': len(repos),
            'repositories': repos
        }
        
        # Save to file
        os.makedirs(CACHE_DIR, exist_ok=True)
        output_file = f'{CACHE_DIR}/{platform}.json'
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Saved {len(repos)} repos to {output_file}")
        
        # Delay between platforms
        time.sleep(3)
    
    print("\n✓ All platforms processed successfully!")

if __name__ == '__main__':
    main()