import os
import sys
import json
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

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

# Platform configurations
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
        'topics': ['linux', 'gtk', 'qt', 'gnome', 'kde'],
        'installer_extensions': ['.appimage', '.deb', '.rpm'],
        'score_keywords': {
            'high': ['linux', 'gtk', 'qt', 'gnome'],
            'medium': ['desktop', 'gnome', 'kde', 'flatpak'],
            'low': ['app', 'unix', 'gui']
        },
        'languages': {
            'primary': ['c++', 'rust', 'c'],
            'secondary': ['python', 'go', 'vala']
        }
    }
}

# Configuration - optimized for QUALITY over speed
MAX_RETRIES = 3
INITIAL_BACKOFF = 2
MAX_WORKERS = 5

# Use absolute path from script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
CACHE_DIR = os.path.join(REPO_ROOT, 'cached-data')
CACHE_VALIDITY_HOURS = 23

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
    recent_stars_velocity: float = 0.0
    latest_release_date: Optional[str] = None
    
    def to_summary(self, category: str = 'trending') -> Dict:
        """Convert to output format with category-specific fields"""
        base = {
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
            'createdAt': self.created_at
        }
        
        # Add category-specific metrics
        if category == 'trending':
            base['trendingScore'] = round(self.score + (self.recent_stars_velocity * 10), 2)
        elif category == 'new-releases':
            base['latestReleaseDate'] = self.latest_release_date
            recency = self._calculate_release_age()
            base['releaseRecency'] = recency
            # Add human-readable time
            if recency == 0:
                base['releaseRecencyText'] = 'Released today'
            elif recency == 1:
                base['releaseRecencyText'] = 'Released yesterday'
            elif recency < 7:
                base['releaseRecencyText'] = f'Released {recency} days ago'
            else:
                base['releaseRecencyText'] = f'Released {recency} days ago'
        elif category == 'most-popular':
            base['popularityScore'] = self.stars + (self.forks * 2)
            
        return base
    
    def _calculate_release_age(self) -> int:
        """Calculate days since latest release"""
        if not self.latest_release_date:
            return 999
        try:
            # Parse the release date - strip timezone for naive datetime
            release_date_str = self.latest_release_date.replace('Z', '')
            if '+' in release_date_str:
                release_date_str = release_date_str.split('+')[0]
            release_date = datetime.fromisoformat(release_date_str)
            
            # Get current UTC time as naive datetime
            now = datetime.utcnow()
            
            # Calculate difference
            age = (now - release_date).days
            return max(0, age)  # Ensure non-negative
        except Exception as e:
            print(f"  ⚠ Error calculating age for {self.latest_release_date}: {e}")
            return 999

def exponential_backoff_sleep(attempt: int, retry_after: Optional[int] = None) -> None:
    """Sleep with exponential backoff"""
    if retry_after:
        sleep_time = retry_after
    else:
        sleep_time = min(INITIAL_BACKOFF * (2 ** attempt), 60)
    time.sleep(sleep_time)

def make_request_with_retry(url: str, params: Optional[Dict] = None, timeout: int = 30) -> Tuple[Optional[requests.Response], Optional[str]]:
    """Make HTTP request with retry logic"""
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

                if is_rate_limit and attempt < MAX_RETRIES - 1:
                    exponential_backoff_sleep(attempt, retry_after_int)
                    continue
                else:
                    return None, f"Rate limit or access denied"

            if 500 <= response.status_code < 600 and attempt < MAX_RETRIES - 1:
                exponential_backoff_sleep(attempt)
                continue

            return None, f"Request failed with status {response.status_code}"

        except requests.Timeout:
            if attempt < MAX_RETRIES - 1:
                exponential_backoff_sleep(attempt)
                continue
            else:
                return None, "Timeout"

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                exponential_backoff_sleep(attempt)
                continue
            else:
                return None, str(e)

    return None, "Max retries exceeded"

def calculate_platform_score(repo: Dict, platform: str) -> int:
    """Calculate relevance score for a repository"""
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

    # Framework bonus (0-10 points)
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
    """Calculate trending velocity metrics"""
    try:
        created = datetime.fromisoformat(repo['created_at'].replace('Z', '+00:00'))
        updated = datetime.fromisoformat(repo['updated_at'].replace('Z', '+00:00'))
        now = datetime.now(created.tzinfo)
        
        age_days = max((now - created).days, 1)
        days_since_update = (now - updated).days
        
        stars = repo['stargazers_count']
        stars_per_day = stars / age_days
        
        # Recency multiplier
        recency_multiplier = 1.0
        if days_since_update <= 7:
            recency_multiplier = 2.0
        elif days_since_update <= 30:
            recency_multiplier = 1.5
        elif days_since_update <= 90:
            recency_multiplier = 1.2
        
        adjusted_velocity = stars_per_day * recency_multiplier
        
        return adjusted_velocity, age_days
        
    except Exception:
        return 0.0, 365

def get_latest_stable_release(owner: str, repo_name: str) -> Tuple[Optional[str], Optional[Dict]]:
    """
    Get the latest STABLE (non-prerelease, non-draft) release
    Returns: (published_at, release_data)
    """
    # Try the /releases/latest endpoint first (fastest, gets latest stable)
    latest_url = f'https://api.github.com/repos/{owner}/{repo_name}/releases/latest'
    response, error = make_request_with_retry(latest_url, timeout=10)
    
    if response and response.status_code == 200:
        try:
            release = response.json()
            # Double-check it's not a pre-release or draft
            if not release.get('draft') and not release.get('prerelease'):
                return release.get('published_at'), release
        except:
            pass
    
    # Fallback: manually search through releases
    releases_url = f'https://api.github.com/repos/{owner}/{repo_name}/releases'
    response, error = make_request_with_retry(releases_url, params={'per_page': 10}, timeout=10)
    
    if response is None:
        return None, None
    
    try:
        releases = response.json()
        
        # Find first stable release (not draft, not prerelease)
        for release in releases:
            if not release.get('draft') and not release.get('prerelease'):
                return release.get('published_at'), release
        
        return None, None
        
    except:
        return None, None

def check_repo_has_installers(owner: str, repo_name: str, platform: str, get_release_date: bool = False) -> Tuple[bool, Optional[str]]:
    """Check if repository has relevant installer files"""
    
    # Get latest stable release
    published_at, release_data = get_latest_stable_release(owner, repo_name)
    
    if not release_data:
        return False, None
    
    # Check if release has installer assets
    assets = release_data.get('assets', [])
    if not assets:
        return False, None
    
    # OPTIMIZATION: Limit assets checked for Linux (it often has 20+ files)
    if platform == 'linux' and len(assets) > 15:
        assets = assets[:15]  # Only check first 15 assets
    
    extensions = PLATFORMS[platform]['installer_extensions']
    has_installer = False
    
    for asset in assets:
        asset_name = asset['name'].lower()
        
        if platform == 'linux':
            # Check common extensions first (most likely to match)
            if asset_name.endswith(('.appimage', '.deb', '.rpm')):
                has_installer = True
                break
        else:
            if any(asset_name.endswith(ext) for ext in extensions):
                has_installer = True
                break
    
    if has_installer:
        return True, published_at if get_release_date else None
    
    return False, None

def check_installers_batch(candidates: List[RepoCandidate], platform: str, get_release_dates: bool = False) -> List[RepoCandidate]:
    """Check installers in parallel"""
    results = []
    
    print(f"  Checking {len(candidates)} repos (this may take a few minutes)...")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_repo = {
            executor.submit(
                check_repo_has_installers,
                candidate.owner_login,
                candidate.name,
                platform,
                get_release_dates
            ): candidate
            for candidate in candidates
        }
        
        checked = 0
        for future in as_completed(future_to_repo):
            candidate = future_to_repo[future]
            checked += 1
            if checked % 50 == 0:
                print(f"  Progress: {checked}/{len(candidates)} checked...")
            try:
                has_installers, release_date = future.result(timeout=15)
                candidate.has_installers = has_installers
                candidate.latest_release_date = release_date
                
                if has_installers:
                    results.append(candidate)
                
            except Exception:
                pass
    
    return results

def fetch_trending_repos(platform: str, desired_count: int = 100) -> List[Dict]:
    """Fetch trending repositories with velocity-based scoring"""
    print(f"\n{'='*60}")
    print(f"Fetching TRENDING repos for {platform.upper()}")
    print(f"{'='*60}")

    url = 'https://api.github.com/search/repositories'
    topics = PLATFORMS[platform]['topics']
    
    all_candidates: List[RepoCandidate] = []
    seen: Set[str] = set()
    
    # More comprehensive search strategies for quality
    search_strategies = [
        {'days': 30, 'min_stars': 100, 'topics': topics, 'max_pages': 8, 'weight': 1.5},
        {'days': 90, 'min_stars': 50, 'topics': topics[:1], 'max_pages': 8, 'weight': 1.2},
        {'days': 180, 'min_stars': 500, 'topics': [], 'max_pages': 5, 'weight': 1.0},
        {'days': 365, 'min_stars': 200, 'topics': topics[:1] if topics else [], 'max_pages': 5, 'weight': 0.9}
    ]
    
    for strategy_idx, strategy in enumerate(search_strategies):
        print(f"Strategy {strategy_idx + 1}: {strategy['days']}d, {strategy['min_stars']}+ stars")
        
        past_date = (datetime.utcnow() - timedelta(days=strategy['days'])).strftime('%Y-%m-%d')
        base_query = f"stars:>{strategy['min_stars']} archived:false pushed:>={past_date}"
        
        if strategy['topics']:
            topic_query = " OR ".join([f"topic:{t}" for t in strategy['topics']])
            query = f"{base_query} ({topic_query})"
        else:
            primary_lang = PLATFORMS[platform]['languages']['primary'][0]
            query = f"{base_query} language:{primary_lang}"
        
        for page in range(1, strategy['max_pages'] + 1):
            params = {'q': query, 'sort': 'stars', 'order': 'desc', 'per_page': 100, 'page': page}
            response, error = make_request_with_retry(url, params=params, timeout=30)
            
            if response is None:
                break
            
            try:
                items = response.json().get('items', [])
                if not items:
                    break
                
                for repo in items:
                    full_name = repo['full_name']
                    if full_name in seen:
                        continue
                    seen.add(full_name)
                    
                    base_score = calculate_platform_score(repo, platform)
                    velocity, age = calculate_trending_metrics(repo)
                    weighted_score = int(base_score * strategy['weight'])
                    
                    if weighted_score < 5:
                        continue
                    
                    candidate = RepoCandidate(
                        id=repo['id'], name=repo['name'], full_name=full_name,
                        owner_login=repo['owner']['login'], owner_avatar=repo['owner']['avatar_url'],
                        description=repo.get('description'), default_branch=repo.get('default_branch', 'main'),
                        html_url=repo['html_url'], stars=repo['stargazers_count'], forks=repo['forks_count'],
                        language=repo.get('language'), topics=repo.get('topics', []),
                        releases_url=repo['releases_url'], updated_at=repo['updated_at'],
                        created_at=repo['created_at'], score=weighted_score, recent_stars_velocity=velocity
                    )
                    all_candidates.append(candidate)
                    
            except Exception:
                break
            
            time.sleep(0.2)  # Reduced sleep for faster execution
    
    all_candidates.sort(key=lambda c: c.score + (c.recent_stars_velocity * 10), reverse=True)
    # Check MORE candidates (5x instead of 4x)
    top_candidates = all_candidates[:min(len(all_candidates), desired_count * 5)]
    verified_repos = check_installers_batch(top_candidates, platform, get_release_dates=False)
    verified_repos.sort(key=lambda c: c.score + (c.recent_stars_velocity * 10), reverse=True)
    final_repos = verified_repos[:desired_count]
    
    print(f"✓ Found {len(final_repos)} trending repos")
    return [repo.to_summary('trending') for repo in final_repos]

def fetch_new_releases(platform: str, desired_count: int = 100) -> List[Dict]:
    """Fetch repos with new STABLE releases in last 21 days"""
    print(f"\n{'='*60}")
    print(f"Fetching NEW RELEASES for {platform.upper()}")
    print(f"{'='*60}")

    url = 'https://api.github.com/search/repositories'
    topics = PLATFORMS[platform]['topics']
    
    all_candidates: List[RepoCandidate] = []
    seen: Set[str] = set()
    
    # Comprehensive search for new releases - prioritize quality over speed
    search_strategies = [
        {'days': 7, 'min_stars': 50, 'topics': topics, 'max_pages': 8},
        {'days': 14, 'min_stars': 30, 'topics': topics, 'max_pages': 8},
        {'days': 21, 'min_stars': 100, 'topics': topics[:1] if topics else [], 'max_pages': 8},
        {'days': 21, 'min_stars': 500, 'topics': [], 'max_pages': 8},
        {'days': 14, 'min_stars': 10, 'topics': topics, 'max_pages': 5}
    ]
    
    for strategy_idx, strategy in enumerate(search_strategies):
        print(f"Strategy {strategy_idx + 1}: Last {strategy['days']} days, {strategy['min_stars']}+ stars")
        
        past_date = (datetime.utcnow() - timedelta(days=strategy['days'])).strftime('%Y-%m-%d')
        base_query = f"stars:>{strategy['min_stars']} archived:false pushed:>={past_date}"
        
        if strategy['topics']:
            topic_query = " OR ".join([f"topic:{t}" for t in strategy['topics']])
            query = f"{base_query} ({topic_query})"
        else:
            primary_lang = PLATFORMS[platform]['languages']['primary'][0]
            query = f"{base_query} language:{primary_lang}"
        
        for page in range(1, strategy['max_pages'] + 1):
            params = {'q': query, 'sort': 'updated', 'order': 'desc', 'per_page': 100, 'page': page}
            response, error = make_request_with_retry(url, params=params, timeout=30)
            
            if response is None:
                break
            
            try:
                items = response.json().get('items', [])
                if not items:
                    break
                
                for repo in items:
                    full_name = repo['full_name']
                    if full_name in seen:
                        continue
                    seen.add(full_name)
                    
                    score = calculate_platform_score(repo, platform)
                    
                    candidate = RepoCandidate(
                        id=repo['id'], name=repo['name'], full_name=full_name,
                        owner_login=repo['owner']['login'], owner_avatar=repo['owner']['avatar_url'],
                        description=repo.get('description'), default_branch=repo.get('default_branch', 'main'),
                        html_url=repo['html_url'], stars=repo['stargazers_count'], forks=repo['forks_count'],
                        language=repo.get('language'), topics=repo.get('topics', []),
                        releases_url=repo['releases_url'], updated_at=repo['updated_at'],
                        created_at=repo['created_at'], score=score
                    )
                    all_candidates.append(candidate)
                    
            except Exception:
                break
            
            time.sleep(0.2)
    
    print(f"\n✓ Collected {len(all_candidates)} candidates")
    
    # Define cutoff BEFORE using it
    twenty_one_days_ago = datetime.utcnow() - timedelta(days=21)
    now = datetime.utcnow()
    
    # Sort and check MORE candidates for better coverage
    all_candidates.sort(key=lambda c: c.updated_at, reverse=True)
    top_candidates = all_candidates[:min(len(all_candidates), desired_count * 4)]  # Check 8x candidates
    
    print(f"Checking {len(top_candidates)} candidates for recent STABLE releases...")
    print(f"Looking for releases published in last 21 days")
    print(f"Current UTC: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Cutoff date: {twenty_one_days_ago.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    verified_repos = check_installers_batch(top_candidates, platform, get_release_dates=True)
    
    print(f"\nValidating {len(verified_repos)} repos with releases...")
    recent_releases = []

    for repo in verified_repos:
        if not repo.latest_release_date:
            continue
        
        try:
            # Parse release date - strip timezone for naive datetime
            release_date_str = repo.latest_release_date.replace('Z', '')
            if '+' in release_date_str:
                release_date_str = release_date_str.split('+')[0]
            release_date_utc = datetime.fromisoformat(release_date_str)
            
            # Calculate actual age in days
            days_ago = (now - release_date_utc).days
            
            # Validate: release must be within last 21 days
            if days_ago <= 21:
                # Validate: release must not be in the future (allow 1 hour clock skew)
                if release_date_utc <= now + timedelta(hours=1):
                    recent_releases.append(repo)
                    print(f"  ✓ {repo.full_name}: Released {days_ago}d ago")
                else:
                    print(f"  ✗ {repo.full_name}: Future release date (skipped)")
            else:
                print(f"  ✗ {repo.full_name}: Too old ({days_ago}d ago, need ≤21d)")
                
        except Exception as e:
            print(f"  ✗ {repo.full_name}: Error - {e}")
            continue
    
    # Sort by release date (newest first)
    recent_releases.sort(key=lambda r: r.latest_release_date or '', reverse=True)
    final_repos = recent_releases[:desired_count]
    
    print(f"\n{'='*60}")
    if len(final_repos) > 0:
        print(f"✓ Found {len(final_repos)} repos with new STABLE releases")
    else:
        print(f"⚠️  Found 0 repos with new releases in last 21 days")
        print(f"   Checked {len(verified_repos)} repos with installers")
    print(f"{'='*60}")
    
    return [repo.to_summary('new-releases') for repo in final_repos]

def fetch_most_popular(platform: str, desired_count: int = 100) -> List[Dict]:
    """Fetch most popular (highest stars) mature repositories"""
    print(f"\n{'='*60}")
    print(f"Fetching MOST POPULAR repos for {platform.upper()}")
    print(f"{'='*60}")

    url = 'https://api.github.com/search/repositories'
    topics = PLATFORMS[platform]['topics']
    
    all_candidates: List[RepoCandidate] = []
    seen: Set[str] = set()
    
    # Search for high-star repos that are mature and active
    six_months_ago = (datetime.utcnow() - timedelta(days=180)).strftime('%Y-%m-%d')
    one_year_ago = (datetime.utcnow() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    # More comprehensive search
    search_strategies = [
        {'min_stars': 5000, 'topics': topics, 'max_pages': 8, 'created_before': six_months_ago},
        {'min_stars': 2000, 'topics': topics[:1] if topics else [], 'max_pages': 8, 'created_before': six_months_ago},
        {'min_stars': 1000, 'topics': [], 'max_pages': 5, 'created_before': one_year_ago}
    ]
    
    for strategy_idx, strategy in enumerate(search_strategies):
        print(f"Strategy {strategy_idx + 1}: {strategy['min_stars']}+ stars, created before {strategy['created_before']}")
        
        base_query = f"stars:>{strategy['min_stars']} archived:false pushed:>={one_year_ago} created:<{strategy['created_before']}"
        
        if strategy['topics']:
            topic_query = " OR ".join([f"topic:{t}" for t in strategy['topics']])
            query = f"{base_query} ({topic_query})"
        else:
            primary_lang = PLATFORMS[platform]['languages']['primary'][0]
            query = f"{base_query} language:{primary_lang}"
        
        for page in range(1, strategy['max_pages'] + 1):
            params = {'q': query, 'sort': 'stars', 'order': 'desc', 'per_page': 100, 'page': page}
            response, error = make_request_with_retry(url, params=params, timeout=30)
            
            if response is None:
                break
            
            try:
                items = response.json().get('items', [])
                if not items:
                    break
                
                for repo in items:
                    full_name = repo['full_name']
                    if full_name in seen:
                        continue
                    seen.add(full_name)
                    
                    score = calculate_platform_score(repo, platform)
                    
                    candidate = RepoCandidate(
                        id=repo['id'], name=repo['name'], full_name=full_name,
                        owner_login=repo['owner']['login'], owner_avatar=repo['owner']['avatar_url'],
                        description=repo.get('description'), default_branch=repo.get('default_branch', 'main'),
                        html_url=repo['html_url'], stars=repo['stargazers_count'], forks=repo['forks_count'],
                        language=repo.get('language'), topics=repo.get('topics', []),
                        releases_url=repo['releases_url'], updated_at=repo['updated_at'],
                        created_at=repo['created_at'], score=score
                    )
                    all_candidates.append(candidate)
                    
            except Exception:
                break
            
            time.sleep(0.2)
    
    # Sort by stars and check MORE candidates
    all_candidates.sort(key=lambda c: c.stars, reverse=True)
    top_candidates = all_candidates[:min(len(all_candidates), desired_count * 5)]
    
    print(f"Checking {len(top_candidates)} candidates for installers...")
    verified_repos = check_installers_batch(top_candidates, platform, get_release_dates=False)
    
    verified_repos.sort(key=lambda c: c.stars, reverse=True)
    final_repos = verified_repos[:desired_count]
    
    print(f"✓ Found {len(final_repos)} most popular repos")
    return [repo.to_summary('most-popular') for repo in final_repos]

def load_cache(category: str, platform: str) -> Optional[Dict]:
    """Load cached data if valid"""
    cache_file = os.path.join(CACHE_DIR, category, f'{platform}.json')
    
    if not os.path.exists(cache_file):
        return None
    
    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        repo_count = data.get('totalCount', 0)
        # Lower threshold for new-releases since it's more volatile
        min_threshold = 10 if category == 'new-releases' else 30
        if repo_count < min_threshold:
            print(f"Cache for {category}/{platform} has insufficient data ({repo_count} repos), refetching...")
            return None
        
        last_updated = datetime.fromisoformat(data['lastUpdated'].replace('Z', '+00:00'))
        age_hours = (datetime.now(last_updated.tzinfo) - last_updated).total_seconds() / 3600
        
        if age_hours < CACHE_VALIDITY_HOURS:
            print(f"✓ Using cache for {category}/{platform} ({age_hours:.1f}h old, {repo_count} repos)")
            return data
        
    except Exception as e:
        print(f"Error loading cache for {category}/{platform}: {e}")
    
    return None

def save_category_data(category: str, platform: str, repos: List[Dict], timestamp: str):
    """Save category data to file"""
    output = {
        'category': category,
        'platform': platform,
        'lastUpdated': timestamp,
        'totalCount': len(repos),
        'repositories': repos
    }
    
    category_dir = os.path.join(CACHE_DIR, category)
    os.makedirs(category_dir, exist_ok=True)
    
    output_file = os.path.join(category_dir, f'{platform}.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved {len(repos)} repos to {output_file}")

def main():
    """Main function to fetch all categories"""
    timestamp = datetime.utcnow().isoformat() + 'Z'
    
    # Check rate limit
    response, _ = make_request_with_retry('https://api.github.com/rate_limit')
    if response:
        rate_data = response.json()
        core_remaining = rate_data.get('resources', {}).get('core', {}).get('remaining', 0)
        print(f"GitHub API rate limit: {core_remaining} requests remaining\n")
        
        if core_remaining < 500:
            print("WARNING: Low rate limit remaining.", file=sys.stderr)
    
    categories = {
        'trending': fetch_trending_repos,
        'new-releases': fetch_new_releases,
        'most-popular': fetch_most_popular
    }
    
    for category_name, fetch_func in categories.items():
        print(f"\n{'#'*70}")
        print(f"# CATEGORY: {category_name.upper()}")
        print(f"{'#'*70}")
        
        for platform in PLATFORMS.keys():
            print(f"\n--- Platform: {platform} ---")
            
            # Check cache
            cached_data = load_cache(category_name, platform)
            if cached_data:
                continue
            
            # Fetch fresh data
            repos = fetch_func(platform, desired_count=100)
            save_category_data(category_name, platform, repos, timestamp)
            
            time.sleep(1)  # Small delay between platforms
    
    print("\n" + "="*70)
    print("✓ ALL CATEGORIES PROCESSED SUCCESSFULLY!")
    print("="*70)

if __name__ == '__main__':
    main()
