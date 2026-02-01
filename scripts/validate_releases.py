#!/usr/bin/env python3
"""
Release Date Validation Script

This script validates that the release dates in new-releases category
are accurate by checking them against GitHub's actual release data.

Usage:
    python validate_releases.py [platform]
    
Examples:
    python validate_releases.py android
    python validate_releases.py  # validates all platforms
"""

import json
import sys
import requests
from datetime import datetime
from pathlib import Path

GITHUB_TOKEN = None  # Will read from environment if needed

def check_release_date(owner: str, repo_name: str, expected_date: str) -> dict:
    """
    Verify the release date for a repository
    Returns dict with validation results
    """
    headers = {}
    if GITHUB_TOKEN:
        headers['Authorization'] = f'token {GITHUB_TOKEN}'
    
    # First try /releases/latest (fastest)
    url = f'https://api.github.com/repos/{owner}/{repo_name}/releases/latest'
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            release = response.json()
            actual_date = release.get('published_at')
            is_prerelease = release.get('prerelease', False)
            is_draft = release.get('draft', False)
            tag = release.get('tag_name', 'unknown')
            
            # Compare dates
            match = actual_date == expected_date
            
            return {
                'success': True,
                'match': match,
                'expected': expected_date,
                'actual': actual_date,
                'is_prerelease': is_prerelease,
                'is_draft': is_draft,
                'tag': tag,
                'status': 'OK' if match else 'MISMATCH'
            }
        else:
            return {
                'success': False,
                'error': f'API returned {response.status_code}',
                'status': 'ERROR'
            }
            
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'status': 'ERROR'
        }

def validate_platform(platform: str) -> dict:
    """Validate all repos in a platform's new-releases file"""
    
    file_path = Path(f'cached-data/new-releases/{platform}.json')
    
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return {'total': 0, 'validated': 0, 'errors': 0, 'mismatches': 0}
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    repos = data.get('repositories', [])
    total = len(repos)
    
    print(f"\n{'='*70}")
    print(f"Validating {platform.upper()} - {total} repositories")
    print(f"{'='*70}\n")
    
    results = {
        'total': total,
        'validated': 0,
        'errors': 0,
        'mismatches': 0,
        'details': []
    }
    
    for idx, repo in enumerate(repos[:10], 1):  # Validate first 10 to avoid rate limits
        full_name = repo['fullName']
        owner, name = full_name.split('/')
        expected_date = repo.get('latestReleaseDate')
        recency = repo.get('releaseRecency', '?')
        
        print(f"[{idx}/{min(10, total)}] Checking {full_name}... ", end='', flush=True)
        
        if not expected_date:
            print("⚠️  No release date in data")
            results['errors'] += 1
            continue
        
        validation = check_release_date(owner, name, expected_date)
        
        if validation['status'] == 'OK':
            print(f"✅ OK (released {recency}d ago)")
            results['validated'] += 1
        elif validation['status'] == 'MISMATCH':
            print(f"❌ MISMATCH")
            print(f"    Expected: {validation['expected']}")
            print(f"    Actual:   {validation['actual']}")
            print(f"    Tag:      {validation['tag']}")
            if validation.get('is_prerelease'):
                print(f"    ⚠️  Actual release is a PRE-RELEASE")
            results['mismatches'] += 1
        else:
            print(f"❌ ERROR: {validation.get('error', 'Unknown')}")
            results['errors'] += 1
        
        results['details'].append({
            'repo': full_name,
            'validation': validation
        })
    
    return results

def main():
    platforms = ['android', 'windows', 'macos', 'linux']
    
    # Check if specific platform requested
    if len(sys.argv) > 1:
        platform = sys.argv[1].lower()
        if platform not in platforms:
            print(f"❌ Invalid platform: {platform}")
            print(f"Valid platforms: {', '.join(platforms)}")
            sys.exit(1)
        platforms = [platform]
    
    print(f"\n🔍 RELEASE DATE VALIDATION")
    print(f"{'='*70}")
    print(f"This will validate release dates against GitHub's API")
    print(f"(Checking first 10 repos per platform to conserve API quota)")
    print(f"{'='*70}")
    
    all_results = {}
    
    for platform in platforms:
        results = validate_platform(platform)
        all_results[platform] = results
    
    # Summary
    print(f"\n{'='*70}")
    print(f"VALIDATION SUMMARY")
    print(f"{'='*70}\n")
    
    total_checked = 0
    total_ok = 0
    total_mismatch = 0
    total_errors = 0
    
    for platform, results in all_results.items():
        total_checked += min(10, results['total'])
        total_ok += results['validated']
        total_mismatch += results['mismatches']
        total_errors += results['errors']
        
        print(f"{platform.upper():<10} | ", end='')
        print(f"✅ {results['validated']:2d} OK  ", end='')
        print(f"❌ {results['mismatches']:2d} Mismatch  ", end='')
        print(f"⚠️  {results['errors']:2d} Errors")
    
    print(f"\n{'='*70}")
    print(f"TOTAL: {total_checked} checked | {total_ok} OK | {total_mismatch} mismatches | {total_errors} errors")
    
    if total_mismatch > 0:
        print(f"\n⚠️  Found {total_mismatch} release date mismatches!")
        print(f"This could indicate:")
        print(f"  - Pre-releases being counted as stable releases")
        print(f"  - Timezone/date parsing issues")
        print(f"  - Stale cached data")
        print(f"\nRecommendation: Re-run the fetch script with force_refresh=true")
    elif total_ok == total_checked:
        print(f"\n✅ All release dates validated successfully!")
    
    print(f"{'='*70}\n")

if __name__ == '__main__':
    # Try to get GitHub token from environment
    import os
    GITHUB_TOKEN = os.environ.get('GITHUB_TOKEN')
    
    if not GITHUB_TOKEN:
        print("⚠️  GITHUB_TOKEN not set - API rate limits will be lower")
        print("Set GITHUB_TOKEN for higher rate limits\n")
    
    main()