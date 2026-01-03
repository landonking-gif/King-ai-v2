#!/usr/bin/env python3
"""
Validate source URLs for deep research.

This script validates URLs before fetching, checking for:
- Valid URL format
- Domain availability
- Known blocked domains

Usage:
    python validate_sources.py <url1> [url2] [url3] ...

Returns JSON with validation results.
"""

import json
import re
import sys
from typing import List, Dict, Any
from urllib.parse import urlparse

# Domains known to block scrapers
BLOCKED_DOMAINS = {
    "linkedin.com",
    "facebook.com",
    "instagram.com",
    "twitter.com",
    "x.com",
    "pinterest.com",
}

# Domains requiring special handling
PAYWALL_DOMAINS = {
    "wsj.com",
    "ft.com",
    "nytimes.com",
    "bloomberg.com",
    "economist.com",
}

# High-quality source domains
AUTHORITATIVE_DOMAINS = {
    "wikipedia.org",
    "reuters.com",
    "bbc.com",
    "gov",
    "edu",
    "nature.com",
    "sciencedirect.com",
    "pubmed.ncbi.nlm.nih.gov",
}


def validate_url(url: str) -> Dict[str, Any]:
    """Validate a single URL."""
    result = {
        "url": url,
        "valid": False,
        "issues": [],
        "domain": None,
        "is_blocked": False,
        "is_paywall": False,
        "is_authoritative": False,
    }

    # Check URL format
    try:
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            result["issues"].append("Invalid URL format")
            return result
        result["domain"] = parsed.netloc.lower()
    except Exception as e:
        result["issues"].append(f"URL parse error: {str(e)}")
        return result

    domain = result["domain"]

    # Check for blocked domains
    for blocked in BLOCKED_DOMAINS:
        if blocked in domain:
            result["is_blocked"] = True
            result["issues"].append(f"Domain {blocked} typically blocks scrapers")
            break

    # Check for paywall domains
    for paywall in PAYWALL_DOMAINS:
        if paywall in domain:
            result["is_paywall"] = True
            result["issues"].append(f"Domain {paywall} may have paywall restrictions")
            break

    # Check for authoritative domains
    for auth in AUTHORITATIVE_DOMAINS:
        if auth in domain:
            result["is_authoritative"] = True
            break

    # URL is valid if no critical issues
    result["valid"] = not result["is_blocked"]

    return result


def main(urls: List[str]) -> Dict[str, Any]:
    """Validate multiple URLs."""
    results = []
    valid_count = 0
    blocked_count = 0
    paywall_count = 0
    authoritative_count = 0

    for url in urls:
        validation = validate_url(url)
        results.append(validation)

        if validation["valid"]:
            valid_count += 1
        if validation["is_blocked"]:
            blocked_count += 1
        if validation["is_paywall"]:
            paywall_count += 1
        if validation["is_authoritative"]:
            authoritative_count += 1

    return {
        "success": True,
        "total_urls": len(urls),
        "valid_urls": valid_count,
        "blocked_urls": blocked_count,
        "paywall_urls": paywall_count,
        "authoritative_urls": authoritative_count,
        "results": results,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"success": False, "error": "No URLs provided"}))
        sys.exit(1)

    urls = sys.argv[1:]
    result = main(urls)
    print(json.dumps(result, indent=2))
