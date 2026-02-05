#!/usr/bin/env python3
"""
Test script for the Poster Extraction API.

The API no longer exposes a file upload endpoint; it polls the database for
ExtractionJob records and processes them. This script tests health endpoints only.

Usage:
    python test_api.py [--url URL] [--endpoint ENDPOINT]

Examples:
    # Test health endpoint
    python test_api.py --endpoint health

    # Test root endpoint
    python test_api.py --endpoint root

    # Test all HTTP endpoints (health + root)
    python test_api.py --endpoint all

    # Use custom API URL
    python test_api.py --url http://localhost:8001
"""

import argparse
import json
import sys

import requests


def test_health_endpoint(base_url: str) -> bool:
    """Test the /health endpoint."""
    print("=" * 60)
    print("Testing /health endpoint")
    print("=" * 60)
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_root_endpoint(base_url: str) -> bool:
    """Test the / endpoint."""
    print("=" * 60)
    print("Testing / endpoint")
    print("=" * 60)
    try:
        response = requests.get(f"{base_url}/", timeout=10)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Test the Poster Extraction API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Base URL of the API (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--endpoint",
        choices=["health", "root", "all"],
        default="all",
        help="Which endpoint to test (default: all)",
    )

    args = parser.parse_args()

    base_url = args.url.rstrip("/")

    print(f"Testing API at: {base_url}\n")

    results = []

    # Test endpoints based on selection
    if args.endpoint in ["all", "health"]:
        results.append(("Health", test_health_endpoint(base_url)))

    if args.endpoint in ["all", "root"]:
        results.append(("Root", test_root_endpoint(base_url)))

    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")

    # Exit with appropriate code
    all_passed = all(result[1] for result in results)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
