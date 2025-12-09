#!/usr/bin/env python3
"""
Test script for the Poster Extraction API.

Usage:
    python test_api.py [--url URL] [--file PATH] [--endpoint ENDPOINT]

Examples:
    # Test health endpoint
    python test_api.py --endpoint health

    # Test root endpoint
    python test_api.py --endpoint root

    # Test /extract endpoint success with a PDF file
    python test_api.py --endpoint extract --file manual_poster_annotation/42/42.pdf

    # Test /extract endpoint success with an image file
    python test_api.py --endpoint extract --file manual_poster_annotation/4737132/4737132.jpg

    # Or use example shortcut
    python test_api.py --endpoint extract --example 42

    # Test extract success validation specifically
    python test_api.py --endpoint extract-success --file manual_poster_annotation/42/42.pdf

    # Use custom API URL
    python test_api.py --url http://localhost:8001 --file manual_poster_annotation/42/42.pdf
"""

import argparse
import json
import sys
from pathlib import Path

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


def test_extract_endpoint(base_url: str, file_path: str) -> bool:
    """Test the /extract endpoint with a file upload and validate success."""
    print("=" * 60)
    print("Testing /extract endpoint - SUCCESS TEST")
    print("=" * 60)

    file_path_obj = Path(file_path)

    if not file_path_obj.exists():
        print(f"Error: File not found: {file_path}")
        return False

    print(f"File: {file_path}")
    print(f"File size: {file_path_obj.stat().st_size / (1024 * 1024):.2f} MB")

    try:
        with open(file_path, "rb") as f:
            files = {"file": (file_path_obj.name, f, file_path_obj.suffix.lower())}
            print(f"\nUploading file to {base_url}/extract...")
            response = requests.post(
                f"{base_url}/extract",
                files=files,
                timeout=300,  # 5 minutes timeout for processing
            )

        print(f"\nStatus Code: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print("\n✓ /extract endpoint returned 200 OK - SUCCESS!")
            print("✓ Extraction completed successfully!")

            print(f"\nResponse: {json.dumps(result, indent=2)}")

            # Validate success response structure
            validation_errors = []
            validation_warnings = []

            # Check for error key (should not be present in successful response)
            if "error" in result:
                validation_errors.append(
                    "Response contains 'error' key - extraction may have failed"
                )

            # Validate expected fields
            expected_fields = ["posterContent"]
            for field in expected_fields:
                if field not in result:
                    validation_warnings.append(f"Missing expected field: {field}")

            # Validate posterContent structure
            if "posterContent" in result:
                poster_content = result["posterContent"]
                if not isinstance(poster_content, dict):
                    validation_errors.append("posterContent should be an object")
                else:
                    if "sections" not in poster_content:
                        validation_warnings.append(
                            "posterContent missing 'sections' field"
                        )
                    elif not isinstance(poster_content["sections"], list):
                        validation_errors.append(
                            "posterContent.sections should be an array"
                        )

            # Print validation results
            if validation_errors:
                print("\n✗ Validation Errors:")
                for error in validation_errors:
                    print(f"  - {error}")
                print("\n⚠ Response may not be valid despite 200 status code")
            elif validation_warnings:
                print("\n⚠ Validation Warnings:")
                for warning in validation_warnings:
                    print(f"  - {warning}")
            else:
                print("\n✓ Response structure validated successfully")

            print(f"\nResponse keys: {list(result.keys())}")

            # Save response to file
            output_file = file_path_obj.stem + "_test_output.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"\nFull response saved to: {output_file}")

            # Print detailed summary
            print("\n" + "=" * 60)
            print("Extraction Summary")
            print("=" * 60)

            if "posterContent" in result:
                print("\nPoster Content:")
                if "posterTitle" in result["posterContent"]:
                    title = result["posterContent"]["posterTitle"]
                    print(f"  Title: {title[:100]}{'...' if len(title) > 100 else ''}")

                if "sections" in result["posterContent"]:
                    sections = result["posterContent"]["sections"]
                    print(f"  Sections found: {len(sections)}")
                    for i, section in enumerate(sections, 1):
                        title = section.get("sectionTitle", "Unknown")
                        content = section.get("sectionContent", "")
                        content_len = len(content)
                        print(f"    {i}. {title}: {content_len} characters")
                        if content_len == 0:
                            validation_warnings.append(
                                f"Section '{title}' has empty content"
                            )

            if "creators" in result:
                creators = result["creators"]
                print(f"\nCreators: {len(creators)}")
                for i, creator in enumerate(creators[:3], 1):  # Show first 3
                    name = creator.get("name", "Unknown")
                    print(f"  {i}. {name}")
                if len(creators) > 3:
                    print(f"  ... and {len(creators) - 3} more")

            if "titles" in result:
                titles = result["titles"]
                print(f"\nTitles: {len(titles)}")
                for i, title_obj in enumerate(titles[:3], 1):  # Show first 3
                    title = title_obj.get("title", "Unknown")
                    print(f"  {i}. {title[:80]}{'...' if len(title) > 80 else ''}")
                if len(titles) > 3:
                    print(f"  ... and {len(titles) - 3} more")

            if "imageCaption" in result:
                captions = result["imageCaption"]
                print(f"\nImage Captions: {len(captions)}")

            if "tableCaption" in result:
                captions = result["tableCaption"]
                print(f"Table Captions: {len(captions)}")

            if "identifiers" in result:
                identifiers = result["identifiers"]
                print(f"Identifiers: {len(identifiers)}")

            # Return True only if no validation errors (successful extraction)
            success = len(validation_errors) == 0
            if success:
                print("\n" + "=" * 60)
                print("✓ /extract SUCCESS TEST PASSED")
                print("=" * 60)
            else:
                print("\n" + "=" * 60)
                print("✗ /extract SUCCESS TEST FAILED (validation errors)")
                print("=" * 60)
            return success
        else:
            print("\nError Response:")
            try:
                error_data = response.json()
                print(json.dumps(error_data, indent=2))
            except Exception:
                print(response.text)
            return False

    except requests.exceptions.Timeout:
        print("Error: Request timed out (processing may take a while)")
        return False
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_extract_success(base_url: str, file_path: str) -> bool:
    """
    Specifically test that the extract endpoint returns a successful response
    with valid structure. This is a more focused test for success cases.
    """
    print("=" * 60)
    print("Testing Extract Success Validation")
    print("=" * 60)

    file_path_obj = Path(file_path)

    if not file_path_obj.exists():
        print(f"Error: File not found: {file_path}")
        return False

    print(f"File: {file_path}")

    try:
        with open(file_path, "rb") as f:
            files = {"file": (file_path_obj.name, f, file_path_obj.suffix.lower())}
            print(f"Uploading to {base_url}/extract...")
            response = requests.post(
                f"{base_url}/extract",
                files=files,
                timeout=300,
            )

        # Check HTTP status
        if response.status_code != 200:
            print(f"✗ Failed: HTTP {response.status_code}")
            try:
                error_data = response.json()
                print(f"Error: {json.dumps(error_data, indent=2)}")
            except Exception:
                print(f"Error: {response.text}")
            return False

        # Parse response
        try:
            result = response.json()
        except json.JSONDecodeError as e:
            print(f"✗ Failed: Invalid JSON response - {e}")
            print(f"Response: {response.text[:500]}")
            return False

        # Validate success indicators
        success_checks = []

        # 1. Should not have error key
        if "error" in result:
            success_checks.append(
                (
                    "No error key",
                    False,
                    f"Found error: {result.get('error', 'Unknown')}",
                )
            )
        else:
            success_checks.append(("No error key", True, None))

        # 2. Should have posterContent
        if "posterContent" in result:
            success_checks.append(("Has posterContent", True, None))
            # 3. Should have sections
            if "sections" in result["posterContent"]:
                sections = result["posterContent"]["sections"]
                if isinstance(sections, list) and len(sections) > 0:
                    success_checks.append(
                        ("Has sections", True, f"{len(sections)} sections found")
                    )
                else:
                    success_checks.append(
                        ("Has sections", False, "Sections array is empty or invalid")
                    )
            else:
                success_checks.append(("Has sections", False, "Missing sections field"))
        else:
            success_checks.append(
                ("Has posterContent", False, "Missing posterContent field")
            )

        # 4. Should have at least creators or titles
        has_creators = "creators" in result and isinstance(result["creators"], list)
        has_titles = "titles" in result and isinstance(result["titles"], list)

        if has_creators or has_titles:
            success_checks.append(
                (
                    "Has creators or titles",
                    True,
                    f"Creators: {len(result.get('creators', []))}, Titles: {len(result.get('titles', []))}",
                )
            )
        else:
            success_checks.append(
                ("Has creators or titles", False, "Missing both creators and titles")
            )

        # Print validation results
        print("\nValidation Results:")
        all_passed = True
        for check_name, passed, detail in success_checks:
            status = "✓" if passed else "✗"
            print(f"  {status} {check_name}", end="")
            if detail:
                print(f" - {detail}")
            else:
                print()
            if not passed:
                all_passed = False

        if all_passed:
            print("\n✓ All success validations passed!")
        else:
            print("\n✗ Some success validations failed")

        return all_passed

    except requests.exceptions.Timeout:
        print("✗ Failed: Request timed out")
        return False
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_error_cases(base_url: str) -> bool:
    """Test error handling cases."""
    print("=" * 60)
    print("Testing Error Cases")
    print("=" * 60)

    all_passed = True

    # Test 1: No file provided
    print("\n1. Testing no file provided...")
    try:
        response = requests.post(f"{base_url}/extract", timeout=10)
        if response.status_code == 400:
            print("   ✓ Correctly returned 400 for missing file")
        else:
            print(f"   ✗ Expected 400, got {response.status_code}")
            all_passed = False
    except Exception as e:
        print(f"   ✗ Error: {e}")
        all_passed = False

    # Test 2: Invalid file type
    print("\n2. Testing invalid file type...")
    try:
        files = {"file": ("test.txt", b"test content", "text/plain")}
        response = requests.post(f"{base_url}/extract", files=files, timeout=10)
        if response.status_code == 400:
            print("   ✓ Correctly returned 400 for invalid file type")
        else:
            print(f"   ✗ Expected 400, got {response.status_code}")
            all_passed = False
    except Exception as e:
        print(f"   ✗ Error: {e}")
        all_passed = False

    return all_passed


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
        "--file",
        type=str,
        help="Path to file to test extraction with (PDF or image)",
    )
    parser.add_argument(
        "--endpoint",
        choices=["health", "root", "extract", "extract-success", "all", "errors"],
        default="all",
        help="Which endpoint to test (default: all)",
    )
    parser.add_argument(
        "--example",
        type=str,
        help="Use an example file from manual_poster_annotation (e.g., '42', '4737132')",
    )

    args = parser.parse_args()

    base_url = args.url.rstrip("/")

    print(f"Testing API at: {base_url}\n")

    # Handle example file selection
    if args.example:
        example_dir = Path("manual_poster_annotation") / args.example
        if not example_dir.exists():
            print(f"Error: Example directory not found: {example_dir}")
            sys.exit(1)

        # Find PDF or image file
        pdf_file = list(example_dir.glob("*.pdf"))
        img_file = (
            list(example_dir.glob("*.jpg"))
            + list(example_dir.glob("*.jpeg"))
            + list(example_dir.glob("*.png"))
        )

        if pdf_file:
            args.file = str(pdf_file[0])
        elif img_file:
            args.file = str(img_file[0])
        else:
            print(f"Error: No PDF or image file found in {example_dir}")
            sys.exit(1)

    results = []

    # Test endpoints based on selection
    if args.endpoint in ["all", "health"]:
        results.append(("Health", test_health_endpoint(base_url)))

    if args.endpoint in ["all", "root"]:
        results.append(("Root", test_root_endpoint(base_url)))

    if args.endpoint in ["all", "extract"]:
        if args.file:
            results.append(("Extract", test_extract_endpoint(base_url, args.file)))
        elif args.endpoint == "extract":
            print("Error: --file is required when testing extract endpoint")
            sys.exit(1)

    if args.endpoint in ["all", "extract-success"]:
        if args.file:
            results.append(
                ("Extract Success", test_extract_success(base_url, args.file))
            )
        elif args.endpoint == "extract-success":
            print("Error: --file is required when testing extract-success endpoint")
            sys.exit(1)

    if args.endpoint in ["all", "errors"]:
        results.append(("Error Cases", test_error_cases(base_url)))

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
