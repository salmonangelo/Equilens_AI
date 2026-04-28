#!/usr/bin/env python3
"""
EquiLens AI — Local API Testing Script

This script demonstrates how to interact with the EquiLens AI API locally
using Python requests. It covers:
  1. Health checks
  2. CSV upload
  3. Fairness analysis
  4. Error handling

Usage:
    # Make sure the backend is running first:
    #   python main.py
    
    # Then run this script:
    python tests/test_api_local.py
"""

import json
import sys
from pathlib import Path

import requests

# =====================================================================
# Configuration
# =====================================================================

BASE_URL = "http://localhost:8080"
TIMEOUT = 10  # seconds

# Colors for terminal output
class Color:
    """ANSI color codes for terminal output."""
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


def print_header(text: str) -> None:
    """Print a colored header."""
    print(f"\n{Color.BOLD}{Color.BLUE}{'=' * 80}{Color.RESET}")
    print(f"{Color.BOLD}{Color.BLUE}{text}{Color.RESET}")
    print(f"{Color.BOLD}{Color.BLUE}{'=' * 80}{Color.RESET}\n")


def print_success(text: str) -> None:
    """Print a success message."""
    print(f"{Color.GREEN}✓ {text}{Color.RESET}")


def print_warning(text: str) -> None:
    """Print a warning message."""
    print(f"{Color.YELLOW}⚠ {text}{Color.RESET}")


def print_error(text: str) -> None:
    """Print an error message."""
    print(f"{Color.RED}✗ {text}{Color.RESET}")


# =====================================================================
# Test Functions
# =====================================================================

def test_health_check() -> bool:
    """Test the health check endpoint."""
    print_header("1. Health Check")

    try:
        response = requests.get(
            f"{BASE_URL}/health",
            timeout=TIMEOUT,
        )

        if response.status_code == 200:
            data = response.json()
            print_success(f"Health check passed")
            print(json.dumps(data, indent=2))
            return True
        else:
            print_error(f"Health check failed: {response.status_code}")
            return False

    except requests.exceptions.ConnectionError:
        print_error(f"Cannot connect to {BASE_URL}")
        print_warning("Make sure the backend is running: python main.py")
        return False
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        return False


def create_sample_csv() -> str:
    """Create a sample CSV file for testing."""
    print_header("2. Creating Sample CSV")

    csv_path = Path("sample_test_data.csv")

    csv_content = """gender,age,approved,predicted,income
M,35,1,1,75000
M,42,1,1,85000
F,28,0,0,55000
F,31,1,0,60000
M,45,1,1,95000
F,26,0,0,45000
M,38,1,1,70000
F,33,1,1,72000
M,50,1,1,120000
F,29,0,1,58000
M,36,1,1,76000
F,32,1,1,71000
M,44,1,1,82000
F,27,0,0,52000
M,41,1,1,88000
F,30,1,0,62000
M,39,1,1,78000
F,25,0,0,48000
M,48,1,1,110000
F,34,1,1,75000
"""

    with open(csv_path, "w") as f:
        f.write(csv_content.strip())

    print_success(f"Created sample CSV: {csv_path}")
    print(f"  Rows: 20 | Columns: 5")
    print(f"  Protected attribute: gender | Target: approved | Prediction: predicted")

    return str(csv_path)


def test_upload_csv(csv_path: str) -> str | None:
    """Test CSV upload endpoint."""
    print_header("3. Upload CSV")

    try:
        with open(csv_path, "rb") as f:
            files = {"file": f}
            response = requests.post(
                f"{BASE_URL}/api/v1/upload",
                files=files,
                timeout=TIMEOUT,
            )

        if response.status_code == 201:
            data = response.json()
            dataset_id = data.get("dataset_id")
            print_success(f"CSV uploaded successfully")
            print(f"  Dataset ID: {dataset_id}")
            print(f"  Rows: {data.get('rows')} | Columns: {data.get('columns')}")
            print(f"  Columns: {', '.join(data.get('column_names', []))}")
            return dataset_id
        else:
            print_error(f"Upload failed: {response.status_code}")
            print(json.dumps(response.json(), indent=2))
            return None

    except FileNotFoundError:
        print_error(f"File not found: {csv_path}")
        return None
    except Exception as e:
        print_error(f"Upload error: {e}")
        return None


def test_analysis(dataset_id: str) -> bool:
    """Test fairness analysis endpoint."""
    print_header("4. Run Fairness Analysis")

    request_body = {
        "dataset_id": dataset_id,
        "protected_attributes": ["gender"],
        "target_column": "approved",
        "prediction_column": "predicted",
        "model_name": "loan_approval_model_v1",
        "skip_anonymization": False,
    }

    print("Request body:")
    print(json.dumps(request_body, indent=2))
    print()

    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/analyze",
            json=request_body,
            timeout=TIMEOUT,
        )

        if response.status_code == 200:
            data = response.json()
            print_success("Analysis completed successfully")
            print(json.dumps(data, indent=2))
            return True
        else:
            print_error(f"Analysis failed: {response.status_code}")
            error_detail = response.json().get("detail", "Unknown error")
            print(json.dumps(error_detail, indent=2))
            return False

    except Exception as e:
        print_error(f"Analysis error: {e}")
        return False


def test_error_handling() -> None:
    """Test error handling scenarios."""
    print_header("5. Error Handling Tests")

    # Test 1: Invalid dataset ID
    print("Test 5a: Invalid dataset ID")
    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/analyze",
            json={
                "dataset_id": "nonexistent_id_12345",
                "protected_attributes": ["gender"],
                "target_column": "approved",
                "prediction_column": "predicted",
                "model_name": "test_model",
            },
            timeout=TIMEOUT,
        )
        if response.status_code == 404:
            print_success("Correctly rejected invalid dataset ID (404)")
        else:
            print_warning(f"Unexpected status: {response.status_code}")
    except Exception as e:
        print_error(f"Error: {e}")

    # Test 2: Missing column
    print("\nTest 5b: Missing column (create dataset first)")
    csv_path = create_sample_csv()
    dataset_id = test_upload_csv(csv_path)

    if dataset_id:
        try:
            response = requests.post(
                f"{BASE_URL}/api/v1/analyze",
                json={
                    "dataset_id": dataset_id,
                    "protected_attributes": ["nonexistent_column"],
                    "target_column": "approved",
                    "prediction_column": "predicted",
                    "model_name": "test_model",
                },
                timeout=TIMEOUT,
            )
            if response.status_code == 422:
                print_success("Correctly rejected missing column (422)")
                error_detail = response.json().get("detail", {})
                if isinstance(error_detail, dict):
                    print(f"  Missing columns: {error_detail.get('missing_columns')}")
            else:
                print_warning(f"Unexpected status: {response.status_code}")
        except Exception as e:
            print_error(f"Error: {e}")


# =====================================================================
# Main Test Suite
# =====================================================================

def main() -> int:
    """Run all tests."""
    print(f"\n{Color.BOLD}{Color.BLUE}EquiLens AI — Local API Test Suite{Color.RESET}")
    print(f"Target: {BASE_URL}\n")

    # Test 1: Health check
    if not test_health_check():
        print_error("Health check failed. Is the backend running?")
        print_warning("Run: python main.py")
        return 1

    # Test 2-4: Upload and analyze
    csv_path = create_sample_csv()
    dataset_id = test_upload_csv(csv_path)

    if dataset_id:
        if not test_analysis(dataset_id):
            print_warning("Analysis failed")
    else:
        print_error("Could not proceed without dataset ID")
        return 1

    # Test 5: Error handling
    test_error_handling()

    # Summary
    print_header("✓ All tests completed")
    print("Next steps:")
    print("  1. Review the API docs: http://localhost:8080/docs")
    print("  2. Try with your own CSV files")
    print("  3. Check logs for detailed information")
    print()

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
