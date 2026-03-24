"""
PRISM Analysis Validation
=========================

Validates JSON analysis files against framework requirements.
Ensures all required categories are present with correct schema.
Generates global and Commonwealth CSV exports.
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any

from PRISM.frameworks.definitions import get_all_sub_elements

logger = logging.getLogger("prism.validation")

# Required fields in each analysis entry
REQUIRED_FIELDS = [
    "category",
    "response",
    "response_page_citations",
    "score",
    "scoring_reasoning",
    "scoring_reasoning_page_citations",
]

# Commonwealth member states
COMMONWEALTH_COUNTRIES = {
    "Australia", "Bangladesh", "Barbados", "Bermuda", "Cameroon",
    "Canada", "Cook Islands", "Cyprus", "Ghana", "India",
    "Isle of Man", "Jamaica", "Kenya", "Malawi", "Malaysia",
    "Maldives", "Malta", "Mauritius", "Mozambique", "New Zealand",
    "Nigeria", "Rwanda", "South Africa", "Sri Lanka", "Togo",
    "United Kingdom", "Zambia", "Zimbabwe",
}


class AnalysisValidator:
    """Validate PRISM analysis JSON files against framework requirements."""

    def __init__(self, domain: str = "cancer"):
        self.domain = domain
        self.required_categories = get_all_sub_elements(domain)

    def validate_file(self, filepath: str) -> Dict[str, Any]:
        """Validate a single analysis JSON file.

        Returns dict with validation results.
        """
        filepath = Path(filepath)
        results = {
            "filename": filepath.name,
            "is_valid": True,
            "total_categories": 0,
            "expected_categories": len(self.required_categories),
            "missing_categories": [],
            "extra_categories": [],
            "duplicate_categories": [],
            "categories_missing_fields": [],
            "errors": [],
        }

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            results["is_valid"] = False
            results["errors"].append(f"JSON parsing error: {e}")
            return results
        except Exception as e:
            results["is_valid"] = False
            results["errors"].append(f"Error reading file: {e}")
            return results

        if not isinstance(data, list):
            results["is_valid"] = False
            results["errors"].append("JSON root is not a list/array")
            return results

        results["total_categories"] = len(data)
        found_categories = []

        for i, entry in enumerate(data):
            if not isinstance(entry, dict):
                results["errors"].append(f"Entry {i} is not a dictionary")
                continue

            if "category" not in entry:
                results["errors"].append(f"Entry {i} missing 'category' field")
                continue

            category = entry["category"]
            found_categories.append(category)

            missing_fields = [
                f for f in REQUIRED_FIELDS if f not in entry
            ]
            if missing_fields:
                results["categories_missing_fields"].append({
                    "category": category,
                    "missing_fields": missing_fields,
                })

        found_set = set(found_categories)
        required_set = set(self.required_categories)

        results["missing_categories"] = sorted(required_set - found_set)
        results["extra_categories"] = sorted(found_set - required_set)

        seen = set()
        duplicates = set()
        for cat in found_categories:
            if cat in seen:
                duplicates.add(cat)
            seen.add(cat)
        results["duplicate_categories"] = sorted(duplicates)

        if (results["missing_categories"] or results["errors"]
                or results["duplicate_categories"]):
            results["is_valid"] = False

        return results

    def validate_folder(
        self, folder_path: str, verbose: bool = True
    ) -> List[Dict[str, Any]]:
        """Validate all JSON files in a folder.

        Returns list of validation results.
        """
        folder = Path(folder_path)
        if not folder.exists() or not folder.is_dir():
            logger.error(f"Invalid folder: {folder_path}")
            return []

        json_files = sorted(folder.glob("*.json"))
        if not json_files:
            logger.warning(f"No JSON files in {folder_path}")
            return []

        print(f"\n{'=' * 70}")
        print(f"PRISM Analysis Validator ({self.domain.upper()})")
        print(f"{'=' * 70}")
        print(f"Folder: {folder_path}")
        print(f"Files: {len(json_files)}")
        print(f"Expected categories: {len(self.required_categories)}")

        all_results = []
        complete = 0

        for json_file in json_files:
            result = self.validate_file(str(json_file))
            all_results.append(result)

            if result["is_valid"]:
                complete += 1
                status = "COMPLETE"
            else:
                status = "INCOMPLETE"

            if verbose:
                print(f"\n  {result['filename']}: {status} "
                      f"({result['total_categories']}/{result['expected_categories']})")

                if result["missing_categories"]:
                    for cat in result["missing_categories"][:5]:
                        print(f"    Missing: {cat}")
                    if len(result["missing_categories"]) > 5:
                        print(f"    ... and {len(result['missing_categories']) - 5} more")

        print(f"\n{'=' * 70}")
        print(f"SUMMARY: {complete}/{len(all_results)} complete")
        print(f"{'=' * 70}")

        return all_results
