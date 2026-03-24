"""
Data Export Utilities
=====================

Converts PRISM analysis JSON outputs to CSV format for the
FrameworkExplorer visualization application.

Output CSV schema:
    country, year, Sub-Element, response, response_page_citations,
    score, reasoning, scoring_reasoning_page_citations
"""

import csv
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set

logger = logging.getLogger("prism.visualization")


class DataExporter:
    """Export PRISM analysis results to various formats."""

    CSV_COLUMNS = [
        "country",
        "year",
        "Sub-Element",
        "response",
        "response_page_citations",
        "score",
        "reasoning",
        "scoring_reasoning_page_citations",
    ]

    @staticmethod
    def extract_country_year(filename: str) -> Tuple[str, str]:
        """Extract country and year from filename.

        Handles formats:
            Country_2019.json        -> ("Country", "2019")
            Country_2021_2031.json   -> ("Country", "2021-2031")
            Country-Name_2019.json   -> ("Country Name", "2019")
        """
        base = filename.replace(".json", "")

        # Year range: Country_YYYY_YYYY
        match = re.match(r"^(.+?)_(\d{4})_(\d{4})$", base)
        if match:
            country = match.group(1).replace("_", " ").replace("-", " ")
            year = f"{match.group(2)}-{match.group(3)}"
            return country, year

        # Single year: Country_YYYY
        match = re.match(r"^(.+?)_(\d{4})$", base)
        if match:
            country = match.group(1).replace("_", " ").replace("-", " ")
            return country, match.group(2)

        return base.replace("_", " ").replace("-", " "), "Unknown"

    @classmethod
    def json_to_csv_rows(
        cls,
        data: List[Dict[str, Any]],
        country: str,
        year: str,
    ) -> List[Dict[str, str]]:
        """Convert a single JSON analysis to CSV rows (one per category)."""
        rows = []
        for entry in data:
            response_citations = entry.get("response_page_citations", [])
            scoring_citations = entry.get("scoring_reasoning_page_citations", [])

            row = {
                "country": country,
                "year": year,
                "Sub-Element": entry.get("category", ""),
                "response": entry.get("response", ""),
                "response_page_citations": (
                    ", ".join(map(str, response_citations))
                    if response_citations else ""
                ),
                "score": str(entry.get("score", "")),
                "reasoning": entry.get("scoring_reasoning", ""),
                "scoring_reasoning_page_citations": (
                    ", ".join(map(str, scoring_citations))
                    if scoring_citations else ""
                ),
            }
            rows.append(row)
        return rows

    @classmethod
    def export_folder_to_csv(
        cls,
        input_folder: str,
        output_path: str,
        country_filter: Optional[Set[str]] = None,
    ) -> int:
        """Convert all JSON analyses in a folder to a single CSV.

        Args:
            input_folder: Path to folder with JSON analysis files.
            output_path: Path for the output CSV.
            country_filter: Optional set of country names to include.

        Returns:
            Number of rows written.
        """
        folder = Path(input_folder)
        json_files = sorted(folder.glob("*.json"))

        if not json_files:
            logger.warning(f"No JSON files found in {input_folder}")
            return 0

        all_rows = []
        for json_file in json_files:
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Skipping {json_file.name}: {e}")
                continue

            country, year = cls.extract_country_year(json_file.name)

            if country_filter and country not in country_filter:
                continue

            rows = cls.json_to_csv_rows(data, country, year)
            all_rows.extend(rows)

        # Write CSV
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=cls.CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(all_rows)

        countries = len({r["country"] for r in all_rows})
        logger.info(
            f"CSV exported: {output_path} "
            f"({countries} countries, {len(all_rows)} rows)"
        )
        return len(all_rows)

    @classmethod
    def export_global_and_commonwealth(
        cls,
        input_folder: str,
        output_dir: str,
        commonwealth_countries: Optional[Set[str]] = None,
    ) -> Tuple[int, int]:
        """Generate both global and Commonwealth CSVs.

        Returns:
            Tuple of (global_rows, commonwealth_rows).
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Global
        global_path = str(out / "global_analyses.csv")
        global_rows = cls.export_folder_to_csv(input_folder, global_path)

        # Commonwealth
        cw_path = str(out / "CW_analyses.csv")
        cw_rows = cls.export_folder_to_csv(
            input_folder, cw_path, country_filter=commonwealth_countries
        )

        return global_rows, cw_rows
