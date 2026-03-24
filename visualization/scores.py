"""
Score Analysis Utilities
========================

Statistical analysis and normalization of PRISM scoring data.
Supports section-level aggregation and cross-country comparison.
"""

import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from PRISM.frameworks.definitions import (
    get_sections,
    get_max_scores,
    normalize_score,
)

logger = logging.getLogger("prism.visualization")


class ScoreAnalyzer:
    """Analyze and aggregate PRISM scores across documents."""

    def __init__(self, domain: str = "cancer"):
        self.domain = domain
        self.sections = get_sections(domain)
        self.max_scores = get_max_scores(domain)

    def compute_section_scores(
        self, analyses: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Compute average normalized score per framework section.

        Args:
            analyses: List of sub-element analysis dicts.

        Returns:
            Dict mapping section name -> average normalized score (0-5).
        """
        # Index by category
        by_category = {a["category"]: a for a in analyses if "category" in a}

        section_scores = {}
        for section_name, sub_elements in self.sections.items():
            scores = []
            for se in sub_elements:
                if se in by_category:
                    raw_score = by_category[se].get("score", 0)
                    normalized = normalize_score(
                        raw_score, se, self.domain
                    )
                    scores.append(normalized)

            if scores:
                section_scores[section_name] = sum(scores) / len(scores)
            else:
                section_scores[section_name] = 0.0

        return section_scores

    def compute_overall_score(
        self, analyses: List[Dict[str, Any]]
    ) -> float:
        """Compute overall comprehensiveness score (median of all normalized scores)."""
        import statistics

        scores = []
        for a in analyses:
            category = a.get("category", "")
            raw_score = a.get("score", 0)
            if category in self.max_scores:
                scores.append(normalize_score(raw_score, category, self.domain))

        if not scores:
            return 0.0
        return statistics.median(scores)

    def compare_countries(
        self, folder_path: str
    ) -> Dict[str, Dict[str, float]]:
        """Compare section scores across all countries in a folder.

        Returns:
            Dict mapping country -> section_name -> avg_normalized_score.
        """
        from PRISM.visualization.export import DataExporter

        folder = Path(folder_path)
        results = {}

        for json_file in sorted(folder.glob("*.json")):
            country, year = DataExporter.extract_country_year(json_file.name)

            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    analyses = json.load(f)
            except (json.JSONDecodeError, IOError):
                continue

            section_scores = self.compute_section_scores(analyses)
            results[f"{country} ({year})"] = section_scores

        return results

    def generate_summary_statistics(
        self, folder_path: str
    ) -> Dict[str, Any]:
        """Generate aggregate statistics across all analyzed plans.

        Returns dict with median scores, highest/lowest elements, etc.
        """
        from PRISM.visualization.export import DataExporter
        import statistics

        folder = Path(folder_path)
        all_overall = []
        element_scores = {}  # category -> list of normalized scores

        for json_file in sorted(folder.glob("*.json")):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    analyses = json.load(f)
            except (json.JSONDecodeError, IOError):
                continue

            overall = self.compute_overall_score(analyses)
            all_overall.append(overall)

            for a in analyses:
                cat = a.get("category", "")
                raw = a.get("score", 0)
                if cat in self.max_scores:
                    norm = normalize_score(raw, cat, self.domain)
                    element_scores.setdefault(cat, []).append(norm)

        if not all_overall:
            return {"error": "No data found"}

        # Section-level aggregation
        section_medians = {}
        for section_name, sub_elements in self.sections.items():
            section_vals = []
            for se in sub_elements:
                if se in element_scores:
                    section_vals.extend(element_scores[se])
            if section_vals:
                section_medians[section_name] = statistics.median(section_vals)

        sorted_sections = sorted(
            section_medians.items(), key=lambda x: x[1], reverse=True
        )

        return {
            "total_plans": len(all_overall),
            "median_overall": round(statistics.median(all_overall), 2),
            "mean_overall": round(statistics.mean(all_overall), 2),
            "highest_sections": [
                {"section": s, "median": round(v, 2)}
                for s, v in sorted_sections[:3]
            ],
            "lowest_sections": [
                {"section": s, "median": round(v, 2)}
                for s, v in sorted_sections[-3:]
            ],
            "section_medians": {
                s: round(v, 2) for s, v in sorted_sections
            },
        }
