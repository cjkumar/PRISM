"""
Framework Loader
================

Loads framework definitions (scoring rubrics, indicators, definitions) from
the project's JSON/TXT framework files for use by the analysis agents.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger("prism.frameworks")


class FrameworkLoader:
    """Loads and provides access to analytical framework definitions."""

    def __init__(self, framework_path: str):
        self.framework_path = Path(framework_path)
        self._framework: List[Dict[str, Any]] = []
        self._by_category: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self):
        """Load framework from JSON file."""
        if not self.framework_path.exists():
            raise FileNotFoundError(
                f"Framework file not found: {self.framework_path}"
            )

        with open(self.framework_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Handle both list format and dict-with-key format
        if isinstance(data, list):
            self._framework = data
        elif isinstance(data, dict) and "framework" in data:
            self._framework = data["framework"]
        else:
            raise ValueError(
                f"Unexpected framework format in {self.framework_path}"
            )

        self._by_category = {
            entry["category"]: entry for entry in self._framework
        }
        logger.info(
            f"Loaded framework with {len(self._framework)} sub-elements "
            f"from {self.framework_path.name}"
        )

    @property
    def categories(self) -> List[str]:
        """Return list of all category names."""
        return [e["category"] for e in self._framework]

    @property
    def num_categories(self) -> int:
        return len(self._framework)

    def get_definition(self, category: str) -> str:
        """Get the definition text for a category."""
        entry = self._by_category.get(category)
        if entry is None:
            raise KeyError(f"Unknown category: {category}")
        return entry.get("definition", "")

    def get_scoring_definitions(self, category: str) -> List[str]:
        """Get the scoring rubric for a category."""
        entry = self._by_category.get(category)
        if entry is None:
            raise KeyError(f"Unknown category: {category}")
        return entry.get("scoring_definitions", [])

    def get_indicators(self, category: str) -> List[str]:
        """Get recommended indicators for a category."""
        entry = self._by_category.get(category)
        if entry is None:
            raise KeyError(f"Unknown category: {category}")
        return entry.get("indicators", [])

    def get_entry(self, category: str) -> Dict[str, Any]:
        """Get the full framework entry for a category."""
        entry = self._by_category.get(category)
        if entry is None:
            raise KeyError(f"Unknown category: {category}")
        return dict(entry)

    def build_rag_knowledge_base(self) -> List[Dict[str, str]]:
        """Build knowledge base entries for RAG ingestion.

        Returns a list of documents with 'content' and 'metadata' fields,
        suitable for embedding and retrieval.
        """
        documents = []
        for entry in self._framework:
            category = entry["category"]
            definition = entry.get("definition", "")
            indicators = entry.get("indicators", [])
            scoring_defs = entry.get("scoring_definitions", [])

            # Main definition document
            content = f"Category: {category}\n\nDefinition: {definition}"
            if indicators:
                content += "\n\nRecommended Indicators:\n"
                content += "\n".join(f"- {ind}" for ind in indicators)
            documents.append({
                "content": content,
                "metadata": {
                    "category": category,
                    "type": "definition",
                },
            })

            # Scoring rubric document
            if scoring_defs:
                rubric = f"Scoring Rubric for {category}:\n\n"
                rubric += "\n".join(scoring_defs)
                documents.append({
                    "content": rubric,
                    "metadata": {
                        "category": category,
                        "type": "scoring_rubric",
                    },
                })

        logger.info(f"Built RAG knowledge base with {len(documents)} documents")
        return documents

    def build_prompt_context(self, category: str) -> str:
        """Build a prompt context string for a specific category.

        Used by Agent 2 to construct analysis prompts with full framework context.
        """
        entry = self.get_entry(category)
        definition = entry.get("definition", "")
        indicators = entry.get("indicators", [])
        scoring_defs = entry.get("scoring_definitions", [])

        parts = [
            f"## Sub-Element: {category}",
            f"\n### Definition\n{definition}",
        ]

        if indicators:
            parts.append("\n### Recommended Indicators")
            for ind in indicators:
                parts.append(f"- {ind}")

        if scoring_defs:
            parts.append("\n### Scoring Criteria")
            for sd in scoring_defs:
                parts.append(f"- {sd}")

        return "\n".join(parts)
