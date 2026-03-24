"""
Page Reference Extractor
========================

Orchestrates the three-method triangulation approach for page reference
extraction and computes final attributions with confidence scoring.

Final set: PageRefs = A ∩ B ∩ C

Fallback hierarchy when three-way intersection is empty:
    1. Pairwise intersections: A∩B, A∩C, B∩C (use largest)
    2. If all pairwise empty: return Set B with reduced confidence
"""

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Set, Optional, Callable, Tuple, Any

from PRISM.page_references.methods import (
    NumericEntityMatcher,
    SemanticSimilarityMatcher,
    GenerativeMatcher,
)

logger = logging.getLogger("prism.page_references")


@dataclass
class PageReference:
    """A page reference attribution with confidence."""
    page_number: int
    confidence: str  # "high", "medium", "low"
    methods_agreed: List[str]  # e.g., ["A", "B", "C"]


@dataclass
class ExtractionResult:
    """Result from page reference extraction for a single sub-element."""
    category: str
    field: str  # "response" or "scoring_reasoning"
    page_references: List[PageReference]
    set_a: Set[int]
    set_b: Set[int]
    set_c: Set[int]
    processing_time_seconds: float

    @property
    def page_numbers(self) -> List[int]:
        """Sorted list of attributed page numbers."""
        return sorted(pr.page_number for pr in self.page_references)

    @property
    def high_confidence_pages(self) -> List[int]:
        return sorted(
            pr.page_number for pr in self.page_references
            if pr.confidence == "high"
        )


class PageReferenceExtractor:
    """Orchestrator for three-method page reference triangulation.

    Combines:
        - Method A: Numeric entity matching
        - Method B: Sliding window semantic similarity
        - Method C: Generative matching with temperature sampling

    Confidence levels:
        - High: A ∩ B ∩ C (all three methods agree)
        - Medium: Two-method intersection
        - Low: Single method only (flagged for review)
    """

    def __init__(self, config, generate_fn: Optional[Callable] = None):
        """
        Args:
            config: PageRefConfig instance.
            generate_fn: Optional callable(prompt, temperature, top_p) -> str
                         for Method C generative validation.
        """
        self.config = config
        self.method_a = NumericEntityMatcher(
            year_range=config.year_range,
        )
        self.method_b = SemanticSimilarityMatcher(
            embedding_model_name=config.embedding_model,
            window_size=config.window_size,
            stride=config.window_stride,
            batch_size=config.batch_size,
        )
        self.method_c = GenerativeMatcher(
            temperature_low=config.generative_temperature_low,
            temperature_high=config.generative_temperature_high,
            top_p=config.generative_top_p,
            n_samples=config.generative_n_samples,
            majority_threshold=config.generative_majority_threshold,
            prefilter_percentile=config.prefilter_percentile,
        )
        self.generate_fn = generate_fn

    def extract_references(
        self,
        llm_text: str,
        page_texts: Dict[int, str],
        category: str,
        field: str = "response",
    ) -> ExtractionResult:
        """Extract page references for a single LLM output field.

        Args:
            llm_text: The LLM-generated text to attribute.
            page_texts: Dict mapping page number -> page text content.
            category: Framework sub-element name.
            field: "response" or "scoring_reasoning".

        Returns:
            ExtractionResult with attributed pages and confidence.
        """
        start_time = time.time()

        if not llm_text or not llm_text.strip():
            return ExtractionResult(
                category=category,
                field=field,
                page_references=[],
                set_a=set(),
                set_b=set(),
                set_c=set(),
                processing_time_seconds=0.0,
            )

        # Method A: Numeric entity matching
        set_a = self.method_a.match_pages(llm_text, page_texts)

        # Method B: Semantic similarity
        set_b, page_similarities = self.method_b.match_pages(
            llm_text, page_texts
        )

        # Method C: Generative matching (if generate_fn provided)
        set_c = set()
        if self.generate_fn is not None:
            set_c = self.method_c.match_pages(
                llm_text, page_texts, page_similarities, self.generate_fn
            )

        # Compute final attribution via set intersection
        page_refs = self._compute_attribution(set_a, set_b, set_c)

        processing_time = time.time() - start_time

        result = ExtractionResult(
            category=category,
            field=field,
            page_references=page_refs,
            set_a=set_a,
            set_b=set_b,
            set_c=set_c,
            processing_time_seconds=processing_time,
        )

        logger.info(
            f"Page refs for {category}/{field}: "
            f"{len(page_refs)} pages "
            f"(A={len(set_a)}, B={len(set_b)}, C={len(set_c)}) "
            f"in {processing_time:.1f}s"
        )
        return result

    def extract_all_references(
        self,
        analyses: List[Dict[str, Any]],
        page_texts: Dict[int, str],
    ) -> Dict[str, Dict[str, ExtractionResult]]:
        """Extract page references for all sub-elements.

        Args:
            analyses: List of analysis dicts (Agent 2 output).
            page_texts: Dict mapping page number -> text.

        Returns:
            Nested dict: {category: {"response": result, "reasoning": result}}
        """
        results = {}
        total = len(analyses)

        for i, analysis in enumerate(analyses):
            category = analysis.get("category", f"unknown_{i}")
            logger.info(
                f"Extracting references [{i + 1}/{total}]: {category}"
            )

            results[category] = {}

            # Response field
            response = analysis.get("response", "")
            results[category]["response"] = self.extract_references(
                response, page_texts, category, "response"
            )

            # Scoring reasoning field
            reasoning = analysis.get("scoring_reasoning", "")
            results[category]["reasoning"] = self.extract_references(
                reasoning, page_texts, category, "scoring_reasoning"
            )

        return results

    def _compute_attribution(
        self, set_a: Set[int], set_b: Set[int], set_c: Set[int]
    ) -> List[PageReference]:
        """Compute final page attribution from method sets.

        Priority: A∩B∩C > pairwise intersections > Set B alone.
        """
        # Three-way intersection
        triple = set_a & set_b & set_c
        if triple:
            return [
                PageReference(p, "high", ["A", "B", "C"])
                for p in sorted(triple)
            ]

        # Pairwise intersections
        ab = set_a & set_b
        ac = set_a & set_c
        bc = set_b & set_c

        pairwise = [(ab, ["A", "B"]), (ac, ["A", "C"]), (bc, ["B", "C"])]
        pairwise.sort(key=lambda x: len(x[0]), reverse=True)

        for pair_set, methods in pairwise:
            if pair_set:
                return [
                    PageReference(p, "medium", methods)
                    for p in sorted(pair_set)
                ]

        # Fallback: Set B with reduced confidence
        if set_b:
            return [
                PageReference(p, "low", ["B"])
                for p in sorted(set_b)
            ]

        # Nothing found
        return []

    @staticmethod
    def update_analyses_with_references(
        analyses: List[Dict[str, Any]],
        ref_results: Dict[str, Dict[str, ExtractionResult]],
    ) -> List[Dict[str, Any]]:
        """Update analysis dicts with extracted page references.

        Modifies the response_page_citations and
        scoring_reasoning_page_citations fields in-place.
        """
        for analysis in analyses:
            category = analysis.get("category", "")
            if category in ref_results:
                cat_refs = ref_results[category]

                if "response" in cat_refs:
                    analysis["response_page_citations"] = (
                        cat_refs["response"].page_numbers
                    )

                if "reasoning" in cat_refs:
                    analysis["scoring_reasoning_page_citations"] = (
                        cat_refs["reasoning"].page_numbers
                    )

        return analyses
