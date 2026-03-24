"""
Agent 3: Quality Assurance and Validation
==========================================

Multi-dimensional quality assessment implementing:
    - Readability Analysis: Flesch-Kincaid metrics
    - Semantic Coherence: BERT-based scoring
    - Coverage Completeness: Framework element coverage verification
    - Schema Compliance: JSON structure validation with auto-correction

Outputs failing composite threshold (<0.75) trigger automatic re-analysis.
Checkpointing at each sub-element ensures continuity.
"""

import json
import logging
import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger("prism.agent3")


@dataclass
class QualityScore:
    """Quality assessment score for a single sub-element."""
    category: str
    readability: float
    coherence: float
    coverage: float
    schema_compliance: float
    composite: float
    issues: List[str]
    passed: bool


@dataclass
class QualityReport:
    """Aggregate quality report for an entire analysis."""
    scores: List[QualityScore]
    overall_composite: float
    pass_rate: float
    failed_categories: List[str]
    total_issues: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_composite": round(self.overall_composite, 4),
            "pass_rate": round(self.pass_rate, 4),
            "failed_categories": self.failed_categories,
            "total_issues": self.total_issues,
            "scores": [
                {
                    "category": s.category,
                    "readability": round(s.readability, 4),
                    "coherence": round(s.coherence, 4),
                    "coverage": round(s.coverage, 4),
                    "schema_compliance": round(s.schema_compliance, 4),
                    "composite": round(s.composite, 4),
                    "passed": s.passed,
                    "issues": s.issues,
                }
                for s in self.scores
            ],
        }


class ReadabilityAnalyzer:
    """Flesch-Kincaid readability metrics for policy analysis outputs."""

    @staticmethod
    def count_syllables(word: str) -> int:
        """Estimate syllable count for a word."""
        word = word.lower().strip()
        if not word:
            return 0

        count = 0
        vowels = "aeiouy"
        prev_vowel = False

        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_vowel:
                count += 1
            prev_vowel = is_vowel

        # Adjust for common patterns
        if word.endswith("e") and count > 1:
            count -= 1
        if word.endswith("le") and len(word) > 2 and word[-3] not in vowels:
            count += 1
        return max(count, 1)

    @classmethod
    def flesch_kincaid_grade(cls, text: str) -> float:
        """Calculate Flesch-Kincaid Grade Level.

        Target for policy documents: 12-16 (accessible to policymakers).
        """
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = text.split()

        if not sentences or not words:
            return 0.0

        total_words = len(words)
        total_sentences = len(sentences)
        total_syllables = sum(cls.count_syllables(w) for w in words)

        grade = (
            0.39 * (total_words / total_sentences)
            + 11.8 * (total_syllables / total_words)
            - 15.59
        )
        return max(grade, 0.0)

    @classmethod
    def flesch_reading_ease(cls, text: str) -> float:
        """Calculate Flesch Reading Ease score (0-100, higher = easier)."""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = text.split()

        if not sentences or not words:
            return 0.0

        total_words = len(words)
        total_sentences = len(sentences)
        total_syllables = sum(cls.count_syllables(w) for w in words)

        score = (
            206.835
            - 1.015 * (total_words / total_sentences)
            - 84.6 * (total_syllables / total_words)
        )
        return max(min(score, 100.0), 0.0)

    @classmethod
    def score_readability(cls, text: str) -> float:
        """Score readability on 0-1 scale.

        Optimal: Grade level 12-16, Reading ease 30-60.
        """
        if not text or len(text.split()) < 10:
            return 0.0

        grade = cls.flesch_kincaid_grade(text)
        ease = cls.flesch_reading_ease(text)

        # Grade level scoring (peak at 12-16)
        if 12 <= grade <= 16:
            grade_score = 1.0
        elif grade < 8 or grade > 22:
            grade_score = 0.3
        else:
            grade_score = 0.7

        # Reading ease scoring (30-60 optimal for technical docs)
        if 30 <= ease <= 60:
            ease_score = 1.0
        elif ease < 10 or ease > 80:
            ease_score = 0.3
        else:
            ease_score = 0.7

        return (grade_score + ease_score) / 2.0


class SemanticCoherenceAnalyzer:
    """BERT-based semantic coherence scoring."""

    def __init__(self, model_name: str = "bert-base-uncased"):
        self.model_name = model_name
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. "
                "Using heuristic coherence scoring."
            )

    def score_coherence(self, text: str) -> float:
        """Score semantic coherence of text on 0-1 scale.

        Measures consistency of meaning across sentences using
        embedding similarity.
        """
        if not text or len(text.split()) < 20:
            return 0.5

        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

        if len(sentences) < 2:
            return 0.5

        self._load_model()

        if self._model is not None:
            return self._score_semantic(sentences)
        return self._score_heuristic(sentences)

    def _score_semantic(self, sentences: List[str]) -> float:
        """Score using sentence embeddings (BERT-based)."""
        import numpy as np

        embeddings = self._model.encode(sentences, normalize_embeddings=True)

        # Calculate pairwise similarities between consecutive sentences
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = float(np.dot(embeddings[i], embeddings[i + 1]))
            similarities.append(sim)

        if not similarities:
            return 0.5

        mean_sim = float(np.mean(similarities))
        # Map similarity (typically 0.2-0.8) to 0-1 scale
        return min(max((mean_sim - 0.1) / 0.7, 0.0), 1.0)

    def _score_heuristic(self, sentences: List[str]) -> float:
        """Simple heuristic coherence scoring without ML models."""
        # Check for vocabulary overlap between consecutive sentences
        overlaps = []
        for i in range(len(sentences) - 1):
            words_a = set(sentences[i].lower().split())
            words_b = set(sentences[i + 1].lower().split())
            if words_a and words_b:
                overlap = len(words_a & words_b) / min(len(words_a), len(words_b))
                overlaps.append(overlap)

        if not overlaps:
            return 0.5

        mean_overlap = sum(overlaps) / len(overlaps)
        return min(mean_overlap * 2.0, 1.0)


class CoverageAnalyzer:
    """Verify that all framework elements receive adequate attention."""

    MINIMUM_RESPONSE_WORDS = 30
    MINIMUM_REASONING_WORDS = 20

    @classmethod
    def score_coverage(
        cls,
        analysis: Dict[str, Any],
        definition: str,
        indicators: List[str],
    ) -> Tuple[float, List[str]]:
        """Score coverage completeness for a sub-element analysis.

        Returns (score, list_of_issues).
        """
        issues = []
        score = 1.0

        response = analysis.get("response", "")
        reasoning = analysis.get("scoring_reasoning", "")

        # Check response length
        resp_words = len(response.split())
        if resp_words < cls.MINIMUM_RESPONSE_WORDS:
            issues.append(
                f"Response too short ({resp_words} words, "
                f"minimum {cls.MINIMUM_RESPONSE_WORDS})"
            )
            score -= 0.3

        # Check reasoning length
        reason_words = len(reasoning.split())
        if reason_words < cls.MINIMUM_REASONING_WORDS:
            issues.append(
                f"Reasoning too short ({reason_words} words, "
                f"minimum {cls.MINIMUM_REASONING_WORDS})"
            )
            score -= 0.2

        # Check indicator coverage
        if indicators:
            mentioned = 0
            response_lower = response.lower()
            for indicator in indicators:
                # Check if indicator or key terms are mentioned
                ind_terms = set(indicator.lower().split())
                # Require at least half the terms to match
                matches = sum(1 for t in ind_terms if t in response_lower)
                if matches >= len(ind_terms) * 0.5:
                    mentioned += 1

            coverage_ratio = mentioned / len(indicators)
            if coverage_ratio < 0.3:
                issues.append(
                    f"Low indicator coverage ({mentioned}/{len(indicators)} "
                    f"indicators referenced)"
                )
                score -= 0.2

        # Check definition coverage
        if definition:
            defn_terms = set(
                w.lower() for w in definition.split()
                if len(w) > 4  # Skip short common words
            )
            resp_terms = set(response.lower().split())
            defn_coverage = len(defn_terms & resp_terms) / max(len(defn_terms), 1)
            if defn_coverage < 0.2:
                issues.append("Response does not address key definition terms")
                score -= 0.2

        return max(score, 0.0), issues


class SchemaValidator:
    """JSON schema validation with automatic error correction."""

    REQUIRED_FIELDS = {
        "category": str,
        "response": str,
        "score": (int, float),
        "scoring_reasoning": str,
        "response_page_citations": list,
        "scoring_reasoning_page_citations": list,
    }

    @classmethod
    def validate_and_repair(
        cls, analysis: Dict[str, Any], category: str
    ) -> Tuple[Dict[str, Any], float, List[str]]:
        """Validate schema and repair if possible.

        Returns (repaired_analysis, compliance_score, issues).
        """
        issues = []
        repaired = dict(analysis)
        fields_present = 0
        total_fields = len(cls.REQUIRED_FIELDS)

        for field_name, expected_type in cls.REQUIRED_FIELDS.items():
            if field_name not in repaired:
                issues.append(f"Missing field: {field_name}")
                # Auto-repair with defaults
                if expected_type == str:
                    repaired[field_name] = ""
                elif expected_type == list:
                    repaired[field_name] = []
                elif expected_type in ((int, float), int, float):
                    repaired[field_name] = 0
            else:
                value = repaired[field_name]
                if isinstance(expected_type, tuple):
                    if not isinstance(value, expected_type):
                        issues.append(
                            f"Type mismatch for {field_name}: "
                            f"expected {expected_type}, got {type(value)}"
                        )
                        try:
                            repaired[field_name] = int(value)
                        except (ValueError, TypeError):
                            repaired[field_name] = 0
                    else:
                        fields_present += 1
                elif not isinstance(value, expected_type):
                    issues.append(
                        f"Type mismatch for {field_name}: "
                        f"expected {expected_type.__name__}, "
                        f"got {type(value).__name__}"
                    )
                else:
                    fields_present += 1

            # Validate citations are lists of integers
            if field_name.endswith("_citations") and field_name in repaired:
                citations = repaired[field_name]
                if isinstance(citations, list):
                    cleaned = []
                    for c in citations:
                        try:
                            cleaned.append(int(c))
                        except (ValueError, TypeError):
                            pass
                    repaired[field_name] = cleaned

        # Ensure category matches
        repaired["category"] = category

        # Validate score range (0-5 for most elements)
        score = repaired.get("score", 0)
        if isinstance(score, (int, float)):
            repaired["score"] = max(0, min(int(score), 5))

        compliance = fields_present / total_fields
        return repaired, compliance, issues


class QualityAssuranceAgent:
    """Agent 3: Quality assurance and validation.

    Implements multi-dimensional quality assessment with configurable
    thresholds and automatic remediation triggering.
    """

    def __init__(self, config, framework_loader):
        self.config = config
        self.framework = framework_loader
        self.readability = ReadabilityAnalyzer()
        self.coherence = SemanticCoherenceAnalyzer(config.bert_model)
        self.coverage = CoverageAnalyzer()
        self.schema = SchemaValidator()

    def validate_analysis(
        self, analyses: List[Dict[str, Any]]
    ) -> QualityReport:
        """Validate a complete analysis against quality thresholds.

        Args:
            analyses: List of sub-element analysis dicts.

        Returns:
            QualityReport with per-element and aggregate scores.
        """
        scores = []
        failed = []
        total_issues = 0

        for analysis in analyses:
            category = analysis.get("category", "Unknown")
            qs = self._validate_single(analysis, category)
            scores.append(qs)
            total_issues += len(qs.issues)
            if not qs.passed:
                failed.append(category)

        # Calculate aggregate metrics
        if scores:
            overall = sum(s.composite for s in scores) / len(scores)
            pass_rate = sum(1 for s in scores if s.passed) / len(scores)
        else:
            overall = 0.0
            pass_rate = 0.0

        report = QualityReport(
            scores=scores,
            overall_composite=overall,
            pass_rate=pass_rate,
            failed_categories=failed,
            total_issues=total_issues,
        )

        logger.info(
            f"Quality validation: composite={overall:.3f}, "
            f"pass_rate={pass_rate:.1%}, "
            f"failed={len(failed)}/{len(scores)}"
        )
        return report

    def _validate_single(
        self, analysis: Dict[str, Any], category: str
    ) -> QualityScore:
        """Validate a single sub-element analysis."""
        all_issues = []

        # 1. Schema validation
        repaired, schema_score, schema_issues = self.schema.validate_and_repair(
            analysis, category
        )
        all_issues.extend(schema_issues)

        response_text = repaired.get("response", "")
        reasoning_text = repaired.get("scoring_reasoning", "")
        combined_text = f"{response_text}\n{reasoning_text}"

        # 2. Readability
        readability_score = self.readability.score_readability(combined_text)

        # 3. Semantic coherence
        coherence_score = self.coherence.score_coherence(combined_text)

        # 4. Coverage completeness
        try:
            definition = self.framework.get_definition(category)
            indicators = self.framework.get_indicators(category)
        except KeyError:
            definition = ""
            indicators = []

        coverage_score, coverage_issues = self.coverage.score_coverage(
            repaired, definition, indicators
        )
        all_issues.extend(coverage_issues)

        # Compute weighted composite
        composite = (
            self.config.readability_weight * readability_score
            + self.config.coherence_weight * coherence_score
            + self.config.coverage_weight * coverage_score
            + self.config.schema_weight * schema_score
        )

        passed = composite >= self.config.composite_threshold

        return QualityScore(
            category=category,
            readability=readability_score,
            coherence=coherence_score,
            coverage=coverage_score,
            schema_compliance=schema_score,
            composite=composite,
            issues=all_issues,
            passed=passed,
        )

    def get_remediation_categories(
        self, report: QualityReport
    ) -> List[str]:
        """Get list of categories needing re-analysis."""
        return report.failed_categories
