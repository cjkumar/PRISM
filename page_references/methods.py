"""
Page Reference Extraction Methods
==================================

Three complementary extraction techniques for source attribution:

Method A - Numeric Entity Matching:
    Extracts percentages, targets, statistics, budget figures, date references
    from LLM output and matches against source page content.

Method B - Sliding Window Semantic Similarity:
    50-token windows with 25-token stride, encoded via all-MiniLM-L6-v2,
    with adaptive threshold selection via K-Means clustering.

Method C - Generative Matching with Temperature Sampling:
    LLM validation of candidate pages using temperature sampling (0.8-1.0),
    majority voting (≥2/3 YES required).
"""

import logging
import re
from typing import Dict, List, Set, Optional, Tuple

import numpy as np

logger = logging.getLogger("prism.page_references")


class NumericEntityMatcher:
    """Method A: Numeric entity matching for high-confidence page attribution.

    Extracts specific numeric values from LLM output and matches them
    against source document pages. Numeric anchors (percentages, years,
    targets) provide strong evidence of source page relevance.
    """

    # Regex patterns for numeric entity extraction
    PATTERNS = {
        "percentage": re.compile(r'(\d+\.?\d*)\s*%'),
        "year": re.compile(r'\b((?:19|20)\d{2})\b'),
        "currency": re.compile(
            r'(?:USD|\$|€|£|¥)\s*(\d[\d,]*\.?\d*)\s*(?:million|billion|trillion|[MBT])?',
            re.IGNORECASE,
        ),
        "large_number": re.compile(
            r'\b(\d{1,3}(?:,\d{3})+)\b'
        ),
        "integer": re.compile(r'\b(\d+)\b'),
    }

    # Context keywords required for small numbers (<10) to avoid false matches
    CONTEXT_KEYWORDS = {
        "score", "level", "stage", "phase", "tier", "grade", "category",
        "priority", "objective", "goal", "target", "pillar", "element",
        "step", "type", "class", "group", "domain", "dimension",
    }

    def __init__(self, year_range: Tuple[int, int] = (1990, 2099)):
        self.year_range = year_range

    def extract_entities(self, text: str) -> Set[str]:
        """Extract numeric entities from text.

        Returns a set of string representations of found entities.
        """
        entities = set()

        # Percentages
        for match in self.PATTERNS["percentage"].finditer(text):
            entities.add(f"{match.group(1)}%")

        # Years
        for match in self.PATTERNS["year"].finditer(text):
            year = int(match.group(1))
            if self.year_range[0] <= year <= self.year_range[1]:
                entities.add(match.group(1))

        # Currency values
        for match in self.PATTERNS["currency"].finditer(text):
            entities.add(match.group(0).strip())

        # Large numbers (with commas)
        for match in self.PATTERNS["large_number"].finditer(text):
            entities.add(match.group(1))

        # Small integers need context validation
        for match in self.PATTERNS["integer"].finditer(text):
            num = int(match.group(1))
            if num >= 10:
                entities.add(match.group(1))
            elif num > 0:
                # Check for adjacent context keywords
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                context = text[start:end].lower()
                if any(kw in context for kw in self.CONTEXT_KEYWORDS):
                    entities.add(match.group(1))

        return entities

    def match_pages(
        self,
        llm_text: str,
        page_texts: Dict[int, str],
    ) -> Set[int]:
        """Find pages containing numeric entities from the LLM output.

        Args:
            llm_text: The LLM-generated text (response or reasoning).
            page_texts: Dict mapping page numbers to their text content.

        Returns:
            Set A: Page numbers containing matching numeric entities.
        """
        llm_entities = self.extract_entities(llm_text)

        if not llm_entities:
            logger.debug("No numeric entities found in LLM output")
            return set()

        matched_pages = set()
        for page_num, page_text in page_texts.items():
            page_entities = self.extract_entities(page_text)
            overlap = llm_entities & page_entities
            if overlap:
                matched_pages.add(page_num)
                logger.debug(
                    f"Page {page_num}: {len(overlap)} entity matches "
                    f"({', '.join(list(overlap)[:5])})"
                )

        logger.info(
            f"Method A: {len(matched_pages)} pages matched "
            f"({len(llm_entities)} entities extracted)"
        )
        return matched_pages


class SemanticSimilarityMatcher:
    """Method B: Sliding window semantic similarity matching.

    Segments LLM output and page text into overlapping windows,
    computes sentence embeddings, and identifies pages with high
    semantic similarity. Adaptive threshold via K-Means clustering.

    Parameters:
        window_size: 50 tokens per window
        stride: 25 tokens (50% overlap)
        embedding_model: all-MiniLM-L6-v2 (384-dim, L2 normalized)
    """

    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        window_size: int = 50,
        stride: int = 25,
        batch_size: int = 32,
    ):
        self.model_name = embedding_model_name
        self.window_size = window_size
        self.stride = stride
        self.batch_size = batch_size
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            logger.info(f"Loaded embedding model: {self.model_name}")
        except ImportError:
            raise ImportError(
                "sentence-transformers required: "
                "pip install sentence-transformers"
            )

    def _create_windows(self, text: str) -> List[str]:
        """Segment text into overlapping token windows."""
        tokens = text.split()
        windows = []
        for i in range(0, max(len(tokens) - self.window_size + 1, 1), self.stride):
            window = " ".join(tokens[i:i + self.window_size])
            if window.strip():
                windows.append(window)
        # Add the last window if we haven't covered it
        if tokens and not windows:
            windows.append(" ".join(tokens))
        return windows

    def _compute_embeddings(self, texts: List[str]) -> np.ndarray:
        """Compute L2-normalized sentence embeddings."""
        self._load_model()
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        return embeddings

    def _adaptive_threshold(self, similarities: np.ndarray) -> float:
        """Determine adaptive similarity threshold via K-Means + Elbow Method.

        Clusters page-level max similarities to find the natural separation
        between relevant and irrelevant pages.
        """
        if len(similarities) < 3:
            return float(np.median(similarities))

        try:
            from sklearn.cluster import KMeans

            # Try k=2 and k=3, use elbow method
            best_k = 2
            best_inertia_drop = 0

            inertias = []
            for k in range(1, min(5, len(similarities))):
                km = KMeans(n_clusters=k, n_init=10, random_state=42)
                km.fit(similarities.reshape(-1, 1))
                inertias.append(km.inertia_)

            # Elbow: largest relative drop
            for i in range(1, len(inertias) - 1):
                drop = inertias[i - 1] - inertias[i]
                if drop > best_inertia_drop:
                    best_inertia_drop = drop
                    best_k = i + 1

            km = KMeans(n_clusters=min(best_k, len(similarities)),
                        n_init=10, random_state=42)
            km.fit(similarities.reshape(-1, 1))

            # Threshold = midpoint between the two highest cluster centers
            centers = sorted(km.cluster_centers_.flatten())
            if len(centers) >= 2:
                threshold = (centers[-1] + centers[-2]) / 2.0
            else:
                threshold = centers[0]

            return float(threshold)

        except ImportError:
            logger.warning("scikit-learn not available, using median threshold")
            return float(np.median(similarities))

    def match_pages(
        self,
        llm_text: str,
        page_texts: Dict[int, str],
    ) -> Tuple[Set[int], Dict[int, float]]:
        """Find pages semantically similar to the LLM output.

        Args:
            llm_text: The LLM-generated text.
            page_texts: Dict mapping page numbers to text.

        Returns:
            Tuple of (Set B: matching page numbers, page_similarity_scores).
        """
        # Create windows from LLM text
        llm_windows = self._create_windows(llm_text)
        if not llm_windows:
            return set(), {}

        llm_embeddings = self._compute_embeddings(llm_windows)

        # Compute per-page maximum similarity
        page_similarities = {}
        for page_num, page_text in page_texts.items():
            if not page_text.strip():
                page_similarities[page_num] = 0.0
                continue

            page_windows = self._create_windows(page_text)
            if not page_windows:
                page_similarities[page_num] = 0.0
                continue

            page_embeddings = self._compute_embeddings(page_windows)

            # Compute similarity matrix: llm_windows x page_windows
            sim_matrix = np.dot(llm_embeddings, page_embeddings.T)
            max_sim = float(np.max(sim_matrix))
            page_similarities[page_num] = max_sim

        if not page_similarities:
            return set(), {}

        # Adaptive threshold
        sim_array = np.array(list(page_similarities.values()))
        threshold = self._adaptive_threshold(sim_array)
        logger.info(f"Method B: adaptive threshold = {threshold:.4f}")

        # Select pages above threshold
        matched = {
            page for page, sim in page_similarities.items()
            if sim >= threshold
        }

        logger.info(
            f"Method B: {len(matched)} pages above threshold "
            f"(out of {len(page_texts)})"
        )
        return matched, page_similarities


class GenerativeMatcher:
    """Method C: Generative matching with temperature sampling.

    Uses LLM inference to validate candidate pages identified by
    Method B. Employs temperature sampling (0.8-1.0) with multiple
    samples and majority voting for robust confirmation.

    Pre-filter: Only pages above 25th percentile in Method B similarity.
    Majority: ≥2/3 YES responses required to confirm match.
    """

    def __init__(
        self,
        temperature_low: float = 0.8,
        temperature_high: float = 1.0,
        top_p: float = 0.95,
        n_samples: int = 3,
        majority_threshold: int = 2,
        prefilter_percentile: float = 0.25,
    ):
        self.temperature_low = temperature_low
        self.temperature_high = temperature_high
        self.top_p = top_p
        self.n_samples = n_samples
        self.majority_threshold = majority_threshold
        self.prefilter_percentile = prefilter_percentile

    def match_pages(
        self,
        llm_text: str,
        page_texts: Dict[int, str],
        page_similarities: Dict[int, float],
        generate_fn,
    ) -> Set[int]:
        """Validate candidate pages via generative matching.

        Args:
            llm_text: The LLM-generated analysis text.
            page_texts: Dict mapping page numbers to text.
            page_similarities: Similarity scores from Method B.
            generate_fn: Callable(prompt, temperature, top_p) -> str

        Returns:
            Set C: Pages confirmed by generative matching.
        """
        # Pre-filter: only evaluate pages above 25th percentile
        if not page_similarities:
            return set()

        sim_values = list(page_similarities.values())
        percentile_threshold = float(
            np.percentile(sim_values, self.prefilter_percentile * 100)
        )

        candidates = {
            page for page, sim in page_similarities.items()
            if sim >= percentile_threshold
        }

        if not candidates:
            return set()

        logger.info(
            f"Method C: evaluating {len(candidates)} candidate pages "
            f"(pre-filtered from {len(page_texts)})"
        )

        matched = set()
        for page_num in sorted(candidates):
            page_text = page_texts.get(page_num, "")
            if not page_text.strip():
                continue

            # Truncate page text if very long
            if len(page_text) > 3000:
                page_text = page_text[:3000] + "..."

            # Majority voting
            yes_count = 0
            for sample_idx in range(self.n_samples):
                temperature = (
                    self.temperature_low
                    + (self.temperature_high - self.temperature_low)
                    * sample_idx / max(self.n_samples - 1, 1)
                )

                prompt = self._build_validation_prompt(
                    llm_text, page_text, page_num
                )

                try:
                    response = generate_fn(prompt, temperature, self.top_p)
                    if self._parse_yes_no(response):
                        yes_count += 1
                except Exception as e:
                    logger.warning(
                        f"Generative validation failed for page {page_num} "
                        f"(sample {sample_idx}): {e}"
                    )

            if yes_count >= self.majority_threshold:
                matched.add(page_num)
                logger.debug(
                    f"Page {page_num}: CONFIRMED ({yes_count}/{self.n_samples})"
                )
            else:
                logger.debug(
                    f"Page {page_num}: REJECTED ({yes_count}/{self.n_samples})"
                )

        logger.info(f"Method C: {len(matched)} pages confirmed")
        return matched

    def _build_validation_prompt(
        self, llm_text: str, page_text: str, page_num: int
    ) -> str:
        """Build the generative validation prompt."""
        # Truncate llm_text if needed
        if len(llm_text) > 2000:
            llm_text = llm_text[:2000] + "..."

        return f"""You are verifying whether a specific page from a policy document was used as a source for an analytical assessment.

## Analytical Assessment (excerpt)
{llm_text}

## Document Page {page_num} Content
{page_text}

## Question
Does the content on Page {page_num} contain information that was directly used in or informed the analytical assessment above? Consider both direct quotes and paraphrased content.

Answer with ONLY "YES" or "NO"."""

    def _parse_yes_no(self, response: str) -> bool:
        """Parse a YES/NO response from the LLM."""
        response = response.strip().upper()
        if response.startswith("YES"):
            return True
        if response.startswith("NO"):
            return False
        # Fuzzy matching
        return "YES" in response and "NO" not in response
