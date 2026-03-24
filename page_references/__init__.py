"""
PRISM Page Reference Extraction
================================

Three-method triangulation approach for automated source attribution:
    - Method A: Numeric Entity Matching
    - Method B: Sliding Window Semantic Similarity
    - Method C: Generative Matching with Temperature Sampling

Final attribution: PageRefs = A ∩ B ∩ C (with fallback hierarchy).
"""

from PRISM.page_references.extractor import PageReferenceExtractor
from PRISM.page_references.methods import (
    NumericEntityMatcher,
    SemanticSimilarityMatcher,
    GenerativeMatcher,
)
