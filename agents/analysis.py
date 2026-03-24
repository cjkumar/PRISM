"""
Agent 2: Policy Analysis and Scoring
=====================================

Employs Llama 4 Scout 70B for policy analysis with Retrieval-Augmented Generation.
Generates structured JSON output for each framework sub-element containing:
    - response: Comprehensive analytical assessment
    - score: Quantitative rating based on scoring criteria
    - scoring_reasoning: Detailed justification with textual evidence

Uses Pydantic models for schema enforcement and the instructor library for
structured output with automatic validation and retries.

Supports domain-specific frameworks for cancer and CVD control plans.

Performance: 0.75-0.82 accuracy on LegalBench, 0.73 F1 on PolicyQA.
"""

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger("prism.agent2")


# ---------------------------------------------------------------------------
# Pydantic models for structured LLM output
# ---------------------------------------------------------------------------

class SubElementResponse(BaseModel):
    """Pydantic schema for a single sub-element LLM response.

    Instructor uses this to enforce JSON mode, validate output,
    and automatically retry on malformed responses.
    """
    category: str = Field(description="The framework sub-element category name")
    response: str = Field(
        min_length=50,
        description=(
            "Comprehensive 2-4 paragraph analytical assessment of how the plan "
            "addresses this sub-element. Must reference specific content from "
            "the document."
        ),
    )
    score: int = Field(
        ge=0, le=5,
        description="Integer score based on the scoring criteria",
    )
    scoring_reasoning: str = Field(
        min_length=30,
        description=(
            "Detailed justification for the score citing textual evidence "
            "and explicitly referencing the scoring criteria."
        ),
    )

    @field_validator("response")
    @classmethod
    def response_must_not_be_template(cls, v: str) -> str:
        """Reject placeholder / template responses echoed back by the model."""
        placeholders = [
            "<Comprehensive analytical assessment",
            "[Provide details about",
            "<Your response here>",
        ]
        for p in placeholders:
            if p in v:
                raise ValueError(
                    f"Response contains template placeholder text: '{p}'. "
                    "Provide a real analytical assessment."
                )
        return v

    @field_validator("scoring_reasoning")
    @classmethod
    def reasoning_must_not_be_template(cls, v: str) -> str:
        placeholders = [
            "<Detailed justification",
            "[Explain the reasoning",
        ]
        for p in placeholders:
            if p in v:
                raise ValueError(
                    f"Scoring reasoning contains template placeholder: '{p}'. "
                    "Provide real justification."
                )
        return v


# ---------------------------------------------------------------------------
# Dataclasses for internal pipeline results (unchanged interface)
# ---------------------------------------------------------------------------

@dataclass
class SubElementAnalysis:
    """Analysis result for a single framework sub-element."""
    category: str
    response: str
    score: int
    scoring_reasoning: str
    response_page_citations: List[int]
    scoring_reasoning_page_citations: List[int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "response": self.response,
            "response_page_citations": self.response_page_citations,
            "score": self.score,
            "scoring_reasoning": self.scoring_reasoning,
            "scoring_reasoning_page_citations": self.scoring_reasoning_page_citations,
        }


@dataclass
class AnalysisResult:
    """Complete analysis result for a document."""
    country: str
    year: str
    domain: str
    sub_elements: List[SubElementAnalysis]
    processing_time_seconds: float
    model_name: str

    def to_dict(self) -> List[Dict[str, Any]]:
        """Return list of sub-element dicts (matches existing JSON output format)."""
        return [se.to_dict() for se in self.sub_elements]

    def to_json(self, path: str):
        """Save analysis to JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"Analysis saved to {path}")


class RAGKnowledgeBase:
    """Retrieval-Augmented Generation knowledge base for framework context.

    Maintains definitions, scoring criteria, and indicators for all
    framework elements, enabling consistent analysis across documents.
    """

    def __init__(self, framework_loader, config):
        self.framework = framework_loader
        self.config = config
        self._embeddings = None
        self._documents = []
        self._embedding_model = None

    def build_index(self):
        """Build the embedding index from framework documents."""
        self._documents = self.framework.build_rag_knowledge_base()

        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np

            self._embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            texts = [doc["content"] for doc in self._documents]
            self._embeddings = self._embedding_model.encode(
                texts, normalize_embeddings=True, show_progress_bar=False
            )
            logger.info(
                f"RAG index built: {len(self._documents)} documents, "
                f"embeddings shape {self._embeddings.shape}"
            )
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. "
                "RAG will use keyword matching fallback."
            )

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, str]]:
        """Retrieve the most relevant knowledge base documents for a query."""
        if self._embeddings is not None and self._embedding_model is not None:
            return self._retrieve_semantic(query, top_k)
        return self._retrieve_keyword(query, top_k)

    def _retrieve_semantic(self, query: str, top_k: int) -> List[Dict[str, str]]:
        """Semantic retrieval using embedding similarity."""
        import numpy as np

        query_embedding = self._embedding_model.encode(
            [query], normalize_embeddings=True
        )
        similarities = np.dot(self._embeddings, query_embedding.T).flatten()
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            doc = self._documents[idx].copy()
            doc["similarity"] = float(similarities[idx])
            results.append(doc)
        return results

    def _retrieve_keyword(self, query: str, top_k: int) -> List[Dict[str, str]]:
        """Simple keyword-based retrieval fallback."""
        query_terms = set(query.lower().split())
        scored = []
        for doc in self._documents:
            content_terms = set(doc["content"].lower().split())
            overlap = len(query_terms & content_terms)
            scored.append((overlap, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:top_k]]


class PolicyAnalysisAgent:
    """Agent 2: Policy analysis, scoring, and reasoning.

    Uses an LLM with RAG to analyze policy documents against
    validated analytical frameworks. Employs instructor + Pydantic
    for structured output with automatic validation and retries.
    """

    def __init__(self, config, framework_loader):
        self.config = config
        self.framework = framework_loader
        self.rag = RAGKnowledgeBase(framework_loader, config)
        self._instructor_client = None

    def initialize(self):
        """Initialize the RAG knowledge base and the instructor client."""
        self.rag.build_index()
        self._init_instructor()
        logger.info("Policy Analysis Agent initialized")

    def _init_instructor(self):
        """Set up the instructor-patched OpenAI client for structured output."""
        import instructor
        from openai import OpenAI

        api_base = os.environ.get("PRISM_API_BASE", "http://localhost:8000/v1")
        api_key = os.environ.get("PRISM_API_KEY", "not-needed")

        base_client = OpenAI(base_url=api_base, api_key=api_key)

        # Use JSON mode for Ollama / local models
        self._instructor_client = instructor.from_openai(
            base_client, mode=instructor.Mode.JSON
        )
        logger.info(f"Instructor client initialized (base_url={api_base})")

    def analyze_document(
        self,
        document_text: str,
        page_texts: Dict[int, str],
        country: str,
        year: str,
        domain: str = "cancer",
        checkpoint_dir: Optional[str] = None,
    ) -> AnalysisResult:
        """Analyze a document against all framework sub-elements."""
        start_time = time.time()
        categories = self.framework.categories
        analyses = []

        # Load any existing checkpoints
        completed = set()
        if checkpoint_dir:
            checkpoint_path = Path(checkpoint_dir)
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            for cp_file in checkpoint_path.glob("*.json"):
                try:
                    with open(cp_file) as f:
                        cp_data = json.load(f)
                    analyses.append(SubElementAnalysis(**cp_data))
                    completed.add(cp_data["category"])
                    logger.info(f"Loaded checkpoint: {cp_data['category']}")
                except Exception as e:
                    logger.warning(f"Failed to load checkpoint {cp_file}: {e}")

        for i, category in enumerate(categories):
            if category in completed:
                continue

            logger.info(
                f"Analyzing [{i + 1}/{len(categories)}]: {category}"
            )

            try:
                analysis = self._analyze_sub_element(
                    category, document_text, page_texts, domain
                )
                analyses.append(analysis)

                # Save checkpoint
                if checkpoint_dir:
                    cp_file = Path(checkpoint_dir) / f"{_safe_filename(category)}.json"
                    with open(cp_file, "w") as f:
                        json.dump(analysis.to_dict(), f, indent=2)

            except Exception as e:
                logger.error(f"Error analyzing {category}: {e}")
                analyses.append(SubElementAnalysis(
                    category=category,
                    response=f"Analysis failed: {e}",
                    score=0,
                    scoring_reasoning=f"Error during analysis: {e}",
                    response_page_citations=[],
                    scoring_reasoning_page_citations=[],
                ))

        processing_time = time.time() - start_time

        return AnalysisResult(
            country=country,
            year=year,
            domain=domain,
            sub_elements=analyses,
            processing_time_seconds=processing_time,
            model_name=self.config.model_name,
        )

    def _analyze_sub_element(
        self,
        category: str,
        document_text: str,
        page_texts: Dict[int, str],
        domain: str,
    ) -> SubElementAnalysis:
        """Analyze a single sub-element using RAG-enhanced prompting."""

        # Retrieve relevant context from RAG knowledge base
        rag_context = self.rag.retrieve(category, top_k=self.config.rag_top_k)
        rag_text = "\n\n".join(doc["content"] for doc in rag_context)

        # Build the framework prompt context
        framework_context = self.framework.build_prompt_context(category)

        # Construct the analysis prompt
        prompt = self._build_analysis_prompt(
            category, framework_context, rag_text, document_text, domain
        )

        # Generate structured output via instructor
        validated = self._generate_structured(prompt, category)

        return SubElementAnalysis(
            category=validated.category,
            response=validated.response,
            score=validated.score,
            scoring_reasoning=validated.scoring_reasoning,
            response_page_citations=[],
            scoring_reasoning_page_citations=[],
        )

    def _build_analysis_prompt(
        self,
        category: str,
        framework_context: str,
        rag_context: str,
        document_text: str,
        domain: str,
    ) -> str:
        """Construct the full analysis prompt with RAG context."""
        domain_label = (
            "National Cancer Control Plan" if domain == "cancer"
            else "National Cardiovascular Disease Control Plan"
        )

        # Truncate document if too long for context window
        max_doc_chars = 80000
        if len(document_text) > max_doc_chars:
            document_text = document_text[:max_doc_chars] + "\n\n[Document truncated]"

        return f"""You are an expert health policy analyst evaluating a {domain_label}.

{framework_context}

## Additional Context from Knowledge Base
{rag_context}

## Document Text
{document_text}

## Instructions

Analyze the document above for the sub-element "{category}".

Provide your analysis as a JSON object with these fields:
- "category": "{category}"
- "response": A comprehensive 2-4 paragraph analytical assessment of how the plan addresses {category}. Reference specific content from the document. Be thorough and detailed.
- "score": An integer score based on the scoring criteria above.
- "scoring_reasoning": Detailed justification for the score. Identify specific strengths, weaknesses, and cite textual evidence from the document. Explicitly reference the scoring criteria.

Rules:
- The score must be an integer within the valid range defined by the scoring criteria.
- The response should be 2-4 paragraphs analyzing the document's coverage of this sub-element.
- The scoring_reasoning must explicitly reference the scoring criteria to justify the assigned score.
- If the document does not address this sub-element at all, assign a score of 0 with appropriate reasoning.
- Be objective and evidence-based. Do not infer content that is not present in the document."""

    def _generate_structured(
        self, prompt: str, category: str
    ) -> SubElementResponse:
        """Generate a validated structured response using instructor.

        Instructor handles: JSON mode enforcement, Pydantic validation,
        and automatic retries (up to max_retries) when validation fails.
        """
        model_name = os.environ.get("PRISM_MODEL_NAME", self.config.model_name)

        try:
            result = self._instructor_client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert health policy analyst. "
                            "Respond with a detailed analytical JSON object. "
                            "Never echo back template placeholders."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                response_model=SubElementResponse,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
                max_retries=self.config.max_retries,
            )
            logger.info(
                f"Structured output validated for {category} (score={result.score})"
            )
            return result

        except Exception as e:
            logger.warning(
                f"Instructor structured generation failed for {category}: {e}. "
                f"Falling back to raw API call."
            )
            return self._fallback_generate(prompt, category)

    def _fallback_generate(self, prompt: str, category: str) -> SubElementResponse:
        """Fallback: raw API call with manual JSON parsing and Pydantic validation."""
        api_base = os.environ.get("PRISM_API_BASE", "http://localhost:8000/v1")
        api_key = os.environ.get("PRISM_API_KEY", "not-needed")
        model_name = os.environ.get("PRISM_MODEL_NAME", self.config.model_name)

        from openai import OpenAI
        client = OpenAI(base_url=api_base, api_key=api_key)

        for attempt in range(1, self.config.max_retries + 1):
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are an expert health policy analyst. "
                                "Respond with ONLY a valid JSON object."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    top_p=self.config.top_p,
                    response_format={"type": "json_object"},
                )
                raw = response.choices[0].message.content

                # Strip markdown fences if present
                json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw)
                if json_match:
                    raw = json_match.group(1)

                # Validate with Pydantic
                result = SubElementResponse.model_validate_json(raw)
                result.category = category
                logger.info(
                    f"Fallback validated for {category} (score={result.score})"
                )
                return result

            except Exception as e:
                logger.warning(
                    f"Fallback attempt {attempt}/{self.config.max_retries} "
                    f"for {category}: {e}"
                )
                if attempt < self.config.max_retries:
                    time.sleep(self.config.retry_delay * (2 ** (attempt - 1)))

        # All retries exhausted — return a minimal valid response
        logger.error(f"All retries exhausted for {category}")
        return SubElementResponse(
            category=category,
            response=f"Analysis could not be completed for {category} after {self.config.max_retries} attempts.",
            score=0,
            scoring_reasoning=f"Unable to generate valid analysis for {category}. Manual review required.",
        )


def _safe_filename(name: str) -> str:
    """Convert a category name to a safe filename."""
    return re.sub(r'[^\w\-]', '_', name).strip('_')
