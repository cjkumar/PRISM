"""
PRISM Pipeline Orchestrator
============================

Implements a directed acyclic graph (DAG) architecture connecting the
three primary agents with feedback loops for quality remediation.

Pipeline flow:
    PDF → Agent 1 (Ingestion) → Agent 2 (Analysis) → Agent 3 (QA)
                                      ↑                    |
                                      └── Remediation ←────┘

    Agent 2 output → Page Reference Extraction → Final Output

The pipeline maintains comprehensive logging at each stage, preserves
full audit trails, and supports per-element checkpointing.
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from PRISM.config import PRISMConfig
from PRISM.agents.ingestion import DocumentIngestionAgent, IngestionResult
from PRISM.agents.analysis import PolicyAnalysisAgent, AnalysisResult
from PRISM.agents.quality import QualityAssuranceAgent, QualityReport
from PRISM.frameworks.loader import FrameworkLoader
from PRISM.page_references.extractor import PageReferenceExtractor

logger = logging.getLogger("prism.pipeline")


class PRISMPipeline:
    """End-to-end pipeline orchestrating PRISM's multi-agent architecture.

    Processes PDF policy documents through:
    1. Document Ingestion (Agent 1)
    2. Policy Analysis (Agent 2)
    3. Quality Assurance (Agent 3) with remediation loop
    4. Page Reference Extraction (optional)
    5. Output Generation (JSON + CSV)
    """

    def __init__(self, config: Optional[PRISMConfig] = None):
        self.config = config or PRISMConfig()
        self._setup_logging()

        # Load framework
        self.framework = FrameworkLoader(self.config.framework_path)

        # Initialize agents
        self.agent1 = DocumentIngestionAgent(self.config.agent1)
        self.agent2 = PolicyAnalysisAgent(self.config.agent2, self.framework)
        self.agent3 = QualityAssuranceAgent(self.config.agent3, self.framework)

        # Page reference extractor
        self.page_ref_extractor = None
        if self.config.enable_page_references:
            self.page_ref_extractor = PageReferenceExtractor(
                self.config.page_ref
            )

        logger.info(
            f"PRISM Pipeline initialized "
            f"(domain={self.config.domain}, "
            f"config_hash={self.config.config_hash()})"
        )

    def _setup_logging(self):
        """Configure pipeline logging."""
        log_dir = Path(self.config.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        logging.basicConfig(
            level=level,
            format="%(asctime)s  %(name)-20s  %(levelname)-8s  %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(
                    log_dir / f"prism_{datetime.now():%Y%m%d_%H%M%S}.log"
                ),
            ],
        )

    def process_document(
        self,
        pdf_path: str,
        country: str,
        year: str,
        output_path: Optional[str] = None,
        lightweight_ingestion: bool = False,
    ) -> Dict[str, Any]:
        """Process a single policy document through the full pipeline.

        Args:
            pdf_path: Path to the PDF document.
            country: Country name.
            year: Plan publication year.
            output_path: Optional output JSON path. If None, auto-generated.
            lightweight_ingestion: Use PyMuPDF-only ingestion (no VL model).

        Returns:
            Dict with analysis results, quality report, and metadata.
        """
        pipeline_start = time.time()
        run_id = f"{country}_{year}_{datetime.now():%Y%m%d_%H%M%S}"

        logger.info(f"{'=' * 70}")
        logger.info(f"PRISM Pipeline: {country} ({year})")
        logger.info(f"Document: {pdf_path}")
        logger.info(f"Run ID: {run_id}")
        logger.info(f"{'=' * 70}")

        # Save pipeline config for reproducibility
        config_path = Path(self.config.log_dir) / f"{run_id}_config.json"
        self.config.save(str(config_path))

        # ── Stage 1: Document Ingestion ──
        logger.info("STAGE 1: Document Ingestion")
        stage1_start = time.time()

        if lightweight_ingestion:
            ingestion_result = self.agent1.process_document_lightweight(pdf_path)
        else:
            ingestion_result = self.agent1.process_document(pdf_path)

        page_texts = ingestion_result.page_texts
        document_text = ingestion_result.full_text

        logger.info(
            f"Stage 1 complete: {ingestion_result.total_pages} pages, "
            f"{time.time() - stage1_start:.1f}s"
        )

        # ── Stage 2: Policy Analysis ──
        logger.info("STAGE 2: Policy Analysis")
        stage2_start = time.time()

        self.agent2.initialize()

        checkpoint_dir = None
        if self.config.enable_checkpointing:
            checkpoint_dir = str(
                Path(self.config.checkpoint_dir) / run_id
            )

        analysis_result = self.agent2.analyze_document(
            document_text=document_text,
            page_texts=page_texts,
            country=country,
            year=year,
            domain=self.config.domain,
            checkpoint_dir=checkpoint_dir,
        )

        logger.info(
            f"Stage 2 complete: {len(analysis_result.sub_elements)} elements, "
            f"{time.time() - stage2_start:.1f}s"
        )

        # ── Stage 3: Quality Assurance ──
        logger.info("STAGE 3: Quality Assurance")
        stage3_start = time.time()

        analyses_dicts = analysis_result.to_dict()
        quality_report = self.agent3.validate_analysis(analyses_dicts)

        # Remediation loop
        remediation_count = 0
        while (
            quality_report.failed_categories
            and remediation_count < self.config.agent3.max_remediation_attempts
        ):
            remediation_count += 1
            failed = quality_report.failed_categories
            logger.info(
                f"Remediation attempt {remediation_count}: "
                f"re-analyzing {len(failed)} categories"
            )

            for category in failed:
                try:
                    new_analysis = self.agent2._analyze_sub_element(
                        category, document_text, page_texts, self.config.domain
                    )
                    # Replace in the analysis list
                    for i, se in enumerate(analysis_result.sub_elements):
                        if se.category == category:
                            analysis_result.sub_elements[i] = new_analysis
                            break
                except Exception as e:
                    logger.error(f"Remediation failed for {category}: {e}")

            analyses_dicts = analysis_result.to_dict()
            quality_report = self.agent3.validate_analysis(analyses_dicts)

        logger.info(
            f"Stage 3 complete: composite={quality_report.overall_composite:.3f}, "
            f"pass_rate={quality_report.pass_rate:.1%}, "
            f"remediations={remediation_count}, "
            f"{time.time() - stage3_start:.1f}s"
        )

        # ── Stage 4: Page Reference Extraction (optional) ──
        ref_results = None
        if self.page_ref_extractor and self.config.enable_page_references:
            logger.info("STAGE 4: Page Reference Extraction")
            stage4_start = time.time()

            ref_results = self.page_ref_extractor.extract_all_references(
                analyses_dicts, page_texts
            )

            # Update analyses with extracted references
            analyses_dicts = PageReferenceExtractor.update_analyses_with_references(
                analyses_dicts, ref_results
            )

            logger.info(
                f"Stage 4 complete: {time.time() - stage4_start:.1f}s"
            )

        # ── Output ──
        if output_path is None:
            output_dir = Path(self.config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = str(output_dir / f"{country}_{year}.json")

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(analyses_dicts, f, indent=2, ensure_ascii=False)

        pipeline_time = time.time() - pipeline_start

        logger.info(f"{'=' * 70}")
        logger.info(f"Pipeline complete: {pipeline_time:.1f}s total")
        logger.info(f"Output: {output_path}")
        logger.info(f"{'=' * 70}")

        # Save quality report
        qr_path = Path(self.config.log_dir) / f"{run_id}_quality.json"
        with open(qr_path, "w") as f:
            json.dump(quality_report.to_dict(), f, indent=2)

        return {
            "country": country,
            "year": year,
            "domain": self.config.domain,
            "output_path": output_path,
            "total_pages": ingestion_result.total_pages,
            "total_elements": len(analyses_dicts),
            "quality_composite": quality_report.overall_composite,
            "quality_pass_rate": quality_report.pass_rate,
            "remediation_attempts": remediation_count,
            "processing_time_seconds": pipeline_time,
            "config_hash": self.config.config_hash(),
            "analyses": analyses_dicts,
            "quality_report": quality_report.to_dict(),
        }

    def process_batch(
        self,
        documents: List[Dict[str, str]],
        lightweight_ingestion: bool = False,
    ) -> List[Dict[str, Any]]:
        """Process multiple documents sequentially.

        Args:
            documents: List of dicts with keys: pdf_path, country, year.
            lightweight_ingestion: Use lightweight ingestion mode.

        Returns:
            List of pipeline result dicts.
        """
        results = []
        total = len(documents)

        logger.info(f"Batch processing: {total} documents")

        for i, doc in enumerate(documents):
            logger.info(f"Document [{i + 1}/{total}]: {doc['country']}")
            try:
                result = self.process_document(
                    pdf_path=doc["pdf_path"],
                    country=doc["country"],
                    year=doc["year"],
                    lightweight_ingestion=lightweight_ingestion,
                )
                results.append(result)
            except Exception as e:
                logger.error(
                    f"Failed to process {doc['country']}: {e}",
                    exc_info=True,
                )
                results.append({
                    "country": doc["country"],
                    "year": doc["year"],
                    "error": str(e),
                })

        # Summary
        successful = sum(1 for r in results if "error" not in r)
        logger.info(
            f"Batch complete: {successful}/{total} successful"
        )
        return results
