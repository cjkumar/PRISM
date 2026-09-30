"""
PRISM Pipeline Orchestrator (LangGraph)
========================================

A LangGraph implementation of the PRISM DAG, provided alongside the original
:mod:`PRISM.pipeline` so the two can be compared on identical inputs.

Graph shape:

    START → ingest ─┬─(Send per category)→ analyze_one ─┐
                    │                                    ↓
                    │                                 quality
                    │                                    │
                    └──────── remediate (failed only) ←──┤
                                                         ↓
                                              page_refs → write_output → END

Differences from :class:`PRISM.pipeline.PRISMPipeline`:

* Sub-element analysis fans out in parallel via LangGraph's ``Send`` API
  (bounded by ``max_concurrency``) instead of running sequentially.
* The remediation loop is a conditional edge rather than a ``while`` block,
  and re-dispatches only the categories that failed quality validation.
* Checkpointing covers *every* stage via a SQLite checkpointer keyed on
  ``run_id``, not just Agent 2's per-category JSON files. Re-invoking with the
  same ``run_id`` resumes from the last completed superstep.
* Ingested document text lives in a disk-backed side store keyed by
  ``run_id`` rather than in graph state, so checkpoint writes stay small.

The agents themselves are untouched: nodes are thin wrappers over
``DocumentIngestionAgent``, ``PolicyAnalysisAgent._analyze_sub_element`` and
``QualityAssuranceAgent``. No LangChain model wrappers are involved; Agent 2
still uses its own instructor + OpenAI client.
"""

import json
import logging
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from PRISM.config import PRISMConfig
from PRISM.agents.ingestion import DocumentIngestionAgent
from PRISM.agents.analysis import PolicyAnalysisAgent
from PRISM.agents.quality import QualityAssuranceAgent
from PRISM.frameworks.loader import FrameworkLoader
from PRISM.page_references.extractor import PageReferenceExtractor

logger = logging.getLogger("prism.pipeline_graph")

DEFAULT_MAX_CONCURRENCY = 4
RECURSION_LIMIT = 50


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

def merge_analyses(
    left: List[Dict[str, Any]], right: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Reducer for the ``analyses`` channel: replace by category, not append.

    Parallel ``analyze_one`` tasks each contribute one analysis dict. A plain
    additive reducer would append duplicates every time a category is
    re-analyzed during remediation, so entries are keyed by category and
    later writes win.
    """
    by_category: Dict[str, Dict[str, Any]] = {
        a["category"]: a for a in left
    }
    for a in right:
        by_category[a["category"]] = a
    return list(by_category.values())


class PRISMGraphState(TypedDict, total=False):
    """Graph state. Every field must be JSON-serializable for checkpointing."""

    run_id: str
    pdf_path: str
    country: str
    year: str
    domain: str
    lightweight: bool
    output_path: str

    # Key into the document side store, not the text itself.
    doc_key: str
    total_pages: int

    analyses: Annotated[List[Dict[str, Any]], merge_analyses]
    quality_report: Optional[Dict[str, Any]]

    # Categories awaiting analysis. Narrowed to the failures on remediation.
    pending_categories: List[str]
    # Number of completed quality passes; remediations = quality_runs - 1.
    quality_runs: int


class _AnalyzeTask(TypedDict):
    """Payload delivered to a single ``analyze_one`` task via ``Send``."""

    doc_key: str
    category: str
    domain: str


# ---------------------------------------------------------------------------
# Document side store
# ---------------------------------------------------------------------------

class DocumentStore:
    """Disk-backed store for ingested document text, keyed by run_id.

    Keeping the full document text out of graph state matters: LangGraph
    serializes state on every superstep, and a 200-page plan re-written a
    dozen times is real disk churn. Backing it with a file (rather than an
    in-process dict) also means a resumed run in a fresh process can still
    reach the text after the ``ingest`` node has been skipped.
    """

    def __init__(self, root: str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._memo: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def _path(self, key: str) -> Path:
        return self.root / f"{key}.json"

    def put(self, key: str, full_text: str, page_texts: Dict[int, str]) -> str:
        payload = {
            "full_text": full_text,
            "page_texts": {str(k): v for k, v in page_texts.items()},
        }
        with open(self._path(key), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        with self._lock:
            self._memo[key] = {
                "full_text": full_text,
                "page_texts": dict(page_texts),
            }
        logger.debug(f"Document stored: {self._path(key)}")
        return key

    def get(self, key: str) -> Dict[str, Any]:
        """Load a stored document, memoized so parallel tasks read disk once."""
        with self._lock:
            cached = self._memo.get(key)
        if cached is not None:
            return cached

        path = self._path(key)
        if not path.exists():
            raise FileNotFoundError(
                f"No stored document for run '{key}' at {path}. "
                f"The ingest stage must run before analysis; if you are "
                f"resuming, keep the checkpoint directory intact."
            )
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        doc = {
            "full_text": payload["full_text"],
            "page_texts": {int(k): v for k, v in payload["page_texts"].items()},
        }
        with self._lock:
            self._memo[key] = doc
        return doc


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class PRISMGraphPipeline:
    """LangGraph-backed equivalent of :class:`PRISM.pipeline.PRISMPipeline`.

    Exposes the same ``process_document`` / ``process_batch`` surface and
    returns the same result dict, so the two implementations can be run
    against each other on identical inputs.
    """

    def __init__(
        self,
        config: Optional[PRISMConfig] = None,
        max_concurrency: Optional[int] = None,
    ):
        self.config = config or PRISMConfig()
        self._setup_logging()

        self.max_concurrency = int(
            max_concurrency
            if max_concurrency is not None
            else os.environ.get("PRISM_MAX_CONCURRENCY", DEFAULT_MAX_CONCURRENCY)
        )

        self.framework = FrameworkLoader(self.config.framework_path)

        self.agent1 = DocumentIngestionAgent(self.config.agent1)
        self.agent2 = PolicyAnalysisAgent(self.config.agent2, self.framework)
        self.agent3 = QualityAssuranceAgent(self.config.agent3, self.framework)

        self.page_ref_extractor = None
        if self.config.enable_page_references:
            self.page_ref_extractor = PageReferenceExtractor(self.config.page_ref)

        self.doc_store = DocumentStore(
            str(Path(self.config.checkpoint_dir) / "documents")
        )

        # Agent 2 builds its RAG index and instructor client lazily, because
        # on a resumed run the ingest node never executes.
        self._agent2_ready = False
        self._agent2_lock = threading.Lock()

        self._category_order = {
            c: i for i, c in enumerate(self.framework.categories)
        }

        self.graph = self._build_graph()

        logger.info(
            f"PRISM LangGraph Pipeline initialized "
            f"(domain={self.config.domain}, "
            f"max_concurrency={self.max_concurrency}, "
            f"config_hash={self.config.config_hash()})"
        )

    # -- setup ------------------------------------------------------------

    def _setup_logging(self):
        """Configure pipeline logging (mirrors PRISMPipeline)."""
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

    def _ensure_agent2(self):
        """Initialize Agent 2 once, safely under parallel fan-out."""
        if self._agent2_ready:
            return
        with self._agent2_lock:
            if not self._agent2_ready:
                self.agent2.initialize()
                self._agent2_ready = True

    # -- nodes ------------------------------------------------------------

    def _node_ingest(self, state: PRISMGraphState) -> Dict[str, Any]:
        """Stage 1: Document Ingestion."""
        logger.info("STAGE 1: Document Ingestion")
        start = time.time()

        if state.get("lightweight"):
            result = self.agent1.process_document_lightweight(state["pdf_path"])
        else:
            result = self.agent1.process_document(state["pdf_path"])

        doc_key = self.doc_store.put(
            state["run_id"], result.full_text, result.page_texts
        )

        logger.info(
            f"Stage 1 complete: {result.total_pages} pages, "
            f"{time.time() - start:.1f}s"
        )

        return {
            "doc_key": doc_key,
            "total_pages": result.total_pages,
            "pending_categories": list(self.framework.categories),
            "quality_runs": 0,
        }

    def _fan_out(self, state: PRISMGraphState):
        """Dispatch one ``analyze_one`` task per pending category."""
        from langgraph.types import Send

        pending = state.get("pending_categories") or []
        if state.get("quality_runs", 0) == 0:
            logger.info(f"STAGE 2: Policy Analysis ({len(pending)} categories)")

        return [
            Send(
                "analyze_one",
                _AnalyzeTask(
                    doc_key=state["doc_key"],
                    category=category,
                    domain=state["domain"],
                ),
            )
            for category in pending
        ]

    def _node_analyze_one(self, task: _AnalyzeTask) -> Dict[str, Any]:
        """Analyze a single framework sub-element.

        Receives only the ``Send`` payload, not full graph state, so the
        document is loaded from the side store.
        """
        self._ensure_agent2()
        category = task["category"]
        doc = self.doc_store.get(task["doc_key"])

        try:
            analysis = self.agent2._analyze_sub_element(
                category,
                doc["full_text"],
                doc["page_texts"],
                task["domain"],
            )
            return {"analyses": [analysis.to_dict()]}
        except Exception as e:
            # Match the original pipeline: one bad category degrades to a
            # zero-score record rather than failing the whole document.
            logger.error(f"Error analyzing {category}: {e}")
            return {
                "analyses": [{
                    "category": category,
                    "response": f"Analysis failed: {e}",
                    "response_page_citations": [],
                    "score": 0,
                    "scoring_reasoning": f"Error during analysis: {e}",
                    "scoring_reasoning_page_citations": [],
                }]
            }

    def _node_quality(self, state: PRISMGraphState) -> Dict[str, Any]:
        """Stage 3: Quality Assurance."""
        logger.info("STAGE 3: Quality Assurance")
        start = time.time()

        analyses = self._ordered(state.get("analyses") or [])
        report = self.agent3.validate_analysis(analyses)
        quality_runs = state.get("quality_runs", 0) + 1

        logger.info(
            f"Stage 3 complete: composite={report.overall_composite:.3f}, "
            f"pass_rate={report.pass_rate:.1%}, "
            f"failed={len(report.failed_categories)}, "
            f"{time.time() - start:.1f}s"
        )

        return {
            "quality_report": report.to_dict(),
            "pending_categories": list(report.failed_categories),
            "quality_runs": quality_runs,
        }

    def _route_after_quality(self, state: PRISMGraphState):
        """Remediate failed categories, or move on to page references.

        Returns ``Send`` objects rather than a node name when remediating.
        A static ``{"remediate": "analyze_one"}`` mapping would hand
        ``analyze_one`` the whole graph state as a single task instead of
        fanning out one task per failed category.
        """
        failed = state.get("pending_categories") or []
        remediations_done = state.get("quality_runs", 1) - 1
        max_attempts = self.config.agent3.max_remediation_attempts

        if failed and remediations_done < max_attempts:
            logger.info(
                f"Remediation attempt {remediations_done + 1}: "
                f"re-analyzing {len(failed)} categories"
            )
            return self._fan_out(state)

        if failed:
            logger.warning(
                f"{len(failed)} categories still failing after "
                f"{max_attempts} remediation attempts"
            )
        return "page_refs"

    def _node_page_refs(self, state: PRISMGraphState) -> Dict[str, Any]:
        """Stage 4: Page Reference Extraction (optional)."""
        analyses = self._ordered(state.get("analyses") or [])

        if not (self.page_ref_extractor and self.config.enable_page_references):
            return {"analyses": analyses}

        logger.info("STAGE 4: Page Reference Extraction")
        start = time.time()

        doc = self.doc_store.get(state["doc_key"])
        ref_results = self.page_ref_extractor.extract_all_references(
            analyses, doc["page_texts"]
        )
        analyses = PageReferenceExtractor.update_analyses_with_references(
            analyses, ref_results
        )

        logger.info(f"Stage 4 complete: {time.time() - start:.1f}s")
        return {"analyses": analyses}

    def _node_write_output(self, state: PRISMGraphState) -> Dict[str, Any]:
        """Write the analysis JSON and the quality report."""
        analyses = self._ordered(state.get("analyses") or [])
        output_path = state["output_path"]

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(analyses, f, indent=2, ensure_ascii=False)

        qr_path = Path(self.config.log_dir) / f"{state['run_id']}_quality.json"
        with open(qr_path, "w") as f:
            json.dump(state.get("quality_report") or {}, f, indent=2)

        logger.info(f"Output: {output_path}")
        return {"analyses": analyses}

    # -- helpers ----------------------------------------------------------

    def _ordered(
        self, analyses: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Restore framework order.

        The reducer accumulates in task-completion order under parallel
        fan-out; the original pipeline emits categories in framework order,
        and output diffs are only meaningful if both agree.
        """
        return sorted(
            analyses,
            key=lambda a: self._category_order.get(
                a.get("category", ""), len(self._category_order)
            ),
        )

    def _build_graph(self):
        from langgraph.graph import StateGraph, START, END

        g = StateGraph(PRISMGraphState)

        g.add_node("ingest", self._node_ingest)
        # ``analyze_one`` receives a Send payload, not graph state, so it
        # needs its own input schema; otherwise LangGraph filters the
        # payload down to the state schema and drops ``category``.
        g.add_node(
            "analyze_one",
            self._node_analyze_one,
            input_schema=_AnalyzeTask,
        )
        g.add_node("quality", self._node_quality)
        g.add_node("page_refs", self._node_page_refs)
        g.add_node("write_output", self._node_write_output)

        g.add_edge(START, "ingest")
        g.add_conditional_edges("ingest", self._fan_out, ["analyze_one"])
        g.add_edge("analyze_one", "quality")
        g.add_conditional_edges(
            "quality",
            self._route_after_quality,
            ["analyze_one", "page_refs"],
        )
        g.add_edge("page_refs", "write_output")
        g.add_edge("write_output", END)

        return g

    def _checkpointer(self):
        """SQLite checkpointer, falling back to in-memory if unavailable."""
        from contextlib import nullcontext

        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
        except ImportError:
            from langgraph.checkpoint.memory import MemorySaver

            logger.warning(
                "langgraph-checkpoint-sqlite not installed; using an "
                "in-memory checkpointer. Runs will not be resumable."
            )
            return nullcontext(MemorySaver())

        db_dir = Path(self.config.checkpoint_dir)
        db_dir.mkdir(parents=True, exist_ok=True)
        return SqliteSaver.from_conn_string(str(db_dir / "prism_graph.db"))

    # -- public API -------------------------------------------------------

    def process_document(
        self,
        pdf_path: str,
        country: str,
        year: str,
        output_path: Optional[str] = None,
        lightweight_ingestion: bool = False,
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Process a single policy document through the graph.

        Args:
            pdf_path: Path to the PDF document.
            country: Country name.
            year: Plan publication year.
            output_path: Optional output JSON path. If None, auto-generated.
            lightweight_ingestion: Use PyMuPDF-only ingestion (no VL model).
            run_id: Reuse a prior run's id to resume from its checkpoint.

        Returns:
            Dict with analysis results, quality report, and metadata —
            the same shape returned by ``PRISMPipeline.process_document``.
        """
        pipeline_start = time.time()
        run_id = run_id or f"{country}_{year}_{datetime.now():%Y%m%d_%H%M%S}"

        if output_path is None:
            output_dir = Path(self.config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = str(output_dir / f"{country}_{year}.json")

        logger.info(f"{'=' * 70}")
        logger.info(f"PRISM LangGraph Pipeline: {country} ({year})")
        logger.info(f"Document: {pdf_path}")
        logger.info(f"Run ID: {run_id}")
        logger.info(f"{'=' * 70}")

        config_path = Path(self.config.log_dir) / f"{run_id}_config.json"
        self.config.save(str(config_path))

        initial: PRISMGraphState = {
            "run_id": run_id,
            "pdf_path": pdf_path,
            "country": country,
            "year": year,
            "domain": self.config.domain,
            "lightweight": lightweight_ingestion,
            "output_path": output_path,
            "analyses": [],
            "quality_report": None,
            "pending_categories": [],
            "quality_runs": 0,
        }

        with self._checkpointer() as checkpointer:
            app = self.graph.compile(checkpointer=checkpointer)
            graph_config = {
                "configurable": {"thread_id": run_id},
                "max_concurrency": self.max_concurrency,
                "recursion_limit": RECURSION_LIMIT,
            }

            # Resuming an interrupted run means invoking with ``None``.
            # Passing the input dict again would restart the thread from
            # START and re-run every completed LLM call.
            snapshot = app.get_state(graph_config)
            if snapshot.next:
                logger.info(
                    f"Resuming run {run_id} at: {', '.join(snapshot.next)}"
                )
                final = app.invoke(None, config=graph_config)
            elif snapshot.values.get("analyses"):
                logger.info(
                    f"Run {run_id} already complete; "
                    f"returning checkpointed result without re-running"
                )
                final = snapshot.values
            else:
                final = app.invoke(initial, config=graph_config)

        analyses = self._ordered(final.get("analyses") or [])
        quality_report = final.get("quality_report") or {}
        remediation_count = max(final.get("quality_runs", 1) - 1, 0)
        pipeline_time = time.time() - pipeline_start

        logger.info(f"{'=' * 70}")
        logger.info(f"Pipeline complete: {pipeline_time:.1f}s total")
        logger.info(f"Output: {output_path}")
        logger.info(f"{'=' * 70}")

        return {
            "country": country,
            "year": year,
            "domain": self.config.domain,
            "output_path": output_path,
            "total_pages": final.get("total_pages", 0),
            "total_elements": len(analyses),
            "quality_composite": quality_report.get("overall_composite", 0.0),
            "quality_pass_rate": quality_report.get("pass_rate", 0.0),
            "remediation_attempts": remediation_count,
            "processing_time_seconds": pipeline_time,
            "config_hash": self.config.config_hash(),
            "run_id": run_id,
            "analyses": analyses,
            "quality_report": quality_report,
        }

    def process_batch(
        self,
        documents: List[Dict[str, str]],
        lightweight_ingestion: bool = False,
    ) -> List[Dict[str, Any]]:
        """Process multiple documents, one graph invocation each.

        The batch stays a plain loop: each document is an independent thread
        of graph state, so modelling the batch itself as a graph would only
        obscure per-document resume.
        """
        results = []
        total = len(documents)

        logger.info(f"Batch processing: {total} documents")

        for i, doc in enumerate(documents):
            logger.info(f"Document [{i + 1}/{total}]: {doc['country']}")
            try:
                results.append(self.process_document(
                    pdf_path=doc["pdf_path"],
                    country=doc["country"],
                    year=doc["year"],
                    lightweight_ingestion=lightweight_ingestion,
                ))
            except Exception as e:
                logger.error(
                    f"Failed to process {doc['country']}: {e}", exc_info=True
                )
                results.append({
                    "country": doc["country"],
                    "year": doc["year"],
                    "error": str(e),
                })

        successful = sum(1 for r in results if "error" not in r)
        logger.info(f"Batch complete: {successful}/{total} successful")
        return results

    def draw_mermaid(self) -> str:
        """Return a Mermaid diagram of the compiled graph."""
        return self.graph.compile().get_graph().draw_mermaid()
