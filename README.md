# PRISM

**P**olicy **R**easoning **I**ntegrated **S**equential **M**odel — a multi-agent LLM system
that scores national disease control plans against structured, expert-validated policy
frameworks, and cites the pages of the source document backing every score.

Health Systems Innovation Lab, Department of Global Health and Population,
Harvard T.H. Chan School of Public Health.

---

## The problem

Assessing a national cancer control plan against a policy framework is expert work measured in
hours. In our own timing study, human reviewers averaged **~11.2 minutes per rubric
sub-element** — roughly **710 expert-hours** to assess 50 national plans across 76
sub-elements each. That cost is why cross-country policy comparison tends to happen once, on a
small sample, and then go stale.

PRISM automates the assessment while keeping it auditable: every score carries a written
justification and page-level citations back into the source PDF, so a human can check the
machine's reasoning against the document rather than taking it on faith.

---

## Architecture

Three agents arranged as a four-stage DAG with a quality-gated feedback loop:

```
PDF ──► Agent 1 ──► Agent 2 ──────► Agent 3 ──► Page Refs ──► JSON + CSV
      (Ingestion)  (Analysis)    (Quality QA)  (Attribution)
                        ▲              │
                        └─ remediation ┘
                        (failed sub-elements only, max 3 passes)
```

### Agent 1 — Document Ingestion (`agents/ingestion.py`)

Two interchangeable paths:

- **Vision-language path** — renders pages to images and runs Qwen2.5-VL-72B-Instruct for OCR
  and layout-aware extraction, with an OpenCV preprocessing chain (grayscale, Gaussian blur,
  median filter, Otsu threshold, Hough-transform deskew via Canny + HoughLines, dilation/erosion). Needs a GPU
  and the optional `transformers`/`accelerate`/`opencv-python` extras.
- **Text-layer path** (`process_document_lightweight`) — PyMuPDF text extraction. No GPU, orders
  of magnitude faster, and the practical default for PDFs that carry a usable text layer.

Both emit the same `IngestionResult`: page-indexed text plus per-page metadata (word count,
table/figure detection, confidence).

### Agent 2 — Policy Analysis (`agents/analysis.py`)

Scores each framework sub-element independently against the rubric.

- **RAG over the framework.** `RAGKnowledgeBase` embeds framework definitions, scoring criteria
  and indicators with `all-MiniLM-L6-v2`, then retrieves top-k by cosine similarity to build
  per-sub-element prompt context. Falls back to token-overlap keyword retrieval if
  sentence-transformers is unavailable.
- **Schema-enforced generation.** Output is constrained to a Pydantic model
  (`SubElementResponse`) through `instructor` in JSON mode: bounded integer score, minimum-length
  response and reasoning, plus custom `field_validator`s that reject the template-placeholder
  text these models echo back under load. Validation failures trigger automatic retries.
- **Defence in depth.** If `instructor` fails entirely, a fallback path issues a raw
  OpenAI-compatible call, strips markdown fences, re-validates against the same Pydantic model,
  and retries with exponential backoff before degrading to a scored-zero record.
- **Checkpointing.** Each completed sub-element is written to disk immediately, so a crash
  partway through a document does not replay the LLM calls already paid for.

### Agent 3 — Quality Assurance (`agents/quality.py`)

Scores each generated analysis on four dimensions, equally weighted into a composite:

| Dimension | Method |
|---|---|
| Readability | Flesch-Kincaid grade level (target band 12–16) |
| Coherence | Mean cosine similarity between consecutive sentence embeddings |
| Coverage | Response/reasoning length minimums plus framework indicator term coverage |
| Schema compliance | Field presence, type and range validation |

Sub-elements below the composite threshold (default 0.75) are returned to Agent 2 for
re-analysis, up to three passes.

> **Scope note:** these dimensions measure the *form* of the output — is it readable, internally
> consistent, complete, well-shaped. They do **not** verify factual accuracy against the source
> document. Factual checking is what the page-attribution layer and human review are for.

### Page Reference Attribution (`page_references/`)

The part of the system that makes outputs auditable. Given a generated passage, which pages of
the source document actually support it?

- **Method A — Numeric entity matching** (`NumericEntityMatcher`). Extracts years, percentages,
  currency amounts and counts from the generated text and locates pages containing the same
  entities. High precision, low recall: exact figures are strong evidence of provenance.
- **Method B — Sliding-window semantic similarity** (`SemanticSimilarityMatcher`). Embeds
  overlapping token windows of each page and the generated text, then sets the relevance
  threshold *adaptively per document* by clustering page-level max similarities with K-Means and
  picking the elbow — the midpoint between the two highest cluster centres. A fixed global
  threshold fails badly here, because similarity distributions differ sharply between a 40-page
  plan and a 500-page one.
- **Method C — Generative validation** (`GenerativeMatcher`). Prefilters candidate pages, then
  asks the LLM to confirm support at multiple temperatures and takes a majority vote.
  **Implemented but not wired into the default pipeline** — see [Limitations](#limitations).

Results combine by set intersection into graded confidence: `A∩B∩C` → high, any pairwise
intersection → medium, Method B alone → low. Low-confidence citations are the ones worth
flagging for human review.

---

## Repository layout

```
PRISM/
├── pipeline.py              # 4-stage orchestrator + remediation loop + batch driver
├── config.py                # Frozen dataclass config tree; SHA-256 config_hash for provenance
├── cli.py                   # analyze | batch | validate | export | summary
├── validation.py            # Schema/completeness auditing of output JSON
├── agents/
│   ├── ingestion.py         # Agent 1: Qwen2.5-VL path + PyMuPDF path, OpenCV preprocessing
│   ├── analysis.py          # Agent 2: RAG + Pydantic/instructor structured output
│   └── quality.py           # Agent 3: readability, coherence, coverage, schema
├── frameworks/
│   ├── definitions.py       # Section/sub-element structure, max scores, normalisation
│   └── loader.py            # Loads framework JSON, builds prompt + RAG context
├── page_references/
│   ├── methods.py           # Methods A, B, C
│   └── extractor.py         # Set-intersection attribution, confidence grading
└── visualization/
    ├── export.py            # JSON → CSV (global + Commonwealth subsets)
    └── scores.py            # Section/overall aggregation, cross-country comparison
```

~4,250 lines of Python across 17 modules.

---

## Installation

```bash
git clone https://github.com/cjkumar/PRISM.git
cd PRISM
pip install -r requirements.txt
```

For the Qwen2.5-VL ingestion path (requires a GPU):

```bash
pip install transformers accelerate opencv-python pdf2image
```

### Two setup requirements that are easy to miss

**1. Run from the parent directory.** The package uses absolute `PRISM.*` imports, so the
*parent* of this directory must be on `sys.path`:

```bash
cd /path/to/parent-of-PRISM
python -m PRISM.cli analyze --help     # works
```

Running `python -m PRISM.cli` from inside the repo raises `ModuleNotFoundError: No module
named 'PRISM'`.

**2. Supply a framework file.** The scoring frameworks are **not bundled** in this repository.
`PRISMConfig` expects JSON (in a `.txt` file) at
`<parent>/NCCP_Frameworks:Mapping/NCCPFramework_Aug9.txt` for cancer, or the CVD equivalent —
a list of objects with `category`, `definition`, `scoring_definitions` and `indicators`.
Override with `--config` or by setting `framework_path` directly.

### Inference endpoint

Agent 2 talks to any OpenAI-compatible endpoint (vLLM, Ollama, TGI, or a hosted API):

```bash
export PRISM_API_BASE=http://localhost:8000/v1      # default
export PRISM_API_KEY=not-needed                     # default
export PRISM_MODEL_NAME=meta-llama/Llama-4-Scout-70B
```

---

## Usage

```bash
# Single document, text-layer ingestion
python -m PRISM.cli analyze \
    --pdf plan.pdf --country "Australia" --year 2023 --lightweight

# CVD domain
python -m PRISM.cli analyze \
    --pdf plan.pdf --country "Brazil" --year 2021 --domain cvd

# Every PDF in a folder
python -m PRISM.cli batch --input-dir ./pdfs --domain cancer

# Audit output completeness against the framework
python -m PRISM.cli validate --folder ./NCCP_Analyses

# Export to CSV, and summary statistics
python -m PRISM.cli export --folder ./NCCP_Analyses --output ./nccp_data
python -m PRISM.cli summary --folder ./NCCP_Analyses --domain cancer
```

As a library:

```python
from PRISM import PRISMPipeline, PRISMConfig

pipeline = PRISMPipeline(PRISMConfig.for_cancer())
result = pipeline.process_document(
    pdf_path="plan.pdf", country="Australia", year="2023",
    lightweight_ingestion=True,
)
print(result["quality_composite"], result["output_path"])
```

---

## Output

One JSON array per document, one object per framework sub-element:

```json
[
  {
    "category": "Cancer Surveillance Systems",
    "response": "The plan establishes a population-based cancer registry with ...",
    "response_page_citations": [23, 24, 31],
    "score": 4,
    "scoring_reasoning": "Scored 4 because the plan specifies registry coverage and ...",
    "scoring_reasoning_page_citations": [24, 31]
  }
]
```

`visualization/export.py` flattens these to CSV for analysis.

---

## Frameworks

| Domain | Sections | Sub-elements |
|---|---|---|
| Cancer (NCCP) | 12 | 76 |
| Cardiovascular disease (CVD) | 11 | 69 |

Both frameworks follow a health-systems structure. Cancer sections, with sub-element counts:
Outcomes (4), Objectives (4), Outputs (4), Functions (3), Threats (8), Opportunities (8),
Strategy (6), Governance and Organisation (6), Financing (11), Resource Management (11),
Health Services (7), Implementation (4).

Maximum scores vary per sub-element; `frameworks/definitions.py:normalize_score` rescales to a
common 0–5 axis for cross-element comparison.

---

## Reliability and reproducibility

The design assumes every component can fail, and that any published number must be
regenerable.

- **Graceful degradation behind every ML dependency.** No sentence-transformers → keyword
  retrieval. No scikit-learn → median similarity threshold. No transformers/GPU → PyMuPDF text
  extraction. No `instructor` success → raw call with manual parse and backoff. The pipeline
  degrades in quality rather than failing closed.
- **Provenance.** `PRISMConfig.config_hash()` is a SHA-256 over the entire config tree; the
  resolved config and the quality report are written to `.logs/` per run, keyed by run id.
- **Checkpointing.** Per-sub-element JSON checkpoints allow resumption mid-document.
- **Schema auditing.** `validation.py` independently re-checks every output file for missing,
  duplicated or malformed sub-elements — a second gate after Agent 3.

### Validation against expert review

The two domains have been validated **separately, against different reviewer panels, and on
different pipeline variants**. The results differ substantially, and the two should not be
quoted interchangeably.

| | Cancer (NCCP) | Cardiovascular disease (CVD) |
|---|---|---|
| System scored | Full multi-agent pipeline | **Single-pass variant**, not the multi-agent pipeline |
| Plans | 6 | 6 |
| Sub-elements | 76 | 69 |
| Expert reviewers | 2 | 3 |
| Paired comparisons | 456 | 403 |
| Exact agreement | 35.1% | 43.7% |
| Within one point | 70.8% | 68.0% |
| Mean abs. difference | 1.04 | 1.22 |
| Correlation | Pearson r ≈ 0.41 (p<0.0001) | Spearman ρ = 0.061 (p=0.22, n.s.) |
| Quadratic-weighted κ | 0.397 | 0.045 |
| Mean bias (PRISM − human) | +0.30 | +0.28 |

**Cancer.** Six plans, each rated by one of two experts across all 76 sub-elements. Agreement
is moderate and statistically significant, though it varies widely by plan (per-plan r from
0.15 to 0.80).

**CVD.** Six plans, each rated by one of three experts. Agreement here is **not
distinguishable from chance**: quadratic-weighted κ = 0.045, Spearman ρ = 0.061 (p = 0.22),
ICC(3,1) = 0.046. Per-reviewer correlation ranged from +0.22 to −0.09. Two properties of the
data explain why the exact-agreement figure (43.7%) looks reasonable while the agreement
statistics do not:

- **A shared floor.** 61.0% of model scores and 62.3% of human scores are 0. Most exact
  agreement is two raters independently assigning zero, not shared discrimination among
  non-zero scores.
- **Compression toward the middle.** Bias is strongly conditional on the human score:
  +0.96 where the human scored 0, but −0.93 at 2, −1.66 at 3, and −3.00 at 4. The model
  neither reproduces the experts' low scores nor their high ones.

Agreement also varied sharply by framework element — 72.9% exact on Health System Threats and
Opportunities, but 16.7% on CVD Strategy and 19.0% on Governance and Organization, the
elements carrying the largest positive bias.

**A limitation common to both studies:** no sub-element was rated by more than one expert
(verified — zero overlapping items in either panel), so neither study estimates
**human–human** agreement. Without that ceiling, the absolute agreement figures above cannot
be judged against what two experts achieve with each other on the same rubric, which on
ordinal policy instruments is itself often moderate.

The CVD analysis reports the full ordinal-statistics suite (Spearman, Kendall τ-b, linear and
quadratic weighted κ, ICC) and explicitly flags Pearson r as inappropriate for ordinal data;
the cancer analysis reports Pearson r. The weighted-κ figures in the table above are computed
on the same scale for both so they can be compared directly.

Reviewer data, per-country results and the analysis scripts live with the respective
manuscripts rather than in this repository.


---

## Limitations

Known and worth stating plainly.

1. **Prompt truncation.** `Agent2._build_analysis_prompt` caps document text at 80,000
   characters. On a 50-plan corpus, 72% of plans exceeded that cap, and the median truncated
   plan was scored on ~43% of its extracted text. Long plans are therefore assessed on a
   prefix, not the whole document. Per-sub-element retrieval over document chunks — rather than
   a single truncated prompt — is the right fix.
2. **Method C is not wired in.** Neither pipeline passes a `generate_fn` to
   `PageReferenceExtractor`, so `set_c` is always empty. In practice attribution runs on
   Methods A and B only, and the `high` confidence tier is unreachable — shipped citations are
   `medium` (A∩B) or `low` (B alone).
3. **No automated test suite and no CI.** Correctness has been checked by manual review and by
   `validation.py` schema auditing, not by unit or integration tests.
4. **Agent 3 measures form, not factual accuracy.** A high composite score does not mean the
   analysis is factually correct about the document. The remediation loop optimises these
   form metrics, so it may improve polish without improving accuracy — that relationship has
   not been measured.
5. **`Agent3Config.bert_model` is dead configuration.** It defaults to `bert-base-uncased`, but
   `SemanticCoherenceAnalyzer` hardcodes `all-MiniLM-L6-v2` and ignores the field.
6. **Benchmark figures in module docstrings are upstream, not ours.** The DocVQA/TextVQA/ChartQA
   numbers in `agents/ingestion.py` and the LegalBench/PolicyQA numbers in `agents/analysis.py`
   are published characteristics of the underlying Qwen and Llama models on generic tasks. They
   are **not** PRISM evaluation results. The only end-to-end evaluation of PRISM is the expert
   agreement study above.
7. **Non-deterministic by default.** Agent 2 runs at temperature 0.1, not 0, and PRISM's own
   run-to-run variance has not been quantified.
8. **The CVD expert comparison did not score this pipeline.** It evaluated a single-pass
   variant, so it does not measure the contribution of the multi-agent architecture, the
   remediation loop, or RAG in the CVD domain. Whether the full pipeline performs better,
   worse, or the same on CVD plans is untested.

---

## Citation

Dataset: [https://doi.org/10.7910/DVN/ETVLMD](https://doi.org/10.7910/DVN/ETVLMD)
