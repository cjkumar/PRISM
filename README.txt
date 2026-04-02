PRISM - Policy Review and Intelligent Scoring Mechanism
========================================================

A multi-agent AI pipeline for structured analysis of national health policy
documents. PRISM processes PDF policy plans through four specialized stages
-- document ingestion, framework-based scoring, quality assurance, and
page-level evidence attribution -- producing structured, citation-backed
assessments against validated analytical frameworks.

Currently supports National Cancer Control Plans (76 sub-elements) and
Cardiovascular Disease Control Plans (69 sub-elements), with frameworks
validated by 67 and 42 international experts respectively.


PIPELINE ARCHITECTURE
---------------------

  PDF --> Agent 1 (Ingestion) --> Agent 2 (Analysis) --> Agent 3 (QA) --> Agent 4 (Page References)
                                                            ^    |
                                                            |    v
                                                        Remediation Loop

  Stage 1: Document Ingestion
    Model:   Qwen2.5-VL-72B-Instruct (vision-language)
    Process: PDF-to-image conversion, multi-stage image preprocessing
             (grayscale, Gaussian blur, Otsu thresholding, Hough Transform
             deskewing, morphological operations), layout recognition,
             text extraction with confidence scoring
    Output:  Page-indexed text with HTML layout, table/figure detection

  Stage 2: Policy Analysis
    Model:   Llama 4 Scout 70B with RAG (512-token chunks, top-5 retrieval)
    Process: Per-sub-element scoring against framework definitions and
             indicators, structured output enforced via Pydantic + Instructor
    Output:  Score (0-5), analytical response, scoring reasoning per element

  Stage 3: Quality Assurance
    Metrics: Readability (Flesch-Kincaid), semantic coherence (BERT),
             coverage completeness, schema compliance
    Process: Weighted composite scoring (0.25 each dimension), automatic
             remediation of failed categories (composite < 0.75, up to 3 retries)
    Output:  Quality report with per-element and aggregate scores

  Stage 4: Page Reference Extraction
    Methods: A) Numeric entity matching (percentages, years, currency)
             B) Sliding-window semantic similarity (all-MiniLM-L6-v2, 384-dim)
             C) Generative validation with majority voting (3 samples)
    Confidence: High (A+B+C agree), Medium (two-method), Low (single method)
    Output:  Page citations per response and per scoring reasoning


DIRECTORY STRUCTURE
-------------------

  PRISM/
  |-- agents/
  |   |-- ingestion.py          Agent 1: Document processing
  |   |-- analysis.py           Agent 2: Policy scoring with RAG
  |   +-- quality.py            Agent 3: Multi-dimensional QA
  |-- frameworks/
  |   |-- loader.py             Framework loading and RAG knowledge base
  |   +-- definitions.py        NCCP (76 elements) and CVD (69 elements)
  |-- page_references/
  |   |-- extractor.py          Three-method triangulation orchestrator
  |   +-- methods.py            Numeric, semantic, and generative matchers
  |-- visualization/
  |   |-- export.py             JSON-to-CSV conversion and filtering
  |   +-- scores.py             Score aggregation and statistics
  |-- pipeline.py               DAG orchestrator
  |-- config.py                 Centralized configuration (5 config classes)
  |-- validation.py             Schema validation for analysis outputs
  |-- cli.py                    Command-line interface (5 commands)
  +-- requirements.txt          Dependencies


INSTALLATION
------------

  git clone https://github.com/cjkumar/PRISM.git
  cd PRISM
  pip install -r requirements.txt

  Optional (GPU required for full Agent 1):
    pip install transformers>=4.37.0 accelerate>=0.25.0
    pip install opencv-python>=4.8.0 pdf2image>=1.16.0

  Environment variables:
    PRISM_API_BASE    API endpoint (default: http://localhost:11434/v1)
    PRISM_API_KEY     API key (default: not-needed for local inference)
    PRISM_MODEL_NAME  Override model for all agents


USAGE
-----

  Single document analysis:
    python -m PRISM analyze \
        --pdf /path/to/plan.pdf \
        --country "Kenya" \
        --year 2023 \
        --domain cancer \
        --output-dir ./results

  Batch processing (auto-detects country/year from filenames):
    python -m PRISM batch \
        --input-dir /path/to/pdfs/ \
        --domain cancer \
        --output-dir ./results

  Validate analysis outputs:
    python -m PRISM validate \
        --folder ./results \
        --domain cancer

  Export to CSV:
    python -m PRISM export \
        --folder ./results \
        --output analysis.csv \
        --commonwealth

  Summary statistics:
    python -m PRISM summary \
        --folder ./results \
        --domain cancer

  Lightweight mode (PyMuPDF-only ingestion, no vision model):
    python -m PRISM analyze --pdf plan.pdf --lightweight


OUTPUT FORMAT
-------------

  JSON (per document):
    [
      {
        "category": "Health",
        "response": "Analytical assessment of the sub-element...",
        "response_page_citations": [1, 3, 5],
        "score": 4,
        "scoring_reasoning": "Justification citing textual evidence...",
        "scoring_reasoning_page_citations": [2, 4]
      },
      ...
    ]

  CSV (batch export):
    country, year, Sub-Element, response, response_page_citations,
    score, reasoning, scoring_reasoning_page_citations

  Quality Report:
    Overall composite score, pass rate, failed categories,
    per-element quality dimensions (readability, coherence,
    coverage, schema compliance)


MODELS
------

  Component              Model                      Parameters
  --------------------   -------------------------  ----------
  Document Ingestion     Qwen2.5-VL-72B-Instruct   72B
  Policy Analysis        Llama 4 Scout 70B          70B
  Embeddings/Similarity  all-MiniLM-L6-v2           22.7M
  Coherence Scoring      BERT-base-uncased          110M
  Readability            Built-in (Flesch-Kincaid)  N/A


FRAMEWORKS
----------

  National Cancer Control Plan (NCCP):
    12 sections, 76 sub-elements
    Based on Atun et al. (2008), validated with 67 experts
    Sections: Outcomes, Objectives, Outputs, Functions, Threats,
    Opportunities, Strategy, Governance, Financing, Resource Management,
    Health Services, Implementation

  Cardiovascular Disease Control Plan (CVD):
    11 sections, 69 sub-elements
    Adapted from NCCP framework, validated with 42 specialists
    across 28 countries


DEPENDENCIES
------------

  Core:
    PyMuPDF >= 1.23.0          PDF text and image extraction
    numpy >= 1.24.0            Array operations
    Pillow >= 10.0.0           Image processing
    sentence-transformers >= 2.2.0   Embeddings
    torch >= 2.0.0             PyTorch backend
    scikit-learn >= 1.3.0      K-Means clustering
    openai >= 1.0.0            OpenAI-compatible API client
    requests >= 2.31.0         HTTP fallback
    pandas >= 2.0.0            Data export

  Optional:
    transformers >= 4.37.0     Full vision-language model (GPU)
    accelerate >= 0.25.0       Model acceleration
    opencv-python >= 4.8.0     Advanced image preprocessing
    pdf2image >= 1.16.0        Alternative PDF rendering


CONFIGURATION
-------------

  All parameters are centralized in config.py with sensible defaults.
  Key settings:

    Domain:           "cancer" or "cvd"
    Agent 2 temp:     0.1 (low for scoring consistency)
    RAG chunk size:   512 tokens, 64-token overlap, top-5 retrieval
    QA threshold:     0.75 composite (triggers remediation below)
    QA weights:       0.25 readability, 0.25 coherence, 0.25 coverage,
                      0.25 schema compliance
    Max remediation:  3 attempts per failed category
    Checkpointing:    Enabled by default (per-element)
    Page references:  Enabled by default (three-method triangulation)

  Configs can be saved/loaded as JSON and include SHA256 hashing
  for reproducibility tracking.


LICENSE
-------

  See LICENSE file for details.
