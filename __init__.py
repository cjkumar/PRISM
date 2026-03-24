"""
PRISM - Policy Reasoning Integrated Sequential Model
=====================================================

Multi-Agent AI System for Systematic Analysis of National Disease Control Plans.

Agents:
    - Agent 1 (DocumentIngestion): Qwen2.5-VL-72B based OCR and visual processing
    - Agent 2 (PolicyAnalysis): Llama 4 Scout 70B based policy scoring with RAG
    - Agent 3 (QualityAssurance): NLP-based quality validation

Modules:
    - page_references: Three-method triangulation for source attribution
    - pipeline: DAG orchestrator connecting all agents
    - frameworks: Domain-specific analytical framework definitions
    - visualization: Data export and visualization utilities

Health Systems Innovation Lab
Department of Global Health and Population
Harvard T.H. Chan School of Public Health
"""

__version__ = "1.0.0"
__author__ = "Health Systems Innovation Lab, Harvard T.H. Chan School of Public Health"

from PRISM.pipeline import PRISMPipeline
from PRISM.config import PRISMConfig
