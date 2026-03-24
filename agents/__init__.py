"""
PRISM Agents
============

Three specialized agents forming the core PRISM pipeline:
    - Agent 1 (DocumentIngestion): Visual document processing via Qwen2.5-VL-72B
    - Agent 2 (PolicyAnalysis): Policy scoring and reasoning via Llama 4 Scout 70B
    - Agent 3 (QualityAssurance): Multi-dimensional validation
"""

from PRISM.agents.ingestion import DocumentIngestionAgent
from PRISM.agents.analysis import PolicyAnalysisAgent
from PRISM.agents.quality import QualityAssuranceAgent
