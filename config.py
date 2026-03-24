"""
PRISM Configuration
===================

Centralized configuration for the PRISM pipeline, including model parameters,
file paths, quality thresholds, and framework settings.
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger("prism.config")

# Base project path
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Agent1Config:
    """Configuration for Document Ingestion Agent (Qwen2.5-VL-72B)."""
    model_name: str = "Qwen/Qwen2.5-VL-72B-Instruct"
    max_pages_per_batch: int = 5
    image_resolution: int = 1024
    gaussian_kernel: tuple = (3, 3)
    median_filter_window: int = 5
    dilation_kernel: tuple = (2, 2)
    erosion_kernel: tuple = (2, 2)
    deskew_enabled: bool = True
    confidence_threshold: float = 0.85
    output_format: str = "html"  # "html" or "plain"


@dataclass
class Agent2Config:
    """Configuration for Policy Analysis Agent (Llama 4 Scout 70B)."""
    model_name: str = "meta-llama/Llama-4-Scout-70B"
    temperature: float = 0.1
    max_tokens: int = 4096
    top_p: float = 0.95
    rag_chunk_size: int = 512
    rag_chunk_overlap: int = 64
    rag_top_k: int = 5
    max_retries: int = 3
    retry_delay: float = 2.0


@dataclass
class Agent3Config:
    """Configuration for Quality Assurance Agent."""
    composite_threshold: float = 0.75
    readability_weight: float = 0.25
    coherence_weight: float = 0.25
    coverage_weight: float = 0.25
    schema_weight: float = 0.25
    max_remediation_attempts: int = 3
    bert_model: str = "bert-base-uncased"


@dataclass
class PageRefConfig:
    """Configuration for Page Reference Extraction."""
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384
    window_size: int = 50
    window_stride: int = 25
    batch_size: int = 32
    generative_temperature_low: float = 0.8
    generative_temperature_high: float = 1.0
    generative_top_p: float = 0.95
    generative_n_samples: int = 3
    generative_majority_threshold: int = 2
    prefilter_percentile: float = 0.25
    year_range: tuple = (1990, 2099)


@dataclass
class PRISMConfig:
    """Master configuration for the entire PRISM pipeline."""

    # Domain: "cancer" or "cvd"
    domain: str = "cancer"

    # Agent configs
    agent1: Agent1Config = field(default_factory=Agent1Config)
    agent2: Agent2Config = field(default_factory=Agent2Config)
    agent3: Agent3Config = field(default_factory=Agent3Config)
    page_ref: PageRefConfig = field(default_factory=PageRefConfig)

    # Paths
    input_dir: str = ""
    output_dir: str = ""
    framework_path: str = ""
    checkpoint_dir: str = ""
    log_dir: str = ""

    # Pipeline settings
    enable_page_references: bool = True
    enable_checkpointing: bool = True
    log_level: str = "INFO"

    def __post_init__(self):
        if not self.input_dir:
            self.input_dir = str(PROJECT_ROOT / "Data")
        if not self.output_dir:
            if self.domain == "cancer":
                self.output_dir = str(PROJECT_ROOT / "NCCP_Analyses")
            else:
                self.output_dir = str(PROJECT_ROOT / "CVD_Analyses")
        if not self.framework_path:
            if self.domain == "cancer":
                self.framework_path = str(
                    PROJECT_ROOT / "NCCP_Frameworks:Mapping" / "NCCPFramework_Aug9.txt"
                )
            else:
                self.framework_path = str(
                    PROJECT_ROOT / "CVD_Frameworks:Mapping" / "CVDFramework_Aug9.txt"
                )
        if not self.checkpoint_dir:
            self.checkpoint_dir = str(
                Path(self.output_dir) / ".checkpoints"
            )
        if not self.log_dir:
            self.log_dir = str(Path(self.output_dir) / ".logs")

    def config_hash(self) -> str:
        """Generate a cryptographic hash of the configuration for versioning."""
        config_json = json.dumps(asdict(self), sort_keys=True, default=str)
        return hashlib.sha256(config_json.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str):
        """Save configuration to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        logger.info(f"Configuration saved to {path} (hash: {self.config_hash()})")

    @classmethod
    def load(cls, path: str) -> "PRISMConfig":
        """Load configuration from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        agent1 = Agent1Config(**data.pop("agent1", {}))
        agent2 = Agent2Config(**data.pop("agent2", {}))
        agent3 = Agent3Config(**data.pop("agent3", {}))
        page_ref = PageRefConfig(**data.pop("page_ref", {}))
        config = cls(agent1=agent1, agent2=agent2, agent3=agent3,
                     page_ref=page_ref, **data)
        logger.info(f"Configuration loaded from {path} (hash: {config.config_hash()})")
        return config

    @classmethod
    def for_cancer(cls, **kwargs) -> "PRISMConfig":
        """Create config preset for cancer control plan analysis."""
        return cls(domain="cancer", **kwargs)

    @classmethod
    def for_cvd(cls, **kwargs) -> "PRISMConfig":
        """Create config preset for cardiovascular disease plan analysis."""
        return cls(domain="cvd", **kwargs)
