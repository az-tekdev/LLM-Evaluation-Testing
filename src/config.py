"""Configuration management for LLM Evaluation Framework."""

import os
from typing import Optional, List, Dict, Any
from pydantic_settings import BaseSettings


class EvaluationConfig(BaseSettings):
    """Configuration for LLM evaluation framework."""
    
    # OpenAI Configuration
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4"
    openai_temperature: float = 0.0
    openai_max_tokens: int = 1000
    openai_timeout: int = 60
    
    # Evaluation Settings
    batch_size: int = 10
    max_workers: int = 4
    retry_attempts: int = 3
    retry_delay: float = 1.0
    
    # Metrics Configuration
    enable_accuracy: bool = True
    enable_f1: bool = True
    enable_bleu: bool = True
    enable_rouge: bool = True
    enable_bertscore: bool = False  # Requires transformers
    enable_hallucination_detection: bool = True
    enable_consistency_check: bool = True
    
    # Hallucination Detection
    hallucination_samples: int = 3  # Number of samples for self-consistency
    hallucination_threshold: float = 0.7  # Consistency threshold
    perturbation_enabled: bool = True
    perturbation_count: int = 2
    
    # Output Configuration
    output_dir: str = "results"
    output_format: str = "json"  # json, csv, both
    generate_visualizations: bool = True
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None
    
    # Dataset Configuration
    dataset_cache_dir: str = ".cache/datasets"
    max_samples: Optional[int] = None  # Limit number of samples for testing
    
    class Config:
        """Pydantic config."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global config instance
config = EvaluationConfig()


def validate_config() -> None:
    """Validate configuration settings."""
    if not config.openai_api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable is required. "
            "Please set it in your .env file or environment."
        )
