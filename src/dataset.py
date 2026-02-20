"""Dataset loading and preprocessing for LLM evaluation."""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator
import pandas as pd

logger = logging.getLogger(__name__)


class EvaluationDataset:
    """Handles loading and preprocessing of evaluation datasets."""
    
    def __init__(self, file_path: Optional[str] = None, data: Optional[List[Dict]] = None):
        """Initialize dataset.
        
        Args:
            file_path: Path to dataset file (JSONL, JSON, or CSV)
            data: Optional list of dictionaries with data
        """
        self.file_path = file_path
        self.data = data or []
        self._load_data()
    
    def _load_data(self) -> None:
        """Load data from file if file_path is provided."""
        if not self.file_path:
            return
        
        path = Path(self.file_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.file_path}")
        
        logger.info(f"Loading dataset from {self.file_path}")
        
        if path.suffix == ".jsonl":
            self._load_jsonl(path)
        elif path.suffix == ".json":
            self._load_json(path)
        elif path.suffix == ".csv":
            self._load_csv(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        
        logger.info(f"Loaded {len(self.data)} samples")
    
    def _load_jsonl(self, path: Path) -> None:
        """Load JSONL file."""
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))
    
    def _load_json(self, path: Path) -> None:
        """Load JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            content = json.load(f)
            if isinstance(content, list):
                self.data = content
            else:
                self.data = [content]
    
    def _load_csv(self, path: Path) -> None:
        """Load CSV file."""
        df = pd.read_csv(path)
        self.data = df.to_dict('records')
    
    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get sample by index."""
        return self.data[idx]
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over samples."""
        return iter(self.data)
    
    def get_prompts(self) -> List[str]:
        """Extract prompts from dataset.
        
        Returns:
            List of prompt strings
        """
        prompts = []
        for sample in self.data:
            # Try common prompt field names
            prompt = (
                sample.get("prompt") or
                sample.get("input") or
                sample.get("question") or
                sample.get("instruction") or
                str(sample.get("text", ""))
            )
            if prompt:
                prompts.append(str(prompt))
            else:
                logger.warning(f"Could not extract prompt from sample: {sample}")
                prompts.append("")
        return prompts
    
    def get_references(self) -> List[str]:
        """Extract reference answers from dataset.
        
        Returns:
            List of reference answer strings
        """
        references = []
        for sample in self.data:
            # Try common reference field names
            reference = (
                sample.get("reference") or
                sample.get("output") or
                sample.get("answer") or
                sample.get("expected") or
                sample.get("target") or
                ""
            )
            references.append(str(reference) if reference else "")
        return references
    
    def get_metadata(self) -> List[Dict[str, Any]]:
        """Extract metadata from dataset.
        
        Returns:
            List of metadata dictionaries
        """
        metadata = []
        for sample in self.data:
            # Extract all fields except prompt/reference fields
            meta = {k: v for k, v in sample.items() 
                   if k not in ["prompt", "input", "question", "instruction", "text",
                               "reference", "output", "answer", "expected", "target"]}
            metadata.append(meta)
        return metadata
    
    def limit(self, max_samples: int) -> 'EvaluationDataset':
        """Limit dataset to max_samples.
        
        Args:
            max_samples: Maximum number of samples
            
        Returns:
            New dataset instance with limited samples
        """
        limited_data = self.data[:max_samples]
        return EvaluationDataset(data=limited_data)
    
    def filter(self, condition: callable) -> 'EvaluationDataset':
        """Filter dataset based on condition.
        
        Args:
            condition: Function that takes a sample dict and returns bool
            
        Returns:
            New filtered dataset instance
        """
        filtered_data = [s for s in self.data if condition(s)]
        return EvaluationDataset(data=filtered_data)


def load_dataset_from_yaml(yaml_path: str) -> EvaluationDataset:
    """Load evaluation suite from YAML file.
    
    Args:
        yaml_path: Path to YAML file
        
    Returns:
        EvaluationDataset instance
    """
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required for YAML support. Install with: pip install pyyaml")
    
    with open(yaml_path, 'r', encoding='utf-8') as f:
        content = yaml.safe_load(f)
    
    # Convert YAML structure to dataset format
    data = []
    if isinstance(content, dict) and "tests" in content:
        for test in content["tests"]:
            data.append({
                "prompt": test.get("prompt", ""),
                "reference": test.get("reference", ""),
                "category": test.get("category", "general"),
                "metadata": test.get("metadata", {})
            })
    elif isinstance(content, list):
        data = content
    
    return EvaluationDataset(data=data)


def create_sample_dataset() -> EvaluationDataset:
    """Create a sample dataset for testing.
    
    Returns:
        EvaluationDataset with sample data
    """
    sample_data = [
        {
            "prompt": "What is the capital of France?",
            "reference": "Paris",
            "category": "factual"
        },
        {
            "prompt": "Calculate 15 * 23",
            "reference": "345",
            "category": "math"
        },
        {
            "prompt": "Who wrote Romeo and Juliet?",
            "reference": "William Shakespeare",
            "category": "factual"
        }
    ]
    return EvaluationDataset(data=sample_data)
