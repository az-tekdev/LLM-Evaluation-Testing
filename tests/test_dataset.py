"""Unit tests for dataset module."""

import pytest
import tempfile
import json
from pathlib import Path

from src.dataset import EvaluationDataset, create_sample_dataset


def test_create_sample_dataset():
    """Test creating a sample dataset."""
    dataset = create_sample_dataset()
    assert len(dataset) > 0
    assert len(dataset.get_prompts()) > 0


def test_dataset_from_data():
    """Test creating dataset from data list."""
    data = [
        {"prompt": "Test 1", "reference": "Answer 1"},
        {"prompt": "Test 2", "reference": "Answer 2"}
    ]
    dataset = EvaluationDataset(data=data)
    assert len(dataset) == 2


def test_get_prompts():
    """Test extracting prompts."""
    data = [
        {"prompt": "Question 1", "reference": "Answer 1"},
        {"question": "Question 2", "answer": "Answer 2"}
    ]
    dataset = EvaluationDataset(data=data)
    prompts = dataset.get_prompts()
    assert len(prompts) == 2
    assert "Question 1" in prompts
    assert "Question 2" in prompts


def test_get_references():
    """Test extracting references."""
    data = [
        {"prompt": "Q1", "reference": "A1"},
        {"prompt": "Q2", "answer": "A2"}
    ]
    dataset = EvaluationDataset(data=data)
    references = dataset.get_references()
    assert len(references) == 2
    assert "A1" in references
    assert "A2" in references


def test_load_jsonl():
    """Test loading JSONL file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        f.write('{"prompt": "Test", "reference": "Answer"}\n')
        f.write('{"prompt": "Test2", "reference": "Answer2"}\n')
        temp_path = f.name
    
    try:
        dataset = EvaluationDataset(file_path=temp_path)
        assert len(dataset) == 2
    finally:
        Path(temp_path).unlink()


def test_load_json():
    """Test loading JSON file."""
    data = [
        {"prompt": "Test 1", "reference": "Answer 1"},
        {"prompt": "Test 2", "reference": "Answer 2"}
    ]
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(data, f)
        temp_path = f.name
    
    try:
        dataset = EvaluationDataset(file_path=temp_path)
        assert len(dataset) == 2
    finally:
        Path(temp_path).unlink()


def test_limit():
    """Test limiting dataset size."""
    data = [{"prompt": f"Q{i}", "reference": f"A{i}"} for i in range(10)]
    dataset = EvaluationDataset(data=data)
    limited = dataset.limit(5)
    assert len(limited) == 5
    assert len(dataset) == 10  # Original unchanged


def test_filter():
    """Test filtering dataset."""
    data = [
        {"prompt": "Q1", "reference": "A1", "category": "math"},
        {"prompt": "Q2", "reference": "A2", "category": "science"},
        {"prompt": "Q3", "reference": "A3", "category": "math"}
    ]
    dataset = EvaluationDataset(data=data)
    filtered = dataset.filter(lambda s: s.get("category") == "math")
    assert len(filtered) == 2
