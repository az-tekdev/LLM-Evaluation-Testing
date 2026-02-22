"""Unit tests for evaluator module."""

import pytest
from unittest.mock import Mock, patch

from src.dataset import EvaluationDataset
from src.evaluator import Evaluator


@pytest.fixture
def mock_model():
    """Create a mock LLM model."""
    model = Mock()
    model.generate_batch.return_value = ["Answer 1", "Answer 2", "Answer 3"]
    model.generate_multiple_samples.return_value = ["Answer", "Answer", "Answer"]
    model.generate_with_perturbation.return_value = ["Answer", "Answer", "Answer"]
    return model


@pytest.fixture
def sample_dataset():
    """Create a sample dataset."""
    data = [
        {"prompt": "Question 1?", "reference": "Answer 1"},
        {"prompt": "Question 2?", "reference": "Answer 2"},
        {"prompt": "Question 3?", "reference": "Answer 3"}
    ]
    return EvaluationDataset(data=data)


def test_evaluator_initialization(mock_model):
    """Test evaluator initialization."""
    from src.metrics import MetricCalculator
    
    evaluator = Evaluator(model=mock_model, metrics=MetricCalculator())
    assert evaluator.model == mock_model


def test_evaluate_basic(mock_model, sample_dataset):
    """Test basic evaluation."""
    from src.metrics import MetricCalculator
    
    evaluator = Evaluator(model=mock_model, metrics=MetricCalculator())
    results = evaluator.evaluate(sample_dataset)
    
    assert "summary" in results
    assert "samples" in results
    assert len(results["samples"]) == 3
    assert "accuracy" in results["summary"]


def test_evaluate_with_consistency(mock_model, sample_dataset):
    """Test evaluation with consistency checking."""
    from src.metrics import MetricCalculator
    
    evaluator = Evaluator(model=mock_model, metrics=MetricCalculator())
    results = evaluator.evaluate(sample_dataset, enable_consistency_check=True)
    
    assert "summary" in results
    assert any("consistency_score" in s for s in results["samples"])


def test_evaluate_with_perturbation(mock_model, sample_dataset):
    """Test perturbation-based evaluation."""
    from src.metrics import MetricCalculator
    
    evaluator = Evaluator(model=mock_model, metrics=MetricCalculator())
    results = evaluator.evaluate_with_perturbation(sample_dataset)
    
    assert "summary" in results
    assert "perturbation_consistency" in results["summary"]
    assert all("perturbation_consistency" in s for s in results["samples"])
