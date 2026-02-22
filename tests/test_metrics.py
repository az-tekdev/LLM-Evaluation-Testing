"""Unit tests for metrics module."""

import pytest
from src.metrics import MetricCalculator


def test_accuracy():
    """Test accuracy calculation."""
    calculator = MetricCalculator()
    
    predictions = ["Paris", "Shakespeare", "345"]
    references = ["Paris", "Shakespeare", "345"]
    accuracy = calculator.accuracy(predictions, references)
    assert accuracy == 1.0
    
    predictions = ["Paris", "Wrong", "345"]
    references = ["Paris", "Shakespeare", "345"]
    accuracy = calculator.accuracy(predictions, references)
    assert accuracy == pytest.approx(2/3, rel=0.01)


def test_f1_score():
    """Test F1 score calculation."""
    calculator = MetricCalculator()
    
    predictions = ["The capital is Paris", "Shakespeare wrote it"]
    references = ["Paris", "Shakespeare"]
    
    f1 = calculator.f1_score(predictions, references)
    assert 0 <= f1 <= 1


def test_consistency_score():
    """Test consistency score calculation."""
    calculator = MetricCalculator()
    
    # High consistency
    predictions = ["Paris", "Paris", "Paris"]
    score = calculator.consistency_score(predictions)
    assert score == 1.0
    
    # Low consistency
    predictions = ["Paris", "London", "Berlin"]
    score = calculator.consistency_score(predictions)
    assert 0 <= score < 1.0


def test_text_similarity():
    """Test text similarity calculation."""
    calculator = MetricCalculator()
    
    sim = calculator._text_similarity("Paris is the capital", "Paris")
    assert 0 < sim <= 1.0
    
    sim = calculator._text_similarity("Paris", "Paris")
    assert sim == 1.0
    
    sim = calculator._text_similarity("Paris", "London")
    assert sim < 1.0


def test_hallucination_detection():
    """Test hallucination detection."""
    calculator = MetricCalculator()
    
    predictions = ["Paris", "Wrong Answer", "345"]
    references = ["Paris", "Correct Answer", "345"]
    
    results = calculator.hallucination_detection(predictions, references)
    
    assert "hallucination_rate" in results
    assert "factual_accuracy" in results
    assert "details" in results
    assert 0 <= results["hallucination_rate"] <= 1
    assert len(results["details"]) == 3


def test_compute_all_metrics():
    """Test computing all metrics."""
    calculator = MetricCalculator()
    
    predictions = ["Paris", "Shakespeare", "345"]
    references = ["Paris", "Shakespeare", "345"]
    
    metrics = calculator.compute_all_metrics(predictions, references)
    
    assert "accuracy" in metrics
    assert "f1_score" in metrics
    assert "bleu" in metrics
    assert "hallucination_rate" in metrics
    assert all(0 <= v <= 1 for k, v in metrics.items() if isinstance(v, (int, float)))
