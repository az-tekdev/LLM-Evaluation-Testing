"""Unit tests for report module."""

import pytest
import tempfile
import json
from pathlib import Path

from src.report import ReportGenerator


@pytest.fixture
def sample_results():
    """Create sample evaluation results."""
    return {
        "summary": {
            "accuracy": 0.8,
            "f1_score": 0.75,
            "bleu": 0.7,
            "hallucination_rate": 0.2
        },
        "samples": [
            {
                "sample_id": 0,
                "prompt": "Test question?",
                "prediction": "Test answer",
                "reference": "Test answer",
                "correct": True
            },
            {
                "sample_id": 1,
                "prompt": "Another question?",
                "prediction": "Wrong answer",
                "reference": "Correct answer",
                "correct": False
            }
        ]
    }


def test_report_generator_initialization():
    """Test report generator initialization."""
    with tempfile.TemporaryDirectory() as tmpdir:
        generator = ReportGenerator(output_dir=tmpdir)
        assert Path(generator.output_dir).exists()


def test_save_json(sample_results):
    """Test saving JSON report."""
    with tempfile.TemporaryDirectory() as tmpdir:
        generator = ReportGenerator(output_dir=tmpdir)
        filepath = generator._save_json(sample_results, "test")
        
        assert Path(filepath).exists()
        with open(filepath, 'r') as f:
            loaded = json.load(f)
        assert "summary" in loaded


def test_save_csv(sample_results):
    """Test saving CSV report."""
    with tempfile.TemporaryDirectory() as tmpdir:
        generator = ReportGenerator(output_dir=tmpdir)
        filepath = generator._save_csv(sample_results, "test")
        
        assert Path(filepath).exists()
        # Check that summary CSV was created
        summary_path = Path(tmpdir) / "test_summary.csv"
        assert summary_path.exists()


def test_generate_report_json(sample_results):
    """Test generating JSON report."""
    with tempfile.TemporaryDirectory() as tmpdir:
        generator = ReportGenerator(output_dir=tmpdir)
        path = generator.generate_report(sample_results, "test", "json")
        
        json_path = Path(tmpdir) / "test.json"
        assert json_path.exists()


def test_generate_report_csv(sample_results):
    """Test generating CSV report."""
    with tempfile.TemporaryDirectory() as tmpdir:
        generator = ReportGenerator(output_dir=tmpdir)
        path = generator.generate_report(sample_results, "test", "csv")
        
        csv_path = Path(tmpdir) / "test_summary.csv"
        assert csv_path.exists()
