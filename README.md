# LLM Evaluation Testing Framework

A comprehensive, production-ready Python framework for evaluating Large Language Model (LLM) quality, with a focus on accuracy, reasoning, coherence, and hallucination detection. Inspired by OpenAI Evals and designed for rigorous LLM assessment.

## Features

- 🎯 **Comprehensive Metrics**: Accuracy, F1, BLEU, ROUGE, and custom hallucination detection
- 🔍 **Hallucination Detection**: Self-consistency checks, perturbation testing, and fact-checking
- 📊 **Multiple Evaluation Modes**: Standard evaluation, consistency checking, and perturbation-based testing
- 📁 **Flexible Dataset Support**: JSONL, JSON, CSV, and YAML formats
- ⚡ **Parallel Processing**: Efficient batch inference with configurable workers
- 📈 **Rich Reporting**: JSON/CSV reports with optional visualizations
- 🧪 **Extensible**: Easy to add custom metrics and evaluation suites
- 🐳 **Docker Support**: Containerized deployment ready

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    CLI (eval.py)                        │
└────────────────────┬────────────────────────────────────┘
                     │
         ┌───────────▼───────────┐
         │     Evaluator         │
         │  (Main Pipeline)      │
         └───────┬───────────────┘
                 │
    ┌────────────┼────────────┐
    │            │            │
┌───▼───┐  ┌────▼────┐  ┌────▼────┐
│Dataset│  │  Model  │  │ Metrics │
│Loader │  │  (LLM)  │  │Calculator│
└───────┘  └─────────┘  └─────────┘
    │            │            │
    │            │            │
    └────────────┼────────────┘
                 │
         ┌───────▼────────┐
         │ Report Generator │
         │  (JSON/CSV/Viz)  │
         └─────────────────┘
```

## Installation

### Prerequisites

- Python 3.10 or higher
- OpenAI API key (or other LLM provider)

### Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd LLM-Evaluation-Testing
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env and add your OPENAI_API_KEY
   ```

## Quick Start

### Basic Evaluation

```bash
# Evaluate on a sample dataset
python -m src.eval --dataset data/sample_qa.jsonl

# With hallucination detection
python -m src.eval --dataset data/sample_qa.jsonl --enable-hallucination --enable-consistency

# Using perturbation testing
python -m src.eval --dataset data/sample_qa.jsonl --perturbation

# Limit number of samples for testing
python -m src.eval --dataset data/sample_qa.jsonl --max-samples 5
```

### Custom Dataset

Create a JSONL file with your test cases:

```jsonl
{"prompt": "What is the capital of France?", "reference": "Paris", "category": "factual"}
{"prompt": "Calculate 10 * 5", "reference": "50", "category": "math"}
```

Then run:
```bash
python -m src.eval --dataset your_dataset.jsonl
```

### YAML Evaluation Suite

Create a YAML file (see `examples/sample_eval_suite.yaml`):

```yaml
name: "My Evaluation Suite"
tests:
  - prompt: "Your question here"
    reference: "Expected answer"
    category: "factual"
```

Run with:
```bash
python -m src.eval --dataset examples/sample_eval_suite.yaml
```

## Usage Examples

### Command-Line Options

```bash
python -m src.eval \
  --dataset data/sample_qa.jsonl \
  --model gpt-4 \
  --output results/my_eval \
  --format both \
  --enable-hallucination \
  --enable-consistency \
  --batch-size 20 \
  --max-samples 100
```

### Programmatic Usage

```python
from src.dataset import EvaluationDataset
from src.model import LLMModel
from src.metrics import MetricCalculator
from src.evaluator import Evaluator
from src.report import ReportGenerator

# Load dataset
dataset = EvaluationDataset("data/sample_qa.jsonl")

# Initialize components
model = LLMModel(model_name="gpt-4")
metrics = MetricCalculator()
evaluator = Evaluator(model=model, metrics=metrics)

# Run evaluation
results = evaluator.evaluate(
    dataset,
    enable_hallucination_detection=True,
    enable_consistency_check=True
)

# Generate report
report_gen = ReportGenerator(output_dir="results")
report_gen.generate_report(results, filename="my_evaluation")
```

## Support

- Telegram: https://t.me/az_tekDev
- Twitter: https://x.com/az_tekDev
