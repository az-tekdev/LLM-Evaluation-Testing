#!/usr/bin/env python3
"""CLI script for running LLM evaluations."""

import argparse
import logging
import sys
from pathlib import Path

from src.config import config, validate_config
from src.dataset import EvaluationDataset, load_dataset_from_yaml
from src.model import LLMModel
from src.metrics import MetricCalculator
from src.evaluator import Evaluator
from src.report import ReportGenerator

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        *([logging.FileHandler(config.log_file)] if config.log_file else [])
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="LLM Evaluation Framework - Evaluate LLM quality and detect hallucinations"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to dataset file (JSONL, JSON, CSV, or YAML)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"Model name (default: {config.openai_model})"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=f"Output directory (default: {config.output_dir})"
    )
    
    parser.add_argument(
        "--format",
        type=str,
        choices=["json", "csv", "both"],
        default=None,
        help=f"Output format (default: {config.output_format})"
    )
    
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate"
    )
    
    parser.add_argument(
        "--enable-hallucination",
        action="store_true",
        help="Enable hallucination detection"
    )
    
    parser.add_argument(
        "--enable-consistency",
        action="store_true",
        help="Enable consistency checking"
    )
    
    parser.add_argument(
        "--perturbation",
        action="store_true",
        help="Use perturbation-based evaluation"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=f"Batch size for processing (default: {config.batch_size})"
    )
    
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Disable visualization generation"
    )
    
    args = parser.parse_args()
    
    try:
        # Validate configuration
        validate_config()
        
        # Override config with CLI args
        if args.model:
            config.openai_model = args.model
        if args.output:
            config.output_dir = args.output
        if args.format:
            config.output_format = args.format
        if args.max_samples:
            config.max_samples = args.max_samples
        if args.batch_size:
            config.batch_size = args.batch_size
        if args.no_viz:
            config.generate_visualizations = False
        
        # Load dataset
        logger.info(f"Loading dataset from {args.dataset}")
        dataset_path = Path(args.dataset)
        
        if dataset_path.suffix == ".yaml" or dataset_path.suffix == ".yml":
            dataset = load_dataset_from_yaml(str(dataset_path))
        else:
            dataset = EvaluationDataset(str(dataset_path))
        
        if len(dataset) == 0:
            logger.error("Dataset is empty")
            sys.exit(1)
        
        logger.info(f"Loaded {len(dataset)} samples")
        
        # Limit samples if specified
        if config.max_samples and len(dataset) > config.max_samples:
            dataset = dataset.limit(config.max_samples)
            logger.info(f"Limited to {config.max_samples} samples")
        
        # Initialize components
        model = LLMModel(model_name=args.model)
        metrics = MetricCalculator()
        evaluator = Evaluator(model=model, metrics=metrics)
        report_generator = ReportGenerator(output_dir=args.output)
        
        # Run evaluation
        if args.perturbation:
            logger.info("Running perturbation-based evaluation")
            results = evaluator.evaluate_with_perturbation(dataset)
        else:
            logger.info("Running standard evaluation")
            results = evaluator.evaluate(
                dataset,
                enable_hallucination_detection=args.enable_hallucination,
                enable_consistency_check=args.enable_consistency
            )
        
        # Generate report
        output_filename = dataset_path.stem + "_eval"
        report_path = report_generator.generate_report(
            results,
            filename=output_filename,
            format=args.format
        )
        
        # Print summary
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        summary = results["summary"]
        for metric, value in summary.items():
            if isinstance(value, (int, float)):
                print(f"{metric:30s}: {value:.4f}")
        print("="*60)
        print(f"\nReport saved to: {report_path}")
        
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
