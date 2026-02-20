"""Main evaluation pipeline."""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from src.dataset import EvaluationDataset
from src.model import LLMModel
from src.metrics import MetricCalculator
from src.config import config

logger = logging.getLogger(__name__)


class Evaluator:
    """Main evaluation pipeline."""
    
    def __init__(
        self,
        model: Optional[LLMModel] = None,
        metrics: Optional[MetricCalculator] = None
    ):
        """Initialize evaluator.
        
        Args:
            model: LLM model instance (creates default if None)
            metrics: Metric calculator instance (creates default if None)
        """
        self.model = model or LLMModel()
        self.metrics = metrics or MetricCalculator()
    
    def evaluate(
        self,
        dataset: EvaluationDataset,
        enable_hallucination_detection: Optional[bool] = None,
        enable_consistency_check: Optional[bool] = None
    ) -> Dict[str, Any]:
        """Run evaluation on dataset.
        
        Args:
            dataset: Evaluation dataset
            enable_hallucination_detection: Enable hallucination detection
            enable_consistency_check: Enable consistency checking
            
        Returns:
            Dictionary with evaluation results
        """
        enable_hallucination = (
            enable_hallucination_detection 
            if enable_hallucination_detection is not None 
            else config.enable_hallucination_detection
        )
        enable_consistency = (
            enable_consistency_check
            if enable_consistency_check is not None
            else config.enable_consistency_check
        )
        
        logger.info(f"Starting evaluation on {len(dataset)} samples")
        
        # Get prompts and references
        prompts = dataset.get_prompts()
        references = dataset.get_references()
        metadata = dataset.get_metadata()
        
        # Limit samples if configured
        if config.max_samples and len(prompts) > config.max_samples:
            logger.info(f"Limiting to {config.max_samples} samples")
            prompts = prompts[:config.max_samples]
            references = references[:config.max_samples]
            metadata = metadata[:config.max_samples]
        
        # Generate predictions
        logger.info("Generating predictions...")
        predictions = self.model.generate_batch(prompts)
        
        # Generate consistency samples if enabled
        consistency_predictions = None
        if enable_consistency:
            logger.info("Generating consistency samples...")
            consistency_predictions = []
            for prompt in prompts:
                samples = self.model.generate_multiple_samples(
                    prompt,
                    num_samples=config.hallucination_samples
                )
                consistency_predictions.append(samples)
        
        # Compute metrics
        logger.info("Computing metrics...")
        all_metrics = self.metrics.compute_all_metrics(
            predictions=predictions,
            references=references,
            consistency_predictions=consistency_predictions
        )
        
        # Prepare detailed results
        results = {
            "summary": all_metrics,
            "samples": []
        }
        
        for i, (pred, ref, meta) in enumerate(zip(predictions, references, metadata)):
            sample_result = {
                "sample_id": i,
                "prompt": prompts[i],
                "prediction": pred,
                "reference": ref,
                "metadata": meta,
                "correct": pred.strip().lower() == ref.strip().lower()
            }
            
            if consistency_predictions:
                sample_result["consistency_samples"] = consistency_predictions[i]
                sample_result["consistency_score"] = self.metrics.consistency_score(
                    consistency_predictions[i]
                )
            
            results["samples"].append(sample_result)
        
        logger.info("Evaluation complete")
        return results
    
    def evaluate_with_perturbation(
        self,
        dataset: EvaluationDataset,
        num_perturbations: Optional[int] = None
    ) -> Dict[str, Any]:
        """Evaluate with prompt perturbations.
        
        Args:
            dataset: Evaluation dataset
            num_perturbations: Number of perturbations per prompt
            
        Returns:
            Dictionary with evaluation results including perturbation analysis
        """
        num_perturbations = num_perturbations or config.perturbation_count
        
        logger.info(f"Starting perturbation-based evaluation on {len(dataset)} samples")
        
        prompts = dataset.get_prompts()
        references = dataset.get_references()
        metadata = dataset.get_metadata()
        
        if config.max_samples and len(prompts) > config.max_samples:
            prompts = prompts[:config.max_samples]
            references = references[:config.max_samples]
            metadata = metadata[:config.max_samples]
        
        # Generate predictions with perturbations
        logger.info("Generating predictions with perturbations...")
        all_predictions = []
        for prompt in prompts:
            pert_results = self.model.generate_with_perturbation(
                prompt,
                num_perturbations=num_perturbations
            )
            all_predictions.append(pert_results)
        
        # Use first prediction as main prediction
        predictions = [preds[0] for preds in all_predictions]
        
        # Compute metrics
        all_metrics = self.metrics.compute_all_metrics(
            predictions=predictions,
            references=references
        )
        
        # Analyze perturbation consistency
        perturbation_scores = []
        for pred_group in all_predictions:
            score = self.metrics.consistency_score(pred_group)
            perturbation_scores.append(score)
        
        all_metrics["perturbation_consistency"] = sum(perturbation_scores) / len(perturbation_scores) if perturbation_scores else 0.0
        
        # Prepare results
        results = {
            "summary": all_metrics,
            "samples": []
        }
        
        for i, (pred_group, ref, meta) in enumerate(zip(all_predictions, references, metadata)):
            sample_result = {
                "sample_id": i,
                "prompt": prompts[i],
                "predictions": pred_group,
                "main_prediction": pred_group[0],
                "reference": ref,
                "metadata": meta,
                "perturbation_consistency": perturbation_scores[i],
                "correct": pred_group[0].strip().lower() == ref.strip().lower()
            }
            results["samples"].append(sample_result)
        
        logger.info("Perturbation evaluation complete")
        return results
