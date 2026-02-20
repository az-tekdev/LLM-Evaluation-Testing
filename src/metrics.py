"""Metrics for LLM evaluation including hallucination detection."""

import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from collections import Counter

try:
    from evaluate import load as load_metric
    EVALUATE_AVAILABLE = True
except ImportError:
    EVALUATE_AVAILABLE = False
    logging.warning("Hugging Face evaluate library not available. Some metrics will be disabled.")

logger = logging.getLogger(__name__)


class MetricCalculator:
    """Calculates various evaluation metrics."""
    
    def __init__(self):
        """Initialize metric calculators."""
        self.bleu_metric = None
        self.rouge_metric = None
        self.bertscore_metric = None
        
        if EVALUATE_AVAILABLE:
            try:
                self.bleu_metric = load_metric("bleu")
                self.rouge_metric = load_metric("rouge")
            except Exception as e:
                logger.warning(f"Could not load some metrics: {e}")
    
    def accuracy(self, predictions: List[str], references: List[str]) -> float:
        """Calculate exact match accuracy.
        
        Args:
            predictions: List of predicted strings
            references: List of reference strings
            
        Returns:
            Accuracy score (0-1)
        """
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have same length")
        
        correct = sum(1 for p, r in zip(predictions, references) if p.strip().lower() == r.strip().lower())
        return correct / len(predictions) if predictions else 0.0
    
    def f1_score(self, predictions: List[str], references: List[str]) -> float:
        """Calculate F1 score based on token overlap.
        
        Args:
            predictions: List of predicted strings
            references: List of reference strings
            
        Returns:
            Average F1 score (0-1)
        """
        f1_scores = []
        for pred, ref in zip(predictions, references):
            pred_tokens = set(pred.lower().split())
            ref_tokens = set(ref.lower().split())
            
            if len(ref_tokens) == 0:
                f1_scores.append(0.0)
                continue
            
            intersection = pred_tokens & ref_tokens
            if len(intersection) == 0:
                f1_scores.append(0.0)
                continue
            
            precision = len(intersection) / len(pred_tokens) if pred_tokens else 0
            recall = len(intersection) / len(ref_tokens)
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            f1_scores.append(f1)
        
        return np.mean(f1_scores) if f1_scores else 0.0
    
    def bleu_score(self, predictions: List[str], references: List[str]) -> float:
        """Calculate BLEU score.
        
        Args:
            predictions: List of predicted strings
            references: List of reference strings (can be list of lists)
            
        Returns:
            Average BLEU score (0-1)
        """
        if not EVALUATE_AVAILABLE or not self.bleu_metric:
            logger.warning("BLEU metric not available, using simple token overlap")
            return self._simple_bleu(predictions, references)
        
        try:
            # Convert references to list of lists format
            refs = [[ref] if isinstance(ref, str) else ref for ref in references]
            results = self.bleu_metric.compute(predictions=predictions, references=refs)
            return results.get("bleu", 0.0)
        except Exception as e:
            logger.warning(f"Error computing BLEU: {e}, using fallback")
            return self._simple_bleu(predictions, references)
    
    def _simple_bleu(self, predictions: List[str], references: List[str]) -> float:
        """Simple BLEU approximation using n-gram overlap."""
        scores = []
        for pred, ref in zip(predictions, references):
            pred_words = pred.lower().split()
            ref_words = ref.lower().split()
            
            if len(ref_words) == 0:
                scores.append(0.0)
                continue
            
            # Calculate 1-gram precision
            pred_counts = Counter(pred_words)
            ref_counts = Counter(ref_words)
            
            matches = sum(min(pred_counts[w], ref_counts[w]) for w in pred_counts)
            precision = matches / len(pred_words) if pred_words else 0
            scores.append(precision)
        
        return np.mean(scores) if scores else 0.0
    
    def rouge_score(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Calculate ROUGE scores.
        
        Args:
            predictions: List of predicted strings
            references: List of reference strings
            
        Returns:
            Dictionary with ROUGE-1, ROUGE-2, ROUGE-L scores
        """
        if not EVALUATE_AVAILABLE or not self.rouge_metric:
            return self._simple_rouge(predictions, references)
        
        try:
            results = self.rouge_metric.compute(predictions=predictions, references=references)
            return {
                "rouge1": results.get("rouge1", 0.0),
                "rouge2": results.get("rouge2", 0.0),
                "rougeL": results.get("rougeL", 0.0)
            }
        except Exception as e:
            logger.warning(f"Error computing ROUGE: {e}, using fallback")
            return self._simple_rouge(predictions, references)
    
    def _simple_rouge(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Simple ROUGE approximation."""
        rouge1_scores = []
        for pred, ref in zip(predictions, references):
            pred_words = set(pred.lower().split())
            ref_words = set(ref.lower().split())
            
            if len(ref_words) == 0:
                rouge1_scores.append(0.0)
                continue
            
            intersection = pred_words & ref_words
            recall = len(intersection) / len(ref_words) if ref_words else 0
            rouge1_scores.append(recall)
        
        avg_rouge1 = np.mean(rouge1_scores) if rouge1_scores else 0.0
        return {
            "rouge1": avg_rouge1,
            "rouge2": avg_rouge1,  # Simplified
            "rougeL": avg_rouge1   # Simplified
        }
    
    def consistency_score(self, predictions: List[str]) -> float:
        """Calculate consistency across multiple predictions (for hallucination detection).
        
        Args:
            predictions: List of predictions (multiple per sample for consistency check)
            
        Returns:
            Consistency score (0-1), higher means more consistent
        """
        if len(predictions) < 2:
            return 1.0  # Single prediction is always consistent
        
        # Group predictions (assuming they come in groups)
        # For simplicity, calculate pairwise similarity
        similarities = []
        for i in range(len(predictions)):
            for j in range(i + 1, len(predictions)):
                sim = self._text_similarity(predictions[i], predictions[j])
                similarities.append(sim)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity using token overlap."""
        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())
        
        if not tokens1 and not tokens2:
            return 1.0
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = tokens1 & tokens2
        union = tokens1 | tokens2
        return len(intersection) / len(union) if union else 0.0
    
    def hallucination_detection(
        self,
        predictions: List[str],
        references: List[str],
        consistency_predictions: Optional[List[List[str]]] = None
    ) -> Dict[str, Any]:
        """Detect potential hallucinations.
        
        Args:
            predictions: List of predicted strings
            references: List of reference strings
            consistency_predictions: Optional list of multiple predictions per sample
            
        Returns:
            Dictionary with hallucination metrics
        """
        results = {
            "hallucination_rate": 0.0,
            "consistency_score": 0.0,
            "factual_accuracy": 0.0,
            "details": []
        }
        
        # Factual accuracy (exact match)
        factual_correct = sum(1 for p, r in zip(predictions, references) 
                             if p.strip().lower() == r.strip().lower())
        results["factual_accuracy"] = factual_correct / len(predictions) if predictions else 0.0
        
        # Consistency check
        if consistency_predictions:
            consistency_scores = []
            for pred_group in consistency_predictions:
                if len(pred_group) > 1:
                    score = self.consistency_score(pred_group)
                    consistency_scores.append(score)
            results["consistency_score"] = np.mean(consistency_scores) if consistency_scores else 0.0
        
        # Hallucination rate (low factual accuracy + low consistency)
        hallucination_rate = 1.0 - (results["factual_accuracy"] * 0.7 + results["consistency_score"] * 0.3)
        results["hallucination_rate"] = max(0.0, min(1.0, hallucination_rate))
        
        # Per-sample details
        for i, (pred, ref) in enumerate(zip(predictions, references)):
            is_hallucination = (
                pred.strip().lower() != ref.strip().lower() and
                self._text_similarity(pred, ref) < 0.5
            )
            results["details"].append({
                "sample_id": i,
                "is_hallucination": is_hallucination,
                "similarity": self._text_similarity(pred, ref)
            })
        
        return results
    
    def compute_all_metrics(
        self,
        predictions: List[str],
        references: List[str],
        consistency_predictions: Optional[List[List[str]]] = None
    ) -> Dict[str, Any]:
        """Compute all available metrics.
        
        Args:
            predictions: List of predicted strings
            references: List of reference strings
            consistency_predictions: Optional list of multiple predictions per sample
            
        Returns:
            Dictionary with all metric scores
        """
        metrics = {
            "accuracy": self.accuracy(predictions, references),
            "f1_score": self.f1_score(predictions, references),
            "bleu": self.bleu_score(predictions, references),
        }
        
        rouge_scores = self.rouge_score(predictions, references)
        metrics.update(rouge_scores)
        
        if consistency_predictions:
            hallucination_results = self.hallucination_detection(
                predictions, references, consistency_predictions
            )
            metrics.update({
                "hallucination_rate": hallucination_results["hallucination_rate"],
                "consistency_score": hallucination_results["consistency_score"],
                "factual_accuracy": hallucination_results["factual_accuracy"]
            })
        else:
            hallucination_results = self.hallucination_detection(predictions, references)
            metrics.update({
                "hallucination_rate": hallucination_results["hallucination_rate"],
                "factual_accuracy": hallucination_results["factual_accuracy"]
            })
        
        return metrics
