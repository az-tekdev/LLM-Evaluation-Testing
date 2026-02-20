"""Report generation and visualization."""

import json
import csv
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logging.warning("Matplotlib not available. Visualizations will be skipped.")

from src.config import config

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates evaluation reports and visualizations."""
    
    def __init__(self, output_dir: Optional[str] = None):
        """Initialize report generator.
        
        Args:
            output_dir: Output directory for reports (defaults to config)
        """
        self.output_dir = Path(output_dir or config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_report(
        self,
        results: Dict[str, Any],
        filename: Optional[str] = None,
        format: Optional[str] = None
    ) -> str:
        """Generate evaluation report.
        
        Args:
            results: Evaluation results dictionary
            filename: Output filename (without extension)
            format: Output format ('json', 'csv', 'both')
            
        Returns:
            Path to generated report file
        """
        filename = filename or "evaluation_report"
        format = format or config.output_format
        
        if format in ["json", "both"]:
            json_path = self._save_json(results, filename)
            logger.info(f"Saved JSON report to {json_path}")
        
        if format in ["csv", "both"]:
            csv_path = self._save_csv(results, filename)
            logger.info(f"Saved CSV report to {csv_path}")
        
        if config.generate_visualizations and MATPLOTLIB_AVAILABLE:
            self._generate_visualizations(results, filename)
        
        return str(self.output_dir / filename)
    
    def _save_json(self, results: Dict[str, Any], filename: str) -> Path:
        """Save results as JSON.
        
        Args:
            results: Results dictionary
            filename: Base filename
            
        Returns:
            Path to saved file
        """
        filepath = self.output_dir / f"{filename}.json"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        return filepath
    
    def _save_csv(self, results: Dict[str, Any], filename: str) -> Path:
        """Save results as CSV.
        
        Args:
            results: Results dictionary
            filename: Base filename
            
        Returns:
            Path to saved file
        """
        # Save summary metrics
        summary_path = self.output_dir / f"{filename}_summary.csv"
        summary_df = pd.DataFrame([results["summary"]])
        summary_df.to_csv(summary_path, index=False)
        
        # Save detailed samples
        samples_path = self.output_dir / f"{filename}_samples.csv"
        if "samples" in results and results["samples"]:
            # Flatten sample data
            samples_data = []
            for sample in results["samples"]:
                flat_sample = {
                    "sample_id": sample.get("sample_id"),
                    "prompt": sample.get("prompt", "")[:100],  # Truncate long prompts
                    "prediction": sample.get("prediction", "")[:200],
                    "reference": sample.get("reference", "")[:200],
                    "correct": sample.get("correct", False),
                }
                
                # Add consistency score if available
                if "consistency_score" in sample:
                    flat_sample["consistency_score"] = sample["consistency_score"]
                
                # Add perturbation consistency if available
                if "perturbation_consistency" in sample:
                    flat_sample["perturbation_consistency"] = sample["perturbation_consistency"]
                
                samples_data.append(flat_sample)
            
            samples_df = pd.DataFrame(samples_data)
            samples_df.to_csv(samples_path, index=False)
        
        return summary_path
    
    def _generate_visualizations(self, results: Dict[str, Any], filename: str) -> None:
        """Generate visualization plots.
        
        Args:
            results: Results dictionary
            filename: Base filename for saving plots
        """
        if not MATPLOTLIB_AVAILABLE:
            return
        
        try:
            # Set style
            sns.set_style("whitegrid")
            plt.rcParams['figure.figsize'] = (12, 8)
            
            # 1. Metrics bar chart
            self._plot_metrics_bar(results["summary"], filename)
            
            # 2. Accuracy distribution
            if "samples" in results:
                self._plot_accuracy_distribution(results["samples"], filename)
            
            # 3. Consistency scores (if available)
            if "samples" in results and any("consistency_score" in s for s in results["samples"]):
                self._plot_consistency_scores(results["samples"], filename)
            
            logger.info(f"Generated visualizations for {filename}")
        except Exception as e:
            logger.warning(f"Error generating visualizations: {e}")
    
    def _plot_metrics_bar(self, summary: Dict[str, Any], filename: str) -> None:
        """Plot metrics as bar chart.
        
        Args:
            summary: Summary metrics dictionary
            filename: Base filename
        """
        # Filter numeric metrics
        metrics_to_plot = {
            k: v for k, v in summary.items()
            if isinstance(v, (int, float)) and k not in ["sample_id"]
        }
        
        if not metrics_to_plot:
            return
        
        fig, ax = plt.subplots()
        names = list(metrics_to_plot.keys())
        values = list(metrics_to_plot.values())
        
        bars = ax.barh(names, values)
        ax.set_xlabel("Score")
        ax.set_title("Evaluation Metrics Summary")
        ax.set_xlim(0, 1.1)
        
        # Add value labels
        for bar, value in zip(bars, values):
            ax.text(value, bar.get_y() + bar.get_height()/2, 
                   f'{value:.3f}', ha='left', va='center')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{filename}_metrics.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_accuracy_distribution(self, samples: List[Dict], filename: str) -> None:
        """Plot accuracy distribution.
        
        Args:
            samples: List of sample results
            filename: Base filename
        """
        correct = [s.get("correct", False) for s in samples]
        correct_count = sum(correct)
        incorrect_count = len(correct) - correct_count
        
        fig, ax = plt.subplots()
        ax.bar(["Correct", "Incorrect"], [correct_count, incorrect_count], 
              color=["green", "red"], alpha=0.7)
        ax.set_ylabel("Count")
        ax.set_title("Accuracy Distribution")
        
        # Add count labels
        ax.text(0, correct_count, str(correct_count), 
               ha='center', va='bottom', fontweight='bold')
        ax.text(1, incorrect_count, str(incorrect_count), 
               ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{filename}_accuracy.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_consistency_scores(self, samples: List[Dict], filename: str) -> None:
        """Plot consistency score distribution.
        
        Args:
            samples: List of sample results
            filename: Base filename
        """
        consistency_scores = [
            s.get("consistency_score", 0) for s in samples
            if "consistency_score" in s
        ]
        
        if not consistency_scores:
            return
        
        fig, ax = plt.subplots()
        ax.hist(consistency_scores, bins=20, edgecolor='black', alpha=0.7)
        ax.set_xlabel("Consistency Score")
        ax.set_ylabel("Frequency")
        ax.set_title("Consistency Score Distribution")
        ax.axvline(sum(consistency_scores) / len(consistency_scores), 
                  color='red', linestyle='--', label='Mean')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{filename}_consistency.png", dpi=150, bbox_inches='tight')
        plt.close()
