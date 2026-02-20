"""LLM model integration for evaluation."""

import logging
import time
from typing import List, Dict, Any, Optional
import openai
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from src.config import config

logger = logging.getLogger(__name__)


class LLMModel:
    """Wrapper for LLM API calls with batch processing and retry logic."""
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ):
        """Initialize LLM model.
        
        Args:
            model_name: Model name (defaults to config)
            api_key: OpenAI API key (defaults to config)
            temperature: Temperature setting (defaults to config)
            max_tokens: Max tokens (defaults to config)
        """
        self.model_name = model_name or config.openai_model
        self.api_key = api_key or config.openai_api_key
        self.temperature = temperature if temperature is not None else config.openai_temperature
        self.max_tokens = max_tokens or config.openai_max_tokens
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        openai.api_key = self.api_key
        self.client = openai.OpenAI(api_key=self.api_key)
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a single response.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional arguments for API call
            
        Returns:
            Generated text
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                timeout=config.openai_timeout
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise
    
    def generate_with_retry(self, prompt: str, max_retries: Optional[int] = None, **kwargs) -> str:
        """Generate response with retry logic.
        
        Args:
            prompt: Input prompt
            max_retries: Maximum retry attempts (defaults to config)
            **kwargs: Additional arguments
            
        Returns:
            Generated text
        """
        max_retries = max_retries or config.retry_attempts
        
        for attempt in range(max_retries):
            try:
                return self.generate(prompt, **kwargs)
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = config.retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay}s: {e}")
                    time.sleep(delay)
                else:
                    logger.error(f"All {max_retries} attempts failed for prompt: {prompt[:50]}...")
                    raise
    
    def generate_batch(
        self,
        prompts: List[str],
        batch_size: Optional[int] = None,
        max_workers: Optional[int] = None,
        show_progress: bool = True
    ) -> List[str]:
        """Generate responses for multiple prompts in parallel.
        
        Args:
            prompts: List of input prompts
            batch_size: Batch size for processing (defaults to config)
            max_workers: Number of worker threads (defaults to config)
            show_progress: Whether to show progress bar
            
        Returns:
            List of generated texts
        """
        batch_size = batch_size or config.batch_size
        max_workers = max_workers or config.max_workers
        
        results = []
        total_batches = (len(prompts) + batch_size - 1) // batch_size
        
        iterator = range(0, len(prompts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Processing batches", total=total_batches)
        
        for i in iterator:
            batch = prompts[i:i + batch_size]
            batch_results = self._process_batch(batch, max_workers)
            results.extend(batch_results)
        
        return results
    
    def _process_batch(self, prompts: List[str], max_workers: int) -> List[str]:
        """Process a batch of prompts in parallel.
        
        Args:
            prompts: List of prompts in batch
            max_workers: Number of worker threads
            
        Returns:
            List of generated texts
        """
        results = [None] * len(prompts)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(self.generate_with_retry, prompt): idx
                for idx, prompt in enumerate(prompts)
            }
            
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.error(f"Error processing prompt {idx}: {e}")
                    results[idx] = f"ERROR: {str(e)}"
        
        return results
    
    def generate_multiple_samples(
        self,
        prompt: str,
        num_samples: int = 3,
        **kwargs
    ) -> List[str]:
        """Generate multiple samples for consistency checking.
        
        Args:
            prompt: Input prompt
            num_samples: Number of samples to generate
            **kwargs: Additional arguments
            
        Returns:
            List of generated texts
        """
        samples = []
        for _ in range(num_samples):
            sample = self.generate_with_retry(prompt, **kwargs)
            samples.append(sample)
        return samples
    
    def generate_with_perturbation(
        self,
        prompt: str,
        num_perturbations: int = 2,
        **kwargs
    ) -> List[str]:
        """Generate responses with prompt perturbations.
        
        Args:
            prompt: Original prompt
            num_perturbations: Number of perturbations to create
            **kwargs: Additional arguments
            
        Returns:
            List of generated texts (original + perturbations)
        """
        results = [self.generate_with_retry(prompt, **kwargs)]
        
        # Simple perturbation: rephrase the prompt
        perturbations = self._create_perturbations(prompt, num_perturbations)
        for pert_prompt in perturbations:
            result = self.generate_with_retry(pert_prompt, **kwargs)
            results.append(result)
        
        return results
    
    def _create_perturbations(self, prompt: str, num: int) -> List[str]:
        """Create prompt perturbations.
        
        Args:
            prompt: Original prompt
            num: Number of perturbations
            
        Returns:
            List of perturbed prompts
        """
        perturbations = []
        
        # Simple rephrasing strategies
        strategies = [
            lambda p: f"Please answer: {p}",
            lambda p: f"Question: {p}",
            lambda p: f"Can you tell me: {p}",
            lambda p: f"I need to know: {p}",
        ]
        
        for i in range(min(num, len(strategies))):
            perturbations.append(strategies[i](prompt))
        
        return perturbations
