#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main Predictor Module

Replaces the MOA Java simulation with LLM-based prediction.
"""

import os
import sys
import json
import pickle
import datetime
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np

# Add pyloader to path for data loading
sys.path.insert(0, str(Path(__file__).parent.parent / 'pyloader'))

from data_loader import DataLoader, load_data
from compressor import DataCompressor, BatchCompressor
from llm_client import SiliconFlowClient, DiskFailurePromptBuilder, create_llm_client
from evaluator import ClassificationEvaluator, RegressionEvaluator, DelayEvaluator, format_metrics_table

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LLMPredictor:
    """
    LLM-based disk failure predictor.
    Replaces the MOA Java simulation component.
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 model: str = "deepseek-ai/DeepSeek-V3.2",
                 threshold: float = 0.5,
                 validation_window: int = 30,
                 use_compression: bool = True,
                 compression_method: str = 'gzip',
                 batch_size: int = 10,
                 bl_delay: bool = True,
                 bl_regression: bool = False):
        """
        Initialize LLM Predictor.
        
        Args:
            api_key: SiliconFlow API key
            model: Model name
            threshold: Classification threshold
            validation_window: Window size for delay evaluation
            use_compression: Whether to compress data before sending
            compression_method: Compression method ('gzip', 'zlib', 'none')
            batch_size: Number of disks to process in one batch
            bl_delay: Enable delay evaluation
            bl_regression: Regression mode (predict days until failure)
        """
        # Initialize LLM client
        self.client = create_llm_client(
            provider='siliconflow',
            api_key=api_key or os.getenv('SILICONFLOW_API_KEY'),
            model=model
        )
        self.prompt_builder = DiskFailurePromptBuilder()
        
        # Configuration
        self.threshold = threshold
        self.validation_window = validation_window
        self.use_compression = use_compression
        self.batch_size = batch_size
        self.bl_delay = bl_delay
        self.bl_regression = bl_regression
        
        # Compression
        self.compressor = DataCompressor(method=compression_method) if use_compression else None
        self.batch_compressor = BatchCompressor(self.compressor, batch_size=batch_size) if use_compression else None
        
        # Evaluators
        if bl_regression:
            self.global_evaluator = RegressionEvaluator()
            self.local_evaluator = RegressionEvaluator()
        else:
            self.global_evaluator = ClassificationEvaluator(threshold=threshold)
            self.local_evaluator = ClassificationEvaluator(threshold=threshold)
        
        self.delay_evaluator = DelayEvaluator(validation_window=validation_window)
        
        # Data tracking
        self.keep_delay = {}  # serial_number -> list of predictions for delay evaluation
        self.current_date = None
        self.iteration = 0
        
        # Paths
        self.train_path = None
        self.test_path = None
        
    def init_paths(self, train_path: str, test_path: Optional[str] = None):
        """Initialize data paths."""
        self.train_path = Path(train_path)
        self.test_path = Path(test_path) if test_path else None
        
    def set_date(self, date: datetime.date):
        """Set current date."""
        self.current_date = date
        
    def load_data_file(self, filename: str, data_type: str = 'train') -> pd.DataFrame:
        """
        Load data file for given date.
        
        Args:
            filename: Date string (YYYY-MM-DD)
            data_type: 'train' or 'test'
            
        Returns:
            DataFrame with data
        """
        if data_type == 'train':
            path = self.train_path
        else:
            path = self.test_path
            
        if path is None:
            raise ValueError(f"Path not set for {data_type}")
        
        # Try ARFF first, then CSV
        for ext in ['.arff', '.csv']:
            file_path = path / (filename + ext)
            if file_path.exists():
                loader = DataLoader(str(file_path))
                return loader.load()
        
        raise FileNotFoundError(f"Data file not found for {filename}")
    
    def predict_single(self, disk_data: Dict[str, Any], 
                      historical_data: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Predict failure probability for a single disk.
        
        Args:
            disk_data: Dictionary with SMART attributes
            historical_data: Optional historical data
            
        Returns:
            Prediction result dictionary
        """
        # Build prompt
        prompt = self.prompt_builder.build_prediction_prompt(disk_data, historical_data)
        
        # Call LLM API
        try:
            response = self.client.predict(
                prompt,
                system_message=self.prompt_builder.SYSTEM_MESSAGE,
                temperature=0.1,
                max_tokens=500
            )
            
            # Parse response
            result = self.prompt_builder.parse_prediction_response(response['text'])
            result['raw_response'] = response.get('text', '')
            result['api_usage'] = response.get('usage', {})
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {
                'failure_probability': 0.5,
                'risk_level': 'unknown',
                'reasoning': f'API error: {str(e)}',
                'confidence': 0.0
            }
    
    def predict_batch(self, disks_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Predict failure probability for a batch of disks.
        
        Args:
            disks_data: List of disk data dictionaries
            
        Returns:
            List of prediction results
        """
        if len(disks_data) <= self.batch_size:
            # Process as single batch
            prompt = self.prompt_builder.build_batch_prompt(disks_data)
            
            try:
                response = self.client.predict(
                    prompt,
                    system_message=self.prompt_builder.SYSTEM_MESSAGE,
                    temperature=0.1,
                    max_tokens=2000
                )
                
                results = self.prompt_builder.parse_batch_response(response['text'])
                
                # Ensure results match input length
                while len(results) < len(disks_data):
                    results.append({
                        'failure_probability': 0.5,
                        'risk_level': 'unknown',
                        'reasoning': 'Missing prediction',
                        'confidence': 0.0
                    })
                
                return results[:len(disks_data)]
                
            except Exception as e:
                logger.error(f"Batch prediction failed: {e}")
                # Return default predictions
                return [{
                    'failure_probability': 0.5,
                    'risk_level': 'unknown',
                    'reasoning': f'API error: {str(e)}',
                    'confidence': 0.0
                } for _ in disks_data]
        else:
            # Split into smaller batches
            all_results = []
            for i in range(0, len(disks_data), self.batch_size):
                batch = disks_data[i:i+self.batch_size]
                results = self.predict_batch(batch)
                all_results.extend(results)
            return all_results
    
    def process_test_data(self, test_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Process test data and make predictions.
        
        Args:
            test_data: Test data DataFrame
            
        Returns:
            Evaluation metrics
        """
        if test_data is None or len(test_data) == 0:
            return {}
        
        logger.info(f"Processing {len(test_data)} test instances")
        
        # Get serial numbers
        serial_numbers = test_data.get('serial_number', [f'disk_{i}' for i in range(len(test_data))])
        
        # Get labels
        label_col = 'failure' if 'failure' in test_data.columns else test_data.columns[0]
        actuals = test_data[label_col].values if label_col in test_data.columns else [0] * len(test_data)
        
        # Convert to list of dicts for batch processing
        feature_cols = [c for c in test_data.columns if c not in ['serial_number', 'failure', 'date', 'model']]
        disks_data = test_data[feature_cols].to_dict('records')
        
        # Make predictions in batches
        predictions = self.predict_batch(disks_data)
        
        # Extract probabilities
        predicted_probs = []
        for pred in predictions:
            prob = pred.get('failure_probability', 0.5)
            # Handle string values
            if isinstance(prob, str):
                try:
                    prob = float(prob)
                except:
                    prob = 0.5
            predicted_probs.append(prob)
        
        # Update evaluators
        for i, (actual, pred_prob) in enumerate(zip(actuals, predicted_probs)):
            # Convert actual to int if needed
            if isinstance(actual, str):
                actual = 1 if actual in ['1', 'c1', 'True', 'true'] else 0
            actual = int(actual)
            
            sn = serial_numbers.iloc[i] if hasattr(serial_numbers, 'iloc') else serial_numbers[i]
            
            self.global_evaluator.add_result(actual, pred_prob, sn)
            self.local_evaluator.add_result(actual, pred_prob, sn)
            
            # Delay evaluation tracking
            if self.bl_delay:
                if sn not in self.keep_delay:
                    self.keep_delay[sn] = []
                self.keep_delay[sn].append({
                    'actual': actual,
                    'predicted_prob': pred_prob,
                    'day': self.iteration
                })
        
        return self.local_evaluator.get_metrics()
    
    def delay_evaluate(self) -> Dict[str, Any]:
        """
        Perform delay evaluation on tracked instances.
        
        Returns:
            Delay evaluation metrics
        """
        if not self.bl_delay:
            return {}
        
        # Process delay windows
        pop_sn = []
        for sn, instances in self.keep_delay.items():
            if len(instances) > 0:
                instances.pop(0)  # Remove oldest
                if len(instances) == 0:
                    pop_sn.append(sn)
        
        for sn in pop_sn:
            del self.keep_delay[sn]
        
        # Return current metrics
        return self.global_evaluator.get_metrics()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current evaluation metrics."""
        metrics = {
            'global': self.global_evaluator.get_metrics(),
            'local': self.local_evaluator.get_metrics(),
            'iteration': self.iteration
        }
        
        if self.current_date:
            metrics['date'] = self.current_date.isoformat()
        
        return metrics
    
    def reset_local_evaluator(self):
        """Reset local evaluator for new evaluation window."""
        if self.bl_regression:
            self.local_evaluator = RegressionEvaluator()
        else:
            self.local_evaluator = ClassificationEvaluator(threshold=self.threshold)
    
    def save_state(self, filepath: str):
        """Save predictor state to file."""
        state = {
            'iteration': self.iteration,
            'current_date': self.current_date,
            'keep_delay': self.keep_delay,
            'global_evaluator_state': self._get_evaluator_state(self.global_evaluator),
            'threshold': self.threshold
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        logger.info(f"State saved to {filepath}")
    
    def load_state(self, filepath: str):
        """Load predictor state from file."""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        self.iteration = state.get('iteration', 0)
        self.current_date = state.get('current_date')
        self.keep_delay = state.get('keep_delay', {})
        self.threshold = state.get('threshold', 0.5)
        
        logger.info(f"State loaded from {filepath}")
    
    def _get_evaluator_state(self, evaluator) -> Dict:
        """Extract evaluator state."""
        if isinstance(evaluator, ClassificationEvaluator):
            return {
                'tp': evaluator.tp,
                'fp': evaluator.fp,
                'tn': evaluator.tn,
                'fn': evaluator.fn,
                'total': evaluator.total_predictions
            }
        return {}
    
    def print_metrics(self):
        """Print current metrics."""
        metrics = self.get_metrics()
        
        print("\n" + "="*60)
        print(f"Evaluation Metrics - Iteration {metrics['iteration']}")
        if 'date' in metrics:
            print(f"Date: {metrics['date']}")
        print("="*60)
        
        print("\nGlobal Metrics:")
        print(format_metrics_table(metrics['global']))
        
        print("\nLocal Metrics (Current Window):")
        print(format_metrics_table(metrics['local']))


def run_simulation(
    start_date: str,
    train_path: str,
    test_path: Optional[str] = None,
    iterations: int = 30,
    api_key: Optional[str] = None,
    model: str = "deepseek-ai/DeepSeek-V3.2",
    threshold: float = 0.5,
    validation_window: int = 30,
    batch_size: int = 10,
    bl_delay: bool = True,
    output_file: Optional[str] = None
):
    """
    Run LLM-based simulation (replacement for MOA Java simulation).
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        train_path: Path to training data files
        test_path: Path to test data files (optional)
        iterations: Number of iterations (days) to simulate
        api_key: SiliconFlow API key
        model: Model name
        threshold: Classification threshold
        validation_window: Window size for delay evaluation
        batch_size: Batch size for API calls
        bl_delay: Enable delay evaluation
        output_file: File to save results
    """
    # Initialize predictor
    predictor = LLMPredictor(
        api_key=api_key,
        model=model,
        threshold=threshold,
        validation_window=validation_window,
        batch_size=batch_size,
        bl_delay=bl_delay
    )
    
    predictor.init_paths(train_path, test_path)
    
    # Parse start date
    cur_date = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
    predictor.set_date(cur_date)
    
    logger.info(f"Starting simulation from {start_date}")
    logger.info(f"Train path: {train_path}")
    logger.info(f"Test path: {test_path}")
    
    # Warm-up period for delay evaluation
    if bl_delay:
        logger.info(f"Warm-up period: {validation_window} days")
        for i in range(validation_window):
            date_str = cur_date.isoformat()
            logger.info(f"Warm-up: {date_str}")
            
            try:
                train_data = predictor.load_data_file(date_str, 'train')
                predictor.process_test_data(train_data)
            except FileNotFoundError:
                logger.warning(f"Data not found for {date_str}")
            
            cur_date += datetime.timedelta(days=1)
            predictor.set_date(cur_date)
            predictor.iteration += 1
    
    # Main simulation loop
    all_results = []
    
    for i in range(iterations):
        date_str = cur_date.isoformat()
        logger.info(f"Iteration {i+1}/{iterations}: {date_str}")
        
        predictor.reset_local_evaluator()
        
        try:
            # Load and process test data
            if test_path:
                test_data = predictor.load_data_file(date_str, 'test')
                metrics = predictor.process_test_data(test_data)
            else:
                # Use training data as test if no separate test set
                train_data = predictor.load_data_file(date_str, 'train')
                metrics = predictor.process_test_data(train_data)
            
            # Delay evaluation
            if bl_delay:
                delay_metrics = predictor.delay_evaluate()
            
            # Get and print metrics
            predictor.print_metrics()
            
            # Store results
            result = {
                'date': date_str,
                'iteration': predictor.iteration,
                'metrics': predictor.get_metrics()
            }
            all_results.append(result)
            
        except FileNotFoundError as e:
            logger.warning(f"Data not found: {e}")
        except Exception as e:
            logger.error(f"Error processing {date_str}: {e}")
        
        # Move to next day
        cur_date += datetime.timedelta(days=1)
        predictor.set_date(cur_date)
        predictor.iteration += 1
    
    # Save results
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        logger.info(f"Results saved to {output_file}")
    
    # Final metrics
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    predictor.print_metrics()
    
    return all_results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='LLM-based Disk Failure Prediction')
    parser.add_argument('-s', '--start-date', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('-p', '--train-path', required=True, help='Training data path')
    parser.add_argument('-t', '--test-path', help='Test data path')
    parser.add_argument('-i', '--iterations', type=int, default=30, help='Number of iterations')
    parser.add_argument('-k', '--api-key', help='SiliconFlow API key')
    parser.add_argument('-m', '--model', default='deepseek-ai/DeepSeek-V3.2', help='Model name')
    parser.add_argument('--threshold', type=float, default=0.5, help='Classification threshold')
    parser.add_argument('--batch-size', type=int, default=10, help='Batch size for API calls')
    parser.add_argument('--no-delay', action='store_true', help='Disable delay evaluation')
    parser.add_argument('-o', '--output', help='Output file for results')
    
    args = parser.parse_args()
    
    run_simulation(
        start_date=args.start_date,
        train_path=args.train_path,
        test_path=args.test_path,
        iterations=args.iterations,
        api_key=args.api_key,
        model=args.model,
        threshold=args.threshold,
        batch_size=args.batch_size,
        bl_delay=not args.no_delay,
        output_file=args.output
    )
