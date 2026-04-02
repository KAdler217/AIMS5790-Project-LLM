#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluator Module

Implements evaluation metrics for disk failure prediction.
Matches the evaluation metrics used in original MOA implementation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import json


class ClassificationEvaluator:
    """
    Classification performance evaluator.
    Computes: FP, FPR, F1-score, Precision, Recall
    """
    
    def __init__(self, threshold: float = 0.5, num_classes: int = 2):
        self.threshold = threshold
        self.num_classes = num_classes
        
        # Confusion matrix components
        self.tp = 0  # True Positives
        self.fp = 0  # False Positives
        self.tn = 0  # True Negatives
        self.fn = 0  # False Negatives
        
        # Tracking
        self.total_predictions = 0
        self.total_positives = 0  # Actual positives
        self.total_negatives = 0  # Actual negatives
        
        # Per-disk tracking for delay evaluation
        self.disk_predictions = defaultdict(list)  # serial_number -> list of predictions
        self.disk_actuals = defaultdict(list)      # serial_number -> list of actuals
        
    def add_result(self, actual: int, predicted_prob: float, serial_number: str = None):
        """
        Add a prediction result.
        
        Args:
            actual: Actual class (0 or 1)
            predicted_prob: Predicted probability of failure (0.0-1.0)
            serial_number: Disk serial number for tracking
        """
        predicted = 1 if predicted_prob >= self.threshold else 0
        
        # Update confusion matrix
        if actual == 1 and predicted == 1:
            self.tp += 1
        elif actual == 0 and predicted == 1:
            self.fp += 1
        elif actual == 0 and predicted == 0:
            self.tn += 1
        elif actual == 1 and predicted == 0:
            self.fn += 1
        
        self.total_predictions += 1
        if actual == 1:
            self.total_positives += 1
        else:
            self.total_negatives += 1
        
        # Track per-disk
        if serial_number:
            self.disk_predictions[serial_number].append(predicted_prob)
            self.disk_actuals[serial_number].append(actual)
    
    def add_batch_results(self, actuals: List[int], predicted_probs: List[float], 
                         serial_numbers: List[str] = None):
        """Add multiple results at once."""
        for i, actual in enumerate(actuals):
            pred_prob = predicted_probs[i] if i < len(predicted_probs) else 0.5
            sn = serial_numbers[i] if serial_numbers and i < len(serial_numbers) else None
            self.add_result(actual, pred_prob, sn)
    
    def get_metrics(self) -> Dict[str, float]:
        """
        Calculate evaluation metrics.
        
        Returns:
            Dictionary with metrics
        """
        metrics = {}
        
        # False Positives
        metrics['FP'] = float(self.fp)
        
        # False Positive Rate = FP / (FP + TN)
        if (self.fp + self.tn) > 0:
            metrics['FPR'] = self.fp / (self.fp + self.tn)
        else:
            metrics['FPR'] = 0.0
        
        # Precision = TP / (TP + FP)
        if (self.tp + self.fp) > 0:
            metrics['Precision'] = self.tp / (self.tp + self.fp)
        else:
            metrics['Precision'] = 0.0
        
        # Recall = TP / (TP + FN)
        if (self.tp + self.fn) > 0:
            metrics['Recall'] = self.tp / (self.tp + self.fn)
        else:
            metrics['Recall'] = 0.0
        
        # F1-score = 2 * (Precision * Recall) / (Precision + Recall)
        if (metrics['Precision'] + metrics['Recall']) > 0:
            metrics['F1-score'] = 2 * (metrics['Precision'] * metrics['Recall']) / \
                                 (metrics['Precision'] + metrics['Recall'])
        else:
            metrics['F1-score'] = 0.0
        
        # Additional metrics
        metrics['Accuracy'] = (self.tp + self.tn) / self.total_predictions if self.total_predictions > 0 else 0.0
        metrics['TP'] = float(self.tp)
        metrics['FN'] = float(self.fn)
        metrics['TN'] = float(self.tn)
        metrics['Total'] = float(self.total_predictions)
        
        return metrics
    
    def reset(self):
        """Reset all counters."""
        self.tp = self.fp = self.tn = self.fn = 0
        self.total_predictions = 0
        self.total_positives = 0
        self.total_negatives = 0
        self.disk_predictions.clear()
        self.disk_actuals.clear()
    
    def __str__(self) -> str:
        """String representation of metrics."""
        metrics = self.get_metrics()
        return (
            f"FP: {metrics['FP']:.2f}, "
            f"FPR: {metrics['FPR']:.6f}, "
            f"F1: {metrics['F1-score']:.6f}, "
            f"Precision: {metrics['Precision']:.6f}, "
            f"Recall: {metrics['Recall']:.6f}"
        )


class RegressionEvaluator:
    """
    Regression performance evaluator for predicting days until failure.
    """
    
    def __init__(self):
        self.errors = []
        self.abs_errors = []
        self.squared_errors = []
        
        self.actuals = []
        self.predictions = []
        
    def add_result(self, actual: float, predicted: float):
        """Add a regression result."""
        error = predicted - actual
        self.errors.append(error)
        self.abs_errors.append(abs(error))
        self.squared_errors.append(error ** 2)
        
        self.actuals.append(actual)
        self.predictions.append(predicted)
    
    def get_metrics(self) -> Dict[str, float]:
        """Calculate regression metrics."""
        metrics = {}
        
        if len(self.errors) == 0:
            return {
                'MAE': 0.0,
                'RMSE': 0.0,
                'Mean': 0.0,
                'Std': 0.0,
                'Max': 0.0,
                'Min': 0.0
            }
        
        # Mean Absolute Error
        metrics['MAE'] = np.mean(self.abs_errors)
        
        # Root Mean Squared Error
        metrics['RMSE'] = np.sqrt(np.mean(self.squared_errors))
        
        # Error statistics
        metrics['Mean'] = np.mean(self.errors)
        metrics['Std'] = np.std(self.errors)
        metrics['Max'] = np.max(self.errors)
        metrics['Min'] = np.min(self.errors)
        
        return metrics
    
    def reset(self):
        """Reset all data."""
        self.errors.clear()
        self.abs_errors.clear()
        self.squared_errors.clear()
        self.actuals.clear()
        self.predictions.clear()
    
    def __str__(self) -> str:
        """String representation of metrics."""
        metrics = self.get_metrics()
        return (
            f"MAE: {metrics['MAE']:.6f}, "
            f"RMSE: {metrics['RMSE']:.6f}, "
            f"Mean: {metrics['Mean']:.6f}, "
            f"Std: {metrics['Std']:.6f}"
        )


class DelayEvaluator:
    """
    Delay evaluator for early failure prediction.
    Tracks predictions over a validation window.
    """
    
    def __init__(self, validation_window: int = 30):
        self.validation_window = validation_window
        self.disk_instances = defaultdict(list)  # serial_number -> list of (actual, predicted_prob, day)
        self.results = []
        
    def add_instance(self, serial_number: str, actual: int, predicted_prob: float, day: int):
        """Add an instance for a disk."""
        self.disk_instances[serial_number].append({
            'actual': actual,
            'predicted_prob': predicted_prob,
            'day': day
        })
        
        # Evaluate if we have a window of data
        instances = self.disk_instances[serial_number]
        if len(instances) >= self.validation_window:
            self._evaluate_disk(serial_number)
    
    def _evaluate_disk(self, serial_number: str):
        """Evaluate a disk's prediction window."""
        instances = self.disk_instances[serial_number]
        
        # Check if disk actually failed in this window
        failed = any(inst['actual'] == 1 for inst in instances)
        
        # Get maximum predicted probability in window
        max_pred_prob = max(inst['predicted_prob'] for inst in instances)
        
        # Check if we predicted failure (any prediction above threshold)
        predicted_failure = max_pred_prob >= 0.5
        
        # Calculate days before failure (if failed)
        days_before_failure = None
        if failed:
            # Find first failure day
            for i, inst in enumerate(instances):
                if inst['actual'] == 1:
                    days_before_failure = len(instances) - i
                    break
        
        self.results.append({
            'serial_number': serial_number,
            'failed': failed,
            'predicted_failure': predicted_failure,
            'max_predicted_prob': max_pred_prob,
            'days_before_failure': days_before_failure
        })
        
        # Clear evaluated instances
        self.disk_instances[serial_number] = []
    
    def get_metrics(self) -> Dict[str, Any]:
        """Calculate delay evaluation metrics."""
        if not self.results:
            return {'count': 0}
        
        df = pd.DataFrame(self.results)
        
        metrics = {}
        metrics['count'] = len(df)
        
        # True Positives: Actually failed and predicted failure
        tp = len(df[(df['failed'] == True) & (df['predicted_failure'] == True)])
        
        # False Positives: Not failed but predicted failure
        fp = len(df[(df['failed'] == False) & (df['predicted_failure'] == True)])
        
        # False Negatives: Failed but not predicted
        fn = len(df[(df['failed'] == True) & (df['predicted_failure'] == False)])
        
        # True Negatives: Not failed and not predicted
        tn = len(df[(df['failed'] == False) & (df['predicted_failure'] == False)])
        
        metrics['TP'] = float(tp)
        metrics['FP'] = float(fp)
        metrics['FN'] = float(fn)
        metrics['TN'] = float(tn)
        
        # Calculate standard metrics
        if tp + fp > 0:
            metrics['Precision'] = tp / (tp + fp)
        else:
            metrics['Precision'] = 0.0
            
        if tp + fn > 0:
            metrics['Recall'] = tp / (tp + fn)
        else:
            metrics['Recall'] = 0.0
            
        if metrics['Precision'] + metrics['Recall'] > 0:
            metrics['F1-score'] = 2 * (metrics['Precision'] * metrics['Recall']) / \
                                 (metrics['Precision'] + metrics['Recall'])
        else:
            metrics['F1-score'] = 0.0
        
        # Average days before failure (for true positives)
        true_positives = df[(df['failed'] == True) & (df['predicted_failure'] == True)]
        if len(true_positives) > 0:
            metrics['avg_days_before_failure'] = true_positives['days_before_failure'].mean()
        else:
            metrics['avg_days_before_failure'] = 0.0
        
        return metrics
    
    def reset(self):
        """Reset evaluator."""
        self.disk_instances.clear()
        self.results.clear()


def format_metrics_table(metrics: Dict[str, float]) -> str:
    """Format metrics as a nice table."""
    lines = []
    lines.append("-" * 60)
    lines.append(f"{'Metric':<20} {'Value':>15}")
    lines.append("-" * 60)
    
    for key, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"{key:<20} {value:>15.6f}")
        else:
            lines.append(f"{key:<20} {value:>15}")
    
    lines.append("-" * 60)
    return "\n".join(lines)


def save_metrics_to_file(metrics: Dict[str, Any], output_file: str):
    """Save metrics to JSON file."""
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=2)
