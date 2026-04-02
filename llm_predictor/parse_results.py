#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Parse and format LLM prediction results.
"""

import json
import sys
from pathlib import Path
import pandas as pd


def parse_results_file(filepath: str):
    """Parse results JSON file and display metrics."""
    
    with open(filepath, 'r') as f:
        results = json.load(f)
    
    if not results:
        print("No results found in file.")
        return
    
    # Extract final metrics
    print("\n" + "="*70)
    print("FINAL EVALUATION METRICS")
    print("="*70)
    
    if isinstance(results, list) and len(results) > 0:
        # Get last iteration's metrics
        last_result = results[-1]
        metrics = last_result.get('metrics', {})
        
        print(f"\nDate: {last_result.get('date', 'N/A')}")
        print(f"Iteration: {last_result.get('iteration', 'N/A')}")
        
        global_metrics = metrics.get('global', {})
        
        print("\n" + "-"*70)
        print(f"{'Metric':<20} {'Value':>15} {'Description':<30}")
        print("-"*70)
        
        metric_descriptions = {
            'FP': 'False Positives - incorrectly predicted failures',
            'FPR': 'False Positive Rate - FP / (FP + TN)',
            'F1-score': 'F1 Score - harmonic mean of precision and recall',
            'Precision': 'Precision - TP / (TP + FP)',
            'Recall': 'Recall - TP / (TP + FN)',
            'Accuracy': 'Accuracy - (TP + TN) / Total',
            'TP': 'True Positives - correctly predicted failures',
            'FN': 'False Negatives - missed failures',
            'TN': 'True Negatives - correctly predicted non-failures'
        }
        
        for key in ['FP', 'FPR', 'F1-score', 'Precision', 'Recall', 'Accuracy']:
            if key in global_metrics:
                value = global_metrics[key]
                desc = metric_descriptions.get(key, '')
                if isinstance(value, float):
                    print(f"{key:<20} {value:>15.6f} {desc:<30}")
                else:
                    print(f"{key:<20} {value:>15} {desc:<30}")
        
        print("-"*70)
        
        # Create summary table for all iterations
        print("\n" + "="*70)
        print("ITERATION SUMMARY")
        print("="*70)
        
        summary_data = []
        for r in results:
            m = r.get('metrics', {}).get('global', {})
            summary_data.append({
                'Date': r.get('date', ''),
                'FP': m.get('FP', 0),
                'FPR': f"{m.get('FPR', 0):.4f}",
                'F1': f"{m.get('F1-score', 0):.4f}",
                'Precision': f"{m.get('Precision', 0):.4f}",
                'Recall': f"{m.get('Recall', 0):.4f}"
            })
        
        df = pd.DataFrame(summary_data)
        print("\n" + df.to_string(index=False))
        
        # Calculate averages
        print("\n" + "="*70)
        print("AVERAGE METRICS ACROSS ALL ITERATIONS")
        print("="*70)
        
        avg_data = {
            'FP': [],
            'FPR': [],
            'F1-score': [],
            'Precision': [],
            'Recall': []
        }
        
        for r in results:
            m = r.get('metrics', {}).get('global', {})
            for key in avg_data.keys():
                if key in m:
                    avg_data[key].append(m[key])
        
        print(f"{'Metric':<20} {'Average':>15} {'Std Dev':>15}")
        print("-"*70)
        for key, values in avg_data.items():
            if values:
                avg = sum(values) / len(values)
                std = (sum((x - avg) ** 2 for x in values) / len(values)) ** 0.5
                print(f"{key:<20} {avg:>15.6f} {std:>15.6f}")
        
        print("="*70)


def create_comparison_table(result_files: list):
    """Create comparison table for multiple experiments."""
    
    print("\n" + "="*80)
    print("COMPARISON OF DIFFERENT MODELS/CONFIGURATIONS")
    print("="*80)
    
    comparison_data = []
    
    for filepath in result_files:
        path = Path(filepath)
        name = path.parent.name or path.stem
        
        try:
            with open(filepath, 'r') as f:
                results = json.load(f)
            
            if results:
                last_result = results[-1]
                metrics = last_result.get('metrics', {}).get('global', {})
                
                comparison_data.append({
                    'Experiment': name,
                    'FP': f"{metrics.get('FP', 0):.2f}",
                    'FPR': f"{metrics.get('FPR', 0):.4f}",
                    'F1': f"{metrics.get('F1-score', 0):.4f}",
                    'Precision': f"{metrics.get('Precision', 0):.4f}",
                    'Recall': f"{metrics.get('Recall', 0):.4f}"
                })
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
    
    if comparison_data:
        df = pd.DataFrame(comparison_data)
        print("\n" + df.to_string(index=False))
    
    print("="*80)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python parse_results.py <results.json> [results2.json ...]")
        print("\nIf multiple files provided, creates a comparison table.")
        sys.exit(1)
    
    if len(sys.argv) == 2:
        # Single file - parse and display
        parse_results_file(sys.argv[1])
    else:
        # Multiple files - comparison
        create_comparison_table(sys.argv[1:])
