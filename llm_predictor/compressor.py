#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data Compressor Module

Compresses data for efficient API transmission to LLM.
Supports multiple compression methods.
"""

import gzip
import zlib
import json
import base64
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from io import StringIO


class DataCompressor:
    """
    Compress data using various methods for efficient API transmission.
    """
    
    def __init__(self, method: str = 'gzip', level: int = 6):
        """
        Initialize compressor.
        
        Args:
            method: Compression method ('gzip', 'zlib', 'none')
            level: Compression level (1-9, higher = better compression but slower)
        """
        self.method = method.lower()
        self.level = level
        
    def compress(self, data: Any) -> bytes:
        """
        Compress data.
        
        Args:
            data: Data to compress (string, dict, DataFrame)
            
        Returns:
            Compressed bytes
        """
        # Convert to string if needed
        if isinstance(data, pd.DataFrame):
            str_data = self._dataframe_to_string(data)
        elif isinstance(data, dict):
            str_data = json.dumps(data, ensure_ascii=False)
        elif isinstance(data, (list, np.ndarray)):
            str_data = json.dumps(data.tolist() if isinstance(data, np.ndarray) else data)
        else:
            str_data = str(data)
        
        raw_bytes = str_data.encode('utf-8')
        
        if self.method == 'gzip':
            return gzip.compress(raw_bytes, compresslevel=self.level)
        elif self.method == 'zlib':
            return zlib.compress(raw_bytes, level=self.level)
        elif self.method == 'none':
            return raw_bytes
        else:
            raise ValueError(f"Unknown compression method: {self.method}")
    
    def decompress(self, compressed: bytes) -> str:
        """
        Decompress data.
        
        Args:
            compressed: Compressed bytes
            
        Returns:
            Decompressed string
        """
        if self.method == 'gzip':
            return gzip.decompress(compressed).decode('utf-8')
        elif self.method == 'zlib':
            return zlib.decompress(compressed).decode('utf-8')
        elif self.method == 'none':
            return compressed.decode('utf-8')
        else:
            raise ValueError(f"Unknown compression method: {self.method}")
    
    def compress_to_base64(self, data: Any) -> str:
        """
        Compress and encode to base64 string for API transmission.
        
        Args:
            data: Data to compress
            
        Returns:
            Base64 encoded compressed string
        """
        compressed = self.compress(data)
        return base64.b64encode(compressed).decode('ascii')
    
    def decompress_from_base64(self, b64_string: str) -> str:
        """
        Decode from base64 and decompress.
        
        Args:
            b64_string: Base64 encoded compressed string
            
        Returns:
            Decompressed string
        """
        compressed = base64.b64decode(b64_string)
        return self.decompress(compressed)
    
    def _dataframe_to_string(self, df: pd.DataFrame) -> str:
        """Convert DataFrame to compact string representation."""
        # Use CSV format for compactness
        output = StringIO()
        df.to_csv(output, index=False)
        return output.getvalue()
    
    def compress_dataframe_summary(self, df: pd.DataFrame, max_rows: int = 100) -> Dict[str, Any]:
        """
        Create a compressed summary of a DataFrame for LLM context.
        
        Args:
            df: DataFrame to summarize
            max_rows: Maximum number of sample rows to include
            
        Returns:
            Dictionary with summary statistics and sample data
        """
        summary = {
            'shape': list(df.shape),
            'columns': list(df.columns),
            'dtypes': {col: str(df[col].dtype) for col in df.columns},
            'statistics': {}
        }
        
        # Calculate statistics for numeric columns
        for col in df.select_dtypes(include=[np.number]).columns:
            stats = {
                'mean': float(df[col].mean()) if not pd.isna(df[col].mean()) else None,
                'std': float(df[col].std()) if not pd.isna(df[col].std()) else None,
                'min': float(df[col].min()) if not pd.isna(df[col].min()) else None,
                'max': float(df[col].max()) if not pd.isna(df[col].max()) else None,
                'median': float(df[col].median()) if not pd.isna(df[col].median()) else None,
                'null_count': int(df[col].isnull().sum())
            }
            summary['statistics'][col] = {k: v for k, v in stats.items() if v is not None}
        
        # Include sample rows (up to max_rows)
        sample_df = df.head(max_rows)
        summary['sample_data'] = sample_df.to_dict(orient='records')
        
        return summary
    
    def compress_timeseries(self, df: pd.DataFrame, time_col: Optional[str] = None, 
                           aggregation: str = 'mean') -> Dict[str, Any]:
        """
        Compress time series data by aggregation.
        
        Args:
            df: DataFrame with time series data
            time_col: Column name for time (if None, uses index)
            aggregation: Aggregation method ('mean', 'median', 'last')
            
        Returns:
            Compressed time series representation
        """
        result = {
            'type': 'timeseries',
            'shape': list(df.shape),
            'columns': list(df.columns)
        }
        
        # Group by serial number if available
        if 'serial_number' in df.columns:
            grouped = df.groupby('serial_number')
            
            aggregated = []
            for sn, group in grouped:
                group_data = {
                    'serial_number': sn,
                    'count': len(group)
                }
                
                # Aggregate numeric columns
                for col in group.select_dtypes(include=[np.number]).columns:
                    if col != 'serial_number':
                        if aggregation == 'mean':
                            group_data[col] = float(group[col].mean())
                        elif aggregation == 'median':
                            group_data[col] = float(group[col].median())
                        elif aggregation == 'last':
                            group_data[col] = float(group[col].iloc[-1])
                        elif aggregation == 'max':
                            group_data[col] = float(group[col].max())
                        elif aggregation == 'min':
                            group_data[col] = float(group[col].min())
                
                aggregated.append(group_data)
            
            result['aggregated_data'] = aggregated
        else:
            # Global aggregation
            result['aggregated_data'] = {
                col: float(df[col].mean()) 
                for col in df.select_dtypes(include=[np.number]).columns
            }
        
        return result


class BatchCompressor:
    """
    Compress data in batches for efficient processing.
    """
    
    def __init__(self, compressor: DataCompressor, batch_size: int = 1000):
        self.compressor = compressor
        self.batch_size = batch_size
    
    def compress_batches(self, df: pd.DataFrame) -> List[str]:
        """
        Split DataFrame into batches and compress each.
        
        Args:
            df: DataFrame to compress
            
        Returns:
            List of base64 compressed strings
        """
        batches = []
        for i in range(0, len(df), self.batch_size):
            batch = df.iloc[i:i+self.batch_size]
            compressed = self.compressor.compress_to_base64(batch)
            batches.append(compressed)
        return batches
    
    def create_context_prompt(self, df: pd.DataFrame, summary_only: bool = True) -> str:
        """
        Create a context prompt from DataFrame for LLM.
        
        Args:
            df: DataFrame to create context from
            summary_only: If True, only include summary statistics
            
        Returns:
            Context string for LLM prompt
        """
        if summary_only:
            summary = self.compressor.compress_dataframe_summary(df, max_rows=50)
            context = {
                'dataset_summary': summary,
                'note': 'This is a disk SMART data summary for failure prediction.'
            }
        else:
            context = {
                'data': df.head(self.batch_size).to_dict(orient='records'),
                'columns': list(df.columns),
                'note': 'This is disk SMART data for failure prediction.'
            }
        
        return json.dumps(context, indent=2, ensure_ascii=False)


# Convenience functions
def compress_for_api(data: Any, method: str = 'gzip') -> str:
    """Compress data for API transmission."""
    compressor = DataCompressor(method=method)
    return compressor.compress_to_base64(data)


def create_compact_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Create a compact summary of DataFrame."""
    compressor = DataCompressor()
    return compressor.compress_dataframe_summary(df, max_rows=100)
