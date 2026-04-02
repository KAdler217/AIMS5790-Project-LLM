#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM API Client Module

Provides unified interface for calling SiliconFlow API with DeepSeek model.
"""

import os
import json
import time
import requests
from typing import Dict, Any, Optional, List, Union
from abc import ABC, abstractmethod
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        self.api_key = api_key
        self.max_retries = kwargs.get('max_retries', 3)
        self.retry_delay = kwargs.get('retry_delay', 1.0)
        self.timeout = kwargs.get('timeout', 60)
        
    @abstractmethod
    def predict(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Send prediction request to LLM.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional parameters
            
        Returns:
            Response dictionary with predictions
        """
        pass
    
    @abstractmethod
    def batch_predict(self, prompts: List[str], **kwargs) -> List[Dict[str, Any]]:
        """
        Send batch prediction requests.
        
        Args:
            prompts: List of input prompts
            **kwargs: Additional parameters
            
        Returns:
            List of response dictionaries
        """
        pass
    
    def _retry_request(self, func, *args, **kwargs):
        """Retry failed requests."""
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise


class SiliconFlowClient(LLMClient):
    """
    Client for SiliconFlow (硅基流动) API.
    API Docs: https://docs.siliconflow.cn/
    
    Compatible with OpenAI API format.
    Default model: deepseek-ai/DeepSeek-V3.2
    """
    
    def __init__(self, api_key: Optional[str] = None, 
                 model: str = "deepseek-ai/DeepSeek-V3.2",
                 base_url: str = "https://api.siliconflow.cn/v1",
                 **kwargs):
        """
        Initialize SiliconFlow client.
        
        Args:
            api_key: SiliconFlow API key (or set SILICONFLOW_API_KEY env var)
            model: Model name, default is deepseek-ai/DeepSeek-V3.2
            base_url: API base URL
            **kwargs: Additional parameters (max_retries, retry_delay, timeout)
        """
        super().__init__(api_key, **kwargs)
        self.api_key = api_key or os.getenv('SILICONFLOW_API_KEY')
        if not self.api_key:
            raise ValueError("API key is required. Set SILICONFLOW_API_KEY environment variable or pass api_key parameter.")
        
        self.base_url = base_url
        self.model = model
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
    def predict(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Send prediction request to SiliconFlow API.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional parameters
                - system_message: System message for context
                - temperature: Sampling temperature (default: 0.1)
                - max_tokens: Max tokens in response (default: 500)
                - top_p: Top-p sampling (default: 0.9)
                
        Returns:
            Response dictionary with predictions
        """
        def _request():
            url = f"{self.base_url}/chat/completions"
            
            messages = [{"role": "user", "content": prompt}]
            
            # Support for system message
            system_message = kwargs.get('system_message')
            if system_message:
                messages.insert(0, {"role": "system", "content": system_message})
            
            data = {
                "model": self.model,
                "messages": messages,
                "temperature": kwargs.get('temperature', 0.1),
                "max_tokens": kwargs.get('max_tokens', 500),
                "top_p": kwargs.get('top_p', 0.9),
                "stream": False
            }
            
            response = requests.post(url, headers=self.headers, json=data, 
                                    timeout=self.timeout)
            response.raise_for_status()
            
            result = response.json()
            return self._parse_response(result)
        
        return self._retry_request(_request)
    
    def batch_predict(self, prompts: List[str], **kwargs) -> List[Dict[str, Any]]:
        """
        Send batch prediction requests.
        
        Args:
            prompts: List of input prompts
            **kwargs: Additional parameters
            
        Returns:
            List of response dictionaries
        """
        results = []
        for prompt in prompts:
            result = self.predict(prompt, **kwargs)
            results.append(result)
            # Small delay to avoid rate limiting
            time.sleep(0.1)
        return results
    
    def _parse_response(self, response: Dict) -> Dict[str, Any]:
        """
        Parse SiliconFlow API response.
        
        Args:
            response: Raw API response
            
        Returns:
            Parsed response dictionary
        """
        if 'choices' in response and len(response['choices']) > 0:
            content = response['choices'][0]['message']['content']
            return {
                'text': content,
                'raw_response': response,
                'usage': response.get('usage', {}),
                'model': response.get('model', self.model)
            }
        return {
            'text': '', 
            'raw_response': response,
            'usage': {},
            'model': self.model
        }


class DiskFailurePromptBuilder:
    """
    Build prompts for disk failure prediction using SMART data.
    """
    
    SYSTEM_MESSAGE = """You are an expert in disk failure prediction using SMART (Self-Monitoring, Analysis and Reporting Technology) data.
Your task is to analyze disk SMART attributes and predict the probability of disk failure.

Key SMART attributes interpretation:
- smart_5_raw / Reallocated_Sector_Ct: Count of reallocated sectors. > 0 indicates potential failure
- smart_187_raw / Reported_Uncorrectable_Errors: Uncorrectable errors. > 0 is critical
- smart_197_raw / Current_Pending_Sector_Ct: Pending sectors to be remapped. > 0 is concerning
- smart_198_raw / Offline_Uncorrectable: Uncorrectable sectors found during offline scan. > 0 is critical
- smart_1_normalized / Raw_Read_Error_Rate: Lower normalized value indicates more errors
- smart_9_raw / Power_On_Hours: Total power-on time
- smart_12_raw / Power_Cycle_Count: Number of power cycles
- smart_197_normalized: Normalized pending sector count, lower is worse

Failure risk indicators (in order of severity):
1. CRITICAL: smart_5_raw > 0 OR smart_187_raw > 0 OR smart_198_raw > 0
2. HIGH: smart_197_raw > 0
3. MEDIUM: smart_1_normalized < 100 OR other normalized values trending down
4. LOW: All attributes within normal ranges

Output format - return ONLY a JSON object:
{
    "failure_probability": <float 0.0-1.0>,
    "risk_level": "low|medium|high|critical",
    "reasoning": "<brief explanation based on SMART attributes>",
    "confidence": <float 0.0-1.0>
}
"""

    def build_prediction_prompt(self, disk_data: Dict[str, Any], 
                               historical_data: Optional[List[Dict]] = None) -> str:
        """
        Build prompt for single disk prediction.
        
        Args:
            disk_data: Dictionary with SMART attributes
            historical_data: Optional list of historical readings
            
        Returns:
            Formatted prompt string
        """
        prompt = "Analyze the following disk SMART data and predict failure probability:\n\n"
        
        # Current data
        prompt += "Current SMART readings:\n"
        prompt += json.dumps(disk_data, indent=2, ensure_ascii=False)
        prompt += "\n\n"
        
        # Historical context if available
        if historical_data and len(historical_data) > 0:
            prompt += f"Historical trend ({len(historical_data)} previous readings):\n"
            # Show only critical attributes for history
            critical_attrs = ['smart_5_raw', 'smart_187_raw', 'smart_197_raw', 'smart_198_raw', 
                             'smart_1_normalized', 'smart_197_normalized']
            history_summary = []
            for h in historical_data[-3:]:  # Last 3 readings
                summary = {k: v for k, v in h.items() if k in critical_attrs and k in h}
                history_summary.append(summary)
            prompt += json.dumps(history_summary, indent=2, ensure_ascii=False)
            prompt += "\n\n"
        
        prompt += "Provide your prediction in the specified JSON format."
        
        return prompt
    
    def build_batch_prompt(self, disks_data: List[Dict[str, Any]]) -> str:
        """
        Build prompt for batch prediction.
        
        Args:
            disks_data: List of dictionaries with SMART attributes
            
        Returns:
            Formatted prompt string
        """
        prompt = f"Analyze the following {len(disks_data)} disks' SMART data and predict failure probability for each:\n\n"
        
        for i, disk in enumerate(disks_data):
            sn = disk.get('serial_number', disk.get('disk_id', f'N/A'))
            prompt += f"Disk {i+1} (SN: {sn}):\n"
            # Filter to relevant SMART attributes
            smart_data = {k: v for k, v in disk.items() if 'smart' in str(k).lower()}
            prompt += json.dumps(smart_data, indent=2, ensure_ascii=False)
            prompt += "\n\n"
        
        prompt += """Provide your predictions in the following JSON format:
{
    "predictions": [
        {
            "disk_index": 1,
            "failure_probability": 0.0-1.0,
            "risk_level": "low|medium|high|critical",
            "reasoning": "brief explanation",
            "confidence": 0.0-1.0
        }
    ]
}
"""
        return prompt
    
    def parse_prediction_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse LLM response to extract prediction.
        
        Args:
            response_text: Raw response text from LLM
            
        Returns:
            Parsed prediction dictionary
        """
        import re
        
        try:
            # Try to find JSON in the response
            json_match = None
            
            # Try code block first
            code_block = re.search(r'```(?:json)?\s*(.*?)\s*```', response_text, re.DOTALL)
            if code_block:
                json_match = code_block.group(1)
            else:
                # Try to find JSON object directly (handle nested braces)
                # Find the outermost JSON object
                depth = 0
                start = -1
                for i, char in enumerate(response_text):
                    if char == '{':
                        if depth == 0:
                            start = i
                        depth += 1
                    elif char == '}':
                        depth -= 1
                        if depth == 0 and start != -1:
                            json_match = response_text[start:i+1]
                            break
            
            if json_match:
                result = json.loads(json_match)
                return result
            else:
                # Try to parse entire response
                return json.loads(response_text)
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Response text: {response_text[:500]}")
            # Return default prediction
            return {
                'failure_probability': 0.5,
                'risk_level': 'unknown',
                'reasoning': f'Failed to parse response: {str(e)[:100]}',
                'confidence': 0.0,
                'raw_response': response_text[:1000]
            }
    
    def parse_batch_response(self, response_text: str) -> List[Dict[str, Any]]:
        """
        Parse batch prediction response.
        
        Args:
            response_text: Raw response text
            
        Returns:
            List of prediction dictionaries
        """
        result = self.parse_prediction_response(response_text)
        if 'predictions' in result:
            return result['predictions']
        return [result]


def create_llm_client(provider: str = "siliconflow", **kwargs) -> LLMClient:
    """
    Factory function to create LLM client.
    
    Args:
        provider: Provider name ('siliconflow')
        **kwargs: Additional configuration
        
    Returns:
        LLMClient instance
    """
    provider = provider.lower()
    
    if provider == 'siliconflow':
        return SiliconFlowClient(**kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}. Supported: siliconflow")
