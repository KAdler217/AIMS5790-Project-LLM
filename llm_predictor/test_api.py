#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script to verify SiliconFlow API connection and LLM predictor setup.
"""

import os
import sys
from llm_client import SiliconFlowClient, DiskFailurePromptBuilder


def test_api_connection():
    """Test API connection."""
    print("="*60)
    print("Testing SiliconFlow API Connection")
    print("="*60)
    
    api_key = os.getenv('SILICONFLOW_API_KEY')
    if not api_key:
        print("❌ Error: SILICONFLOW_API_KEY environment variable not set")
        print("\nPlease set it with:")
        print("  export SILICONFLOW_API_KEY='your-api-key-here'")
        return False
    
    print(f"✓ API key found (length: {len(api_key)})")
    
    try:
        client = SiliconFlowClient(api_key=api_key)
        print(f"✓ Client initialized")
        print(f"  - Base URL: {client.base_url}")
        print(f"  - Model: {client.model}")
        
        # Test simple prompt
        print("\nSending test prompt...")
        response = client.predict(
            "Hello! Please respond with a simple 'API test successful' message.",
            max_tokens=50
        )
        
        print("✓ API call successful")
        print(f"\nResponse:")
        print(f"  {response['text'][:200]}...")
        print(f"\nUsage:")
        print(f"  {response.get('usage', {})}")
        
        return True
        
    except Exception as e:
        print(f"❌ API call failed: {e}")
        return False


def test_disk_prediction():
    """Test disk failure prediction prompt."""
    print("\n" + "="*60)
    print("Testing Disk Failure Prediction")
    print("="*60)
    
    api_key = os.getenv('SILICONFLOW_API_KEY')
    if not api_key:
        print("❌ Error: SILICONFLOW_API_KEY not set")
        return False
    
    try:
        client = SiliconFlowClient(api_key=api_key)
        prompt_builder = DiskFailurePromptBuilder()
        
        # Sample disk data
        disk_data = {
            'serial_number': 'TEST123',
            'smart_1_normalized': 114,
            'smart_5_raw': 0,
            'smart_5_normalized': 100,
            'smart_9_raw': 1234,
            'smart_187_raw': 0,
            'smart_197_raw': 0,
            'smart_197_normalized': 100,
            'failure': 0
        }
        
        print("Sample disk data:")
        for k, v in disk_data.items():
            print(f"  {k}: {v}")
        
        print("\nBuilding prediction prompt...")
        prompt = prompt_builder.build_prediction_prompt(disk_data)
        
        print("Sending to LLM...")
        response = client.predict(
            prompt,
            system_message=prompt_builder.SYSTEM_MESSAGE,
            temperature=0.1,
            max_tokens=500
        )
        
        print("✓ Prediction received")
        print(f"\nRaw response:")
        print(f"  {response['text']}")
        
        # Parse response
        result = prompt_builder.parse_prediction_response(response['text'])
        print(f"\nParsed result:")
        print(f"  Failure Probability: {result.get('failure_probability', 'N/A')}")
        print(f"  Risk Level: {result.get('risk_level', 'N/A')}")
        print(f"  Confidence: {result.get('confidence', 'N/A')}")
        print(f"  Reasoning: {result.get('reasoning', 'N/A')[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Prediction test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_loading():
    """Test data loading from pyloader output."""
    print("\n" + "="*60)
    print("Testing Data Loading")
    print("="*60)
    
    try:
        from data_loader import DataLoader
        
        # Check if sample data exists
        import os as os_module
        train_path = "./pyloader/train/"
        
        if not os_module.path.exists(train_path):
            print(f"⚠ Training path not found: {train_path}")
            print("  Run pyloader first to generate data files")
            return None
        
        # Find first data file
        files = os_module.listdir(train_path)
        data_files = [f for f in files if f.endswith('.arff') or f.endswith('.csv')]
        
        if not data_files:
            print("⚠ No data files found in train path")
            return None
        
        test_file = os_module.path.join(train_path, data_files[0])
        print(f"Loading: {test_file}")
        
        loader = DataLoader(test_file)
        df = loader.load()
        
        print(f"✓ Data loaded successfully")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Sample data:")
        print(df.head(2).to_string())
        
        return True
        
    except Exception as e:
        print(f"❌ Data loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    print("\n" + "="*60)
    print("LLM Predictor Test Suite")
    print("="*60)
    
    results = {
        'API Connection': test_api_connection(),
        'Disk Prediction': test_disk_prediction(),
        'Data Loading': test_data_loading()
    }
    
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "❌ FAIL" if result is False else "⚠ SKIP"
        print(f"{test_name:<20} {status}")
    
    all_passed = all(r is True for r in results.values() if r is not None)
    
    if all_passed:
        print("\n✓ All tests passed! Ready to run prediction.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        sys.exit(1)
