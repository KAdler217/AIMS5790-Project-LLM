# LLM Predictor for StreamDFP

This module replaces the MOA Java component with a Python-based LLM (Large Language Model) prediction system using SiliconFlow API with DeepSeek-V3.2 model.

## Overview

The LLM Predictor maintains the same workflow as the original StreamDFP:
1. **Python Preprocessing** (`pyloader/`) - unchanged
2. **LLM Prediction** (`llm_predictor/`) - replaces MOA Java

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Raw Data      │ ──→ │  pyloader       │ ──→ │  llm_predictor  │
│  (SMART logs)   │     │  (preprocess)   │     │  (LLM predict)  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                              ARFF/CSV                  SiliconFlow
                            format files                 DeepSeek-V3.2
```

## File Structure

```
llm_predictor/
├── __init__.py           # Module initialization
├── config.yaml           # Configuration file
├── data_loader.py        # ARFF/CSV data loader
├── compressor.py         # Data compression for API
├── llm_client.py         # SiliconFlow API client
├── evaluator.py          # Evaluation metrics
├── predictor.py          # Main predictor class
├── parse_results.py      # Results parser
└── test_api.py           # API test script
```

## Setup

### 1. Install Dependencies

```bash
pip install pandas numpy requests pyyaml
```

### 2. Configure API Key

Set your SiliconFlow API key as an environment variable:

```bash
export SILICONFLOW_API_KEY="your-api-key-here"
```

Or modify `config.yaml` (not recommended for production).

### 3. Test API Connection

```bash
cd llm_predictor
python test_api.py
```

## Usage

### Quick Start

```bash
# Set API key
export SILICONFLOW_API_KEY="your-api-key"

# Run prediction
python llm_predictor/predictor.py \
    -s "2015-01-30" \
    -p "./pyloader/train/" \
    -t "./pyloader/test/" \
    -i 10 \
    --threshold 0.5 \
    --batch-size 10 \
    -o "output/results.json"
```

### Using Shell Script

```bash
# Make script executable
chmod +x run_hi7_llm.sh

# Set API key
export SILICONFLOW_API_KEY="your-api-key"

# Run for Hitachi HDS722020ALA330 (hi7)
./run_hi7_llm.sh
```

### Full Workflow Example

```bash
# Step 1: Preprocess data with pyloader
cd pyloader
python run.py \
    -s "2015-01-30" \
    -a 6 \
    -p "~/trace/smart/all/" \
    -r "../pyloader/train/" \
    -e "../pyloader/test/" \
    -c "features_erg/hi7_all.txt" \
    -d "HDS722020ALA330" \
    -i 10

cd ..

# Step 2: Run LLM prediction
export SILICONFLOW_API_KEY="your-api-key"

python llm_predictor/predictor.py \
    -s "2015-01-30" \
    -p "./pyloader/train/" \
    -t "./pyloader/test/" \
    -i 10 \
    -o "hi7_llm/results.json"

# Step 3: Parse results
python llm_predictor/parse_results.py hi7_llm/results.json
```

## Command Line Arguments

| Argument | Short | Description | Default |
|----------|-------|-------------|---------|
| `--start-date` | `-s` | Start date (YYYY-MM-DD) | Required |
| `--train-path` | `-p` | Training data path | Required |
| `--test-path` | `-t` | Test data path | None |
| `--iterations` | `-i` | Number of iterations (days) | 30 |
| `--api-key` | `-k` | SiliconFlow API key | env var |
| `--model` | `-m` | Model name | deepseek-ai/DeepSeek-V3.2 |
| `--threshold` | | Classification threshold | 0.5 |
| `--batch-size` | | Batch size for API calls | 10 |
| `--no-delay` | | Disable delay evaluation | False |
| `--output` | `-o` | Output file path | None |

## Configuration

Edit `config.yaml` to customize:

```yaml
api:
  provider: "siliconflow"
  model: "deepseek-ai/DeepSeek-V3.2"
  timeout: 60
  max_retries: 3

prediction:
  threshold: 0.5
  validation_window: 30
  bl_delay: true

data:
  batch_size: 10
  use_compression: true
  compression_method: "gzip"
```

## Evaluation Metrics

The LLM Predictor computes the same metrics as the original MOA implementation:

| Metric | Description |
|--------|-------------|
| **FP** | False Positives - incorrectly predicted failures |
| **FPR** | False Positive Rate - FP / (FP + TN) |
| **F1-score** | F1 Score - harmonic mean of precision and recall |
| **Precision** | Precision - TP / (TP + FP) |
| **Recall** | Recall - TP / (TP + FN) |
| **Accuracy** | Accuracy - (TP + TN) / Total |

## Data Compression

To reduce API payload size, the module supports data compression:

- **gzip**: Default, good balance of speed and compression
- **zlib**: Alternative compression method
- **none**: No compression (for debugging)

Compression is applied automatically when `use_compression: true` in config.

## Prompt Engineering

The LLM is instructed with a specialized system message for disk failure prediction:

```
You are an expert in disk failure prediction using SMART data...
```

The prompt includes:
- SMART attribute explanations
- Risk level indicators
- Required output format (JSON)

## Batch Processing

To optimize API usage, disks are processed in batches:

- Default batch size: 10 disks per API call
- Adjustable via `--batch-size` argument
- Larger batches = fewer API calls but longer response time

## Comparison with Original MOA

| Aspect | MOA Java | LLM Predictor |
|--------|----------|---------------|
| Algorithm | Adaptive Random Forest | DeepSeek-V3.2 LLM |
| Training | Incremental learning | In-context learning |
| Latency | Low (local) | Higher (API) |
| Interpretability | Limited | High (natural language reasoning) |
| Cost | Free | Per API call |

## Troubleshooting

### API Key Not Found
```
Error: SILICONFLOW_API_KEY environment variable is not set
```
**Solution**: Set the environment variable:
```bash
export SILICONFLOW_API_KEY="your-api-key"
```

### Data File Not Found
```
FileNotFoundError: Data file not found for 2015-01-30
```
**Solution**: Run pyloader first to generate data files.

### API Rate Limiting
If you encounter rate limiting, increase the retry delay in config:
```yaml
api:
  retry_delay: 2.0  # seconds between retries
```

### Out of Memory
Reduce batch size for large datasets:
```bash
python llm_predictor/predictor.py --batch-size 5 ...
```

## API Costs

SiliconFlow API charges per token. Estimated costs:

- Single disk prediction: ~500-1000 tokens
- Batch of 10 disks: ~2000-4000 tokens

Monitor your API usage in the SiliconFlow dashboard.

## Extending

### Using Different LLM Provider

Modify `llm_client.py` to add new providers:

```python
class NewProviderClient(LLMClient):
    def predict(self, prompt, **kwargs):
        # Implement API call
        pass
```

### Custom Prompts

Modify `DiskFailurePromptBuilder` to customize prompts:

```python
def build_custom_prompt(self, disk_data):
    return f"Custom prompt with {disk_data}"
```

## References

- SiliconFlow API Docs: https://docs.siliconflow.cn/
- DeepSeek Model: https://www.deepseek.com/
- Original StreamDFP: https://github.com/shujiehan/StreamDFP

## License

Same as StreamDFP - GNU General Public License 3.0
