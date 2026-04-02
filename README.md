# StreamDFP-2.2.0

StreamDFP is a general stream mining framework for disk failure prediction with concept-drift adaptation. It includes feature extraction, labeling of samples, as well as training of a prediction model.

StreamDFP is designed to support a variety of learning algorithms, based on three key techniques: online labeling, concept-drift-aware training, and general prediction.

## What's New in StreamDFP-2.2.0

### 🚀 Major Update: LLM Predictor (New)

StreamDFP-2.2.0 introduces a revolutionary **LLM-based prediction module** that leverages Large Language Models for disk failure prediction, providing an alternative to traditional machine learning approaches.

**Key Features:**
- **SiliconFlow API Integration**: Powered by DeepSeek-V3.2 model
- **Natural Language Reasoning**: LLM provides interpretable predictions with explanations
- **Zero Training Required**: Uses in-context learning instead of incremental training
- **Drop-in Replacement**: Compatible with existing pyloader preprocessing pipeline

### Architecture Comparison

```
Traditional StreamDFP:                    LLM-Enhanced StreamDFP:
┌─────────────┐    ┌─────────────┐        ┌─────────────┐    ┌─────────────┐
│  Raw Data   │───→│  pyloader   │        │  Raw Data   │───→│  pyloader   │
│ (SMART logs)│    │(preprocess) │        │ (SMART logs)│    │(preprocess) │
└─────────────┘    └──────┬──────┘        └─────────────┘    └──────┬──────┘
                          │                                          │
                          ▼                                          ▼
                   ┌─────────────┐                            ┌─────────────┐
                   │ MOA/Java    │                            │  LLM        │
                   │(ARF/MLP/RNN)│                            │ Predictor   │
                   └─────────────┘                            └─────────────┘
```

### Historical Versions

- **StreamDFP-2.0.0**: Online transfer learning for minority disk models
- **StreamDFP-2.1.0**: Multilayer Perceptron (MLP) with backpropagation + SSD dataset support
- **StreamDFP-2.2.0**: 
  - Recurrent Neural Network (RNN) with BPTT
  - **LLM Predictor** (SiliconFlow + DeepSeek-V3.2)

---

## Prerequisites

- Python 3.x with numpy, pandas, requests, pyyaml
- Java: jdk-1.8.0 (for MOA-based methods)
- SiliconFlow API Key (for LLM Predictor)

```bash
pip install numpy pandas requests pyyaml
```

---

## Quick Start

### Option 1: LLM Predictor (Recommended for Quick Prototyping)

```bash
# 1. Set API key
export SILICONFLOW_API_KEY="your-api-key-here"

# 2. Preprocess data
cd pyloader
python run.py -s "2015-01-30" -a 6 -p "~/trace/smart/all/" \
    -r "../pyloader/train/" -e "../pyloader/test/" \
    -c "features_erg/hi7_all.txt" -d "HDS722020ALA330" -i 10
cd ..

# 3. Run LLM prediction
python llm_predictor/predictor.py \
    -s "2015-01-30" -p "./pyloader/train/" -t "./pyloader/test/" \
    -i 10 -o "hi7_llm/results.json"

# 4. Parse results
python llm_predictor/parse_results.py hi7_llm/results.json
```

### Option 2: Traditional MOA-based Methods

```bash
# Using shell script
./run_hi7.sh          # Adaptive Random Forest
./run_hi7_rnn.sh      # Recurrent Neural Network
./run_mc1_mlp.sh      # Multilayer Perceptron (SSD)
```

---

## LLM Predictor Detailed Guide

### Setup

1. **Install Dependencies**
   ```bash
   pip install pandas numpy requests pyyaml
   ```

2. **Configure API Key**
   ```bash
   export SILICONFLOW_API_KEY="your-api-key-here"
   ```

3. **Test Connection**
   ```bash
   python llm_predictor/test_api.py
   ```

### Configuration

Edit `llm_predictor/config.yaml`:

```yaml
api:
  provider: "siliconflow"
  model: "deepseek-ai/DeepSeek-V3.2"
  timeout: 60
  max_retries: 3

prediction:
  threshold: 0.5          # Classification threshold
  validation_window: 30   # Evaluation window
  bl_delay: true          # Enable delay evaluation

data:
  batch_size: 10          # Disks per API call
  use_compression: true   # Reduce payload size
  compression_method: "gzip"
```

### Command Line Arguments

| Argument | Short | Description | Default |
|----------|-------|-------------|---------|
| `--start-date` | `-s` | Start date (YYYY-MM-DD) | Required |
| `--train-path` | `-p` | Training data path | Required |
| `--test-path` | `-t` | Test data path | None |
| `--iterations` | `-i` | Number of iterations | 30 |
| `--api-key` | `-k` | SiliconFlow API key | env var |
| `--model` | `-m` | Model name | DeepSeek-V3.2 |
| `--threshold` | | Classification threshold | 0.5 |
| `--batch-size` | | Batch size for API | 10 |
| `--output` | `-o` | Output file path | None |

### Using Shell Script

```bash
chmod +x run_hi7_llm.sh
export SILICONFLOW_API_KEY="your-api-key"
./run_hi7_llm.sh
```

---

## Method Comparison

| Aspect | MOA Java (ARF) | LLM Predictor |
|--------|----------------|---------------|
| **Algorithm** | Adaptive Random Forest | DeepSeek-V3.2 |
| **Training** | Incremental learning | In-context learning |
| **Latency** | Low (local) | Higher (API) |
| **Interpretability** | Limited | High (natural language) |
| **Cost** | Free | Per API call |
| **Setup** | Java + Maven | Python + API Key |
| **Best For** | Production, large scale | Prototyping, explainability |

---

## Dataset

### HDD Models (Backblaze)
- Seagate ST3000DM001, ST4000DM000, ST12000NM0007
- Seagate ST8000DM002, ST8000NM0055
- HGST HMS5C4040BLE640
- Seagate ST31500541AS, ST4000DX000
- Hitachi HDS722020ALA330, HDS5C3030ALA630, HDS723030ALA640

### SSD Models (Alibaba)
- MA1, MB1, MC1

---

## Usage Examples

### Example 1: LLM Prediction (Hitachi HDS722020ALA330)

```bash
# Preprocess
cd pyloader
python run.py -s "2015-01-30" -a 6 -p "~/trace/smart/all/" \
    -r "../pyloader/train/" -e "../pyloader/test/" \
    -c "features_erg/hi7_all.txt" -d "HDS722020ALA330" -i 10
cd ..

# Predict with LLM
export SILICONFLOW_API_KEY="your-api-key"
python llm_predictor/predictor.py \
    -s "2015-01-30" -p "./pyloader/train/" -t "./pyloader/test/" \
    -i 10 --threshold 0.5 --batch-size 10 \
    -o "hi7_llm/results.json"

# Parse results
python llm_predictor/parse_results.py hi7_llm/results.json
```

**Expected Output:**
```
days        FP          FPR         F1-score    Precision   Recall
10.487608   6.658536    0.678112    39.982908   30.785494   57.017281
```

### Example 2: Traditional ARF Classification

```bash
# Preprocess
cd pyloader
bash run_hi7_loader.sh
cd ..

# Train and predict
bash run_hi7.sh

# Parse results
python parse.py hi7_example/example.txt
```

### Example 3: RNN Prediction

```bash
cd pyloader
bash run_hi7_loader.sh
cd ..
bash run_hi7_rnn.sh
python parse.py hi7_rnn/example.txt
```

### Example 4: Online Transfer Learning

```bash
cd pyloader
bash run_hi7_loader_pre.sh      # Source model
bash run_hi640_transfer_loader.sh  # Target model
cd ..
bash run_hi640_transfer.sh
python parse.py hi640_transfer/example.txt
```

---

## Project Structure

```
streamdfp-2.2.0/
├── pyloader/                   # Python preprocessing
│   ├── run.py                  # Main preprocessing script
│   ├── core_utils/             # Core utilities
│   ├── utils/                  # Helper functions
│   └── *.sh                    # Loader scripts
├── moa/                        # Java MOA framework
│   └── src/main/java/moa/      # Classifiers, drift detection, etc.
├── simulate/                   # Java simulator
│   └── src/main/java/simulate/
├── llm_predictor/              # NEW: LLM-based prediction
│   ├── predictor.py            # Main predictor
│   ├── llm_client.py           # API client
│   ├── data_loader.py          # Data loading
│   ├── compressor.py           # Data compression
│   ├── evaluator.py            # Metrics evaluation
│   ├── config.yaml             # Configuration
│   └── README.md               # Detailed LLM docs
├── *.sh                        # Run scripts
├── parse.py                    # Result parser (classification)
├── parse_reg.py                # Result parser (regression)
└── pom.xml                     # Maven configuration
```

---

## Evaluation Metrics

All methods report the same metrics:

| Metric | Description |
|--------|-------------|
| **Days** | Average days before failure |
| **FP** | False Positives |
| **FPR** | False Positive Rate |
| **F1-score** | Harmonic mean of precision and recall |
| **Precision** | TP / (TP + FP) |
| **Recall** | TP / (TP + FN) |

---

## Troubleshooting

### LLM Predictor

**API Key Not Found:**
```
Error: SILICONFLOW_API_KEY environment variable is not set
```
Solution: `export SILICONFLOW_API_KEY="your-api-key"`

**Data File Not Found:**
```
FileNotFoundError: Data file not found for 2015-01-30
```
Solution: Run pyloader first to generate data files.

**API Rate Limiting:**
Increase retry delay in `config.yaml`:
```yaml
api:
  retry_delay: 2.0
```

### Traditional Methods

**Java Class Not Found:**
```bash
mvn clean package
```

---

## API Costs (LLM Predictor)

SiliconFlow API charges per token:

- Single disk prediction: ~500-1000 tokens
- Batch of 10 disks: ~2000-4000 tokens

Monitor usage in the SiliconFlow dashboard.

---

## Extending

### Adding New LLM Provider

Modify `llm_predictor/llm_client.py`:

```python
class NewProviderClient(LLMClient):
    def predict(self, prompt, **kwargs):
        # Implement API call
        pass
```

### Custom Prompts

Modify `llm_predictor/predictor.py` to customize prompts for specific disk models.

---

## References

- SiliconFlow API: https://docs.siliconflow.cn/
- DeepSeek Model: https://www.deepseek.com/
- MOA Framework: https://moa.cms.waikato.ac.nz/
- Original StreamDFP: https://github.com/shujiehan/StreamDFP
- Backblaze Dataset: https://www.backblaze.com/b2/hard-drive-test-data.html
- Alibaba SSD Dataset: https://github.com/alibaba-edu/dcbrain/tree/master/ssd_smart_logs

---

## License

GNU General Public License 3.0

---

## Contact

Shujie Han (shujiehan@pku.edu.cn)

For LLM Predictor issues, please also check `llm_predictor/README.md`.
