# AIN-710: NLP in the Wild - Sentiment & Confidence Lab

This project is a robust, observable NLP laboratory focused on sentiment classification using OpenAI's Structured Outputs API. It demonstrates best practices for building production-ready LLM integrations, including instrumentation, stability testing, and automated documentation.

## 🚀 Key Features

- **Structured Sentiment Analysis**: Uses OpenAI's latest Responses API with strict JSON schemas to ensure reliable output (label, confidence, and rationale).
- **"Under the Hood" Instrumentation**: Lightweight logging of every request and response to JSONL files for auditability and performance tracking.
- **Batch Processing**: Tools to run sentiment analysis on large datasets with support for repeated runs to test model stability.
- **Confidence & Stability Metrics**: Advanced analysis of results to identify "flipping" labels and low-confidence predictions.
- **Developer Experience**: PEP 257 compliant docstrings, modular code structure, and comprehensive smoke tests.

## 📁 Project Structure

```text
.
├── data/                  # Input datasets (JSONL format)
├── notebooks/             # Interactive exploration and labs
├── outputs/               # Generated logs and result files
│   ├── runs.jsonl         # Detailed event logs (requests, responses, errors)
│   ├── sentiment_results.jsonl  # Raw classification results
│   └── confidence_summary.json  # Aggregated stability metrics
├── src/
│   ├── lab/               # Core library modules
│   │   ├── client.py      # OpenAI client & config management
│   │   ├── sentiment.py   # Single-item classification logic
│   │   ├── sentiment_batch.py # Batch processing engine
│   │   ├── confidence.py  # Statistical analysis utilities
│   │   └── logging_utils.py # JSONL logging and timing helpers
│   └── scripts/           # Executable entry points
│       ├── run_batch.py   # Execute batch sentiment runs
│       ├── run_confidence.py # Generate stability reports
│       └── test_*.py      # Smoke tests and environment verification
├── .env                   # Local environment variables (not committed)
└── README.md              # Project documentation
```

## 🛠️ Setup & Installation

### 1. Prerequisites
- Python 3.9+
- OpenAI API Key

### 2. Install Dependencies
```powershell
# Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\activate

# Install required packages
pip install openai python-dotenv
```

### 3. Configuration
Create a `.env` file in the project root:
```env
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4o
PROJECT_NAME=ain710-nlp-lab
```

## 📈 Usage

### Environment Verification
Run the smoke tests to ensure your API key and logging are working:
```powershell
python -m src.scripts.test_env
python -m src.scripts.test_log
```

### Running Batch Sentiment
To classify a list of texts and save results:
```powershell
python -m src.scripts.run_batch
```

### Analyzing Confidence & Stability
After running a batch, generate a stability summary:
```powershell
python -m src.scripts.run_confidence
```

## 🔍 Observability
All interactions are logged to `outputs/runs.jsonl`. Each record includes:
- **Timestamp (UTC)**
- **Run ID**: Unique identifier for the session.
- **Payload**: Redacted request/response data (no API keys leaked).
- **Extra**: Performance metrics like `elapsed_ms`.

## 🛡️ Best Practices
This project adheres to:
- **PEP 8**: Standard Python style guide.
- **PEP 257**: Docstring conventions.
- **Security**: Automatic redaction of secrets in log files.
- **Modularity**: Separation of concerns between client, logic, and scripts.
