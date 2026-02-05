# AIN-710: NLP in the Wild - Sentiment & Confidence Lab

This project is a robust, observable NLP laboratory focused on sentiment classification using OpenAI's Structured Outputs API. It demonstrates best practices for building production-ready LLM integrations, including instrumentation, stability testing, and automated documentation.

## 🚀 Key Features

- **Structured Sentiment Analysis**: Uses OpenAI's latest Responses API with strict JSON schemas to ensure reliable output (label, confidence, and rationale).
- **"Under the Hood" Instrumentation**: Lightweight logging of every request and response to JSONL files for auditability and performance tracking.
- **Batch Processing**: Tools to run sentiment analysis on large datasets with support for repeated runs to test model stability.
- **Confidence & Stability Metrics**: Advanced analysis of results to identify "flipping" labels and low-confidence predictions.
- **Embeddings Laboratory**: Specialized tools for converting text into high-dimensional vectors, including L2-norm verification and named entity recognition.
- **Automated Summaries & Narrative**: Generates structured summaries of sentiment and embedding runs, including AI-generated narrative insights for educational analysis.
- **Developer Experience**: PEP 257 compliant docstrings, modular code structure, and comprehensive smoke tests.

## 📁 Project Structure

```text
.
├── data/                  # Input datasets (JSONL format)
├── notebooks/             # Interactive exploration and labs
├── outputs/               # Generated logs and result files
│   ├── evidence/          # Exported figures and tables for reporting
│   ├── runs.jsonl         # Detailed event logs (requests, responses, errors)
│   ├── sentiment_results.jsonl  # Raw classification results
│   ├── confidence_summary.json  # Aggregated stability metrics
│   ├── embedding_results.jsonl  # Raw embedding vectors and metadata
│   └── embeddings_summary.json  # Aggregated embedding metrics and narrative
├── src/
│   ├── lab/               # Core library modules
│   │   ├── client.py      # OpenAI client & config management
│   │   ├── sentiment.py   # Single-item classification logic
│   │   ├── sentiment_batch.py # Batch processing engine
│   │   ├── confidence.py  # Statistical analysis utilities
│   │   ├── embeddings.py  # Embedding generation and summarization
│   │   └── logging_utils.py # JSONL logging and timing helpers
│   └── scripts/           # Executable entry points
│       ├── run_batch.py   # Execute batch sentiment runs
│       ├── run_confidence.py # Generate stability reports
│       ├── run_embeddings.py # Generate text embeddings
│       ├── run_embedding_summary.py # Generate embedding reports
│       └── test_*.py      # Smoke tests and environment verification
├── .env                   # Local environment variables (not committed)
└── README.md              # Project documentation
```

## 🛠️ Setup & Installation

### 1. Prerequisites
- Python 3.9+
- OpenAI API Key

### 2. Install Dependencies
```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/Scripts/activate

# Install required packages
pip install openai python-dotenv
```

### 3. Configuration
Create a `.env` file in the project root:
```env
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4o
EMBEDDING_MODEL=text-embedding-3-small
PROJECT_NAME=ain710-nlp-lab
```

### 4. Choice of Model
- **`OPENAI_MODEL`**: Used for chat completions, sentiment analysis, and extracting entities. The project defaults to `gpt-4o` for high-quality reasoning and structured outputs.
- **`EMBEDDING_MODEL`**: Used for generating numerical vector representations of text. We use `text-embedding-3-small` because:
    - **Efficiency**: It is OpenAI's most cost-effective and fastest embedding model.
    - **Normalization**: It produces unit-normalized vectors (magnitude of 1.0), which simplifies similarity calculations like Cosine Similarity.
    - **Performance**: Despite being "small", it captures complex semantic relationships effectively for most use cases.

When using chat tools and LLMs, choosing the right model is critical. Larger models like `gpt-4o` provide better reasoning but higher cost/latency, while smaller specialized models like `text-embedding-3-small` are optimized for specific mathematical transformations.

## 📈 Usage

### Environment Verification
Run the smoke tests to ensure your API key and logging are working:
```bash
python -m src.scripts.test_env
python -m src.scripts.test_log
```

### Running Batch Sentiment
To classify a list of texts and save results:
```bash
python -m src.scripts.run_batch
```

### Analyzing Confidence & Stability
After running a batch, generate a stability summary:
```bash
python -m src.scripts.run_confidence
```

### Generating Embeddings
To convert text to vectors and extract entities:
```bash
python -m src.scripts.run_embeddings
```

### Analyzing Embeddings
Generate a summary and narrative overview of embedding results:
```bash
python -m src.scripts.run_embedding_summary
```

### Workflow: Suggested Execution Order
To experience the full capabilities of the laboratory, it is recommended to run the scripts in the following order:

1.  **Verification**: `python -m src.scripts.test_env` — Confirms your API key is valid.
2.  **Sentiment Analysis**: `python -m src.scripts.run_batch` — Processes the input data.
3.  **Stability Report**: `python -m src.scripts.run_confidence` — Analyzes the consistency of the sentiment results.
4.  **Vectorization**: `python -m src.scripts.run_embeddings` — Generates embeddings and extracts entities.
5.  **Embedding Summary**: `python -m src.scripts.run_embedding_summary` — Produces the mathematical and narrative report.
6.  **Reporting**: Open `notebooks/01_sentiment_api_lab_export_evidence.ipynb` to generate final visualizations.

### Resetting the Lab
If you want to start over with fresh data, you can clear the contents of the files in the `outputs/` directory (e.g., `runs.jsonl`, `sentiment_results.jsonl`, `embedding_results.jsonl`). 
**Note**: Do not delete the files themselves, just delete the text within them. This ensures the directory structure remains intact.

### Exporting Evidence
Use the Jupyter notebook in `notebooks/` to generate and export figures and tables:
- `01_sentiment_api_lab_export_evidence.ipynb`: Generates visualizations and CSV tables in `outputs/evidence/`.

## 🔍 Observability
All interactions are logged to `outputs/runs.jsonl`. Each record includes:
- **Timestamp (UTC)**
- **Run ID**: Unique identifier for the session.
- **Payload**: Redacted request/response data (no API keys leaked).
- **Extra**: Performance metrics like `elapsed_ms`.

## 📝 Assignment: NLP in the Wild – A Case Study

This repository was created to support the following assignment:

### 5.2 Discussion: NLP in the Wild – A Case Study
**Prompt:**
Natural Language Processing (NLP) is everywhere—from email autocomplete and grammar checkers to chatbots and voice assistants. This week, you’ll explore how NLP works in the real world through a hands-on mini case study.

#### Step 1 – Explore on Your Own (Not Graded, Supports Your Post)
Before writing your post, complete the following activities to build your understanding:

1. **Choose a real-world NLP tool** you use (e.g., Grammarly, Siri, ChatGPT, Google Translate, etc.).
2. **Describe how you believe it works.**
3. **Which NLP tasks is it performing?** (e.g., sentiment analysis, named entity recognition, translation, summarization)
4. **Apply a no-code sentiment analysis tool** (e.g., [LiveChat AI](https://www.livechat.com/ai-links/) or [Formulabot](https://formulabot.com/)) to a short text of your choice.
   - Input example: a movie review, a tweet, an email, or a product description.
   - Note the model’s output (positive, negative, neutral)

This experience will help you answer the prompt more thoughtfully.

## 🛡️ Best Practices
This project adheres to:
- **PEP 8**: Standard Python style guide.
- **PEP 257**: Docstring conventions.
- **Security**: Automatic redaction of secrets in log files.
- **Modularity**: Separation of concerns between client, logic, and scripts.
