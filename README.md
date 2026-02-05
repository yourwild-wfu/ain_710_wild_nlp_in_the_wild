# AIN-710: NLP in the Wild - Sentiment & Confidence Lab

This project is a dedicated, hands-on laboratory for the AIN-710 "NLP in the Wild" assignment. It provides a real-world testing ground for exploring how modern AI understands human emotion and language. By using professional-grade tools, students can observe the inner workings of AI integrations, test their reliability, and learn how to manage AI behavior through configuration rather than complex code.

## 🚀 Executive Summary of Features

- **Reliable Sentiment Analysis**: Automatically categorizes text as positive, negative, or neutral with high precision, providing both a confidence score and a clear "reasoning" for every decision.
- **Operational Transparency**: Every single interaction with the AI is recorded in detail. This allows for full auditability—seeing exactly what went in and what came out—similar to a black-box flight recorder for AI.
- **Stress Testing & Stability**: Unlike simple tools, this lab can process large batches of text multiple times to see if the AI is consistent. It identifies "flipping" labels and low-confidence guesses to ensure the results are trustworthy.
- **AI "Deep Thinking" (Embeddings)**: A specialized lab that shows how AI converts words into mathematical "vectors." It includes tools to extract key entities (people, places, things) and mathematically proves how the AI organizes information.
- **Automated Insights**: The system doesn't just produce numbers; it generates narrative explanations of its findings, acting as an automated "AI Educator" to help students interpret the data.
- **Plug-and-Play Experimentation**: Students can change the AI's "personality" or "rules" simply by editing a settings file. This makes it easy to experiment with "Prompt Engineering" without needing to be a programmer.

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
│       ├── run_sentiment_batch.py   # Execute batch sentiment runs
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
  - **Note on Costs**: If you have a paid ChatGPT Plus account, you can generate an API key and use these features for a nominal fee. For the purposes of this class and the "NLP in the Wild" assignment, experimenting with this laboratory will typically cost **less than $1.00 USD**.
  - **Pricing Reference**: You can find detailed pricing information at the [OpenAI Pricing Page](https://platform.openai.com/docs/pricing).
  - **Cost Formula**: To estimate the cost for 1,000 tokens, use this formula: `(Price per 1M tokens / 1,000)`.
    - *Example (gpt-4o)*: If the price is $2.50 per 1M input tokens, then 1,000 tokens = `$2.50 / 1,000 = $0.0025`.
    - *Example (embeddings)*: If `text-embedding-3-small` is $0.02 per 1M tokens, then 1,000 tokens = `$0.02 / 1,000 = $0.00002`.

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

# Optional sentiment defaults
SENTIMENT_TEMPERATURE=0.2
SENTIMENT_MAX_OUTPUT_TOKENS=200
SENTIMENT_INCLUDE_LOGPROBS=true

# Input strings for labs
TEST_TEXT="I love this new feature!"
# Note: Separate each text with a semicolon (;)
BATCH_TEXTS="\"I love the design, but the setup was frustrating.\";\"This was a complete waste of time.\""
EMBEDDING_TEXTS="\"The quick brown fox jumps over the lazy dog.\";\"I love NLP.\""
```

### 4. Choice of Model
- **`OPENAI_MODEL`**: Used for chat completions, sentiment analysis, and extracting entities. The project defaults to `gpt-4o` for high-quality reasoning and structured outputs.
- **`EMBEDDING_MODEL`**: Used for generating numerical vector representations of text. We use `text-embedding-3-small` because:
    - **Efficiency**: It is OpenAI's most cost-effective and fastest embedding model.
    - **Normalization**: It produces unit-normalized vectors (magnitude of 1.0), which simplifies similarity calculations like Cosine Similarity.
    - **Performance**: Despite being "small", it captures complex semantic relationships effectively for most use cases.

When using chat tools and LLMs, choosing the right model is critical. Larger models like `gpt-4o` provide better reasoning but higher cost/latency, while smaller specialized models like `text-embedding-3-small` are optimized for specific mathematical transformations.

### 5. Prompts & Engineering
All system instructions and key behavior used by the laboratory are configurable via the `.env` file. This allows students to experiment without modifying the Python code:

- **Regular Expressions (Regex)**: The lab uses regex to handle complex text processing:
  - **Parsing**: `src/lab/client.py` uses a specialized regex pattern `r'"((?:\\.|[^"\\])*)"'` to find quoted strings while respecting "escaped" characters (like `\"`). This allows you to include semicolons and quotes within your batch inputs.
  - **Security**: `src/lab/logging_utils.py` uses regex to automatically find and "redact" OpenAI API keys from logs, ensuring your secrets are never saved to disk.
- **Prompts**:
  - **`SENTIMENT_PROMPT`**: Defines how the model should classify text sentiment.
  - **`ENTITY_EXTRACTION_PROMPT`**: Instructions for identifying people, places, and things in text.
  - **`NARRATIVE_PROMPT`**: The template used to generate educational summaries of embedding runs.
- Sentiment behavior (defaults, per-run overridable):
  - **`SENTIMENT_TEMPERATURE`** (default 0.2)
  - **`SENTIMENT_MAX_OUTPUT_TOKENS`** (default 200)
  - **`SENTIMENT_INCLUDE_LOGPROBS`** (default true)
- Customizable Inputs:
  - **`TEST_TEXT`**: The text used for single-item tests (e.g., `test_single_sentiment.py`).
  - **`BATCH_TEXTS`**: A list of texts used for batch sentiment processing (`run_sentiment_batch.py`).
  - **`EMBEDDING_TEXTS`**: A list of texts used for embedding generation (`run_embeddings.py`).
  - **Formatting**: Separate texts with a semicolon (`;`). To include a semicolon within a text or ensure robust parsing, wrap each entry in escaped double quotes inside a quoted string: `"\"text1\";\"text2\""`.

Students are encouraged to modify these in `.env` to see how the model's behavior changes!

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
python -m src.scripts.run_sentiment_batch
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
2.  **Sentiment Analysis**: `python -m src.scripts.run_sentiment_batch` — Processes the input data.
3.  **Stability Report**: `python -m src.scripts.run_confidence` — Analyzes the consistency of the sentiment results.
4.  **Vectorization**: `python -m src.scripts.run_embeddings` — Generates embeddings and extracts entities.
5.  **Embedding Summary**: `python -m src.scripts.run_embedding_summary` — Produces the mathematical and narrative report.
6.  **Reporting**: Open `notebooks/01_sentiment_api_lab_export_evidence.ipynb` to generate final visualizations.

### Starting Fresh & Resetting the Lab
If you want to clear your experimental history and start with a "clean slate"—perhaps after changing your input strings or tweaking your prompts—follow these guidelines:

1.  **Clear Results Files (Content Only)**:
    - Open the files in the `outputs/` directory (e.g., `runs.jsonl`, `sentiment_results.jsonl`, `embedding_results.jsonl`).
    - Delete all the text inside these files and save them.
    - **Note**: Do not delete the files themselves, just the text within them. This ensures the directory structure remains intact and the scripts can immediately write new data.

2.  **Clear Summary Reports**:
    - You can also clear `confidence_summary.json` and `embeddings_summary.json` in the same way. These files are overwritten by the summary scripts, but clearing them ensures you don't accidentally view old data.

3.  **Clear Evidence Folder**:
    - You can safely delete the contents of the `outputs/evidence/` folder (such as the `.csv` and `.png` files).
    - These files are "snapshots" generated by the Jupyter notebook. Deleting them ensures that your next notebook run produces fresh charts and tables based strictly on your newest data.

By clearing these files, you avoid mixing old results with new experiments, ensuring your stability reports and visualizations are accurate and easy to interpret.

### Exporting Evidence & Final Reporting
The Jupyter notebook located in the `notebooks/` directory is the final stage of the laboratory workflow:
- **Source Data**: The notebook (`01_sentiment_api_lab_export_evidence.ipynb`) reads the raw results from `outputs/sentiment_results.jsonl` and the aggregated metrics from `outputs/confidence_summary.json`.
- **Analysis**: It performs data cleaning, calculates additional distributions, and prepares the data for visualization.
- **Evidence Generation**: When executed, the notebook automatically generates and saves CSV tables and high-resolution figures (charts) to the `outputs/evidence/` directory. These files are intended to be used as evidence for your class discussion posts and assignments.

By using the notebook, you transform raw AI logs into professional, easy-to-read reports that demonstrate your findings.

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
