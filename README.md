# OpsRAG — Multi-Source Operations Knowledge & RAG Evaluation Lab

OpsRAG is a local-first RAG lab for operational knowledge. The system separates document preparation, indexing, retrieval, prompting, generation, and evaluation so each stage can be inspected and tested independently.

The implemented path is:

```text
public-source notes / PDFs / TXT / Markdown
        ↓
text extraction + metadata
        ↓
character chunking (500 / 80 by default)
        ↓
Sentence Transformers embeddings
        ↓
Elasticsearch index
        ↓
BM25 / vector / hybrid-RRF retrieval
        ↓
bounded retrieved context
        ↓
baseline or improved prompt
        ↓
local Ollama chat model
        ↓
answer + exact source labels
        ↓
Recall@5 / MRR / token F1 / citation coverage
```

## What is actually in this repository

The repository contains the working source modules from the project build:

- direct Elasticsearch access through the official Python client
- BM25 retrieval over text and metadata
- dense-vector retrieval with Sentence Transformers embeddings
- Elasticsearch-native reciprocal-rank fusion (RRF) for the hybrid path
- local Ollama generation with two prompt variants
- deterministic retrieval and answer metrics
- a 35-question evaluation set with single-source, multi-source, and unanswerable cases
- a small Streamlit interface for querying and viewing measured evaluation output
- unit tests for chunking, retrieval request construction, and evaluation metrics

There is no LangChain or LlamaIndex dependency. Retrieval and generation are explicit calls in the project code.

## Why the project is structured this way

The project is meant to expose the mechanics of a RAG system rather than hide them behind an orchestration framework. The boundaries are deliberate:

| Stage | Implementation | Repository location |
|---|---|---|
| Extraction | PDF, TXT, Markdown | `src/ingestion.py` |
| Chunking | deterministic character windows | `src/chunking.py` |
| Embeddings | Sentence Transformers | `src/embeddings.py` |
| Indexing | Elasticsearch mapping + bulk indexing | `scripts/create_index.py`, `src/ingestion.py` |
| Retrieval | BM25, kNN, native RRF | `src/retrieval.py` |
| Prompting | baseline/improved templates | `src/prompting.py`, `prompts/` |
| Generation | local Ollama chat | `src/generation.py` |
| Evaluation | deterministic metrics | `src/evaluation.py`, `scripts/evaluate.py` |
| UI | Streamlit | `app/streamlit_app.py` |

## Data and evaluation corpus

`data/sample/` contains 10 short, original paraphrase notes. Each note includes a source URL in its header and is intentionally much smaller than the source publication it references. The corpus covers Emergency Operations Centers, ICS, NIMS, incident action planning, resource management, emergency communications, forms, and plain-language public-safety communications.

`data/evaluation/questions.json` contains 35 questions: 34 answerable items with 37 total relevant-chunk references, plus one deliberately unanswerable inventory question (`q035`). The set includes single-source and multi-source questions so retrieval can be evaluated separately from generation.

## Requirements

- Python 3.12+
- Elasticsearch 9.x
- `elasticsearch` Python client 9.5.1
- Ollama
- Sentence Transformers
- Streamlit
- pandas
- pypdf
- pytest for the unit-test suite

The development project was run against a native Elasticsearch 9.5.4 server with the 9.5.1 Python client. The repository is not tied to that machine's install path or certificate location.

## Setup

### 1. Create the Python environment

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

macOS/Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Configure Elasticsearch and Ollama

Copy the safe configuration template:

```powershell
Copy-Item .env.example .env
```

For a loopback HTTP Elasticsearch service, the template defaults are sufficient.

For an authenticated HTTPS Elasticsearch installation, set `ELASTICSEARCH_URL`, authentication values, and a local CA path in `.env`. `ELASTICSEARCH_CA_CERT` is deliberately an environment setting so a certificate file never needs to enter Git.

Start Ollama and set `OLLAMA_MODEL` to a model that is installed locally. The development machine had `qwen3:8b` available; an earlier generation check also succeeded with `qwen3:4b`. Model availability is therefore treated as a local dependency rather than a bundled artifact.

### 3. Create the Elasticsearch index

```bash
python scripts/create_index.py
```

The script asks the configured embedding model for its output dimension and creates the `opsrag_chunks` index with text, metadata, and an indexed `dense_vector` field.

### 4. Ingest the sample collection

```bash
python scripts/ingest_documents.py
```

The ingestion layer accepts `.pdf`, `.txt`, and `.md` files. Text/Markdown files can carry these optional first-line metadata fields:

```text
TITLE: My document
SOURCE: https://example.org/document
TOPIC: emergency planning
SECTION: Operations
```

The default chunk settings are 500 characters with 80 characters of overlap. Change them through `.env` rather than editing source code.

### 5. Run a smoke evaluation

```bash
python scripts/evaluate.py --limit 3
```

That runs both experiment configurations over the first three evaluation questions and writes generated CSVs under `results/`.

For the full set:

```bash
python scripts/evaluate.py
```

The two tracked experiment definitions are:

```text
baseline = BM25 + baseline prompt
improved = Hybrid RRF + improved prompt
```

The result files are intentionally ignored by Git. This keeps generated answers, metrics, and run-specific data out of the public repository.

### 6. Run the unit tests

```bash
pytest -q
```

The unit suite does not require a live Elasticsearch node or Ollama service because the retrieval tests use a fake client and the metric tests are deterministic.

### 7. Start the Streamlit UI

```bash
streamlit run app/streamlit_app.py
```

The **Ask** view exposes BM25, vector, and hybrid retrieval. The **Evaluation** view reads generated CSVs when a live evaluation has populated `results/`.

## Publish to GitHub

The repository is prepared for a normal Git workflow. From the repository root, after creating the empty GitHub repository:

```bash
git init -b main
git add .
git commit -m "Initial OpsRAG repository"
git remote add origin <your-repository-url>
git push -u origin main
```

With GitHub CLI, the repository can also be created and pushed from the same directory:

```bash
gh repo create opsrag --source . --remote origin --push
```

Do not add `.env`, certificates, model files, generated `results/`, or local diagnostic files before pushing. The repository intentionally contains no project `LICENSE` file because no licensing choice was established during development.

## Example query

```text
What are the goals of the National Emergency Communications Plan?
```

The intended answer path is the NECP chunk, with an exact source label such as:

```text
[National Emergency Communications Plan — NECP goals]
```

The exact generated prose depends on the local Ollama model. The repository does not treat an example answer as a benchmark result.

## Evaluation design

The evaluation code reports four primary metrics:

- **Recall@5** — whether the expected relevant chunk IDs appear in the first five retrieved results.
- **MRR** — reciprocal rank of the first relevant chunk.
- **Answer Correctness** — token-level F1 between generated and reference answers.
- **Citation Coverage** — fraction of source labels attached to retrieved relevant chunks that appear verbatim in the answer; this checks label exposure, not semantic support.

The unanswerable question is excluded from Recall@5 and MRR. For that item, citation coverage is satisfied only when the answer uses an insufficient-evidence response.

Failure labels are diagnostic thresholds, not human-judged truth labels: `irrelevant retrieval`, `missing evidence`, `unsupported answer`, `incomplete answer`, and `wrong source`.

See [`docs/evaluation.md`](docs/evaluation.md) for the exact evaluation contract and [`docs/verification.md`](docs/verification.md) for the development evidence record.

## Development verification record

The archived project work contains these concrete checks:

- Elasticsearch server version observed: **9.5.4**.
- Elasticsearch Python client version tuple observed: **9.5.1**.
- A complete sample ingestion run returned **10 documents, 10 chunks, 10 indexed, 0 failures**.
- Elasticsearch `count` then returned **10** indexed documents.
- BM25 retrieval for `emergency communications` returned the `emergency_communications-c0001` chunk.
- Vector retrieval for the same query returned `plain_language_comms` first and `emergency_communications` in the returned set, showing that the two retrieval modes can rank the same small corpus differently.
- A later hybrid probe returned `emergency_communications-c0001`; an earlier hybrid call had errored during debugging, so the repository does not claim an exhaustive live hybrid benchmark.
- An Ollama generation check returned HTTP 200 with `qwen3:4b` configured.

These are development-time observations, not permanently reproducible benchmark claims. Live performance depends on the local Elasticsearch index, embedding model, Ollama model, and runtime environment.

## Limitations

The sample corpus is deliberately small: 10 original source notes, each short enough to remain near the default chunk size. It is a lab dataset, not a production knowledge base.

The evaluation uses deterministic lexical metrics. Token F1 does not replace human assessment, and citation coverage checks label presence rather than semantic support. The retrieval results are also sensitive to chunk boundaries, embedding model choice, and index contents.

## Repository layout

```text
opsrag/
├── README.md
├── .env.example
├── .gitignore
├── requirements.txt
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── sample/
│   │   └── 10 source-note files
│   └── evaluation/
│       └── questions.json
│
├── docs/
│   ├── architecture.md
│   ├── evaluation.md
│   ├── experiment_log.md
│   └── verification.md
│
├── prompts/
│   ├── baseline.txt
│   └── improved.txt
│
├── scripts/
│   ├── create_index.py
│   ├── evaluate.py
│   └── ingest_documents.py
│
├── src/
│   ├── __init__.py
│   ├── chunking.py
│   ├── config.py
│   ├── elasticsearch_client.py
│   ├── embeddings.py
│   ├── evaluation.py
│   ├── generation.py
│   ├── ingestion.py
│   ├── prompting.py
│   └── retrieval.py
│
└── tests/
    ├── conftest.py
    ├── test_chunking.py
    ├── test_evaluation.py
    ├── test_ingestion.py
    ├── test_retrieval.py
    └── test_generation.py
```
