# StackOne RAG System

A RAG system that answers questions about StackOne's 7 unified API specs (HRIS, ATS, LMS, IAM, CRM, Marketing, and the StackOne platform API).

## Setup

### Requirements

- [uv](https://docs.astral.sh/uv/getting-started/installation/) for dependency management
- [Docker](https://docs.docker.com/engine/install/) (optional, for containerized runs)
- An OpenAI API key

### Install

```bash
make install
```

### Configure

Copy `.env_example` to `.env` and fill in your values:

```
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o
EMBEDDINGS_MODEL=text-embedding-3-small
COLLECTION_NAME=documents
CHUNK_SIZE=4000
K_NEIGHBORS=5
DEBUG_MODE=false
```

Set `DEBUG_MODE=true` to include token usage in `/chat` responses.

## Running

### Start the server

```bash
make dev-api
```

Or with Docker: `make start-api`

### Load documents

Once the server is running, load the 7 API specs into the vector store:

```bash
curl http://localhost:80/load
```

### Chat

```bash
curl -X POST http://localhost:80/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "How do you authenticate to the StackOne API?"}'
```

### Streamlit frontend

```bash
make start-app
```

## Evaluation

The eval system has two parts: a generation eval (judges the final answer) and a retrieval eval (checks if the right chunks were retrieved).

### Generation eval

Sends each question to the running server, then uses 3 separate LLM judges (correctness, completeness, faithfulness) to score each answer against predefined ground truth. Results are saved as timestamped CSVs in `ai_exercise/evals/results/`.

```bash
make eval
```

Requires the server to be running with documents loaded.

### Retrieval eval

Calls the retrieval functions directly (no server needed, but documents must be loaded in `.chroma_db`). Checks if the expected endpoint chunk appears in the top-k results for each question.

```bash
make eval-retrieval
```

### Test dataset

20 hand-written test cases in `ai_exercise/evals/dataset.jsonl` across 10 categories: auth, params, field_detail, negative, cross_api, ambiguous, nested, create, specific_api, error.

## Testing and code quality

```bash
make test        # pytest
make lint        # ruff
make format      # ruff --fix
make typecheck   # mypy
```

## Project structure

```
ai_exercise/
  main.py                  # FastAPI routes: /health, /load, /chat
  constants.py             # Settings, OpenAI + Chroma clients
  models.py                # Pydantic models
  loading/
    openapi_chunker.py     # Converts OpenAPI specs to natural language chunks
    document_loader.py     # Fetches specs, chunks, splits, stores
  retrieval/
    retrieval.py           # Query rewriting + vector search
    vector_store.py        # Chroma collection setup
  llm/
    completions.py         # LLM generation
    embeddings.py          # OpenAI embedding function
    prompt.py              # RAG prompt + query rewrite prompt
  evals/
    run_eval.py            # Generation eval with LLM judges
    run_retrieval_eval.py  # Retrieval accuracy eval
    prompt.py              # Judge prompts (correctness, completeness, faithfulness)
    dataset.jsonl          # 20 test cases with ground truth
    results/               # Timestamped CSV outputs
```

## Documentation

- [IMPROVEMENTS.md](IMPROVEMENTS.md) -- what I'd improve with more time
