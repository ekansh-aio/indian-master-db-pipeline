# Legal Document Processing Pipeline

End-to-end pipeline for processing legal documents from Azure Data Lake Storage (ADLS) and indexing them in Azure AI Search with semantic embeddings.

## 🎯 Overview

This pipeline performs the following operations:

1. **Fetch** - Retrieves legal documents from ADLS Gen2
2. **Clean** - Removes headers, citations, and formatting noise
3. **Chunk** - Splits documents into semantic chunks using similarity-based grouping
4. **Embed** - Generates vector embeddings using sentence transformers
5. **Index** - Creates/updates Azure AI Search index
6. **Upload** - Uploads chunks with embeddings to Azure AI Search

## 📋 Architecture

```
ADLS (Azure Data Lake Storage)
    │
    ├─ Raw JSON documents in nested folder structure
    │
    ▼
Pipeline Processing
    │
    ├─ Legal Text Cleaner (removes headers, citations, noise)
    ├─ Semantic Chunker (groups sentences by similarity)
    ├─ Embedding Generator (sentence-transformers)
    ├─ Optional: Top-K Selector (selects most relevant chunks)
    │
    ▼
Azure AI Search
    │
    └─ Indexed chunks with vector embeddings for retrieval
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or download the repository
# cd to the project directory

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Create a `.env` file in the project root (copy from `.env.example`):

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```env
# Azure Data Lake Storage
ADLS_ACCOUNT_NAME=your_storage_account
ADLS_ACCOUNT_KEY=your_account_key
ADLS_CONTAINER_NAME=your_container
ADLS_INPUT_PATH=raw/newapp/dataset=bill/bill/queensland/1992

# Azure AI Search
SEARCH_ENDPOINT=https://your-search.search.windows.net
SEARCH_KEY=your_admin_key
INDEX_NAME=legal-documents-index
```

### 3. Run Pipeline

**Simple execution (uses config.py defaults):**

```bash
python pipeline.py
```

That's it! The pipeline will:
- ✓ Fetch documents from ADLS
- ✓ Process and chunk them
- ✓ Generate embeddings
- ✓ Create/update the search index
- ✓ Upload everything to Azure AI Search

## ⚙️ Configuration

All settings are in `config.py` and can be overridden via environment variables.

### Key Settings

**Processing:**
```python
BATCH_SIZE=10              # Documents per batch
NUM_WORKERS=4              # Parallel workers
USE_PARALLEL=true          # Enable parallel processing
```

**Chunking:**
```python
SIMILARITY_THRESHOLD=0.8   # Semantic similarity threshold (0-1)
MIN_SENTENCES_PER_CHUNK=2  # Minimum sentences per chunk
MAX_SENTENCES_PER_CHUNK=10 # Maximum sentences per chunk
TOP_K_CHUNKS=5            # Select top K chunks (optional, None = all)
```

**Embeddings:**
```python
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIMENSIONS=384
```

**Output:**
```python
SAVE_INTERMEDIATE=true     # Save intermediate JSON files
INTERMEDIATE_DIR=output/intermediate
FINAL_DIR=output/final
```

## 📁 Project Structure

```
├── pipeline.py              # Main pipeline orchestrator
├── config.py                # Configuration file
├── .env                     # Environment variables (create this)
├── requirements.txt         # Python dependencies
│
├── adls_fetcher.py         # Fetch documents from ADLS
├── legal_text_cleaner.py   # Clean legal text
├── semantic_chunker.py     # Semantic chunking logic
├── semantic_processor.py   # Document processor (batch mode)
├── search_uploader.py      # Upload to Azure AI Search
├── create_index.py         # Index creation utility
├── generate_embeddings.py  # Standalone embedding generator
│
└── output/                 # Output directory (auto-created)
    ├── intermediate/       # Intermediate processing files
    │   ├── all_chunks.json
    │   └── top_k_chunks.json
    └── final/             # Final statistics
        └── processing_stats.json
```

## 🔧 Advanced Usage

### Test with Limited Documents

Set in `.env`:
```env
MAX_DOCUMENTS=10
```

### Skip Index Upload (Testing Only)

Set in `.env`:
```env
UPLOAD_TO_SEARCH=false
```

### Change Embedding Model

Set in `.env`:
```env
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
EMBEDDING_DIMENSIONS=384
```

### Use Top-K Selection

To process only the most relevant chunks per document:

Set in `.env`:
```env
TOP_K_CHUNKS=5
TOP_K_METHOD=doc_similarity
```

## 📊 Output Files

**Intermediate Outputs** (`output/intermediate/`):
- `all_chunks.json` - All chunks from all documents
- `top_k_chunks.json` - Top-k selected chunks (if enabled)

**Final Outputs** (`output/final/`):
- `processing_stats.json` - Pipeline statistics and metrics

**Logs:**
- `pipeline.log` - Detailed execution log

## 🔍 Index Schema

The Azure AI Search index contains:

**Metadata Fields:**
- `id` (key) - Unique chunk identifier
- `chunk_id` - Chunk identifier
- `version_id` - Document version
- `type` - Document type
- `jurisdiction` - Legal jurisdiction
- `date` - Document date
- `source_file` - Original ADLS file path

**Content Fields:**
- `text` - Chunk text (searchable)
- `citation` - Legal citation (searchable)

**Structural Fields:**
- `start_char` - Start position in document
- `end_char` - End position in document
- `num_sentences` - Number of sentences in chunk

**Similarity Metrics:**
- `avg_similarity` - Average sentence similarity in chunk
- `doc_similarity` - Chunk-to-document similarity score

**Vector Field:**
- `embedding` - 384-dimensional vector embedding

## 🧩 Component Details

### 1. ADLS Fetcher (`adls_fetcher.py`)

Handles all interactions with Azure Data Lake Storage:
- Lists files recursively
- Reads JSON documents
- Supports batch and streaming modes
- Adds source file metadata

### 2. Legal Text Cleaner (`legal_text_cleaner.py`)

Cleans legal documents by removing:
- Law report headers
- Judge names and panels
- Page headers and footers
- Citations and references
- Hyphenated line breaks
- Excess whitespace

### 3. Semantic Chunker (`semantic_chunker.py`)

Advanced chunking with semantic understanding:
- Splits text into sentences
- Generates sentence embeddings
- Groups sentences by cosine similarity
- Computes chunk-to-document similarity
- Supports top-k selection based on relevance

### 4. Search Uploader (`search_uploader.py`)

Manages Azure AI Search operations:
- Creates/updates search index with vector configuration
- Uploads documents in batches
- Handles retries and error recovery
- Validates document structure

### 5. Pipeline Orchestrator (`pipeline.py`)

Main controller that:
- Validates configuration
- Runs all stages in sequence
- Handles errors gracefully
- Generates statistics and reports
- Saves intermediate outputs

## 🏗️ Standard Pipeline Components (Reference)

You have implemented a solid retrieval system. Here are **additional components** that are commonly built in production systems:

### Already Implemented ✓
1. **Data Ingestion** - ADLS fetcher
2. **Text Cleaning** - Legal text cleaner
3. **Semantic Chunking** - Semantic chunker with similarity
4. **Embedding Generation** - Sentence transformers
5. **Vector Index** - Azure AI Search with HNSW
6. **Batch Processing** - Parallel document processing

### Missing Components (Optional Enhancements)

#### 1. **Query Pipeline** (Retrieval)
Create `query_engine.py` for searching indexed documents:
- Hybrid search (vector + keyword)
- Re-ranking with cross-encoders
- Query expansion
- Filters and facets

#### 2. **Monitoring & Observability**
Create `monitoring.py`:
- Pipeline metrics (throughput, latency)
- Data quality checks
- Error tracking and alerting
- Cost monitoring (Azure usage)

#### 3. **Evaluation Framework**
Create `evaluation.py`:
- Retrieval quality metrics (MRR, NDCG, Recall@K)
- A/B testing framework
- Ground truth comparisons
- Embedding quality assessment

#### 4. **Incremental Updates**
Create `incremental_processor.py`:
- Track processed documents
- Process only new/modified files
- Delta indexing
- Version management

#### 5. **Data Validation**
Create `validators.py`:
- Schema validation
- Data quality checks
- Duplicate detection
- Completeness checks

#### 6. **Caching Layer**
Create `cache_manager.py`:
- Cache embeddings for repeated documents
- Query result caching
- Intermediate result caching

#### 7. **Export & Backup**
Create `export_manager.py`:
- Export index to files
- Backup/restore functionality
- Migration tools

#### 8. **UI/API Layer** (If needed later)
- FastAPI/Flask REST API
- Streamlit/Gradio UI for testing
- Swagger documentation

## 🐛 Troubleshooting

### Issue: "Configuration errors: ADLS_ACCOUNT_NAME not set"
**Solution:** Create `.env` file with all required variables

### Issue: "Failed to connect to ADLS"
**Solution:** Verify account name, key, and container name in `.env`

### Issue: "Index creation failed"
**Solution:** Check Search endpoint URL and admin key permissions

### Issue: "Out of memory during embedding generation"
**Solution:** Reduce `EMBEDDING_BATCH_SIZE` or `BATCH_SIZE` in config

### Issue: "Upload failed with timeout"
**Solution:** Reduce `SEARCH_UPLOAD_BATCH_SIZE` in config

### Issue: "Empty documents after cleaning"
**Solution:** Check if source documents have text field, adjust cleaning patterns if needed

## 📈 Performance Tuning

**For faster processing:**
```env
BATCH_SIZE=20
NUM_WORKERS=8
EMBEDDING_BATCH_SIZE=64
USE_PARALLEL=true
```

**For memory-constrained environments:**
```env
BATCH_SIZE=5
NUM_WORKERS=2
EMBEDDING_BATCH_SIZE=16
```

**For large-scale production:**
```env
BATCH_SIZE=50
NUM_WORKERS=16
SEARCH_UPLOAD_BATCH_SIZE=1000
CHECKPOINT_INTERVAL=500
```

## 📝 Logging

Logs are written to both console and `pipeline.log` by default.

**Change log level:**
```env
LOG_LEVEL=DEBUG  # DEBUG, INFO, WARNING, ERROR
```

**Disable file logging:**
```env
LOG_FILE=
```

## 🔐 Security Notes

- Store `.env` file securely (never commit to git)
- Use Azure Key Vault for production secrets
- Rotate access keys regularly
- Use Managed Identity when possible
- Implement network restrictions (firewall rules)

## 🤝 Contributing

This is a backend pipeline designed for developers. Key principles:

1. **No UI dependencies** - Pure Python processing
2. **Configuration-driven** - Most settings in config.py
3. **Minimal command-line arguments** - Use .env instead
4. **Error handling** - Graceful failures with detailed logs
5. **Batch processing** - Efficient parallel execution

## 📄 License

[Your license here]

## 🆘 Support

For issues or questions:
1. Check the troubleshooting section
2. Review logs in `pipeline.log`
3. Verify configuration in `config.py` and `.env`
4. Check Azure portal for service status
