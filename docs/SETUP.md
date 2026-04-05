# Quick Setup Guide

## 🚀 Getting Started in 3 Steps

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Configure Environment
```bash
# Copy the example file
cp .env.example .env

# Edit .env with your credentials
# Required variables:
# - ADLS_ACCOUNT_NAME
# - ADLS_ACCOUNT_KEY
# - ADLS_CONTAINER_NAME
# - ADLS_INPUT_PATH
# - SEARCH_ENDPOINT
# - SEARCH_KEY
# - INDEX_NAME
```

### Step 3: Run Pipeline
```bash
# Option 1: Use the run script (recommended)
chmod +x run.sh
./run.sh

# Option 2: Run directly
python pipeline.py
```

## 📂 Project Structure

```
.
├── pipeline.py              # 🎯 Main orchestrator (START HERE)
├── config.py                # ⚙️  All configuration settings
├── .env                     # 🔐 Your credentials (create this)
├── requirements.txt         # 📦 Python dependencies
├── run.sh                   # 🏃 Quick start script
│
├── Core Components:
│   ├── adls_fetcher.py     # Fetch from Azure Data Lake
│   ├── legal_text_cleaner.py   # Clean legal text
│   ├── semantic_chunker.py     # Semantic chunking
│   ├── semantic_processor.py   # Batch processor
│   └── search_uploader.py      # Upload to AI Search
│
├── Utilities:
│   ├── config_loader.py        # Config loader (legacy)
│   ├── create_index.py         # Standalone index creator
│   └── generate_embeddings.py  # Standalone embedder
│
└── Documentation:
    ├── README.md               # Complete documentation
    ├── ADDITIONAL_COMPONENTS.md  # What to build next
    └── SETUP.md               # This file
```

## 🔧 Configuration Overview

All settings are in `config.py` but can be overridden in `.env`:

### Must Configure (Required)
```env
ADLS_ACCOUNT_NAME=your_storage_account
ADLS_ACCOUNT_KEY=your_key
ADLS_CONTAINER_NAME=raw
ADLS_INPUT_PATH=raw/newapp/dataset=bill/bill/queensland/1992

SEARCH_ENDPOINT=https://your-search.search.windows.net
SEARCH_KEY=your_admin_key
INDEX_NAME=legal-documents-index
```

### Optional Tuning
```env
# Processing
BATCH_SIZE=10               # Docs per batch
NUM_WORKERS=4              # Parallel workers
MAX_DOCUMENTS=100          # Limit for testing

# Chunking
SIMILARITY_THRESHOLD=0.8   # Semantic similarity (0-1)
TOP_K_CHUNKS=5            # Select top K chunks per doc

# Performance
EMBEDDING_BATCH_SIZE=32    # Embedding batch size
SEARCH_UPLOAD_BATCH_SIZE=100  # Upload batch size
```

## ✅ Verification

### Test ADLS Connection
```python
python -c "
from adls_fetcher import ADLSFetcher
from dotenv import load_dotenv
import os

load_dotenv()
fetcher = ADLSFetcher(
    os.getenv('ADLS_ACCOUNT_NAME'),
    os.getenv('ADLS_ACCOUNT_KEY'),
    os.getenv('ADLS_CONTAINER_NAME')
)
files = fetcher.list_files(os.getenv('ADLS_INPUT_PATH', ''))
print(f'✓ Found {len(files)} files')
"
```

### Test Search Connection
```python
python -c "
from search_uploader import SearchIndexManager
from dotenv import load_dotenv
import os

load_dotenv()
manager = SearchIndexManager(
    os.getenv('SEARCH_ENDPOINT'),
    os.getenv('SEARCH_KEY')
)
exists = manager.index_exists(os.getenv('INDEX_NAME', 'legal-documents-index'))
print(f'✓ Connected to Azure Search')
"
```

## 🎯 Running Modes

### Full Production Run
```bash
# Default mode - processes everything
python pipeline.py
```

### Test Mode (Limited Documents)
```bash
# Set in .env:
MAX_DOCUMENTS=10
UPLOAD_TO_SEARCH=false

python pipeline.py
```

### Dry Run (No Upload)
```bash
# Set in .env:
UPLOAD_TO_SEARCH=false
SAVE_INTERMEDIATE=true

python pipeline.py
```

## 📊 Output Files

After running, check these locations:

```
output/
├── intermediate/
│   ├── all_chunks.json          # All generated chunks
│   └── top_k_chunks.json        # Top-K selected chunks
│
└── final/
    └── processing_stats.json     # Pipeline statistics

pipeline.log                       # Detailed execution log
```

## 🐛 Common Issues

### "Configuration errors: ADLS_ACCOUNT_NAME not set"
→ Create `.env` file with required variables

### "Failed to connect to ADLS"
→ Check account name, key, and container name

### "Out of memory"
→ Reduce `BATCH_SIZE` or `EMBEDDING_BATCH_SIZE`

### "Upload timeout"
→ Reduce `SEARCH_UPLOAD_BATCH_SIZE`

## 🔄 What Happens When You Run

1. **Validation** - Checks configuration
2. **ADLS Fetch** - Downloads JSON documents
3. **Processing** - Cleans, chunks, generates embeddings
4. **Index Creation** - Creates/updates search index
5. **Upload** - Uploads to Azure AI Search
6. **Statistics** - Saves performance metrics

## 📝 Next Steps

After successful run:

1. Check `processing_stats.json` for metrics
2. Test search using Azure portal or API
3. Review `ADDITIONAL_COMPONENTS.md` for query engine
4. Build evaluation framework

## 🆘 Getting Help

1. Check `README.md` for detailed documentation
2. Review `pipeline.log` for error details
3. See `ADDITIONAL_COMPONENTS.md` for enhancement ideas
4. Check configuration in `config.py`

## 🎓 Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    Azure Data Lake                       │
│  (JSON documents in nested folder structure)            │
└──────────────┬──────────────────────────────────────────┘
               │ adls_fetcher.py
               ▼
┌─────────────────────────────────────────────────────────┐
│                  Document Processing                     │
│  ┌─────────────────────────────────────────────────┐   │
│  │  1. Clean (legal_text_cleaner.py)               │   │
│  │  2. Chunk (semantic_chunker.py)                 │   │
│  │  3. Embed (sentence-transformers)               │   │
│  │  4. Select Top-K (optional)                     │   │
│  └─────────────────────────────────────────────────┘   │
└──────────────┬──────────────────────────────────────────┘
               │ search_uploader.py
               ▼
┌─────────────────────────────────────────────────────────┐
│              Azure AI Search Index                       │
│  - Vector embeddings (384-dim)                          │
│  - Metadata (jurisdiction, date, citation)              │
│  - Full-text search capability                          │
│  - HNSW algorithm for fast retrieval                    │
└─────────────────────────────────────────────────────────┘
```

## 💡 Pro Tips

1. **Start Small**: Test with `MAX_DOCUMENTS=10` first
2. **Monitor Logs**: Watch `pipeline.log` in real-time with `tail -f pipeline.log`
3. **Save Intermediate**: Keep `SAVE_INTERMEDIATE=true` for debugging
4. **Tune Performance**: Adjust batch sizes based on your hardware
5. **Use Top-K**: Enable `TOP_K_CHUNKS=5` to reduce index size

Happy Processing! 🚀
