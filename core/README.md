# core/

Shared infrastructure modules used by all pipelines. Nothing in here is a pipeline entry point.

| File | Purpose |
|---|---|
| `adls_fetcher.py` | Read files from Azure Data Lake Storage (list, stream, read JSON) |
| `adls_uploader.py` | Write files to ADLS (upload JSON, write marker files) |
| `adls_paths.py` | Single source of truth for all ADLS paths and ES index names across collections |
| `elasticsearch_client.py` | ES client setup, index management, bulk upload, GPU embedding helpers |
| `inventory.py` | Load/save `_inventory.json` and `_inventory_es.json`; compute processing deltas |
| `progress_tracker.py` | Append-only JSONL progress log; used by pipelines to track completed doc IDs |
| `role_classifier.py` | Fine-tuned RoBERTa inference — classifies each chunk into one of 9 legal roles |
| `semantic_chunker.py` | Groups sentences into semantically coherent chunks using embedding similarity |
