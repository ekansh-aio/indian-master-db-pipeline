# India Legal Data Pipeline

Ingests Indian legal documents from Azure Data Lake Storage (ADLS), chunks and embeds them, and indexes them into a self-hosted Elasticsearch cluster.

## Collections

| Collection | Source | ES Index | Ingest entry point | Chunk+embed entry point |
|---|---|---|---|---|
| High Court (HC) | S3 public bucket → ADLS | `hc_judgements` | `pipelines/hc/ingest.py` | `pipelines/hc/run.py` |
| Supreme Court (SC) | ADLS `app/Supreme_Court_Judgements/` | `sc_judgements` | — | `pipelines/sc/run.py` |
| Central Acts (CA) | indiacode.nic.in scrape | `central_acts` | — | `pipelines/central_acts/run.py` |
| State Acts (SA) | indiacode.nic.in scrape (36 states) | `state_acts` | — | `pipelines/state_acts/run.py` |
| Legal Dictionary | Azure AI Search migration (one-time) | `legal_dictionary` | — | `pipelines/legal_dictionary/run.py` |

## Quick Start

```bash
cp .env.example .env          # fill in ADLS and ES credentials
pip install -r requirements.txt

# HC — two stages (must run ingest first to populate ADLS)
python pipelines/hc/ingest.py --year 2024       # S3 → ADLS
python pipelines/hc/run.py --year-range 2024 2024  # ADLS → ES

# Other collections (ADLS already populated)
python pipelines/sc/run.py --year-range 2020 2026
python pipelines/central_acts/run.py
python pipelines/state_acts/run.py --all-states
```

## Repository Layout

```
masdb_pipeline-ind/
├── pipelines/              # One subdirectory per collection
│   ├── hc/
│   │   ├── ingest.py           # Stage 1: S3 download → PDF extract → ADLS upload
│   │   ├── run.py              # Stage 2: ADLS chunk+embed → ES
│   │   ├── upload_tracker.py   # SQLite upload dedup tracker
│   │   └── upload_tracker.db   # created on first ingest run — tracks uploaded files
│   ├── sc/
│   ├── central_acts/
│   ├── state_acts/
│   └── legal_dictionary/
├── core/                   # Shared infrastructure
├── schemas/                # Elasticsearch index mappings (one per collection)
├── utils/                  # Small shared utilities
├── final_model/            # Fine-tuned RoBERTa role classifier weights
└── logs/                   # Runtime logs
├── config.py               # All config loaded from .env
└── requirements.txt
```

## Credentials

All credentials live in `.env` (never committed). Required variables:

```
ADLS_ACCOUNT_NAME / ADLS_ACCOUNT_KEY / ADLS_CONTAINER_NAME
ES_URL / ES_API_KEY
```

See `.env` for the full list including GPU parallelism tuning.

## Resume Behaviour

**HC ingest (`ingest.py`):** Tracks uploaded files in `hc_upload_tracker.db` (SQLite). Re-runs skip already-uploaded files automatically. Stage 0 refreshes the tracker from ADLS before processing (use `--skip-inventory` to skip this scan on subsequent runs of the same year).

**All other pipelines:** Progress is tracked via ADLS inventory files:
- `_inventory.json` — which ADLS source files have been chunked
- `_inventory_es.json` — which docs have been uploaded to ES

Re-running any pipeline skips already-completed work. Use `--no-resume` to force a full reprocess.
