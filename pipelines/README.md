# pipelines/

One subdirectory per collection. Each has a `run.py` as the single entry point.

## HC — High Court Judgments

The HC pipeline has two stages that run independently. There are three ways to
get PDFs into ADLS (stages 0a/0b/0c below); only one is needed.

### Stage 0a — Initial PDF mirror: S3 → ADLS (`sync_pdfs.py`)

One-time bulk copy of raw PDFs from the public S3 bucket to ADLS `pdf/`.
Use this to seed ADLS before switching to the self-hosted scraper permanently.

```bash
python pipelines/hc/sync_pdfs.py --year 2024
python pipelines/hc/sync_pdfs.py --year 2024 --court 10_8 --dry-run
```

**ADLS output:** `pdf/High_Court_Judgements/year={Y}/court={C}/bench={B}/{file}.pdf`

---

### Stage 0b — Self-hosted scraper: eCourts → ADLS (`scrape/run.py`)

Scrapes judgments directly from `judgments.ecourts.gov.in` and uploads PDFs +
raw metadata JSON to ADLS. Replaces S3 dependency permanently.

**Prerequisites:**
1. Clone `https://github.com/vanga/indian-high-court-judgments` somewhere.
2. Set `SCRAPER_REPO=/path/to/repo` in `.env` (or pass `--scraper-repo`).
3. `pip install onnx onnxruntime pillow lxml bs4 requests`

```bash
# Scrape one court for a date range
python pipelines/hc/scrape/run.py --court 33~10 --start-date 2024-01-01 --end-date 2024-01-31

# Scrape all 25 courts (parallelised, auto-resume from last scraped date)
python pipelines/hc/scrape/run.py --all-courts --end-date 2024-03-31

# Show resume progress
python pipelines/hc/scrape/run.py --status

# Dry run
python pipelines/hc/scrape/run.py --court 33~10 --start-date 2024-01-01 --end-date 2024-01-07 --dry-run
```

**ADLS output:**
- `pdf/High_Court_Judgements/year={Y}/court={C}/bench={B}/{file}.pdf`
- `pdf/High_Court_Judgements/year={Y}/court={C}/bench={B}/{file}.json` (raw portal HTML metadata)

**Resume:** `scrape/scrape.db` — individual file uploads tracked in `files` table;
per-court last-scraped date in `scrape_progress` table.

**File structure:**
```
scrape/
  run.py          ← entry point (CLI)
  downloader.py   ← CourtScraper: search loop + PDF download + ADLS upload
  session.py      ← ECourtSession: cookies, CAPTCHA solving, request_api
  tracker.py      ← SQLite: files + scrape_progress tables
  scrape.db       ← created on first run (gitignored)
```

---

### Stage 0c — Ingest from S3 (text extraction): S3 → ADLS (`ingest.py`)

Downloads PDFs from the public S3 bucket, extracts text, cleans JSON, and uploads
to ADLS `app/High_Court_Judgements/`. Tracks progress in a local SQLite DB
(`hc_upload_tracker.db`) so re-runs skip already-uploaded files.

```bash
python pipelines/hc/ingest.py --year 2024                         # basic run
python pipelines/hc/ingest.py --year 2024 --use-gpu-ocr           # GPU OCR for scanned PDFs
python pipelines/hc/ingest.py --year 2024 --court 10_8 --dry-run  # test one court
python pipelines/hc/ingest.py --year 2024 --skip-inventory        # skip ADLS scan
python pipelines/hc/ingest.py --year 2024 --batch-size 8 --pdf-workers 96
```

**Data flow (Stage 1):**
S3 `s3://indian-high-court-judgments/data/pdf/year={Y}/court={C}/bench={B}/*.pdf`
→ pypdf extract (GPU OCR fallback for scanned docs)
→ clean JSON (remove duplicate root fields, fix escape sequences)
→ ADLS `app/High_Court_Judgements/year={Y}/court={C}/bench={B}/{doc_id}.json`
→ recorded in `hc_upload_tracker.db`

### Stage 2 — Chunk + Embed: ADLS → ES (`run.py`)

Picks up from ADLS and pushes to Elasticsearch.

```bash
python pipelines/hc/run.py --year-range 1950 2026   # full run
python pipelines/hc/run.py --stage chunk --year-range 2020 2026
python pipelines/hc/run.py --stage embed --year-range 2020 2026
python pipelines/hc/run.py --no-resume --year-range 2024 2026
```

**Data flow (Stage 2):**
ADLS `app/High_Court_Judgements/year={Y}/court={C}/bench={B}/{doc_id}.json`
→ sentence split → GPU encode → semantic chunk → GPU role-classify
→ ADLS `processed/High_Court_Judgements/year={Y}/court={C}/{doc_id}_all_chunks.json`
→ top-12 select → GPU embed → ES `hc_judgements`

### Full HC pipeline for a year

```bash
python pipelines/hc/ingest.py --year 2024
python pipelines/hc/run.py --year-range 2024 2024
```

## SC — Supreme Court Judgments

```bash
python pipelines/sc/run.py --year-range 1950 2026
python pipelines/sc/run.py --stage chunk|embed|all
```

**Data flow:** Identical to HC.
ADLS `app/Supreme_Court_Judgements/{Y}/{doc_id}.json` → ... → ES `sc_judgements`

Note: SC uses bare year directories (`1950/`), not `year=1950/` like HC.

## Central Acts

```bash
python pipelines/central_acts/run.py                     # full: scrape + process + ES
python pipelines/central_acts/run.py --stage scrape
python pipelines/central_acts/run.py --stage process
python pipelines/central_acts/run.py --stage es
python pipelines/central_acts/run.py --stage es --recreate-index
```

**Data flow:**
indiacode.nic.in → download PDFs locally → PyMuPDF extract (Tesseract OCR fallback)
→ semantic chunk + embed
→ ADLS `processed/central_acts/{Y}/{doc_id}.json`
→ ES `central_acts`

PDFs are not stored in ADLS — only the processed JSON chunks are uploaded.

## State Acts

```bash
python pipelines/state_acts/run.py --all-states               # all 36 states
python pipelines/state_acts/run.py --states karnataka,delhi   # specific states
python pipelines/state_acts/run.py --stage scrape --all-states
python pipelines/state_acts/run.py --stage process --all-states --gpu-workers 8
python pipelines/state_acts/run.py --stage es --all-states
```

**Data flow:**
indiacode.nic.in (36 states/UTs) → download PDFs locally
→ Surya OCR (image PDFs) + IndicTrans2 translation (non-English)
→ semantic chunk + embed
→ ADLS `state_acts/processed/state={S}/year={Y}/{doc_id}.json`
→ ES `state_acts`

Processing uses a GPU queue scheduler: states are assigned round-robin across `--gpu-workers` GPUs (default 8). Each GPU processes one state at a time.

PDFs are not stored in ADLS.

## Legal Dictionary

```bash
python pipelines/legal_dictionary/run.py
python pipelines/legal_dictionary/run.py --recreate-index
python pipelines/legal_dictionary/run.py --no-resume
```

**Data flow:**
Azure AI Search `legal-dictionary-index-india` (~5,014 terms)
→ embed definition text → ES `legal_dictionary`

This is a one-time migration from AI Search, not an ongoing ingestion pipeline.
Resume is tracked locally in `pipeline_progress/done_ids_legal_dictionary.jsonl`.
